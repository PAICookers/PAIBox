"""Operator IR nodes for chip deployment.

Each :class:`OfflineCoreOp` represents a computation unit that maps to a single
chip offline core (v2.0 or v2.5): a compute operation plus a neuron / activation.
The IR is version-agnostic; the backend handles target-specific lowering.

Node types:

- :class:`SequentialOp` -- compute -> neuron/lut
- :class:`AccumulateOp` -- multi-path compute -> add/sub -> neuron/lut
- :class:`AddOp` -- element-wise add/sub (potential output)
- :class:`StandaloneCompOp` -- compute only (potential output)
- :class:`StandaloneActOp` -- neuron/lut only
"""

import math
from collections.abc import Sequence

import torch
from paicorelib import (
    AddPotentialMode,
    LeakMultiInputMode,
    OutputType,
    PoolingMode,
    SNNMode,
)
from torch import Tensor, nn

from .avgpool_compensation import (
    apply_avgpool_leak_params,
    compensate_avgpool_lut,
    compensate_avgpool_neuron,
)
from .calc_params import LutData, NeuronParams, OfflineCoreParams, OnlineCoreParams
from .core_neuron import CoreNeuronV25
from .ir_base import PAIIRNode

__all__ = [
    "OpNode",
    "OfflineCoreOp",
    "SequentialOp",
    "AccumulateOp",
    "AddOp",
    "ConcatOp",
    "StandaloneCompOp",
    "StandaloneActOp",
    "OnlineCoreOp",
    "CPUOp",
]


def _get_bias(comp: nn.Module) -> Tensor | None:
    """Extract the bias tensor from a compute module, or ``None``."""
    bias = getattr(comp, "bias", None)
    if isinstance(bias, Tensor):
        return bias.data
    return None


def _get_pooling_mode(comp: nn.Module) -> PoolingMode:
    """Infer pooling mode from the compute operation."""
    if isinstance(comp, (nn.MaxPool1d, nn.MaxPool2d)):
        return PoolingMode.MAX
    return PoolingMode.AVERAGE


def _is_avgpool(comp: nn.Module) -> bool:
    """Check if a compute module is an average pooling operation."""
    return isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d))


def _get_pool_window_size(comp: nn.Module) -> int:
    """Return the number of elements in the pooling window (product of kernel dimensions)."""
    assert isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d))
    ks = comp.kernel_size
    if isinstance(ks, int):
        ks = (ks,) * (2 if isinstance(comp, nn.AvgPool2d) else 1)
    return math.prod(ks)


def _apply_avgpool_neuron_compensation(
    params: NeuronParams, comp: nn.Module, act: CoreNeuronV25
) -> NeuronParams:
    """Apply AvgPool-specific neuron parameter compensation.

    Sets leak registers for division-by-shift and, in SNN mode,
    scales thresholds to account for pre-shift accumulation.
    """
    window_size = _get_pool_window_size(comp)
    params = apply_avgpool_leak_params(params, window_size, is_ann=act.lut is not None)
    if act.lut is None:  # SNN mode needs threshold compensation
        decay_input = act.leak_multi_input == LeakMultiInputMode.ENABLE
        params = compensate_avgpool_neuron(
            params, window_size, decay_input, act._original_tau
        )

    return params


def _apply_avgpool_lut_compensation(lut: LutData, comp: nn.Module) -> LutData:
    """Scale LUT thresholds to compensate for AvgPool shift approximation."""
    window_size = _get_pool_window_size(comp)
    return compensate_avgpool_lut(lut, window_size)


class OpNode(nn.Module, PAIIRNode):
    """Abstract base class for all operator IR nodes.

    Provides shape and axis-ordering metadata shared by all operator
    variants (offline core, online core, CPU fallback, etc.).

    Attributes:
        input_shapes: Tensor shapes at each input port.
        output_shape: Output tensor shape.
        input_dims: Axis ordering at each input port.
        output_dims: Output axis ordering.
    """

    def __init__(self) -> None:
        super().__init__()
        super(nn.Module, self).__init__()

        # Shape info, populated during graph construction
        self.input_shapes: list[tuple[int, ...]] = []
        self.output_shape: tuple[int, ...] = ()

        # Axis ordering, populated by DimsProp
        self.input_dims: list[tuple[int, ...]] = []
        self.output_dims: tuple[int, ...] = ()

    def extra_repr(self) -> str:
        parts = [f"name='{self.name}'"]
        if self.input_shapes:
            parts.append(f"input_shapes={self.input_shapes}")
        if self.output_shape:
            parts.append(f"output_shape={self.output_shape}")
        return ", ".join(parts)


class OfflineCoreOp(OpNode):
    """Base class for offline-core operators.

    A single OfflineCoreOp represents the complete computation executed on one
    offline core.  Both v2.0 and v2.5 chips have offline cores with the same
    computational pattern (weight matrix * input + neuron/LUT); the backend
    handles version-specific parameter encoding.

    Attributes:
        core_params: Offline core configuration.
    """

    def __init__(self, core_params: OfflineCoreParams | None = None) -> None:
        super().__init__()
        self.core_params = core_params or OfflineCoreParams()

    def _make_identity_weight(self) -> Tensor:
        """Create identity weight matrix matching output dimensions."""
        assert self.output_shape, "output_shape must be set before accessing weights"
        n = math.prod(self.output_shape[1:])  # exclude batch dim
        return torch.eye(n, dtype=torch.int8)

    @property
    def weights(self) -> list[Tensor] | None:
        """Weight tensors, one per input path.

        Subclasses with compute modules return their raw parameter tensors.
        Weightless ops (pool etc.) return ``None``; pass-through ops
        (StandaloneActOp, AddOp) return identity matrices.
        """
        return [self._make_identity_weight()]

    def extra_repr(self) -> str:
        return super().extra_repr()

    @property
    def neuron_params(self) -> NeuronParams:
        """Neuron configuration for the backend.

        Subclasses with an activation module override this to delegate to
        ``self.act.to_neuron_params()``.  The base implementation returns
        a pass-through configuration (output_type=POTENTIAL).
        """
        return NeuronParams(output_type=OutputType.POTENTIAL)

    @property
    def lut_data(self) -> LutData | None:
        """LUT table data. None for non-LUT ops."""
        return None


class SequentialOp(OfflineCoreOp):
    """Sequential computation: compute -> activation.

    Standard pattern of a compute operation followed by a neuron or LUT
    activation, e.g. ``Conv2d -> LIFNodeV25`` or ``Linear -> LutReLU``.

    Args:
        comp: Compute operation (Conv2d, Linear, MaxPool2d, etc.).
        act: Neuron or LUT activation.
        core_params: Offline core parameters (SNN mode and pooling mode
            are inferred automatically when not provided).
    """

    def __init__(
        self,
        comp: nn.Module,
        act: CoreNeuronV25,
        core_params: OfflineCoreParams | None = None,
    ) -> None:
        core_params = core_params or OfflineCoreParams()
        core_params.snn_mode = act.snn_mode
        core_params.pooling_mode = _get_pooling_mode(comp)

        super().__init__(core_params)
        self.comp = comp
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.comp(x))

    @property
    def weights(self) -> list[Tensor] | None:
        w = getattr(self.comp, "weight", None)
        if isinstance(w, Tensor):
            return [w.data.to(torch.int8)]
        return None

    @property
    def neuron_params(self) -> NeuronParams:
        params = self.act.to_neuron_params(bias=_get_bias(self.comp))
        if _is_avgpool(self.comp):
            params = _apply_avgpool_neuron_compensation(params, self.comp, self.act)
        return params

    @property
    def lut_data(self) -> LutData | None:
        if self.act.lut is None:
            return None

        data = self.act.export_lut()
        assert data is not None

        if _is_avgpool(self.comp):
            data = _apply_avgpool_lut_compensation(data, self.comp)
        return data

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, comp={type(self.comp).__name__}, act={type(self.act).__name__}"


class AccumulateOp(OfflineCoreOp):
    """Multi-path accumulation: comps -> add/sub -> activation.

    Accumulates outputs of multiple compute operations with per-path signs,
    then feeds the result into a neuron / activation.
    E.g. ``Conv_a(x1) + Conv_b(x2) -> LIFNodeV25``.

    Args:
        comps: List of compute operations (one per input path).
        act: Neuron or LUT activation.
        op_signs: Per-path sign. ``(1, 1)`` = add, ``(1, -1)`` = subtract.
        core_params: Offline core parameters.
    """

    def __init__(
        self,
        comps: Sequence[nn.Module],
        act: CoreNeuronV25,
        op_signs: tuple[int, ...] | None = None,
        core_params: OfflineCoreParams | None = None,
    ) -> None:
        if op_signs is None:
            op_signs = (1,) * len(comps)
        if len(op_signs) != len(comps):
            raise ValueError(
                f"'op_signs' length ({len(op_signs)}) != comps length ({len(comps)})"
            )

        core_params = core_params or OfflineCoreParams()
        core_params.snn_mode = act.snn_mode
        core_params.pooling_mode = _get_pooling_mode(comps[0])

        super().__init__(core_params)
        self.comps = nn.ModuleList(comps)
        self.act = act
        self.signs = tuple(op_signs)

    def forward(self, *xs: Tensor) -> Tensor:
        acc: Tensor | None = None
        for sign, op, x in zip(self.signs, self.comps, xs):
            term = sign * op(x)
            acc = term if acc is None else acc + term

        assert acc is not None, "AccumulateOp requires at least one input"
        return self.act(acc)

    @property
    def weights(self) -> list[Tensor] | None:
        result = []
        for comp in self.comps:
            w = getattr(comp, "weight", None)
            if isinstance(w, Tensor):
                result.append(w.data.to(torch.int8))
            else:
                return None
        return result

    @property
    def neuron_params(self) -> NeuronParams:
        fused_bias: Tensor | None = None
        for sign, comp in zip(self.signs, self.comps):
            b = _get_bias(comp)
            if b is not None:
                term = sign * b
                fused_bias = term if fused_bias is None else fused_bias + term

        return self.act.to_neuron_params(bias=fused_bias)

    @property
    def lut_data(self) -> LutData | None:
        return self.act.export_lut()

    def extra_repr(self) -> str:
        ops = ", ".join(type(op).__name__ for op in self.comps)
        return f"{super().extra_repr()}, comps=[{ops}], signs={self.signs}, act={type(self.act).__name__}"


class AddOp(OfflineCoreOp):
    """Element-wise add / subtract.

    Outputs membrane potential (not spikes).
    Maps to ``AddPotentialMode.DIRECT_ADD`` on chip.

    Args:
        op_signs: ``(1, 1)`` for add, ``(1, -1)`` for subtract.
        core_params: Offline core parameters.
    """

    def __init__(
        self,
        op_signs: tuple[int, int] = (1, 1),
        core_params: OfflineCoreParams | None = None,
    ) -> None:
        core_params = core_params or OfflineCoreParams()
        core_params.add_potential = AddPotentialMode.DIRECT_ADD

        super().__init__(core_params)
        self.signs = tuple(op_signs)

    def forward(self, *xs: Tensor) -> Tensor:
        acc: Tensor = torch.zeros([1])
        for sign, x in zip(self.signs, xs):
            acc += sign * x

        return acc

    @property
    def weights(self) -> list[Tensor]:
        eye = self._make_identity_weight()
        return [eye] * len(self.signs)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, signs={self.signs}"


class ConcatOp(OpNode):
    """Order-preserving concatenation along a given dimension.

    A routing-only operation that does not map to any offline core.
    The backend uses the port-ordered predecessor list and ``dim`` to
    determine axon address ranges for the destination core.

    Input ordering is defined by ``dst_port`` on each incoming edge and
    is preserved by :meth:`PAIIRGraph.predecessors`.

    Args:
        dim: Concatenation dimension (typically 1 for channel-dim).
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *xs: Tensor) -> Tensor:
        return torch.cat(xs, dim=self.dim)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"


class StandaloneCompOp(OfflineCoreOp):
    """Standalone compute operation (no activation).

    Contains only a compute operation (Conv2d, Linear, Pool, etc.) and
    outputs membrane potential.

    Args:
        comp: The compute operation.
        core_params: Offline core parameters.
    """

    def __init__(
        self, comp: nn.Module, core_params: OfflineCoreParams | None = None
    ) -> None:
        core_params = core_params or OfflineCoreParams()
        core_params.snn_mode = SNNMode.ANN
        core_params.pooling_mode = _get_pooling_mode(comp)

        super().__init__(core_params)
        self.comp = comp

    def forward(self, x: Tensor) -> Tensor:
        return self.comp(x)

    @property
    def weights(self) -> list[Tensor] | None:
        w = getattr(self.comp, "weight", None)
        if isinstance(w, Tensor):
            return [w.data.to(torch.int8)]
        return None

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, comp={type(self.comp).__name__}"


class StandaloneActOp(OfflineCoreOp):
    """Standalone activation operation (no compute).

    Contains only a neuron or LUT activation function.

    Args:
        act: Neuron or LUT activation.
        core_params: Offline core parameters.
    """

    def __init__(
        self, act: CoreNeuronV25, core_params: OfflineCoreParams | None = None
    ) -> None:
        core_params = core_params or OfflineCoreParams()
        core_params.snn_mode = act.snn_mode

        super().__init__(core_params)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x)

    @property
    def neuron_params(self) -> NeuronParams:
        return self.act.to_neuron_params()

    @property
    def lut_data(self) -> LutData | None:
        return self.act.export_lut()

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, act={type(self.act).__name__}"


class OnlineCoreOp(OpNode):
    """Placeholder for online (learning) core operators.

    Both v2.0 and v2.5 chips have online cores supporting STDP-based
    on-chip learning.  The backend handles version-specific configuration.
    """

    def __init__(self, core_params: OnlineCoreParams | None = None) -> None:
        super().__init__()
        self.core_params = core_params or OnlineCoreParams()


class CPUOp(OpNode):
    """Placeholder for CPU fallback operators.

    Maps to the integrated RISC-V CPU available on v2.5 chips.
    The backend must verify that the target chip has a CPU.
    """

    pass
