"""Operator IR nodes for chip deployment.

Each :class:`OfflineCoreOp` represents a computation unit that maps to a single
chip offline core (v2.0 or v2.5): a compute operation plus a neuron / activation.
The IR is version-agnostic; the backend handles target-specific lowering.

Node types:

- :class:`SequentialOp` -- compute -> neuron/lut
- :class:`AccumulateOp` -- multi-path compute -> add/sub -> neuron/lut
- :class:`StandaloneCompOp` -- compute only (potential output)
- :class:`StandaloneActOp` -- neuron/lut only
- routing ops such as :class:`TransformOp`, :class:`ConcatOp`, and
  :class:`SplitOp`

Add-specific IR nodes live in :mod:`paibox.paiir.ir.add_ops`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import torch
from paicorelib import OutputType, PoolingMode
from torch import Tensor, nn
from torch.nn import functional as F

from .calc_params import LutData, NeuronParams, OfflineCoreParams, OnlineCoreParams
from .core_neuron import CoreNeuronV25
from .ir_base import FormatFlow, PAIIRNode, TensorLayout
from .maxpool_export import (
    MaxPoolExportKind,
    build_identity_lut_data,
    build_identity_lut_neuron_params,
    build_spike_identity_neuron_params,
    is_maxpool_comp,
    refresh_maxpool_export_kind,
)
from .reshape_semantics import materialize_logical_layout
from .signal_domain import SignalDomain

if TYPE_CHECKING:
    from ..pipeline.avgpool.metadata import AvgPoolDeployMetadata

__all__ = [
    "TensorLayout",
    "FormatFlow",
    "OpNode",
    "RoutingOp",
    "OfflineCoreOp",
    "SequentialOp",
    "AccumulateOp",
    "ConcatOp",
    "PadOp",
    "SplitOp",
    "TransformOp",
    "LayoutStage",
    "ShapeStage",
    "StandaloneCompOp",
    "StandaloneActOp",
    "OnlineCoreOp",
    "CPUOp",
]


def _ensure_float(x: Tensor) -> Tensor:
    """Convert int8/uint8 tensor to float32 for PyTorch ops.

    Used for chip-accurate simulation: inputs are quantized (int8/uint8)
    but PyTorch conv/linear require floating-point tensors.
    """
    return x if x.is_floating_point() else x.to(torch.float32)


def _run_comp(comp: nn.Module, x: Tensor) -> Tensor:
    """Execute one compute module with the closest chip-side dtype semantics.

    MaxPool preserves its discrete VALUE-domain code directly on chip, and
    PyTorch supports integer execution for some MaxPool kernels on CPU. Keep
    the incoming dtype where possible so standalone/preceding-value MaxPool
    simulation does not spuriously widen into float.

    Some MaxPool modules are special cases on the current PyTorch CPU build:
    integer ``Byte``/``Char`` inputs raise ``NotImplementedError``. For those
    cases we execute the pool in float32 and cast the exact max values back to
    the original integer dtype.

    Other compute ops such as Conv/Linear still require floating-point tensors
    in PyTorch and therefore use :func:`_ensure_float`.
    """
    if isinstance(comp, (nn.MaxPool1d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
        if x.is_floating_point():
            return comp(x)
        return comp(x.to(torch.float32)).to(x.dtype)

    if isinstance(comp, nn.MaxPool2d):
        return comp(x)
    return comp(_ensure_float(x))


def _prepare_act_input(act: CoreNeuronV25, x: Tensor) -> Tensor:
    """Normalize activation input into a membrane-safe dtype when needed."""
    if act.lut is None and not x.is_floating_point():
        return x.to(torch.int32)
    return x


def _get_bias(comp: nn.Module) -> Tensor | None:
    """Extract the bias tensor from a compute module, or ``None``."""
    bias = getattr(comp, "bias", None)
    if torch.is_tensor(bias):
        return bias.data
    return None


def _get_weight_tensor(comp: nn.Module) -> Tensor | None:
    """Extract the graph-side weight tensor from a compute module.

    The PAIIR conv lowering contract now relies on canonical PyTorch module
    parameters only. Function-form convs are materialized into standard
    ``nn.Conv1d`` / ``nn.Conv2d`` modules before they reach graph-side weight
    queries, so ``weight`` is the only supported source here.
    """
    weight = getattr(comp, "weight", None)
    if torch.is_tensor(weight):
        return weight.data
    return None


def _get_pooling_mode(comp: nn.Module) -> PoolingMode:
    """Infer pooling mode from the compute operation."""
    if is_maxpool_comp(comp):
        return PoolingMode.MAX
    return PoolingMode.AVERAGE


def _tensor_value_range(tensor: Tensor) -> tuple[int, int]:
    """Return integer min/max after applying the same int8 cast used by export paths."""
    qt = tensor.detach().to(torch.int8)
    return int(qt.min().item()), int(qt.max().item())


@dataclass(frozen=True, slots=True)
class LayoutStage:
    """One logical layout materialization step inside a transform op."""

    dims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ShapeStage:
    """One shape-change step inside a transform op."""

    shape_fn: Callable[[torch.Size], torch.Size] | None = None


TransformStage = LayoutStage | ShapeStage


def _apply_transform_stage(x: Tensor, stage: TransformStage) -> Tensor:
    if isinstance(stage, LayoutStage):
        return materialize_logical_layout(x, stage.dims)
    if stage.shape_fn is None:
        return x.flatten()
    return x.reshape(stage.shape_fn(x.shape))


class OpNode(nn.Module, PAIIRNode):
    """Abstract base class for all operator IR nodes.

    Provides shape and axis-ordering metadata shared by all operator
    variants (offline core, online core, CPU fallback, etc.).

    Class attributes:
        __deploy__: Whether this node should be deployed to a chip core.
            Set to False for simulation-only routing transforms.

    Attributes:
        input_layouts: Tensor layouts at each input port.
        output_layouts: Tensor layouts at each output port.
    """

    __deploy__: ClassVar[bool] = True
    input_layouts: tuple[TensorLayout, ...]
    output_layouts: tuple[TensorLayout, ...]

    def __init__(self) -> None:
        super().__init__()
        super(nn.Module, self).__init__()
        self.input_layouts = ()
        self.output_layouts = ()

    @property
    def num_inputs(self) -> int:
        return len(self.input_layouts)

    @property
    def num_outputs(self) -> int:
        return len(self.output_layouts)

    def extra_repr(self) -> str:
        parts = [f"name='{self.name}'"]
        if self.input_layouts:
            parts.append(f"input_layouts={self.input_layouts}")
        if self.output_layouts:
            parts.append(f"output_layouts={self.output_layouts}")
        return ", ".join(parts)


class RoutingOp(OpNode):
    """Base class for routing-only operations.

    Routing operations (concat, reshape, transpose, etc.) do not map to
    any offline core. They are used for:

    1. **Simulation**: Execute tensor transformations for accurate shape
       propagation between compute layers.
    2. **Deployment**: Provide metadata to the backend for correct memory
       layout and axon routing.

    On chip, routing operations are implicit - they affect memory layout
    interpretation but require no actual computation.

    Class attributes:
        __deploy__: False - routing ops are not deployed to any core.

    Subclasses:
    - :class:`TransformOp` - ordered layout / shape reinterpretation
    - :class:`ConcatOp` - concatenation
    - :class:`SplitOp` - split branch selection
    """

    __deploy__: ClassVar[bool] = False
    __format_flow__: ClassVar[FormatFlow] = FormatFlow.PASS_THROUGH
    __tick_depth__: ClassVar[int] = 0


class TransformOp(RoutingOp):
    """Explicit tensor view/layout transform routing op.

    ``TransformOp`` is the general PAIIR carrier for ordered tensor
    reinterpretation steps that do not map to chip compute cores but do affect
    graph-level simulation and backend remap construction.
    """

    stages: tuple[TransformStage, ...]

    def __init__(self, stages: Sequence[TransformStage] = ()) -> None:
        super().__init__()
        self.stages = tuple(stages)

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = _apply_transform_stage(x, stage)
        return x

    def extra_repr(self) -> str:
        parts = [super().extra_repr()]
        if self.stages:
            parts.append(f"stages={self.stages}")
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

    def override_compile_state(self, other: OfflineCoreParams) -> None:
        """Override non-semantic compile-time state from another params object."""
        self.core_params.override_compile_state_from(other)

    def _with_domain_derived_output_type(self, params: NeuronParams) -> NeuronParams:
        """Derive backend-visible output type from propagated frontend domain.

        ``signal_semantics.output_domain`` is the frontend semantic source of truth for whether a
        node emits VALUE- or POTENTIAL-domain data. When that annotation is
        available, keep backend-visible ``output_type`` aligned with it.
        """
        if self.signal_semantics.output_domain is None:
            return params

        if self.signal_semantics.output_domain is SignalDomain.VALUE:
            return replace(params, output_type=OutputType.VALUE)
        else:
            return replace(params, output_type=OutputType.POTENTIAL)

    @property
    def weights(self) -> list[Tensor] | None:
        """Raw parameter weight tensors, one per input path.

        This property is graph-side operator metadata only. It returns explicit
        parameter tensors owned by the op itself and does not synthesize
        backend-expanded connectivity matrices.

        Subclasses with parameter-bearing compute modules return their raw
        parameter tensors. Pooling ops, activation-only ops, and add/pass-through
        ops return ``None``.
        """
        return None

    def get_weight_value_range(self) -> tuple[int, int] | None:
        """Cheap min/max summary for weight-format inference.

        Default offline-core behavior is pass-through-like, so the implicit
        transfer-coefficient range is ``[0, 1]`` without materializing any
        dense identity matrix. This is compile-time export metadata rather than
        the range of :attr:`weights`.
        """
        return 0, 1

    @property
    def neuron_params(self) -> NeuronParams:
        """Neuron configuration for the backend.

        Default: potential output (no neuron/activation).
        """
        return self._with_domain_derived_output_type(
            NeuronParams(output_type=OutputType.POTENTIAL)
        )

    @property
    def lut_data(self) -> LutData | None:
        """LUT table data for ANN mode.

        Default: None (no LUT).
        """
        return None


class SequentialOp(OfflineCoreOp):
    """Sequential computation: compute -> activation.

    Standard pattern of a compute operation followed by a neuron or LUT
    activation, e.g. ``Conv2d -> LIFNodeV25`` or ``Linear -> LutReLU``.

    Args:
        comp: Compute operation (Conv2d, Linear, MaxPool2d, etc.).
        act: Neuron or LUT activation.

    The public constructor derives semantic core parameters from ``comp`` and
    ``act``. Advanced callers that need to preserve prepared compile-time
    state should construct the node normally, then call
    :meth:`OfflineCoreOp.override_compile_state`.
    """

    comp: nn.Module
    act: CoreNeuronV25
    avgpool_deploy_metadata: "AvgPoolDeployMetadata | None"

    def __init__(self, comp: nn.Module, act: CoreNeuronV25) -> None:
        core_params = OfflineCoreParams()
        core_params.snn_mode = act.snn_mode
        core_params.pooling_mode = _get_pooling_mode(comp)

        super().__init__(core_params)
        self.comp = comp
        self.act = act
        self.avgpool_deploy_metadata = None

    def forward(self, x: Tensor) -> Tensor:
        x = _run_comp(self.comp, x)
        return self.act(_prepare_act_input(self.act, x))

    @property
    def weights(self) -> list[Tensor] | None:
        w = _get_weight_tensor(self.comp)
        if torch.is_tensor(w):
            return [w.to(torch.int8)]
        return None

    def get_weight_value_range(self) -> tuple[int, int] | None:
        w = _get_weight_tensor(self.comp)
        if torch.is_tensor(w):
            return _tensor_value_range(w)
        return None

    @property
    def lut_data(self) -> LutData | None:
        """LUT table data for backend export."""
        return self.act.export_lut()

    @property
    def neuron_params(self) -> NeuronParams:
        """Neuron configuration for the backend."""
        return self._with_domain_derived_output_type(
            self.act.to_neuron_params(bias=_get_bias(self.comp))
        )

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

    The public constructor derives semantic core parameters from ``comps`` and
    ``act``. Advanced callers that need to preserve prepared compile-time
    state should construct the node normally, then call
    :meth:`OfflineCoreOp.override_compile_state`.
    """

    def __init__(
        self,
        comps: Sequence[nn.Module],
        act: CoreNeuronV25,
        op_signs: tuple[int, ...] | None = None,
    ) -> None:
        if op_signs is None:
            op_signs = (1,) * len(comps)
        if len(op_signs) != len(comps):
            raise ValueError(
                f"'op_signs' length ({len(op_signs)}) != comps length ({len(comps)})"
            )

        core_params = OfflineCoreParams()
        core_params.snn_mode = act.snn_mode
        core_params.pooling_mode = _get_pooling_mode(comps[0])

        super().__init__(core_params)
        self.comps = nn.ModuleList(comps)
        self.act = act
        self.signs = tuple(op_signs)

    def forward(self, *xs: Tensor) -> Tensor:
        acc: Tensor | None = None
        for sign, op, x in zip(self.signs, self.comps, xs):
            term = sign * _run_comp(op, x)
            acc = term if acc is None else acc + term

        assert acc is not None, "AccumulateOp requires at least one input"
        return self.act(_prepare_act_input(self.act, acc))

    @property
    def weights(self) -> list[Tensor] | None:
        result = []
        for comp in self.comps:
            w = _get_weight_tensor(comp)
            if torch.is_tensor(w):
                result.append(w.to(torch.int8))
            else:
                return None
        return result

    def get_weight_value_range(self) -> tuple[int, int] | None:
        ranges: list[tuple[int, int]] = []
        for comp in self.comps:
            w = _get_weight_tensor(comp)
            if not torch.is_tensor(w):
                return None
            ranges.append(_tensor_value_range(w))

        if not ranges:
            return None

        return min(lo for lo, _ in ranges), max(hi for _, hi in ranges)

    @property
    def lut_data(self) -> LutData | None:
        """LUT table data for backend export."""
        return self.act.export_lut()

    @property
    def neuron_params(self) -> NeuronParams:
        """Neuron configuration with fused bias from all compute ops."""
        fused_bias: Tensor | None = None
        for sign, comp in zip(self.signs, self.comps):
            b = _get_bias(comp)
            if b is not None:
                term = sign * b
                fused_bias = term if fused_bias is None else fused_bias + term

        return self._with_domain_derived_output_type(
            self.act.to_neuron_params(bias=fused_bias)
        )

    def extra_repr(self) -> str:
        ops = ", ".join(type(op).__name__ for op in self.comps)
        return f"{super().extra_repr()}, comps=[{ops}], signs={self.signs}, act={type(self.act).__name__}"


class ConcatOp(RoutingOp):
    """Order-preserving concatenation along a given dimension.

    A routing-only operation that does not map to any offline core.
    The backend uses the port-ordered predecessor list and ``dim`` to
    determine axon address ranges for the destination core.

    Input ordering is defined by ``dst_port`` on each incoming edge and
    is preserved by :meth:`PAIIRGraph.predecessors`.

    Args:
        dim: Concatenation dimension (typically 1 for channel-dim).
    """

    __format_flow__: ClassVar[FormatFlow] = FormatFlow.MERGE

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *xs: Tensor) -> Tensor:
        return torch.cat(xs, dim=self.dim)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"


class PadOp(RoutingOp):
    """Static constant-zero padding routing op.

    ``padding`` follows :func:`torch.nn.functional.pad`: values are specified
    in pairs starting from the last dimension.
    """

    padding: tuple[int, ...]

    def __init__(self, padding: Sequence[int]) -> None:
        super().__init__()
        normalized = tuple(int(p) for p in padding)
        if len(normalized) == 0 or len(normalized) % 2 != 0:
            raise ValueError(
                f"{self.__class__.__name__} padding must contain pairs, got {normalized}"
            )
        if any(p < 0 for p in normalized):
            raise ValueError(
                f"{self.__class__.__name__} only supports non-negative padding, got {normalized}"
            )
        self.padding = normalized

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, self.padding, mode="constant", value=0)

    def output_shape_for(self, input_shape: torch.Size) -> torch.Size:
        if len(self.padding) // 2 > len(input_shape):
            raise ValueError(
                f"{self.__class__.__name__} padding {self.padding} is too long for input rank {len(input_shape)}"
            )

        shape = list(input_shape)
        for pair_idx in range(0, len(self.padding), 2):
            left = self.padding[pair_idx]
            right = self.padding[pair_idx + 1]
            axis = len(shape) - 1 - pair_idx // 2
            shape[axis] += left + right
        return torch.Size(shape)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, padding={self.padding}"


class SplitOp(RoutingOp):
    """Represent one static ``torch.split`` producer with multiple logical outputs.

    ``SplitOp`` is a frontend-only routing placeholder. It keeps the original
    split producer as one graph node. Downstream branch selection is expressed
    on outgoing edges via ``Edge.src_port`` rather than node-local metadata.

    Args:
        sections: Static ``torch.split`` partition spec.
        dim: Split dimension.
    """

    sections: int | tuple[int, ...]

    def __init__(self, sections: int | Sequence[int], dim: int = 0) -> None:
        super().__init__()
        if isinstance(sections, int):
            self.sections = sections
        else:
            self.sections = tuple(sections)

        self.dim = dim

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        sections = (
            list(self.sections) if isinstance(self.sections, tuple) else self.sections
        )
        # Simulation returns the full split result. Downstream consumers pick
        # their branch through the edge's `src_port`.
        return tuple(torch.split(x, sections, self.dim))

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, sections={self.sections}, dim={self.dim}"


class StandaloneCompOp(OfflineCoreOp):
    """Standalone compute operation (no activation).

    Contains only a compute operation (Conv2d, Linear, Pool, etc.) and
    outputs membrane potential.

    Args:
        comp: The compute operation.

    The public constructor derives semantic core parameters from ``comp``.
    Advanced callers that need to preserve prepared compile-time state should
    construct the node normally, then call
    :meth:`OfflineCoreOp.override_compile_state`.
    """

    comp: nn.Module

    def __init__(self, comp: nn.Module) -> None:
        core_params = OfflineCoreParams()
        core_params.pooling_mode = _get_pooling_mode(comp)

        super().__init__(core_params)
        self.comp = comp

    def forward(self, x: Tensor) -> Tensor:
        return _run_comp(self.comp, x)

    @property
    def weights(self) -> list[Tensor] | None:
        w = _get_weight_tensor(self.comp)
        if torch.is_tensor(w):
            return [w.to(torch.int8)]
        return None

    def get_weight_value_range(self) -> tuple[int, int] | None:
        w = _get_weight_tensor(self.comp)
        if torch.is_tensor(w):
            return _tensor_value_range(w)
        return None

    @property
    def neuron_params(self) -> NeuronParams:
        """Backend-visible neuron metadata for standalone compute ops.

        Standalone compute ops have no explicit activation stage, so their
        backend-visible ``output_type`` must be derived from the propagated
        graph semantic domain instead of using a hard-coded default.
        """
        if (
            is_maxpool_comp(self.comp)
            and self.signal_semantics.output_domain is SignalDomain.VALUE
        ):
            kind = refresh_maxpool_export_kind(self)
            if kind in (MaxPoolExportKind.U_SPIKE, MaxPoolExportKind.S_SPIKE):
                return self._with_domain_derived_output_type(
                    build_spike_identity_neuron_params(kind)
                )
            if kind is MaxPoolExportKind.LUT:
                return self._with_domain_derived_output_type(
                    build_identity_lut_neuron_params(
                        self.core_params.input_sign,
                        self.core_params.input_width,
                        self.signal_semantics.known_code_range,
                    )
                )

        return self._with_domain_derived_output_type(
            NeuronParams(output_type=OutputType.POTENTIAL)
        )

    @property
    def lut_data(self) -> LutData | None:
        """Standalone MaxPool may synthesize an identity LUT for wider VALUE code."""
        if (
            is_maxpool_comp(self.comp)
            and self.signal_semantics.output_domain is SignalDomain.VALUE
        ):
            export = refresh_maxpool_export_kind(self)
            if export is MaxPoolExportKind.LUT:
                return build_identity_lut_data(
                    self.core_params.input_sign,
                    self.core_params.input_width,
                    self.signal_semantics.known_code_range,
                )
        return None

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, comp={type(self.comp).__name__}"


class StandaloneActOp(OfflineCoreOp):
    """Standalone activation operation (no compute).

    Contains only a neuron or LUT activation function.

    Args:
        act: Neuron or LUT activation.

    The public constructor derives semantic core parameters from ``act``.
    Advanced callers that need to preserve prepared compile-time state should
    construct the node normally, then call
    :meth:`OfflineCoreOp.override_compile_state`.
    """

    act: CoreNeuronV25

    def __init__(self, act: CoreNeuronV25) -> None:
        core_params = OfflineCoreParams()
        core_params.snn_mode = act.snn_mode

        super().__init__(core_params)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        # Standalone spike neurons integrate into a signed membrane domain
        # even when the predecessor emits unsigned VALUEs (for example
        # split-core AvgPool). Cast here so membrane initialization and
        # FLOOR/negative-threshold handling do not inherit a narrow integer dtype.
        return self.act(_prepare_act_input(self.act, x))

    @property
    def lut_data(self) -> LutData | None:
        """LUT table data for backend export."""
        return self.act.export_lut()

    @property
    def neuron_params(self) -> NeuronParams:
        """Neuron configuration for the backend."""
        return self._with_domain_derived_output_type(self.act.to_neuron_params())

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, act={type(self.act).__name__}"


class OnlineCoreOp(OpNode):
    """Placeholder for online (learning) core operators.

    Both v2.0 and v2.5 chips have online cores supporting STDP-based
    on-chip learning. The backend handles version-specific configuration.
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
