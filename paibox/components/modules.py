import sys
import typing
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np
from paicorelib import TM, HwConfig

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import NotSupportedError, RegisterError, ShapeError
from paibox.types import SpikeType, VoltageType
from paibox.utils import check_elem_unique, shape2num

from .projection import InputProj

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if typing.TYPE_CHECKING:
    from paibox.network import DynSysGroup

__all__ = ["BuildingModule"]

MultiInputsType: TypeAlias = List[SpikeType]  # Type of inputs of `NeuModule`.
BuiltComponentType: TypeAlias = List[Union[SynSys, NeuDyn]]


@dataclass
class ModuleIntf:
    """Module interface. The interface of the module stores the information about where the module  \
        gets input and where it outputs. This information will be used when building the module.
    """

    operands: List[Union[NeuDyn, InputProj]] = field(default_factory=list)
    """TODO can operands be a `NeuModule`?"""
    output: List[SynSys] = field(default_factory=list)
    """A list of synapses."""

    @property
    def num_in(self) -> int:
        return sum(op.num_out for op in self.operands)

    @property
    def num_out(self) -> int:
        return sum(out.num_in for out in self.output)


class BuildingModule:
    module_intf: ModuleIntf

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        """Construct the actual basic components and add to the network. Called in the backend ONLY."""
        raise NotImplementedError

    def register_operands(self, *op: Union[NeuDyn, InputProj]) -> None:
        """Register operands to the interface."""
        self.module_intf.operands.extend(op)

    def unregister_operand(self, op: Union[NeuDyn, InputProj]) -> None:
        """Remove a operand from the interface."""
        self.module_intf.operands.remove(op)

    def register_output(self, syn: SynSys) -> None:
        """Register the output synapses."""
        self.module_intf.output.append(syn)

    def unregister_output(self, syn: SynSys) -> None:
        """Remove an output synapses."""
        self.module_intf.output.remove(syn)

    @property
    def n_op(self) -> int:
        return len(self.module_intf.operands)

    @property
    def n_output(self) -> int:
        return len(self.module_intf.output)


class NeuModule(NeuDyn, BuildingModule):
    __gh_build_ignore__ = True

    n_return: ClassVar[int]
    """#N of outputs."""
    inherent_delay: int = 0
    """Internal delay of the module, relative to the external."""

    def __init__(
        self,
        delay: int,
        tick_wait_start: int,
        tick_wait_end: int,
        unrolling_factor: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.module_intf = ModuleIntf()
        self._delay = delay
        self._tws = tick_wait_start
        self._twe = tick_wait_end
        self._uf = unrolling_factor

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_state(self, *args, **kwargs) -> None:
        return self.reset_memory()

    def get_inputs(self, *args, **kwargs) -> MultiInputsType:
        """Function used to describe getting inputs of the module."""
        raise NotImplementedError

    def spike_func(self, *args, **kwargs):
        """Function used to describe generating output spike of the module."""
        raise NotImplementedError

    @property
    def source(self) -> List[Union[NeuDyn, InputProj]]:
        return self.module_intf.operands

    @property
    def dest(self):
        return self  # will be deprecated at anytime in the future.

    @property
    def target(self):
        return self

    @property
    def external_delay(self) -> int:
        """Equivalent delay relative to the external of the module."""
        return self._delay + self.inherent_delay


class FunctionalModule(NeuModule):
    """Basic functional module. Only used in SNN mode."""

    n_return = 1

    def __init__(
        self,
        *operands: Union[NeuDyn, InputProj],
        shape_out: Tuple[int, ...],
        keep_shape: bool,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("delay", 1)
        kwargs.setdefault("tick_wait_start", 1)
        kwargs.setdefault("tick_wait_end", 0)
        kwargs.setdefault("unrolling_factor", 1)  # TODO Currently, fixed.

        # Unique operands only. Otherwise, multiedge will be created.
        if not check_elem_unique(operands):
            raise RegisterError(
                "duplicate input nodes are not allowed to be connected to modules."
            )

        for op in operands:
            if isinstance(op, FunctionalModule) and op.n_output > 1:
                # TODO Connection between modules with `n_output` > 1.
                raise NotSupportedError(
                    "The connection between the module & the module with output node "
                    "greater than 1 is not supported yet."
                )

        super().__init__(**kwargs, name=name)

        self.keep_shape = keep_shape
        self._shape_out = shape_out
        self.register_operands(*operands)

        # Set memory for only 1 output node.
        # TODO how to handle with more than 1 output nodes
        self.set_memory("_inner_spike", np.zeros((self.num_out,), dtype=np.bool_))
        # Delay registers
        self.set_memory(
            "delay_registers",
            np.zeros(
                (HwConfig.N_TIMESLOT_MAX,) + self._inner_spike.shape, dtype=np.bool_
            ),
        )
        # Set a deque for the `synin` to implement the delay of `inherent_delay` for the module.
        if self.inherent_delay > 0:
            _init_synin = [
                self.n_op
                * [np.zeros(self.module_intf.operands[0].num_out, dtype=np.bool_)]
            ]
        else:
            _init_synin = []

        self.set_memory(
            "synin_deque", deque(_init_synin, maxlen=1 + self.inherent_delay)
        )

    def get_inputs(self) -> MultiInputsType:
        synin = []

        for op in self.module_intf.operands:
            # Retrieve the spike at index `timestamp` of the dest neurons
            if self.is_working():
                if isinstance(op, InputProj):
                    synin.append(op.output.copy())
                else:
                    idx = self.timestamp % HwConfig.N_TIMESLOT_MAX
                    synin.append(op.output[idx].copy())
            else:
                # Retrieve 0 to the dest neurons if it is not working
                synin.append(np.zeros_like(op.spike))

        self.synin_deque.append(synin)  # Append to the right of the deque.

        return self.synin_deque.popleft()  # Pop the left of the deque.

    def update(self, *args, **kwargs) -> Optional[SpikeType]:
        if not self.is_working():
            self._inner_spike = np.zeros((self.num_out,), dtype=np.bool_)
            return None

        synin = self.get_inputs()
        self._inner_spike = self.spike_func(*synin).ravel()

        idx = (self.timestamp + self.delay_relative - 1) % HwConfig.N_TIMESLOT_MAX
        self.delay_registers[idx] = self._inner_spike.copy()

        return self._inner_spike

    @property
    def shape_in(self):
        # TODO Return a tuple (shape_in_of_op1, shape_in_of_op2, ...)?
        raise NotImplementedError

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape_out

    @property
    def num_in(self) -> int:
        return self.module_intf.num_in

    @property
    def num_out(self) -> int:
        return shape2num(self._shape_out)

    @property
    def output(self) -> SpikeType:
        return self.delay_registers

    @property
    def spike(self) -> SpikeType:
        return self._inner_spike

    @property
    def feature_map(self) -> SpikeType:
        return self._inner_spike.reshape(self.varshape)


class FunctionalModule2to1(FunctionalModule):
    """Functional module with two operands."""

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if neuron_a.num_out != neuron_b.num_out:
            raise ShapeError(
                f"two operands must have the same size: {neuron_a.num_out} != {neuron_b.num_out}."
            )

        if keep_shape:
            if neuron_a.shape_out != neuron_b.shape_out:
                raise ShapeError(
                    f"two operands must have the same shape: {neuron_a.shape_out} != {neuron_b.shape_out}.\n"
                    f"When two operands have different shapes, set 'keep_shape=False' and the output will "
                    f"not retain shape information."
                )

            _shape_out = neuron_a.shape_out
        else:
            _shape_out = (neuron_a.num_out,)

        super().__init__(
            neuron_a,
            neuron_b,
            shape_out=_shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)


class TransposeModule(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        shape_in: Tuple[int, ...],
        axes: Optional[Sequence[int]] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if axes is None:
            axes = range(len(shape_in))[::-1]

        axes = tuple(axes)

        if not check_elem_unique(axes):
            raise ValueError(f"repeated axis in transpose.")

        if len(axes) != len(shape_in):
            raise ValueError(f"axes don't match array.")

        if keep_shape:
            shape_out = tuple(shape_in[i] for i in axes)
        else:
            shape_out = (neuron.num_out,)

        self._shape_in = shape_in
        self.axes = axes
        super().__init__(
            neuron, shape_out=shape_out, keep_shape=keep_shape, name=name, **kwargs
        )

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self._shape_in


class FunctionalModule2to1WithV(FunctionalModule2to1):
    """Functional module with two operands.

    NOTE: Compared to `FunctionalModule2to1`, The difference is that we also take the \
        membrane potential voltage(vjt) into consideration.
    """

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(neuron_a, neuron_b, keep_shape, name, **kwargs)

        self.set_memory("_vjt", np.zeros((self.num_out,), dtype=np.int32))
        self.thres_mode = np.full((self.num_out,), TM.NOT_EXCEEDED, dtype=np.uint8)

    def synaptic_integr(self, *args, **kwargs) -> VoltageType:
        """Functions used to describe synaptic integration of the module."""
        raise NotImplementedError

    def update(self, *args, **kwargs) -> Optional[SpikeType]:
        if not self.is_working():
            self._inner_spike = np.zeros((self.num_out,), dtype=np.bool_)
            return None

        synin = self.get_inputs()
        incoming_v = self.synaptic_integr(*synin, self._vjt)
        _is, self._vjt = self.spike_func(incoming_v)
        self._inner_spike = _is.ravel()

        idx = (self.timestamp + self.delay_relative - 1) % HwConfig.N_TIMESLOT_MAX
        self.delay_registers[idx] = self._inner_spike.copy()

        return self._inner_spike

    @property
    def voltage(self) -> VoltageType:
        return self._vjt.reshape(self.varshape)
