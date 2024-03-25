from collections import deque
from dataclasses import dataclass, field
import numpy as np
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

from paicorelib import HwConfig

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import NotSupportedError, ShapeError
from paibox.types import SpikeType
from paibox.utils import check_elem_unique, shape2num

from .projection import InputProj

__all__ = ["BuildingModule"]


@dataclass
class ModuleIntf:
    """Module interface. The interface of a module always records the output that   \
        the module gets from the neuron and the synapses that receives the output   \
        of the module. When building the backend, the component information recorded\
        inside will be used.
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

    def build(self, *args, **kwargs) -> None:
        """Build the actual basic components and add to the network. \
            Only called in the backend.
        """
        raise NotImplementedError

    def register_operands(self, *op: Union[NeuDyn, InputProj]) -> None:
        """Register a operand to the interface."""
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
    # n_arg: ClassVar[int]
    """#N of arguments."""
    n_return: ClassVar[int]
    """#N of outputs."""
    inherent_delay: ClassVar[int] = 0
    """Internal delay of the module, relative to the external."""

    def __init__(
        self,
        module_delay: int,
        module_base_tws: int,
        module_base_twe: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.module_intf = ModuleIntf()
        self._delay = module_delay
        self._tws = module_base_tws
        self._twe = module_base_twe

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_state(self, *args, **kwargs) -> None:
        return self.reset_memory()

    def get_inputs(self, *args, **kwargs):
        """Gather inputs from operands in the interface."""
        raise NotImplementedError

    def op_func(self, *args, **kwargs):
        """Specified function of module."""
        raise NotImplementedError

    @property
    def external_delay(self) -> int:
        """Equivalent delay relative to the external of the module."""
        return self._delay + self.inherent_delay


class FunctionalModule(NeuModule):
    """Functional module. Only used in SNN mode."""

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
        kwargs.setdefault("unrolling_factor", 1)

        for op in operands:
            if isinstance(op, FunctionalModule) and op.n_output > 1:
                # TODO Connection between modules with `n_output` > 1.
                raise NotSupportedError(
                    "The connection between the module & the module with output node "
                    "greater than 1 is not supported yet."
                )

        super().__init__(
            module_delay=kwargs["delay"],
            module_base_tws=kwargs["tick_wait_start"],
            module_base_twe=kwargs["tick_wait_end"],
            name=name,
        )

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

    def _get_inputs(self) -> List[SpikeType]:
        synin = []

        for op in self.module_intf.operands:
            # Retrieve the spike at index `timestamp` of the dest neurons
            if self.is_working:
                if isinstance(op, InputProj):
                    synin.append(op.output.copy())
                else:
                    idx = self.timestamp % HwConfig.N_TIMESLOT_MAX
                    synin.append(op.output[idx].copy())
            else:
                # Retrieve 0 to the dest neurons if it is not working
                synin.append(np.zeros_like(op.spike))

        return synin

    def get_input_and_delay(self) -> List[SpikeType]:
        synin = self._get_inputs()
        self.synin_deque.append(synin)  # Append to the right of the deque.

        return self.synin_deque.popleft()  # Pop the left of the deque.

    def update(self, *args, **kwargs) -> Optional[SpikeType]:
        if not self.is_working:
            self._inner_spike = np.zeros((self.num_out,), dtype=np.bool_)
            return None

        synin = self.get_input_and_delay()
        self._inner_spike = self.op_func(*synin)

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
        return self._inner_spike.reshape(self._shape_out)


class FunctionalModule2to1(FunctionalModule):
    """Functional module with two arguments."""

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
