import sys
import typing
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, ClassVar, Literal, Optional, TypeVar, Union

import numpy as np
from paicorelib import TM, CoreMode, HwConfig, SNNModeEnable, get_core_mode

from paibox.base import NeuDyn
from paibox.exceptions import NotSupportedError, RegisterError, ShapeError
from paibox.types import NEUOUT_U8_DTYPE, NeuOutType, VoltageType
from paibox.utils import check_elem_unique, shape2num

from .neuron.utils import RTModeKwds, _input_width_format, _spike_width_format
from .projection import InputProj

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if typing.TYPE_CHECKING:
    from paibox.network import DynSysGroup

    from .neuron import Neuron
    from .synapses import FullConnectedSyn

__all__ = ["BuildingModule"]

MultiInputsType: TypeAlias = list[NeuOutType]  # Type of inputs of `NeuModule`.
BuiltComponentType: TypeAlias = list[Union["FullConnectedSyn", "Neuron"]]


@dataclass
class ModuleIntf:
    """Module interface. The interface of the module stores the information about where the module  \
        gets input and where it outputs. This information will be used when building the module.
    """

    operands: list[Union[NeuDyn, InputProj]] = field(default_factory=list)
    output: list[Union["FullConnectedSyn", "NeuModule"]] = field(default_factory=list)

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

    def register_operand(self, *op: Union[NeuDyn, InputProj]) -> None:
        """Register operands to the interface."""
        self.module_intf.operands.extend(op)

    def unregister_operand(self, op: Union[NeuDyn, InputProj]) -> None:
        """Remove a operand from the interface."""
        self.module_intf.operands.remove(op)

    def register_output(self, *output: Union["FullConnectedSyn", "NeuModule"]) -> None:
        """Register the output."""
        self.module_intf.output.append(*output)

    def unregister_output(self, output: Union["FullConnectedSyn", "NeuModule"]) -> None:
        """Remove an output."""
        self.module_intf.output.remove(output)

    @property
    def n_op(self) -> int:
        return len(self.module_intf.operands)

    @property
    def n_output(self) -> int:
        return len(self.module_intf.output)


class NeuModule(NeuDyn, BuildingModule):
    __gh_build_ignore__ = True

    n_return: ClassVar[int] = 1
    """#N of outputs."""
    inherent_delay: int = 0
    """Internal delay of the module, relative to the external."""
    rt_mode_kwds: RTModeKwds
    mode: CoreMode

    def __init__(
        self,
        delay: int,
        tick_wait_start: int,
        tick_wait_end: int,
        unrolling_factor: int,
        keep_shape: bool,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.module_intf = ModuleIntf()
        self._delay = delay
        self._tws = tick_wait_start
        self._twe = tick_wait_end
        self._uf = unrolling_factor
        self.keep_shape = keep_shape

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

    def is_outputing(self) -> bool:
        return (self.timestamp - self.inherent_delay) >= 0

    @property
    def source(self) -> list[Union[NeuDyn, InputProj]]:
        return self.module_intf.operands

    @property
    def dest(self) -> list[Union["FullConnectedSyn", "NeuModule"]]:
        return self.module_intf.output  # will be deprecated at anytime in the future.

    @property
    def target(self) -> list[Union["FullConnectedSyn", "NeuModule"]]:
        return self.module_intf.output

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
        shape_out: tuple[int, ...],
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
            if isinstance(op, BuildingModule):
                if op.n_output > 1:
                    # TODO Connection between modules with `n_output` > 1.
                    raise NotSupportedError(
                        "The connection between the module & the module with output node "
                        "greater than 1 is not supported yet."
                    )

                op.register_output(self)

        super().__init__(**kwargs, keep_shape=keep_shape, name=name)
        self._shape_out = shape_out
        self.register_operand(*operands)

        # Set memory for only 1 output node.
        # TODO how to handle with more than 1 output nodes
        self.set_memory("_neu_out", np.zeros((self.num_out,), dtype=NEUOUT_U8_DTYPE))
        # Delay registers
        self.set_memory(
            "delay_registers",
            np.zeros(
                (HwConfig.N_TIMESLOT_MAX,) + self._neu_out.shape, dtype=NEUOUT_U8_DTYPE
            ),
        )
        # Set a deque for the `synin` to implement the delay of `inherent_delay` for the module.
        if self.inherent_delay > 0:
            _init_synin = [
                self.n_op * [np.zeros(self.source[0].num_out, dtype=NEUOUT_U8_DTYPE)]
            ]
        else:
            _init_synin = []

        self.set_memory(
            "synin_deque", deque(_init_synin, maxlen=1 + self.inherent_delay)
        )

    def get_inputs(self) -> None:
        synin = []

        for op in self.source:
            # Retrieve the spike at index `timestamp` of the dest neurons
            if self.is_working():
                if isinstance(op, InputProj):
                    synin.append(op.output)
                else:
                    idx = self.timestamp % HwConfig.N_TIMESLOT_MAX
                    synin.append(op.delay_registers[idx])
            else:
                # Retrieve 0 to the dest neurons if it is not working
                synin.append(np.zeros_like(op.spike))

        self.synin_deque.append(synin)  # Append to the right of the deque.

    def update(self, *args, **kwargs) -> Optional[NeuOutType]:
        if not self.is_working():
            self._neu_out.fill(0)
            return None

        self.get_inputs()

        if self.is_outputing():
            synin = self.synin_deque.popleft()  # Pop the left of the deque.
            self._neu_out = self.spike_func(*synin).ravel()
            idx = (
                self.timestamp - self.inherent_delay + self.delay_relative - 1
            ) % HwConfig.N_TIMESLOT_MAX
            self.delay_registers[idx] = self._neu_out.copy()

        return self._neu_out

    def _rebuild_out_intf(
        self,
        network: "DynSysGroup",
        out_neuron: "Neuron",
        *generated: Union[NeuDyn, "FullConnectedSyn"],
        **build_options,
    ) -> None:
        from .synapses import FullConnectedSyn

        for out in self.target:
            if isinstance(out, FullConnectedSyn):
                out.source = out_neuron
            else:
                out.unregister_operand(self)
                out.register_operand(out_neuron)

        network._add_components(*generated)
        network._remove_components(self)

    @property
    def shape_in(self):
        # TODO Return a tuple (shape_in_of_op1, shape_in_of_op2, ...)?
        raise NotImplementedError

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape_out

    @property
    def num_in(self) -> int:
        return self.module_intf.num_in

    @property
    def num_out(self) -> int:
        return shape2num(self._shape_out)

    @property
    def output(self) -> NeuOutType:
        return self.delay_registers

    @property
    def spike(self) -> NeuOutType:
        return self._neu_out

    @property
    def feature_map(self) -> NeuOutType:
        return self._neu_out.reshape(self.varshape)

    @property
    def varshape(self) -> tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)


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

        super().__init__(
            neuron_a,
            neuron_b,
            shape_out=_shape_check2(neuron_a, neuron_b, keep_shape),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    @property
    def varshape(self) -> tuple[int, ...]:
        return self.shape_out if self.keep_shape else (self.num_out,)


class TransposeModule(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        shape_in: tuple[int, ...],
        axes: Optional[Sequence[int]] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if axes is None:
            axes = range(len(shape_in))[::-1]

        axes = tuple(axes)

        if not check_elem_unique(axes):
            raise ValueError("repeated axis in transpose.")

        if len(axes) != len(shape_in):
            raise ValueError("axes don't match array.")

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
    def shape_in(self) -> tuple[int, ...]:
        return self._shape_in


class FunctionalModuleWithV(FunctionalModule):
    """Functional module with two operands.

    NOTE: Compared to `FunctionalModule`, the difference is that it takes the \
        membrane potential voltage into consideration.
    """

    def __init__(
        self,
        *operands: Union[NeuDyn, InputProj],
        shape_out: tuple[int, ...],
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *operands, shape_out=shape_out, keep_shape=keep_shape, name=name, **kwargs
        )
        self.set_memory("_vjt", np.zeros((self.num_out,), dtype=np.int32))
        self.thres_mode = np.full((self.num_out,), TM.NOT_EXCEEDED, dtype=np.uint8)

    def synaptic_integr(self, *args, **kwargs) -> VoltageType:
        """Functions used to describe synaptic integration of the module."""
        raise NotImplementedError(
            "'synaptic_integr' should be implemented in the subclasses."
        )

    def update(self, *args, **kwargs) -> Optional[NeuOutType]:
        if not self.is_working():
            self._neu_out.fill(0)
            return None

        self.get_inputs()

        if self.is_outputing():
            synin = self.synin_deque.popleft()  # Pop the left of the deque.
            incoming_v = self.synaptic_integr(*synin, self._vjt)
            _is, self._vjt = self.spike_func(incoming_v)
            self._neu_out = _is.ravel()

            idx = (
                self.timestamp - self.inherent_delay + self.delay_relative - 1
            ) % HwConfig.N_TIMESLOT_MAX
            self.delay_registers[idx] = self._neu_out.copy()

        return self._neu_out

    @property
    def voltage(self) -> VoltageType:
        return self._vjt.reshape(self.varshape)


class FunctionalModule2to1WithV(FunctionalModuleWithV):
    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            neuron_a,
            neuron_b,
            shape_out=_shape_check2(neuron_a, neuron_b, keep_shape),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


L = Literal
_T = TypeVar("_T", bound=NeuModule)


def set_rt_mode(
    input_width: L[1, 8], spike_width: L[1, 8], snn_en: L[0, 1]
) -> Callable[[type[_T]], type[_T]]:
    def wrapper(cls: type[_T]) -> type[_T]:
        iw = _input_width_format(input_width)
        sw = _spike_width_format(spike_width)
        sen = SNNModeEnable(snn_en)

        cls.mode = get_core_mode(iw, sw, sen)
        cls.rt_mode_kwds = {"input_width": iw, "spike_width": sw, "snn_en": sen}
        return cls

    return wrapper


set_rt_mode_snn = partial(set_rt_mode, input_width=1, spike_width=1, snn_en=1)
set_rt_mode_ann = partial(set_rt_mode, input_width=8, spike_width=8, snn_en=0)


def _shape_check2(
    neuron_a: Union[NeuDyn, InputProj],
    neuron_b: Union[NeuDyn, InputProj],
    keep_shape: bool,
) -> tuple[int, ...]:
    if keep_shape:
        if neuron_a.shape_out != neuron_b.shape_out:
            raise ShapeError(
                f"two operands must have the same shape: {neuron_a.shape_out} != {neuron_b.shape_out}. "
                f"When two operands have different shapes, set 'keep_shape=False' and the output will "
                f"not retain shape information."
            )

        shape_out = neuron_a.shape_out
    else:
        shape_out = (neuron_a.num_out,)

    return shape_out
