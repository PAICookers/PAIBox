import warnings
from functools import partial
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import LCM, NTM, RM, TM

from paibox.base import NeuDyn, NodeList
from paibox.exceptions import FunctionalError, PAIBoxWarning, ShapeError
from paibox.network import DynSysGroup
from paibox.types import SpikeType, VoltageType
from paibox.utils import as_shape, shape2num

from .modules import (
    BuiltComponentType,
    FunctionalModule,
    FunctionalModule2to1,
    FunctionalModule2to1WithV,
    TransposeModule,
)
from .neuron import Neuron
from .neuron.neurons import *
from .neuron.utils import VJT_MIN_LIMIT, _is_vjt_overflow
from .projection import InputProj
from .synapses import FullConnSyn
from .synapses import GeneralConnType as GConnType
from .synapses.conv_types import _Size2Type
from .synapses.conv_utils import _fm_ndim2_check, _pair
from .synapses.transforms import _Pool2dForward

__all__ = [
    "BitwiseAND",
    "BitwiseNOT",
    "BitwiseOR",
    "BitwiseXOR",
    "DelayChain",
    "SpikingAdd",
    "SpikingAvgPool2d",
    "SpikingMaxPool2d",
    "SpikingSub",
    "Transpose2d",
    "Transpose3d",
]


_L_SADD = 1  # Literal value for spiking addition.
_L_SSUB = -1  # Literal value for spiking subtraction.
VJT_OVERFLOW_ERROR_TEXT = "Membrane potential overflow causes spiking addition errors."


class BitwiseAND(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise AND module. Do a bitwise AND of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.
            - delay: delay between module & another module(or neuron). Default is 1.
            - tick_wait_start: set the moodule to start at timestep `T`. 0 means not working. Default is 1.
            - tick_wait_end: set the module to turn off at time `T`. 0 means always working. Default is 0.
            - unrolling_factor: argument related to the backend. It represents the degree to which modules  \
                are expanded. The larger the value, the more cores required for deployment, but the lower   \
                the latency & the higher the throughput. Default is 1.
            - keep_shape: whether to maintain size information when recording data in the simulation.       \
                Default is `False`.
            - name: name of the module. Optional.

        NOTE: the inherent delay of the module is 0. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+1.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 & x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        # 1. Instantiate neurons & synapses & connect the source
        n1_and = Neuron(
            self.shape_out,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            leak_v=-1,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_and,
            1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_and,
            1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_and, syn1, syn2]

        # 2. Connect the source of all backward synapses to output neuron.
        for syn in self.module_intf.output:
            syn.source = n1_and

        # 3. Add the components to the network & remove the module itself.
        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class BitwiseNOT(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise NOT module. Do a bitwise NOT of the output spike of one neuron & output.

        Args:
            - neuron: the operand.

        NOTE: the inherent delay of the module is 0.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return ~x1

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_not = Neuron(
            self.shape_out,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            leak_v=-1,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_not,
            weights=-1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )

        generated = [n1_not, syn1]

        for syns in self.module_intf.output:
            syns.source = n1_not

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class BitwiseOR(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise OR module. Do a bitwise OR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 | x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_or = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_or,
            1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_or,
            1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_or, syn1, syn2]

        for syns in self.module_intf.output:
            syns.source = n1_or

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class BitwiseXOR(FunctionalModule2to1):
    inherent_delay = 1

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Bitwise XOR module. Do a bitwise XOR of the output spike of two neurons & output.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.

        NOTE: the inherent delay of the module is 1. It means that under the default delay(=1) setting, the \
            input data is input at time T, and the result output at time T+2.
        """
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, x1: SpikeType, x2: SpikeType, **kwargs) -> SpikeType:
        return x1 ^ x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        # If neuron_a is of shape (h1, w1) = N, and neuron_b is of shape (h2, w2) = N.
        # The output shape of the module is (N,) or (h1, w1)(if h1 == h2).
        # The shape of n1 is (2N,) or (2, h1, w1).
        n1_aux = SpikingRelu(
            (2,) + self.shape_out,
            delay=1,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=False,
            name=f"n0_{self.name}",
        )

        identity = np.identity(self.num_out, dtype=np.int8)
        # weight of syn1, (-1*(N,), 1*(N,))
        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_aux,
            weights=np.hstack([-1 * identity, identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )
        # weight of syn2, (1*(N,), -1*(N,))
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_aux,
            weights=np.hstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s1_{self.name}",
        )

        # The shape of n2 is (N,) or (h1, w1).
        n2_xor = SpikingRelu(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n1_{self.name}",
        )

        # weight of syn3, identity matrix with shape (2N, N)
        syn3 = FullConnSyn(
            n1_aux,
            n2_xor,
            weights=np.vstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=GConnType.MatConn,
            name=f"s2_{self.name}",
        )

        generated = [n1_aux, n2_xor, syn1, syn2, syn3]

        for syns in self.module_intf.output:
            syns.source = n2_xor

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class DelayChain(FunctionalModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        chain_level: int = 1,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Delay chain. It will add extra neurons (and identity synapses) as buffer.

        Args:
            - neuron: the target neuron to be delayed.
            - chain_level: the level of delay chain.

        NOTE: the inherent delay of the module depends on `chain_level`.
        """
        if keep_shape:
            shape_out = neuron.shape_out
        else:
            shape_out = (neuron.num_out,)

        if chain_level < 1:
            raise ValueError(
                f"the level of delay chain must be positive, but got {chain_level}."
            )

        self.inherent_delay = chain_level

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return x1

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n_delaychain = NodeList()
        s_delaychain = NodeList()

        # Delay chain of length #D.
        for i in range(self.inherent_delay - 1):
            n_delay = SpikingRelu(
                self.shape_out,
                tick_wait_start=self.tick_wait_start + i,
                tick_wait_end=self.tick_wait_end,
                delay=1,
                name=f"n{i}_{self.name}",
            )
            n_delaychain.append(n_delay)

        # delay = delay_relative for output neuron
        n_out = SpikingRelu(
            self.shape_out,
            tick_wait_start=self.tick_wait_start + i,
            tick_wait_end=self.tick_wait_end,
            delay=self.delay_relative,
            name=f"n{self.inherent_delay-1}_{self.name}",
        )
        n_delaychain.append(n_out)  # Must append to the last.

        syn_in = FullConnSyn(
            self.module_intf.operands[0],
            n_delaychain[0],
            1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )

        for i in range(self.inherent_delay - 1):
            s_delay = FullConnSyn(
                n_delaychain[i],
                n_delaychain[i + 1],
                1,
                conn_type=GConnType.One2One,
                name=f"s{i+1}_{self.name}",
            )

            s_delaychain.append(s_delay)

        generated = [*n_delaychain, syn_in, *s_delaychain]

        for syns in self.module_intf.output:
            syns.source = n_out

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class SpikingAdd(FunctionalModule2to1WithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        overflow_strict: bool = False,
        **kwargs,
    ) -> None:
        """Spiking Addition module. The result will be reflected in time dimension.

        Args:
            - neuron_a: the first operand.
            - neuron_b: the second operand.
            - overflow_strict: flag of whether to strictly check overflow. If enabled, an exception will be \
                raised if the result overflows during simulation.

        NOTE: the inherent delay of the module is 0.
        """
        self.overflow_strict = overflow_strict
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, vjt: VoltageType, **kwargs) -> Tuple[SpikeType, VoltageType]:
        """Simplified neuron computing mechanism as the operator function."""
        return _spike_func_sadd_ssub(vjt)

    def synaptic_integr(
        self, x1: SpikeType, x2: SpikeType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, vjt_pre, _L_SADD, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_sadd = Neuron(
            self.shape_out,
            reset_mode=RM.MODE_LINEAR,
            neg_thres_mode=NTM.MODE_SATURATION,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_sadd,
            1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_sadd,
            1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_sadd, syn1, syn2]

        for syns in self.module_intf.output:
            syns.source = n1_sadd

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class _SpikingPool2d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size2Type] = None,
        # padding: _Size2Type = 0,
        # fm_order: _Order3d = "CHW",
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d pooling for spike.

        Args:
            - neuron: of which the pooling will be performed.
            - kernel_size: the size of the window to take a max over.
            - pool_type: type of pooling, "avg" or "max".
            - stride: the stride of the window. Default value is `kernel_size`.

        NOTE: the inherent delay of the module is 0.
        """
        if pool_type not in ("avg", "max"):
            raise ValueError("type of pooling must be 'avg' or 'max'.")

        # if fm_order not in ("CHW", "HWC"):
        #     raise ValueError("feature map order must be 'CHW' or 'HWC'.")

        # C,H,W
        cin, ih, iw = _fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(0)

        oh = (ih + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        ow = (iw + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (cin, oh, ow)
        else:
            shape_out = (cin * oh * ow,)

        self.tfm = _Pool2dForward(
            cin, (ih, iw), (oh, ow), _ksize, _stride, _padding, pool_type
        )

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        return self.tfm(x1)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        if self.tfm.pool_type == "avg":
            n1_mp = Neuron(
                self.shape_out,
                leak_comparison=LCM.LEAK_BEFORE_COMP,
                leak_v=-(shape2num(self.tfm.ksize) // 2),
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
            )
        else:  # "max"
            n1_mp = SpikingRelu(
                self.shape_out,
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n0_{self.name}",
            )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_mp,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )

        generated = [n1_mp, syn1]

        for syns in self.module_intf.output:
            syns.source = n1_mp

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class SpikingAvgPool2d(_SpikingPool2d):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        # padding: _Size2Type = 0,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d average pooling for spike. The input feature map is in 'CHW' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(neuron, kernel_size, "avg", stride, keep_shape, name, **kwargs)


class SpikingMaxPool2d(_SpikingPool2d):
    """
    XXX: By enabling `MaxPoolingEnable` in neurons, the max pooling function can also be implemented.       \
        However, since the second-level cache of the input buffer before the physical core is in 144*8bit   \
        format, it is extremely wasteful when the input data is 1bit (i.e., spike). Therefore, we still     \
        under SNN mode when implementing max pooling of 1-bit input data.
    """

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        # padding: _Size2Type = 0,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d max pooling for spike.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(neuron, kernel_size, "max", stride, keep_shape, name, **kwargs)


class SpikingSub(FunctionalModule2to1WithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        overflow_strict: bool = False,
        **kwargs,
    ) -> None:
        """Spiking subtraction module. The result will be reflected in time dimension.

        Args:
            - neuron_a: the first operand. It is the minuend.
            - neuron_b: the second operand. It is the subtracter.
            - overflow_strict: flag of whether to strictly check overflow. If enabled, an exception will be \
                raised if the result overflows during simulation.

        NOTE: the inherent delay of the module is 0.
        """
        self.overflow_strict = overflow_strict
        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, vjt: VoltageType, **kwargs) -> Tuple[SpikeType, VoltageType]:
        """Simplified neuron computing mechanism to generate output spike."""
        return _spike_func_sadd_ssub(vjt)

    def synaptic_integr(
        self, x1: SpikeType, x2: SpikeType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, vjt_pre, _L_SSUB, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_ssub = Neuron(
            self.shape_out,
            neg_threshold=VJT_MIN_LIMIT,
            reset_mode=RM.MODE_LINEAR,
            neg_thres_mode=NTM.MODE_SATURATION,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_ssub,
            1,
            conn_type=GConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_ssub,
            weights=-1,
            conn_type=GConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_ssub, syn1, syn2]

        for syns in self.module_intf.output:
            syns.source = n1_ssub

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class Transpose2d(TransposeModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim2_check(neuron.shape_out),
            (1, 0),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.T

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t2d = Neuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t2d,
            weights=_transpose2d_mapping(self.shape_in),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )

        generated = [n1_t2d, syn1]

        for syns in self.module_intf.output:
            syns.source = n1_t2d

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


class Transpose3d(TransposeModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        axes: Optional[Sequence[int]] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """3d transpose module.

        Args:
            - neuron: the neuron of which output spike will be transposed.
            - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1] \
                where N is the number of axes of output shape of neuron. If not specified, defaults to      \
                `range(ndim)[::-1]`, where `ndim` is the dimension of the output shape, which reverses the  \
                order of the axes.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            _shape_ndim3_check(neuron.shape_out),
            axes,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: SpikeType, **kwargs) -> SpikeType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.transpose(self.axes)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t3d = Neuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t3d,
            weights=_transpose3d_mapping(self.shape_in, self.axes),
            conn_type=GConnType.MatConn,
            name=f"s0_{self.name}",
        )

        generated = [n1_t3d, syn1]

        for syns in self.module_intf.output:
            syns.source = n1_t3d

        network._add_components(*generated)
        # network._remove_components(self)

        return generated


def _spike_func_sadd_ssub(vjt: VoltageType) -> Tuple[SpikeType, VoltageType]:
    """Function `spike_func()` in spiking addition & subtraction."""
    # Fire
    thres_mode = np.where(
        vjt >= 1,
        TM.EXCEED_POSITIVE,
        np.where(vjt < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
    )
    spike = np.equal(thres_mode, TM.EXCEED_POSITIVE)
    # Reset
    v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, vjt - 1, vjt)

    return spike, v_reset


def _sum_inputs_sadd_ssub(
    x1: SpikeType,
    x2: SpikeType,
    vjt_pre: VoltageType,
    add_or_sub: Literal[1, -1],
    strict: bool,
) -> VoltageType:
    """Function `sum_input()` for spiking addition & subtraction."""
    # Charge
    incoming_v = (vjt_pre + x1 * 1 + x2 * add_or_sub).astype(np.int32)

    # NOTE: In most cases, membrane potential overflow won't occur, otherwise the result is incorrect.
    if _is_vjt_overflow(incoming_v):
        if strict:
            raise FunctionalError(VJT_OVERFLOW_ERROR_TEXT)
        else:
            warnings.warn(VJT_OVERFLOW_ERROR_TEXT, PAIBoxWarning)

    return incoming_v


def _shape_check(shape: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    if len(shape) > ndim:
        raise ShapeError(
            f"expected shape to have dimensions <= {ndim}, but got {len(shape)}."
        )

    return as_shape(shape, min_dim=ndim)


_shape_ndim2_check = partial(_shape_check, ndim=2)
_shape_ndim3_check = partial(_shape_check, ndim=3)


def _transpose2d_mapping(op_shape: Tuple[int, ...]) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 2d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y) order.

    Return: transposed index matrix with shape (X*Y, Y*X).
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    for idx in np.ndindex(op_shape):
        mt[idx[0] * op_shape[1] + idx[1], idx[1] * op_shape[0] + idx[0]] = 1

    return mt


def _transpose3d_mapping(
    op_shape: Tuple[int, ...], axes: Tuple[int, ...]
) -> NDArray[np.bool_]:
    """Get the mapping matrix for transpose of 3d array.

    Argument:
        - op_shape: the shape of matrix to be transposed, flattened in (X,Y,Z) order.
        - axes: If specified, it must be a tuple or list which contains a permutation of [0, 1, …, N-1]     \
            where N is the number of axes of a.

    Return: transposed index matrix with shape (N, N) where N=X*Y*Z.
    """
    size = shape2num(op_shape)
    mt = np.zeros((size, size), dtype=np.bool_)

    shape_t = tuple(op_shape[i] for i in axes)

    size12 = op_shape[1] * op_shape[2]
    size12_t = shape_t[1] * shape_t[2]

    for idx in np.ndindex(op_shape):
        mt[
            idx[0] * size12 + idx[1] * op_shape[2] + idx[2],
            idx[axes[0]] * size12_t + idx[axes[1]] * shape_t[2] + idx[axes[2]],
        ] = 1

    return mt
