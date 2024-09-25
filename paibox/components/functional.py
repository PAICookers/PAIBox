import math
import sys
from collections.abc import Sequence
from functools import partial
from typing import ClassVar, Optional, Union

import numpy as np
from paicorelib import NTM, RM, TM

from paibox.base import NeuDyn, NodeList
from paibox.exceptions import PAIBoxDeprecationWarning, ResourceError, ShapeError
from paibox.network import DynSysGroup
from paibox.types import (
    LEAK_V_DTYPE,
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    WEIGHT_DTYPE,
    DataType,
    IntScalarType,
    NeuOutType,
    VoltageType,
    WeightType,
)
from paibox.utils import arg_check_pos, as_shape, shape2num

from ._modules import *
from .modules import (
    BuiltComponentType,
    FunctionalModule,
    FunctionalModule2to1,
    FunctionalModule2to1WithV,
    TransposeModule,
    set_rt_mode_ann,
    set_rt_mode_snn,
)
from .neuron import Neuron
from .neuron.base import MetaNeuron
from .neuron.neurons import *
from .neuron.utils import vjt_overflow
from .projection import InputProj
from .synapses import ConnType, Conv2dSemiFoldedSyn, FullConnSyn, MaxPool2dSemiFoldedSyn
from .synapses.conv_types import _Size1Type, _Size2Type
from .synapses.conv_utils import _pair

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

__all__ = [
    "BitwiseAND",
    "BitwiseNOT",
    "BitwiseOR",
    "BitwiseXOR",
    "SpikingAdd",
    "SpikingAvgPool1d",
    "SpikingAvgPool1dWithV",
    "SpikingMaxPool1d",
    "SpikingAvgPool2d",
    "SpikingAvgPool2dWithV",
    "SpikingMaxPool2d",
    "SpikingSub",
    "Transpose2d",
    "Transpose3d",
    "Conv2dSemiFolded",
    "Filter",
    "Linear",
    "MaxPool2dSemiFolded",
    "AvgPool2dSemiFolded",
]


@set_rt_mode_snn()
class BitwiseAND(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
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

    def spike_func(self, x1: NeuOutType, x2: NeuOutType, **kwargs) -> NeuOutType:
        return x1 & x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_and = LIF(
            self.shape_out,
            threshold=1,
            leak_v=-1,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_and,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_and,
            1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_and, syn1, syn2]
        self._rebuild_out_intf(network, n1_and, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class BitwiseNOT(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
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

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        return x1 == 0  # x1 is an array in uint8

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_not = LIF(
            self.shape_out,
            threshold=1,
            leak_v=1,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_not,
            weights=-1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )

        generated = [n1_not, syn1]
        self._rebuild_out_intf(network, n1_not, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class BitwiseOR(FunctionalModule2to1):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
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

    def spike_func(self, x1: NeuOutType, x2: NeuOutType, **kwargs) -> NeuOutType:
        return x1 | x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_or = BypassNeuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_or,
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_or,
            1,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_or, syn1, syn2]
        self._rebuild_out_intf(network, n1_or, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class BitwiseXOR(FunctionalModule2to1):
    inherent_delay = 1

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        *,
        keep_shape: bool = True,
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

    def spike_func(self, x1: NeuOutType, x2: NeuOutType, **kwargs) -> NeuOutType:
        return x1 ^ x2

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        # If neuron_a is of shape (h1, w1) = N, and neuron_b is of shape (h2, w2) = N.
        # The output shape of the module is (N,) or (h1, w1)(if h1 == h2).
        # The shape of n1 is (2N,) or (2, h1, w1).
        n1_aux = BypassNeuron(
            (2,) + self.shape_out,
            delay=1,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=False,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        identity = np.identity(self.num_out, dtype=np.int8)
        # weight of syn1, (-1*(N,), 1*(N,))
        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_aux,
            weights=np.hstack([-1 * identity, identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )
        # weight of syn2, (1*(N,), -1*(N,))
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_aux,
            weights=np.hstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s1_{self.name}",
        )

        # The shape of n2 is (N,) or (h1, w1).
        n2_xor = BypassNeuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=n1_aux.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n1_{self.name}",
            **self.rt_mode_kwds,
        )

        # weight of syn3, identity matrix with shape (2N, N)
        syn3 = FullConnSyn(
            n1_aux,
            n2_xor,
            weights=np.vstack([identity, identity], casting="safe", dtype=np.int8),
            conn_type=ConnType.All2All,
            name=f"s2_{self.name}",
        )

        generated = [n1_aux, n2_xor, syn1, syn2, syn3]
        self._rebuild_out_intf(network, n2_xor, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class SpikingAdd(FunctionalModule2to1WithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron_a: Union[NeuDyn, InputProj],
        neuron_b: Union[NeuDyn, InputProj],
        factor_a: IntScalarType = 1,
        factor_b: IntScalarType = 1,
        pos_thres: IntScalarType = 1,
        reset_v: Optional[int] = None,
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
            - factor_a: positive factor of neuron_a. Default is 1.
            - factor_b: positive factor of neuron_b. Default is 1.
            - pos_thres: positive threshold. Default is 1.
            - reset_v: if not specified, neurons will do soft reset after firing, v - threshold. If         \
                specified, neurons will do hard reset after firing, v = reset_v.
            - overflow_strict: flag of whether to strictly check overflow. If enabled, an exception will be \
                raised if the result overflows during simulation.

        NOTE: the inherent delay of the module is 0.
        """
        self.factor_a = arg_check_pos(int(factor_a), "factor_a")
        self.factor_b = arg_check_pos(int(factor_b), "factor_b")
        self.reset_v = reset_v
        self.pos_threshold = arg_check_pos(int(pos_thres), "pos_threshold")
        self.overflow_strict = overflow_strict

        super().__init__(neuron_a, neuron_b, keep_shape=keep_shape, name=name, **kwargs)

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[NeuOutType, VoltageType]:
        """Simplified neuron computing mechanism as the operator function."""
        return _spike_func_sadd_ssub(vjt, self.pos_threshold, self.reset_v)

    def synaptic_integr(
        self, x1: NeuOutType, x2: NeuOutType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, self.factor_a, self.factor_b, vjt_pre, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_sadd = IF(
            self.shape_out,
            self.pos_threshold,
            self.reset_v,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_sadd,
            self.factor_a,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_sadd,
            self.factor_b,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_sadd, syn1, syn2]
        self._rebuild_out_intf(network, n1_sadd, *generated, **build_options)

        return generated


class SpikingAvgPool1d(_SpikingPool1d):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """1d average pooling for spike. The input feature map is in 'CL' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 1  \
                integer.
            - threshold: if specified, the pooling result is o = (sum of the pooling window > threshold).   \
                Otherwise the threshold is kernel_size // 2.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "avg",
            stride,
            padding,
            threshold,
            keep_shape,
            name,
            **kwargs,
        )


class SpikingAvgPool1dWithV(_SpikingPool1dWithV):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """1d average pooling for spike with voltage at the previous timestep. The input feature map is in  \
            'CL' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 1  \
                integer.
            - threshold: if specified, the pooling result is o = (sum of the pooling window >= threshold).  \
                Otherwise the threshold is kernel_size // 2.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron, kernel_size, stride, padding, threshold, keep_shape, name, **kwargs
        )


class SpikingMaxPool1d(_SpikingPool1d):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """1d max pooling for spike. The input feature map is in 'CL' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of a  \
                integer.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "max",
            stride,
            padding,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class SpikingAvgPool2d(_SpikingPool2d):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d average pooling for spike. The input feature map is in 'CHW' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2  \
                integers.
            - threshold: if specified, the pooling result is o = (sum of the pooling window >= threshold).  \
                Otherwise the threshold is kernel_size // 2.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "avg",
            stride,
            padding,
            threshold,
            keep_shape,
            name,
            **kwargs,
        )


class SpikingAvgPool2dWithV(_SpikingPool2dWithV):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d average pooling for spike with voltage at the previous timestep. The input feature map is in  \
            'CHW' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2  \
                integers.
            - threshold: if specified, the pooling result is o = (sum of the pooling window >= threshold).  \
                Otherwise the threshold is kernel_size // 2.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron, kernel_size, stride, padding, threshold, keep_shape, name, **kwargs
        )


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
        padding: _Size2Type = 0,
        # fm_order: _Order3d = "CHW",
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d max pooling for spike. The input feature map is in 'CHW' order by default.

        Args:
            - neuron: the target neuron to be pooled.
            - kernel_size: the size of the window to take a max over.
            - stride: the stride of the window. Default value is `kernel_size`.
            - padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2  \
                integers.

        NOTE: the inherent delay of the module is 0.
        """
        super().__init__(
            neuron,
            kernel_size,
            "max",
            stride,
            padding,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


@set_rt_mode_snn()
class SpikingSub(FunctionalModule2to1WithV):
    inherent_delay = 0
    factor_a: int = 1
    factor_b: int = -1
    pos_threshold: int = 1

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

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[NeuOutType, VoltageType]:
        """Simplified neuron computing mechanism to generate output spike."""
        return _spike_func_sadd_ssub(vjt, self.pos_threshold)

    def synaptic_integr(
        self, x1: NeuOutType, x2: NeuOutType, vjt_pre: VoltageType
    ) -> VoltageType:
        return _sum_inputs_sadd_ssub(
            x1, x2, self.factor_a, self.factor_b, vjt_pre, strict=self.overflow_strict
        )

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_ssub = Neuron(
            self.shape_out,
            reset_mode=RM.MODE_LINEAR,
            neg_thres_mode=NTM.MODE_SATURATION,
            pos_threshold=self.pos_threshold,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_ssub,
            self.factor_a,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.module_intf.operands[1],
            n1_ssub,
            self.factor_b,
            conn_type=ConnType.One2One,
            name=f"s1_{self.name}",
        )

        generated = [n1_ssub, syn1, syn2]
        self._rebuild_out_intf(network, n1_ssub, *generated, **build_options)

        return generated


@deprecated(
    "'Transpose2d' will be removed in version 1.2.0. Use 'MatMul2d' instead.",
    category=PAIBoxDeprecationWarning,
)
@set_rt_mode_snn()
class Transpose2d(TransposeModule):
    inherent_delay = 0

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

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.T

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t2d = BypassNeuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t2d,
            weights=_transpose2d_mapping(self.shape_in),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_t2d, syn1]
        self._rebuild_out_intf(network, n1_t2d, *generated, **build_options)

        return generated


@deprecated(
    "'Transpose3d' will be removed in version 1.2.0. Use 'MatMul2d' instead.",
    category=PAIBoxDeprecationWarning,
)
@set_rt_mode_snn()
class Transpose3d(TransposeModule):
    inherent_delay = 0

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

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        _x1 = x1.reshape(self.shape_in)

        return _x1.transpose(self.axes)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        n1_t3d = BypassNeuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            n1_t3d,
            weights=_transpose3d_mapping(self.shape_in, self.axes),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_t3d, syn1]
        self._rebuild_out_intf(network, n1_t3d, *generated, **build_options)

        return generated


class LinearSemiFolded(_LinearBase, _SemiFoldedModule):
    "That operator is used on the first fully-connected layer after the semi-folded convolution."

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        raise NotImplementedError

    def build(
        self, network: DynSysGroup, valid_interval: int, **build_options
    ) -> BuiltComponentType:
        assert len(self.module_intf.operands[0].shape_out) == 2
        self.valid_interval = valid_interval

        in_ch, in_h = self.module_intf.operands[0].shape_out
        if in_ch * in_h * in_h * valid_interval > 18432:
            raise ResourceError(
                f"The {self.name} input size is too large. Please adjust the input size or the number of channels."
            )
        n_delays = NodeList()
        s_delays = NodeList()
        s_weight = NodeList()

        n_fc = ANNNeuron(
            self.shape_out,
            self.bias,
            self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        for i in range(in_h):
            neuron = ANNBypassNeuron(
                shape=(in_ch, in_h),
                delay=valid_interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # Delay synapses
            syn1 = FullConnSyn(
                self.module_intf.operands[0],
                neuron,
                weights=_delay_mapping(in_h, in_ch),
                conn_type=ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)

            w = self.weights[in_h - i - 1 :: in_h, :]
            syn2 = FullConnSyn(
                neuron,
                n_fc,
                weights=w,
                conn_type=self.conn_type,
                name=f"s{i}_{self.name}",
            )
            s_weight.append(syn2)

        generated = [n_fc, *n_delays, *s_delays, *s_weight]
        self._rebuild_out_intf(network, n_fc, *generated, **build_options)

        return generated


class Conv2dSemiFolded(_SemiFoldedModule):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel: np.ndarray,
        stride: _Size2Type = 1,
        padding: _Size2Type = 0,
        bias: DataType = 0,
        bit_trunc: int = 8,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d semi-folded convolution for ANN mode."""
        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        self.kernel = kernel
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._w_padding_check(self.padding[1], neuron_s)

        self.bit_trunc = bit_trunc

        if isinstance(bias, np.ndarray):
            _bias = np.atleast_1d(bias).astype(LEAK_V_DTYPE)
        else:
            _bias = int(bias)

        self.bias = _bias

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out
        # XXX Do not consider the case when the shape of source neurons needs to be changed, for now.
        # neuron_s.shape_change((in_ch, in_h))

        cout, cin, kh, _ = kernel.shape
        out_h = (in_h - kh + 2 * self.padding[0]) // self.stride[0] + 1

        if in_ch != cin:
            raise ShapeError(f"The channels mismatch: {in_ch} != {cin}.")

        super().__init__(
            neuron_s,
            shape_out=(cout, out_h),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        raise NotImplementedError

    def build(
        self,
        network: DynSysGroup,
        valid_interval: int,
        input_valid: int,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.module_intf.operands[0].shape_out) == 2
        # if len(self.module_intf.operands[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(
        #         self.module_intf.operands[0].shape_out, "CHW"
        #     )
        #     self.module_intf.operands[0].shape_change((in_ch, in_h))
        self.valid_interval = valid_interval
        _, in_h = self.module_intf.operands[0].shape_out
        _, cin, _, kw = self.kernel.shape
        ts_1st_valid = input_valid + (kw - 1 - self.padding[0]) * valid_interval
        self.ts_1st_valid = ts_1st_valid
        tick_wait_end = (
            1 + ts_1st_valid + (self.shape_out[1] - 1) * valid_interval * self.stride[1]
        )
        if cin * in_h * kw * valid_interval > 18432:
            raise ResourceError(
                f"The {self.name} input size is too large. Please adjust the input size or the number of channels."
            )
        n_delays = NodeList()
        n_copies = NodeList()
        s_delays = NodeList()
        s_kernel = NodeList()

        n_conv2d = ANNNeuron(
            self.shape_out,
            self.bias,
            self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        for i in range(kw):
            neuron = ANNBypassNeuron(
                (cin, in_h),
                delay=valid_interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_delay_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.module_intf.operands[0],
                n_delays[i],
                weights=_delay_mapping(in_h, cin),
                conn_type=ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)

            syn2 = Conv2dSemiFoldedSyn(  # cin, ih -> cout * oh
                neuron,
                n_conv2d,
                self.kernel[:, :, :, kw - i - 1],
                self.stride,
                self.padding,
                "OIL",
                name=f"s{i}_{self.name}",
            )
            s_kernel.append(syn2)

        if input_valid > 0:
            for i in range(self.padding[0]):
                neuron = ANNBypassNeuron(
                    (cin, in_h),
                    delay=valid_interval * (kw - 1 - i) + 1,
                    tick_wait_start=self.tick_wait_start,
                    tick_wait_end=input_valid,
                    keep_shape=self.keep_shape,
                    name=f"n{i}_copy_{self.name}",
                )

                n_copies.append(neuron)
                # delay synapses
                syn1 = FullConnSyn(
                    self.module_intf.operands[0],
                    n_copies[i],
                    weights=_delay_mapping(in_h, cin),
                    conn_type=ConnType.All2All,
                    name=f"s{i}_copy_{self.name}",
                )
                s_delays.append(syn1)

                syn2 = Conv2dSemiFoldedSyn(  # cin, ih -> cout * oh
                    n_copies[i],
                    n_conv2d,
                    -(self.kernel[:, :, :, i]),
                    self.stride,
                    self.padding,
                    "OIL",
                    name=f"neg_s{i}_{self.name}",
                )
                s_kernel.append(syn2)
        generated = [n_conv2d, *n_delays, *n_copies, *s_delays, *s_kernel]
        self._rebuild_out_intf(network, n_conv2d, *generated, **build_options)

        return generated


@deprecated(
    "The backend currently does not support 'Filter', please use it in a future version",
    category=PAIBoxDeprecationWarning,
)
@set_rt_mode_ann()
class Filter(FunctionalModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        time_to_fire: int,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """ """
        shape_out = neuron.shape_out
        self.time_to_fire = time_to_fire
        self.cur_time = 0
        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        if self.cur_time != self.time_to_fire:
            self.cur_time += 1
            return np.zeros_like(x1)
        else:
            self.cur_time = 0
            return x1

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        inp1 = Always1Neuron((2,))
        n1_filter = Neuron(
            self.shape_out,
            leak_v=0,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            input_width=self.input_width,
            spike_width=self.spike_width,
            snn_en=self.snn_en,
            keep_shape=self.keep_shape,
            name="filter",
        )

        syn1 = FullConnSyn(
            self.module_intf.operands[0],  # (10,0)
            n1_filter,  # (10,0)
            weights=1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            inp1,  # (2,0)
            n1_filter,  # (10,0)
            weights=-128,
            conn_type=ConnType.All2All,
            name=f"s1_{self.name}",
        )
        network._add_components(n1_filter, syn1, syn2)
        network._remove_components(self)
        generated = [n1_filter, syn1, syn2]
        return generated


@set_rt_mode_ann()
class Linear(_LinearBase):
    "Linear layer for ANN."

    inherent_delay = 0

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        output = x1 @ self.weights.astype(VOLTAGE_DTYPE)
        output = output + self.bias
        output = np.where(output >= 1, MetaNeuron._truncate(output, self.bit_trunc), 0)

        return output.astype(NEUOUT_U8_DTYPE)

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
        neuron_d = ANNNeuron(
            self.shape_out,
            self.bias,
            self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        syn1 = FullConnSyn(
            self.module_intf.operands[0],
            neuron_d,
            weights=self.weights,
            conn_type=ConnType.All2All,
            name=f"syn1_{self.name}",
        )

        generated = [neuron_d, syn1]
        self._rebuild_out_intf(network, neuron_d, *generated, **build_options)

        return generated


class MaxPool2dSemiFolded(_SemiFoldedModule):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d semi-folded max pooling for ANN mode."""
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            _stride = self.kernel_size
        else:
            _stride = _pair(stride)

        self.stride = _stride
        self.padding = _pair(padding)
        # self._w_padding_check(self.padding[1], neuron_s)

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out

        out_h = (in_h - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_h),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        raise NotImplementedError

    def build(
        self,
        network: DynSysGroup,
        valid_interval: int,
        input_valid: int,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.module_intf.operands[0].shape_out) == 2
        # if len(self.module_intf.operands[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(
        #         self.module_intf.operands[0].shape_out, "CHW"
        #     )
        #     self.module_intf.operands[0].shape_change((in_ch, in_h))
        self.valid_interval = valid_interval

        in_ch, in_h = self.module_intf.operands[0].shape_out
        cin = in_ch
        _, kw = self.kernel_size

        ts_1st_valid = input_valid + (kw - 1) * valid_interval
        self.ts_1st_valid = ts_1st_valid
        tick_wait_end = (
            1 + ts_1st_valid + (self.shape_out[1] - 1) * valid_interval * self.stride[1]
        )

        if cin * in_h * kw * valid_interval > 18432:
            raise ResourceError(
                f"The {self.name} input size is too large. Please adjust the input size or the number of channels."
            )

        n_delays = NodeList()
        s_delays = NodeList()

        pool2d = ANNNeuron(
            self.shape_out,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=tick_wait_end,
            pool_max=True,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        for i in range(kw):
            neuron = ANNBypassNeuron(
                (cin, in_h),
                delay=valid_interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.module_intf.operands[0],
                n_delays[i],
                weights=_delay_mapping(in_h, cin),
                conn_type=ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)
            syn2 = MaxPool2dSemiFoldedSyn(
                neuron,
                pool2d,
                weights=_poo2d_semifolded_mapping(
                    cin, in_h, self.shape_out[1], self.kernel_size[0], self.stride, self.padding
                ),
                name=f"s{i}_{self.name}",
            )
            s_delays.append(syn2)

        generated = [pool2d, *n_delays, *s_delays]
        self._rebuild_out_intf(network, pool2d, *generated, **build_options)

        return generated


class AvgPool2dSemiFolded(_SemiFoldedModule):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d AvgPool2d_semimap for spike."""
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            _stride = self.kernel_size
        else:
            _stride = _pair(stride)

        self.stride = _stride
        self.padding = _pair(padding)
        # self._w_padding_check(self.padding[1], neuron_s)

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out

        out_h = (in_h - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_h),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        raise NotImplementedError

    def build(
        self,
        network: DynSysGroup,
        valid_interval: int,
        input_valid: int,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.module_intf.operands[0].shape_out) == 2
        # if len(self.module_intf.operands[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(
        #         self.module_intf.operands[0].shape_out, "CHW"
        #     )
        #     self.module_intf.operands[0].shape_change((in_ch, in_h))
        self.valid_interval = valid_interval

        in_ch, in_h = self.module_intf.operands[0].shape_out
        cin = in_ch
        kh, kw = self.kernel_size
        if cin * in_h * kw * valid_interval > 18432:
            raise ResourceError(
                f"The {self.name} input size is too large. Please adjust the input size or the number of channels."
            )

        ts_1st_valid = (
                input_valid
                + (kw - 1 - self.padding[0]) * valid_interval
        )
        self.ts_1st_valid = ts_1st_valid
        tick_wait_end = 1 + ts_1st_valid + (self.shape_out[1] - 1) * valid_interval * self.stride[1]

        E = math.ceil(math.log2(cin * in_h * kw / 144))
        E = 0 if E < 0 else E
        if kw * valid_interval > 256 / (2 ** E):
            raise ResourceError(
                f"The {self.name} input size is too large. Please adjust the input size or the number of channels.")


        # NOTE: Division is achieved with the help of truncation operation.
        # It can only be approximated to a power of an integer of 2.
        bit_trunc = 8 + (kh * kw).bit_length() - 1

        n_delays = NodeList()
        n_copies = NodeList()
        s_delays = NodeList()
        s_kernel = NodeList()

        pool2d = ANNNeuron(
            self.shape_out,
            delay=self.delay_relative,
            bit_trunc=bit_trunc,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        for i in range(kw):
            neuron = ANNBypassNeuron(
                (cin, in_h),
                delay=valid_interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.module_intf.operands[0],
                n_delays[i],
                weights=_delay_mapping(in_h, cin),
                conn_type=ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)
            syn2 = FullConnSyn(
                neuron,
                pool2d,
                weights=_poo2d_semifolded_mapping(
                    cin, in_h, self.shape_out[1], self.kernel_size[0], self.stride, self.padding
                ),
                conn_type=ConnType.All2All,
                name=f"s{i}_{self.name}",
            )
            s_delays.append(syn2)
        if input_valid > 0:
            for i in range(self.padding[0]):
                neuron = ANNBypassNeuron(
                    (cin, in_h),
                    delay=valid_interval * (kw-1-i) + 1,
                    tick_wait_start=self.tick_wait_start,
                    tick_wait_end=input_valid,
                    keep_shape=self.keep_shape,
                    name=f"n{i}_copy_{self.name}",
                )

                n_copies.append(neuron)
                # delay synapses
                syn1 = FullConnSyn(
                    self.module_intf.operands[0],
                    n_copies[i],
                    weights=_delay_mapping(in_h, cin),
                    conn_type=ConnType.All2All,
                    name=f"s{i}_copy_{self.name}",
                )
                s_delays.append(syn1)

                syn2 = FullConnSyn(  # cin, ih -> cout * oh
                    n_copies[i],
                    pool2d,
                    weights=-(_poo2d_semifolded_mapping(
                        cin, in_h, self.shape_out[1], self.kernel_size[0], self.stride, self.padding)),
                    conn_type=ConnType.All2All,
                    name=f"neg_s{i}_{self.name}",
                )
                s_kernel.append(syn2)
        generated = [pool2d, *n_delays, *s_delays]
        self._rebuild_out_intf(network, pool2d, *generated, **build_options)

        return generated


def _spike_func_sadd_ssub(
    vjt: VoltageType, pos_thres: int, reset_v: Optional[int] = None
) -> tuple[NeuOutType, VoltageType]:
    """Function `spike_func()` in spiking addition & subtraction."""
    # Fire
    thres_mode = np.where(
        vjt >= pos_thres,
        TM.EXCEED_POSITIVE,
        np.where(vjt < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
    )
    # Reset
    if reset_v is None:
        v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, vjt - pos_thres, vjt)
    else:
        v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, reset_v, vjt)

    # Spike
    spike = thres_mode == TM.EXCEED_POSITIVE

    return spike.astype(NEUOUT_U8_DTYPE), v_reset


def _spike_func_avg_pool(
    vjt: VoltageType, pos_thres: int
) -> tuple[NeuOutType, VoltageType]:
    """Function `spike_func()` in spiking addition & subtraction."""
    # Fire
    thres_mode = np.where(
        vjt >= pos_thres,
        TM.EXCEED_POSITIVE,
        np.where(vjt < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
    )
    spike = thres_mode == TM.EXCEED_POSITIVE
    # Reset
    v_reset = np.where(thres_mode == TM.EXCEED_POSITIVE, 0, vjt)

    return spike.astype(NEUOUT_U8_DTYPE), v_reset


def _sum_inputs_sadd_ssub(
    x1: NeuOutType, x2: NeuOutType, f1: int, f2: int, vjt_pre: VoltageType, strict: bool
) -> VoltageType:
    """Function `sum_input()` for spiking addition & subtraction."""
    incoming_v = (vjt_pre + x1 * f1 + x2 * f2).astype(VOLTAGE_DTYPE)
    return vjt_overflow(incoming_v, strict)


def _shape_check(shape: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(shape) > ndim:
        raise ShapeError(
            f"expected shape to have dimensions <= {ndim}, but got {len(shape)}."
        )

    return as_shape(shape, min_dim=ndim)


_shape_ndim2_check = partial(_shape_check, ndim=2)
_shape_ndim3_check = partial(_shape_check, ndim=3)


def _transpose2d_mapping(op_shape: tuple[int, ...]) -> WeightType:
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
    op_shape: tuple[int, ...], axes: tuple[int, ...]
) -> WeightType:
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


def _delay_mapping(h: int, cin: int) -> WeightType:
    return np.eye(cin * h, dtype=WEIGHT_DTYPE)


def _poo2d_semifolded_mapping(
    cin: int, ih: int, oh: int, kh: int, stride: tuple[int, int], padding: tuple[int, int]
) -> WeightType:
    cout = cin

    m = np.zeros((cin * ih, cout * oh), dtype=WEIGHT_DTYPE)
    m_block = np.zeros((ih+2*padding[0], oh), dtype=WEIGHT_DTYPE)

    for j in range(oh):
        m_block[j * stride[1] : j * stride[1] + kh, j] =1
    if padding[0] > 0:
        m_block = np.delete(
            m_block,
            np.hstack((np.arange(padding[0]), np.arange(ih + padding[0], ih+2*padding[0]))),
            axis=0,
        )

    for i in range(cout):
        m[i*ih: i*ih+ih, i*oh:i*oh+oh] = m_block

    return m
