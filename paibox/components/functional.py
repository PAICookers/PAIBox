import sys
import typing
from collections.abc import Sequence
from functools import partial
from typing import ClassVar, Optional, Union

import numpy as np
from paicorelib import NTM, RM, TM

from paibox.base import NeuDyn, NodeList
from paibox.exceptions import PAIBoxDeprecationWarning, ShapeError
from paibox.types import (
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
from .synapses import ConnType, Conv2dSemiFoldedSyn, FullConnSyn, MaxPoolSyn
from .synapses.conv_types import _Size1Type, _Size2Type
from .synapses.conv_utils import _pair

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

if typing.TYPE_CHECKING:
    from paibox.network import DynSysGroup

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
    "Linear",
    "LinearSemiFolded",
    "Conv2dSemiFolded",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool2dSemiFolded",
    "AvgPool1d",
    "AvgPool2d",
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0], n1_and, 1, ConnType.One2One, name=f"s0_{self.name}"
        )
        syn2 = FullConnSyn(
            self.source[1],
            n1_and,
            1,
            ConnType.One2One,
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0], n1_not, -1, ConnType.One2One, name=f"s0_{self.name}"
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0], n1_or, 1, ConnType.One2One, name=f"s0_{self.name}"
        )
        syn2 = FullConnSyn(
            self.source[1], n1_or, 1, ConnType.One2One, name=f"s1_{self.name}"
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            n1_aux,
            np.hstack([-1 * identity, identity], casting="safe", dtype=np.int8),
            ConnType.All2All,
            name=f"s0_{self.name}",
        )
        # weight of syn2, (1*(N,), -1*(N,))
        syn2 = FullConnSyn(
            self.source[1],
            n1_aux,
            np.hstack([identity, -1 * identity], casting="safe", dtype=np.int8),
            ConnType.All2All,
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
            np.vstack([identity, identity], casting="safe", dtype=np.int8),
            ConnType.All2All,
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            n1_sadd,
            self.factor_a,
            ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.source[1],
            n1_sadd,
            self.factor_b,
            ConnType.One2One,
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            n1_ssub,
            self.factor_a,
            ConnType.One2One,
            name=f"s0_{self.name}",
        )
        syn2 = FullConnSyn(
            self.source[1],
            n1_ssub,
            self.factor_b,
            ConnType.One2One,
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            n1_t2d,
            _transpose2d_mapping(self.shape_in),
            ConnType.All2All,
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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            n1_t3d,
            _transpose3d_mapping(self.shape_in, self.axes),
            ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_t3d, syn1]
        self._rebuild_out_intf(network, n1_t3d, *generated, **build_options)

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

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
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
            self.source[0],
            neuron_d,
            self.weights,
            ConnType.All2All,
            name=f"syn1_{self.name}",
        )

        generated = [neuron_d, syn1]
        self._rebuild_out_intf(network, neuron_d, *generated, **build_options)

        return generated


class LinearSemiFolded(_LinearBase, _SemiFoldedModule):
    "This operator is used on the first fully-connected layer after the semi-folded convolution."

    def build(
        self,
        network: "DynSysGroup",
        incoming_flow_format: SemiFoldedDataFlowFormat,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.source[0].shape_out) == 2
        # For semi-folded linear, the valid output is at only one timestep.
        self._oflow_format = SemiFoldedDataFlowFormat(
            incoming_flow_format.t_last_vld, 1, 1
        )
        twe = 1 + self._oflow_format.t_last_vld

        ich, ih = self.source[0].shape_out
        self._input_buffer_len_check(ich, ih, ih, incoming_flow_format.interval)

        n_delays = NodeList()
        s_delays = NodeList()
        s_weight = NodeList()

        n_linear = ANNNeuron(
            self.shape_out,
            self.bias,
            self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        n_linear.set_oflow_format(
            self._oflow_format.t_1st_vld,
            self._oflow_format.interval,
            self._oflow_format.n_vld,
        )

        for i in range(ih):
            neuron = ANNBypassNeuron(
                shape=(ich, ih),
                delay=incoming_flow_format.interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=(
                    twe
                    if not self.rin_buffer_option
                    else twe - incoming_flow_format.interval * i
                ),
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # Delay synapses
            syn1 = FullConnSyn(
                self.source[0],
                neuron,
                _delay_mapping_mask(ih, ich),
                ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)

            w = self.weights[ih - i - 1 :: ih, :]
            syn2 = FullConnSyn(
                neuron, n_linear, w, ConnType.All2All, name=f"s{i}_{self.name}"
            )
            s_weight.append(syn2)

        generated = [n_linear, *n_delays, *s_delays, *s_weight]
        self._rebuild_out_intf(network, n_linear, *generated, **build_options)

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
        groups: int = 1,
        bit_trunc: int = 8,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d semi-folded convolution for ANN mode.

        Args:
            neuron_s: source neuron. The dimensions need to be expressed explicitly as (C,H) or (C,W).
            kernel: convolution kernel in (O,I,H,W) order.
            stride: the step size of the kernel sliding. It can be a scalar or a tuple of 2 integers.
            padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2 integers.
            bias: it can be a scalar or an array of the same size as the output.
            bit_trunc: the bit truncation position. By default, bits 7 to 0 are truncated.
        """
        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        self.kernel = kernel
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups
        self.bit_trunc = bit_trunc

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out
        # XXX Do not consider the case when the shape of source neurons needs to be changed, for now.
        # neuron_s.shape_change((in_ch, in_h))

        cout, cin, kh, kw = kernel.shape
        out_h = (in_h - kh + 2 * self.padding[0]) // self.stride[0] + 1

        assert self.padding[0] < kh and self.padding[1] < kw

        if in_ch % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if cout % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if in_ch != groups * cin:
            raise ShapeError(f"the channels mismatch: {in_ch} != {cin}.")

        _shape_out = (cout, out_h)
        self.bias = bias

        super().__init__(
            neuron_s, shape_out=_shape_out, keep_shape=keep_shape, name=name, **kwargs
        )

    def build(
        self,
        network: "DynSysGroup",
        incoming_flow_format: SemiFoldedDataFlowFormat,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.source[0].shape_out) == 2
        # if len(self.source[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(
        #         self.source[0].shape_out, "CHW"
        #     )
        #     self.source[0].shape_change((in_ch, in_h))
        ic, ih = self.source[0].shape_out
        _, cin, _, kw = self.kernel.shape
        _, ow = self.shape_out

        self._oflow_format = SemiFoldedDataFlowFormat(
            incoming_flow_format.t_at_n(kw - self.padding[0]),
            incoming_flow_format.interval * self.stride[1],
            ow,
        )
        twe = 1 + self._oflow_format.t_last_vld

        self._input_buffer_len_check(cin, ih, kw, incoming_flow_format.interval)

        n_delays = NodeList()
        n_neg_padding = NodeList()
        s_delays = NodeList()
        s_kernel = NodeList()
        s_neg_padding = NodeList()

        n_conv2d = ANNNeuron(
            self.shape_out,
            self.bias,
            self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=twe,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        n_conv2d.set_oflow_format(
            self._oflow_format.t_1st_vld,
            self._oflow_format.interval,
            self._oflow_format.n_vld,
        )

        for i in range(kw):
            neuron = ANNBypassNeuron(
                (ic, ih),
                delay=incoming_flow_format.interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=(
                    twe
                    if not self.rin_buffer_option
                    else twe - incoming_flow_format.interval * i
                ),
                name=f"n{i}_delay_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.source[0],
                neuron,
                _delay_mapping_mask(ih, ic),
                ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)

            syn2 = Conv2dSemiFoldedSyn(
                neuron,
                n_conv2d,
                self.kernel[:, :, :, kw - i - 1],
                self.stride,
                self.padding,
                self.groups,
                "OIL",
                name=f"s{i}_{self.name}",
            )
            s_kernel.append(syn2)

        # Add additional negative padding layer to eliminate the incorrect output
        # NOTE: `t_1st_vld` = 0 & `padding[0]` > 0 means the previous layer is
        # an input node. No need to add negative padding layer for this case.
        if incoming_flow_format.t_1st_vld > 0:
            for p in range(self.padding[0]):
                neuron = ANNBypassNeuron(
                    (ic, ih),
                    delay=1 + incoming_flow_format.interval * (kw - 1 - p),
                    tick_wait_start=self.tick_wait_start,
                    tick_wait_end=incoming_flow_format.t_1st_vld,
                    keep_shape=self.keep_shape,
                    name=f"n{p}_pad_{self.name}",
                )
                n_neg_padding.append(neuron)
                # delay synapses
                syn1 = FullConnSyn(
                    self.source[0],
                    neuron,
                    _delay_mapping_mask(ih, ic),
                    ConnType.All2All,
                    name=f"s{p}_pad_{self.name}",
                )
                s_delays.append(syn1)

                syn2 = Conv2dSemiFoldedSyn(
                    neuron,
                    n_conv2d,
                    -(self.kernel[:, :, :, p]),
                    self.stride,
                    self.padding,
                    self.groups,
                    "OIL",
                    name=f"neg_s{p}_{self.name}",
                )
                s_neg_padding.append(syn2)

        generated = [
            n_conv2d,
            *n_delays,
            *n_neg_padding,
            *s_delays,
            *s_kernel,
            *s_neg_padding,
        ]
        self._rebuild_out_intf(network, n_conv2d, *generated, **build_options)

        return generated


class MaxPool1d(_Pool1d):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        bit_trunc: int = 8,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """1d max pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window to take a max over.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: implicit negative infinity padding to be added on both sides. It can be a scalar or    \
                a tuple of 1 integer.
            bit_trunc: the bit truncation position. By default, bits 7 to 0 are truncated.
        """
        super().__init__(
            neuron_s,
            kernel_size,
            "max",
            stride,
            padding,
            bit_trunc,
            keep_shape,
            name,
            **kwargs,
        )

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        cin, in_l = self.source[0].shape_out
        k = self.kernel_size[0]
        _, o_l = self.shape_out

        pool_1d = ANNNeuron(
            self.shape_out,
            bit_trunc=self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            pool_max=True,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        syn1 = MaxPoolSyn(
            self.source[0],
            pool_1d,
            _poo1d_mapping_mask(cin, in_l, o_l, k, self.stride, self.padding),
            name=f"s0_{self.name}",
        )

        generated = [pool_1d, syn1]
        self._rebuild_out_intf(network, pool_1d, *generated, **build_options)

        return generated


class MaxPool2d(_Pool2d):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        bit_trunc: Optional[int] = 8,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d max pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window to take a max over.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: implicit negative infinity padding to be added on both sides. It can be a scalar or    \
                a tuple of 2 integers.
            bit_trunc: the bit truncation position. By default, bits 7 to 0 are truncated.
        """
        super().__init__(
            neuron_s,
            kernel_size,
            "max",
            stride,
            padding,
            bit_trunc,
            keep_shape,
            name,
            **kwargs,
        )

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        cin, ih, iw = self.source[0].shape_out
        kh, kw = self.kernel_size
        _, oh, ow = self.shape_out

        pool_2d = ANNNeuron(
            self.shape_out,
            bit_trunc=self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            pool_max=True,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        syn1 = MaxPoolSyn(
            self.source[0],
            pool_2d,
            _poo2d_mapping_mask(cin, ih, iw, oh, ow, kh, kw, self.stride, self.padding),
            name=f"s0_{self.name}",
        )

        generated = [pool_2d, syn1]
        self._rebuild_out_intf(network, pool_2d, *generated, **build_options)

        return generated


class MaxPool2dSemiFolded(_SemiFoldedModule):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        bit_trunc: int = 8,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d semi-folded max pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window to take a max over.
            stride: the stride of the window. Default value is `kernel_size`.
            bit_trunc: the bit truncation position. By default, bits 7 to 0 are truncated.

        NOTE: Since the semi-folded max pooling in the ANN mode is implemented using comparators, it is not \
            possible to use negative padding layer to eliminate the incorrect results of the padding part.
        """
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.bit_trunc = bit_trunc

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out
        out_h = (in_h - self.kernel_size[0]) // self.stride[0] + 1

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_h),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def build(
        self,
        network: "DynSysGroup",
        incoming_flow_format: SemiFoldedDataFlowFormat,
        **build_options,
    ) -> BuiltComponentType:
        assert len(self.source[0].shape_out) == 2
        # if len(self.source[0].shape_out) != 2:
        #     in_ch, in_h, in_w = _fm_ndim2_check(
        #         self.source[0].shape_out, "CHW"
        #     )
        #     self.source[0].shape_change((in_ch, in_h))
        cin, ih = self.source[0].shape_out
        kh, kw = self.kernel_size
        _, ow = self.shape_out

        self._oflow_format = SemiFoldedDataFlowFormat(
            incoming_flow_format.t_at_n(kw),
            incoming_flow_format.interval * self.stride[1],
            ow,
        )
        twe = 1 + self._oflow_format.t_last_vld

        self._input_buffer_len_check(cin, ih, kw, incoming_flow_format.interval)

        n_delays = NodeList()
        s_delays = NodeList()

        n_pool2d = ANNNeuron(
            self.shape_out,
            bit_trunc=self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=twe,
            pool_max=True,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        n_pool2d.set_oflow_format(
            self._oflow_format.t_1st_vld,
            self._oflow_format.interval,
            self._oflow_format.n_vld,
        )

        for i in range(kw):
            neuron = ANNBypassNeuron(
                (cin, ih),
                delay=incoming_flow_format.interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=(
                    twe
                    if not self.rin_buffer_option
                    else twe - incoming_flow_format.interval * i
                ),
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.source[0],
                neuron,
                _delay_mapping_mask(ih, cin),
                ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)
            syn2 = MaxPoolSyn(
                neuron,
                n_pool2d,
                _poo2d_semifolded_mapping_mask(cin, ih, ow, kh, self.stride, (0, 0)),
                name=f"s{i}_{self.name}",
            )
            s_delays.append(syn2)

        generated = [n_pool2d, *n_delays, *s_delays]
        self._rebuild_out_intf(network, n_pool2d, *generated, **build_options)

        return generated


class AvgPool1d(_Pool1d):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        bit_trunc: Optional[int] = None,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """1d average pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2    \
                integers.
            bit_trunc: the bit truncation position. By default, bit_trunc = 8 + ksize.bit_length() - 1.
        """
        super().__init__(
            neuron_s,
            kernel_size,
            "avg",
            stride,
            padding,
            bit_trunc,
            keep_shape,
            name,
            **kwargs,
        )

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        cin, in_l = self.source[0].shape_out
        k = self.kernel_size[0]
        _, o_l = self.shape_out

        pool_1d = ANNNeuron(
            self.shape_out,
            bit_trunc=self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        syn1 = FullConnSyn(
            self.source[0],
            pool_1d,
            _poo1d_mapping_mask(cin, in_l, o_l, k, self.stride, self.padding),
            ConnType.All2All,
            name=f"s1_{self.name}",
        )

        generated = [pool_1d, syn1]
        self._rebuild_out_intf(network, pool_1d, *generated, **build_options)

        return generated


class AvgPool2d(_Pool2d):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        bit_trunc: Optional[int] = None,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d average pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window to take a max over.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2    \
                integers.
            bit_trunc: the bit truncation position. By default, bit_trunc = 8 + ksize.bit_length() - 1.
        """
        super().__init__(
            neuron_s,
            kernel_size,
            "avg",
            stride,
            padding,
            bit_trunc,
            keep_shape,
            name,
            **kwargs,
        )

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        cin, ih, iw = self.source[0].shape_out
        kh, kw = self.kernel_size
        _, oh, ow = self.shape_out

        pool_2d = ANNNeuron(
            self.shape_out,
            bit_trunc=self.bit_trunc,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )

        syn1 = FullConnSyn(
            self.source[0],
            pool_2d,
            _poo2d_mapping_mask(cin, ih, iw, oh, ow, kh, kw, self.stride, self.padding),
            ConnType.All2All,
            name=f"s1_{self.name}",
        )

        generated = [pool_2d, syn1]
        self._rebuild_out_intf(network, pool_2d, *generated, **build_options)

        return generated


class AvgPool2dSemiFolded(_SemiFoldedModule):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        bit_trunc: Optional[int] = None,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """2d semi-folded average pooling for ANN mode.

        Args:
            neuron_s: the input neuron to be pooled.
            kernel_size: the size of the window.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: the amount of zero-padding applied to the input. It can be a scalar or a tuple of 2    \
                integers.
            bit_trunc: the bit truncation position. By default, bit_trunc = 8 + ksize.bit_length() - 1.
        """
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.padding = _pair(padding)

        # NOTE: Division is achieved with the help of output truncation.
        # TODO Since division with a divisor that is an integer power of 2 can only be implemented by
        # truncating the output, when the pooling window is not an integer power of 2 (which is the
        # usual case), additional processing is required before instantiating these operators.
        # For example,
        # 1. The pooling window size is 3x3, but the chip can only accurately implement result/8.
        # 2. bit_trunc=8 for the output neurons of this pooling layer, but for the next layer, the
        # weights becomes w*8/9, where w is the original weights.
        # 3. The alternative is bit_tunc=16 for this layer & w*16/9 for the next layer?
        # NOTE: The resulting linear transformation of weights of the next layer needs to be considered
        # during quantization.
        ksize = shape2num(self.kernel_size)
        self.bit_trunc = 8 + ksize.bit_length() - 1 if bit_trunc is None else bit_trunc

        assert len(neuron_s.shape_out) == 2
        in_ch, in_h = neuron_s.shape_out
        out_h = (in_h - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        kh, kw = self.kernel_size
        assert self.padding[0] < kh and self.padding[1] < kw

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_h),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def build(
        self,
        network: "DynSysGroup",
        incoming_flow_format: SemiFoldedDataFlowFormat,
        **build_options,
    ) -> BuiltComponentType:
        cin, ih = self.source[0].shape_out
        kh, kw = self.kernel_size
        _, ow = self.shape_out

        self._oflow_format = SemiFoldedDataFlowFormat(
            incoming_flow_format.t_at_n(kw - self.padding[0]),
            incoming_flow_format.interval * self.stride[1],
            ow,
        )
        twe = 1 + self._oflow_format.t_last_vld

        # if build_options.get("check_before_compile"):
        self._input_buffer_len_check(cin, ih, kw, incoming_flow_format.interval)

        n_delays = NodeList()
        n_neg_padding = NodeList()
        s_delays = NodeList()
        s_neg_padding = NodeList()

        n_pool2d = ANNNeuron(
            self.shape_out,
            delay=self.delay_relative,
            bit_trunc=self.bit_trunc,
            tick_wait_start=self.tick_wait_start + 1,
            tick_wait_end=twe,
            keep_shape=self.keep_shape,
            name=f"nd_{self.name}",
        )
        n_pool2d.set_oflow_format(
            self._oflow_format.t_1st_vld,
            self._oflow_format.interval,
            self._oflow_format.n_vld,
        )

        for i in range(kw):
            neuron = ANNBypassNeuron(
                (cin, ih),
                delay=incoming_flow_format.interval * i + 1,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=(
                    twe
                    if not self.rin_buffer_option
                    else twe - incoming_flow_format.interval * i
                ),
                keep_shape=self.keep_shape,
                name=f"n{i}_{self.name}",
            )
            n_delays.append(neuron)
            # delay synapses
            syn1 = FullConnSyn(
                self.source[0],
                neuron,
                _delay_mapping_mask(ih, cin),
                ConnType.All2All,
                name=f"s{i}_delay_{self.name}",
            )
            s_delays.append(syn1)
            syn2 = FullConnSyn(
                neuron,
                n_pool2d,
                _poo2d_semifolded_mapping_mask(
                    cin, ih, ow, kh, self.stride, self.padding
                ),
                ConnType.All2All,
                name=f"s{i}_{self.name}",
            )
            s_delays.append(syn2)

        # Add additional negative padding layer to eliminate the incorrect output
        if incoming_flow_format.t_1st_vld > 0:
            for p in range(self.padding[0]):
                neuron = ANNBypassNeuron(
                    (cin, ih),
                    delay=1 + incoming_flow_format.interval * (kw - 1 - p),
                    tick_wait_start=self.tick_wait_start,
                    tick_wait_end=incoming_flow_format.t_1st_vld,
                    keep_shape=self.keep_shape,
                    name=f"n{p}_pad_{self.name}",
                )
                n_neg_padding.append(neuron)
                # delay synapses
                syn1 = FullConnSyn(
                    self.source[0],
                    neuron,
                    _delay_mapping_mask(ih, cin),
                    ConnType.All2All,
                    name=f"s{p}_pad_{self.name}",
                )
                s_delays.append(syn1)

                syn2 = FullConnSyn(
                    neuron,
                    n_pool2d,
                    -_poo2d_semifolded_mapping_mask(
                        cin, ih, ow, kh, self.stride, self.padding
                    ),
                    ConnType.All2All,
                    name=f"neg_s{i}_{self.name}",
                )
                s_neg_padding.append(syn2)

        generated = [n_pool2d, *n_delays, *n_neg_padding, *s_delays, *s_neg_padding]
        self._rebuild_out_intf(network, n_pool2d, *generated, **build_options)

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
    incoming_v = (
        vjt_pre + x1.astype(VOLTAGE_DTYPE) * f1 + x2.astype(VOLTAGE_DTYPE) * f2
    ).astype(VOLTAGE_DTYPE)
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

    return mt.astype(WEIGHT_DTYPE)


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

    return mt.astype(WEIGHT_DTYPE)


def _delay_mapping_mask(h: int, cin: int) -> WeightType:
    return np.eye(cin * h, dtype=WEIGHT_DTYPE)


def _poo2d_semifolded_mapping_mask(
    cin: int,
    ih: int,
    oh: int,
    kh: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> WeightType:
    cout = cin

    m = np.zeros((cin * ih, cout * oh), dtype=WEIGHT_DTYPE)
    m_block = np.zeros((ih + 2 * padding[0], oh), dtype=WEIGHT_DTYPE)

    for j in range(oh):
        m_block[j * stride[1] : j * stride[1] + kh, j] = 1

    if padding[0] > 0:
        m_block = np.delete(
            m_block,
            np.hstack(
                (np.arange(padding[0]), np.arange(ih + padding[0], ih + 2 * padding[0]))
            ),
            axis=0,
        )

    for i in range(cout):
        m[i * ih : i * ih + ih, i * oh : i * oh + oh] = m_block

    return m


def _poo1d_mapping_mask(
    cin: int,
    in_l: int,
    o_l: int,
    kernel_size: int,
    stride: tuple[int],
    padding: tuple[int],
) -> WeightType:
    n_input = cin * in_l
    n_output = cin * o_l

    weights = np.zeros((n_input, n_output), dtype=WEIGHT_DTYPE)

    for c in range(cin):
        for o in range(o_l):
            start = o * stride[0] - padding[0]

            for k in range(kernel_size):
                pos = start + k

                if 0 <= pos < in_l:
                    input_idx = c * in_l + pos
                    output_idx = c * o_l + o
                    weights[input_idx, output_idx] = 1

    return weights


def _poo2d_mapping_mask(
    cin: int,
    ih: int,
    iw: int,
    oh: int,
    ow: int,
    kh: int,
    kw: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> WeightType:
    n_input = cin * ih * iw
    n_output = cin * oh * ow
    weights = np.zeros((n_input, n_output), dtype=WEIGHT_DTYPE)

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    for c in range(cin):
        for h_out in range(oh):
            for w_out in range(ow):
                h_start = h_out * stride_h - pad_h
                w_start = w_out * stride_w - pad_w

                for dh in range(kh):
                    for dw in range(kw):
                        h_in = h_start + dh
                        w_in = w_start + dw

                        if 0 <= h_in < ih and 0 <= w_in < iw:
                            input_idx = c * (ih * iw) + h_in * iw + w_in
                            output_idx = c * (oh * ow) + h_out * ow + w_out
                            weights[input_idx, output_idx] = 1

    return weights
