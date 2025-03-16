import math
import typing
from typing import Literal, Optional, Union

import numpy as np
from paicorelib import TM, HwConfig

from paibox.base import DataFlowFormat, NeuDyn, NodeList
from paibox.exceptions import ResourceError
from paibox.types import (
    NEUOUT_U8_DTYPE,
    WEIGHT_DTYPE,
    DataType,
    NeuOutType,
    Shape,
    VoltageType,
)
from paibox.utils import (
    arg_check_non_neg,
    arg_check_pos,
    as_shape,
    shape2num,
    typical_round,
)

from .modules import (
    BuiltComponentType,
    FunctionalModule,
    FunctionalModuleWithV,
    set_rt_mode_ann,
    set_rt_mode_snn,
)
from .neuron import Neuron
from .neuron.neurons import *
from .neuron.utils import vjt_overflow
from .projection import InputProj
from .synapses import ConnType, FullConnSyn
from .synapses.conv_types import _Size1Type, _Size2Type
from .synapses.conv_utils import _fm_ndim1_check, _fm_ndim2_check, _pair, _single
from .synapses.transforms import (
    Conv1dForward,
    Conv2dForward,
    _Pool1dForward,
    _Pool2dForward,
)

if typing.TYPE_CHECKING:
    from paibox.network import DynSysGroup

__all__ = [
    "_DelayChainANN",
    "_DelayChainSNN",
    "_SpikingPool1d",
    "_SpikingPool1dWithV",
    "_SpikingPool2d",
    "_SpikingPool2dWithV",
    "_SemiFoldedModule",
    "_LinearBase",
    "_Pool1d",
    "_Pool2d",
    "SemiFoldedDataFlowFormat",
]


class _DelayChainBase(FunctionalModule):
    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        chain_level: int = 1,
        *,
        keep_shape: bool = True,
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

        self.chain_level = arg_check_pos(chain_level, "chain level")
        self.inherent_delay = chain_level - 1

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        return x1

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        n_delaychain = NodeList()
        s_delaychain = NodeList()

        # Delay chain of length #D.
        for i in range(self.chain_level - 1):
            n_delay = BypassNeuron(
                self.shape_out,
                tick_wait_start=self.tick_wait_start + i,
                tick_wait_end=self.tick_wait_end,
                delay=1,
                name=f"n{i}_{self.name}",
                **self.rt_mode_kwds,
            )
            n_delaychain.append(n_delay)

        # delay = delay_relative for output neuron
        n_out = BypassNeuron(
            self.shape_out,
            tick_wait_start=self.tick_wait_start + i + 1,
            tick_wait_end=self.tick_wait_end,
            delay=self.delay_relative,
            name=f"n{i + 1}_{self.name}",
            **self.rt_mode_kwds,
        )
        n_delaychain.append(n_out)  # Must append to the last.

        syn_in = FullConnSyn(
            self.source[0],
            n_delaychain[0],
            1,
            conn_type=ConnType.One2One,
            name=f"s0_{self.name}",
        )

        for i in range(self.chain_level - 1):
            s_delay = FullConnSyn(
                n_delaychain[i],
                n_delaychain[i + 1],
                1,
                conn_type=ConnType.One2One,
                name=f"s{i + 1}_{self.name}",
            )

            s_delaychain.append(s_delay)

        generated = [*n_delaychain, syn_in, *s_delaychain]
        self._rebuild_out_intf(network, n_out, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class _DelayChainSNN(_DelayChainBase):
    pass


@set_rt_mode_ann()
class _DelayChainANN(_DelayChainBase):
    pass


class SemiFoldedDataFlowFormat(DataFlowFormat):
    pass


@set_rt_mode_ann()
class _SemiFoldedModule(FunctionalModule):
    """Functional modules with interfaces in semi-folded form. Use `build()` of class `HasSemiFoldedIntf`."""

    inherent_delay = 1
    _oflow_format: SemiFoldedDataFlowFormat

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        shape_out: tuple[int, ...],
        keep_shape: bool = False,
        name: Optional[str] = None,
        rin_buffer_option: bool = False,
        **kwargs,
    ) -> None:
        self.rin_buffer_option = rin_buffer_option
        super().__init__(
            neuron_s, shape_out=shape_out, keep_shape=keep_shape, name=name, **kwargs
        )

    def build(
        self,
        network: "DynSysGroup",
        incoming_flow_format: SemiFoldedDataFlowFormat,
        **build_options,
    ) -> BuiltComponentType:
        raise NotImplementedError

    def _input_buffer_len_check(
        self, in_channels: int, in_h: int, kw: int, valid_interval: int
    ) -> None:
        """Check the limit of the semi-folded operators on the input buffer length of the core during the build phase.

        NOTE: If the condition is not met, an expection will be raised in the subsequent compilation phase.
        """
        E = math.ceil(
            math.log2(
                math.ceil(in_channels * in_h * kw / HwConfig.N_FANIN_PER_DENDRITE_ANN)
            )
        )
        rin_deep = min(in_h - kw, kw - 1) * valid_interval + 1
        if not HwConfig.N_TIMESLOT_MAX / (2**E) > rin_deep:
            raise ResourceError(
                f"the input size of {self.name} is too large. Please adjust the input size or the number of channels."
            )
        buffer_deep = kw * valid_interval
        if buffer_deep > HwConfig.N_TIMESLOT_MAX / (2**E):
            self.rin_buffer_option = True
        if self.rin_buffer_option:
            print("rin buffer has been enabled.")


class _LinearBase(FunctionalModule):
    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        out_features: Shape,
        weights: np.ndarray,
        bias: DataType = 0,
        bit_trunc: int = 8,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic linear layer for ANN mode.

        Args:
            neuron_s: the input neuron.
            out_features: the output shape.
            weights: the weight matrix.
            bias: it can be a scalar or an array of the same size as the output.
            bit_trunc: the bit truncation position. By default, bits 7 to 0 are truncated.
        """
        self.weights = weights
        self.bit_trunc = bit_trunc
        self.bias = bias

        super().__init__(
            neuron_s,
            shape_out=as_shape(out_features),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


@set_rt_mode_snn()
class _SpikingPool1d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        threshold: Optional[int] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 1d spiking pooling."""
        _pool_type_check(pool_type)
        cin, il = _fm_ndim1_check(neuron.shape_out, "CL")

        _ksize = _single(kernel_size)
        _stride = _single(stride) if stride is not None else _ksize
        _padding = _single(padding)

        ol = (il + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1

        if keep_shape:
            shape_out = (cin, ol)
        else:
            shape_out = (cin * ol,)

        self.tfm = _Pool1dForward(
            cin, (il,), (ol,), _ksize, _stride, _padding, pool_type, threshold
        )

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        return self.tfm(x1)

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        if self.tfm.pool_type == "avg":
            n1_p1d = Neuron(
                self.shape_out,
                leak_v=1 - self.tfm.threshold,
                neg_threshold=0,
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n0_{self.name}",
                **self.rt_mode_kwds,
            )
        else:  # "max"
            n1_p1d = BypassNeuron(
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
            n1_p1d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p1d, syn1]
        self._rebuild_out_intf(network, n1_p1d, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class _SpikingPool1dWithV(FunctionalModuleWithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        pos_thres: Optional[int] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 1d spiking pooling with voltage at the previous timestep."""

        cin, il = _fm_ndim1_check(neuron.shape_out, "CL")

        _ksize = _single(kernel_size)
        _kernel = np.ones((cin, cin, *_ksize), dtype=WEIGHT_DTYPE)
        _stride = _single(stride) if stride is not None else _ksize
        _padding = _single(padding)

        ol = (il + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1

        if keep_shape:
            shape_out = (cin, ol)
        else:
            shape_out = (cin * ol,)

        if isinstance(pos_thres, int):
            self.pos_thres = arg_check_non_neg(pos_thres, "positive threshold")
        else:
            self.pos_thres = typical_round(shape2num(_ksize) / 2)

        self.tfm = Conv1dForward((il,), (ol,), _kernel, _stride, _padding)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[NeuOutType, VoltageType]:
        return _spike_func_avg_pool(vjt, self.pos_thres)

    def synaptic_integr(self, x1: NeuOutType, vjt_pre: VoltageType) -> VoltageType:
        return vjt_overflow(vjt_pre + self.tfm(x1).ravel())

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        n1_p1d = IF(
            self.shape_out,
            threshold=self.pos_thres,
            reset_v=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.source[0],
            n1_p1d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p1d, syn1]
        self._rebuild_out_intf(network, n1_p1d, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class _SpikingPool2d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        threshold: Optional[int] = None,
        # fm_order: _Order3d = "CHW",
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling."""
        _pool_type_check(pool_type)
        cin, ih, iw = _fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        oh = (ih + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        ow = (iw + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (cin, oh, ow)
        else:
            shape_out = (cin * oh * ow,)

        self.tfm = _Pool2dForward(
            cin, (ih, iw), (oh, ow), _ksize, _stride, _padding, pool_type, threshold
        )

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, x1: NeuOutType, **kwargs) -> NeuOutType:
        return self.tfm(x1)

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        if self.tfm.pool_type == "avg":
            n1_p2d = Neuron(
                self.shape_out,
                leak_v=1 - self.tfm.threshold,
                neg_threshold=0,
                delay=self.delay_relative,
                tick_wait_start=self.tick_wait_start,
                tick_wait_end=self.tick_wait_end,
                keep_shape=self.keep_shape,
                name=f"n0_{self.name}",
                **self.rt_mode_kwds,
            )
        else:  # "max"
            n1_p2d = BypassNeuron(
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
            n1_p2d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p2d, syn1]
        self._rebuild_out_intf(network, n1_p2d, *generated, **build_options)

        return generated


@set_rt_mode_snn()
class _SpikingPool2dWithV(FunctionalModuleWithV):
    inherent_delay = 0

    def __init__(
        self,
        neuron: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        pos_thres: Optional[int] = None,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling with voltage at the previous timestep.

        NOTE: This is not a regular average pooling operator. It is just to correspond to the operators \
            that appear in PAIFLOW.
        """
        cin, ih, iw = _fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _kernel = np.ones((cin, cin, *_ksize), dtype=WEIGHT_DTYPE)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        oh = (ih + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        ow = (iw + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (cin, oh, ow)
        else:
            shape_out = (cin * oh * ow,)

        if isinstance(pos_thres, int):
            self.pos_thres = arg_check_non_neg(pos_thres, "positive threshold")
        else:
            self.pos_thres = typical_round(shape2num(_ksize) / 2)

        self.tfm = Conv2dForward((ih, iw), (oh, ow), _kernel, _stride, _padding)

        super().__init__(
            neuron,
            shape_out=shape_out,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )

    def spike_func(self, vjt: VoltageType, **kwargs) -> tuple[NeuOutType, VoltageType]:
        return _spike_func_avg_pool(vjt, self.pos_thres)

    def synaptic_integr(self, x1: NeuOutType, vjt_pre: VoltageType) -> VoltageType:
        return vjt_overflow(vjt_pre + self.tfm(x1).ravel())

    def build(self, network: "DynSysGroup", **build_options) -> BuiltComponentType:
        n1_p2d = IF(
            self.shape_out,
            threshold=self.pos_thres,
            reset_v=0,
            neg_threshold=0,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            keep_shape=self.keep_shape,
            name=f"n0_{self.name}",
            **self.rt_mode_kwds,
        )

        syn1 = FullConnSyn(
            self.source[0],
            n1_p2d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p2d, syn1]
        self._rebuild_out_intf(network, n1_p2d, *generated, **build_options)

        return generated


@set_rt_mode_ann()
class _Pool1d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size1Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size1Type] = None,
        padding: _Size1Type = 0,
        bit_trunc: Optional[int] = None,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 1d ANN pooling."""
        _pool_type_check(pool_type)
        in_ch, in_l = _fm_ndim1_check(neuron_s.shape_out, "CL")

        self.kernel_size = _single(kernel_size)
        self.stride = _single(kernel_size if stride is None else stride)
        self.padding = _single(padding)

        # NOTE: Division is achieved with the help of output truncation.
        # See comments in `AvgPool2dSemiFolded` in functional.py for more details.
        ksize = shape2num(self.kernel_size)
        self.bit_trunc = 8 + ksize.bit_length() - 1 if bit_trunc is None else bit_trunc

        out_l = (in_l + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        k = self.kernel_size[0]
        assert 0 <= self.padding[0] <= k / 2 and 0 <= self.padding[0] <= k / 2

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_l),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


@set_rt_mode_ann()
class _Pool2d(FunctionalModule):
    inherent_delay = 0

    def __init__(
        self,
        neuron_s: Union[NeuDyn, InputProj],
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: Optional[_Size2Type] = None,
        padding: _Size2Type = 0,
        bit_trunc: Optional[int] = None,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Basic 2d ANN pooling."""
        _pool_type_check(pool_type)
        in_ch, in_h, in_w = _fm_ndim2_check(neuron_s.shape_out, "CHW")

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        self.padding = _pair(padding)

        # NOTE: Division is achieved with the help of output truncation.
        # See comments in `AvgPool2dSemiFolded` in functional.py for more details.
        ksize = shape2num(self.kernel_size)
        self.bit_trunc = 8 + ksize.bit_length() - 1 if bit_trunc is None else bit_trunc

        out_h = (in_h - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_w = (in_w - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        kh, kw = self.kernel_size
        assert self.padding[0] < kh and self.padding[1] < kw

        super().__init__(
            neuron_s,
            shape_out=(in_ch, out_h, out_w),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


def _spike_func_avg_pool(
    vjt: VoltageType, pos_thres: int
) -> tuple[NeuOutType, VoltageType]:
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


def _pool_type_check(pool_type: str) -> None:
    if pool_type not in ("avg", "max"):
        raise ValueError("type of pooling must be 'avg' or 'max'.")
