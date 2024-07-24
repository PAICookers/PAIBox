from typing import Literal, Optional, Union

import numpy as np
from paicorelib import TM

from paibox.base import NeuDyn
from paibox.network import DynSysGroup
from paibox.types import NEUOUT_U8_DTYPE, WEIGHT_DTYPE, NeuOutType, VoltageType
from paibox.utils import arg_check_non_neg, shape2num, typical_round

from .modules import (
    BuiltComponentType,
    FunctionalModule,
    FunctionalModuleWithV,
    set_rt_mode_snn,
)
from .neuron import Neuron
from .neuron.neurons import *
from .neuron.utils import vjt_overflow
from .projection import InputProj
from .synapses import ConnType, FullConnSyn
from .synapses.conv_types import _Size1Type, _Size2Type
from .synapses.conv_utils import _fm_ndim1_check, _fm_ndim2_check, _single, _pair
from .synapses.transforms import (
    Conv1dForward,
    Conv2dForward,
    _Pool1dForward,
    _Pool2dForward,
)

__all__ = [
    "_SpikingPool1d",
    "_SpikingPool1dWithV",
    "_SpikingPool2d",
    "_SpikingPool2dWithV",
]


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
        if pool_type not in ("avg", "max"):
            raise ValueError("type of pooling must be 'avg' or 'max'.")

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

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
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
            n1_p1d = SpikingRelu(
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

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
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
            self.module_intf.operands[0],
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
        if pool_type not in ("avg", "max"):
            raise ValueError("type of pooling must be 'avg' or 'max'.")

        # if fm_order not in ("CHW", "HWC"):
        #     raise ValueError("feature map order must be 'CHW' or 'HWC'.")

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

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
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
            n1_p2d = SpikingRelu(
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

    def build(self, network: DynSysGroup, **build_options) -> BuiltComponentType:
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
            self.module_intf.operands[0],
            n1_p2d,
            weights=self.tfm.connectivity.astype(np.bool_),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p2d, syn1]
        self._rebuild_out_intf(network, n1_p2d, *generated, **build_options)

        return generated


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
