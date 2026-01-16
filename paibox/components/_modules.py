import math
import typing
from typing import Literal

import numpy as np
from paicorelib import OffCoreCfg

from paibox.base import DataFlowFormat, NeuDyn, NodeList
from paibox.exceptions import ResourceError, ShapeError
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
from .neuron import OfflineNeuron
from .neuron.neurons import IF, BypassNeuron
from .neuron.utils import NeuFireState, v_overflow
from .projection import InputProj
from .synapses import ConnType, FullConnSyn
from .synapses.conv_types import SizeAnyType, _Size1Type, _Size2Type
from .synapses.conv_utils import (
    _conv1d_oshape,
    _conv2d_oshape,
    _pair,
    _single,
    fm_ndim1_check,
    fm_ndim2_check,
)
from .synapses.transforms import (
    Conv1dForward,
    Conv2dForward,
    _Pool1dForward,
    _Pool2dForward,
    _PoolNdForward,
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
        neuron: NeuDyn | InputProj,
        chain_level: int = 1,
        *,
        keep_shape: bool = True,
        name: str | None = None,
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
        neuron_s: NeuDyn | InputProj,
        shape_out: tuple[int, ...],
        keep_shape: bool = False,
        name: str | None = None,
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
        self, ci: int, hi: int, kw: int, valid_interval: int
    ) -> None:
        """Check the limit of the semi-folded operators on the input buffer length of the core during the build phase.

        NOTE: If the condition is not met, an expection will be raised in the subsequent compilation phase.
        """
        E = math.ceil(
            math.log2(math.ceil(ci * hi * kw / OffCoreCfg.N_FANIN_PER_DENDRITE_ANN))
        )
        rin_deep = min(hi - kw, kw - 1) * valid_interval + 1
        if not OffCoreCfg.N_TIMESLOT_MAX / (2**E) > rin_deep:
            raise ResourceError(
                f"the input size of {self.name} is too large. Please adjust the input size or the number of channels."
            )
        buffer_deep = kw * valid_interval
        if buffer_deep > OffCoreCfg.N_TIMESLOT_MAX / (2**E):
            self.rin_buffer_option = True
        if self.rin_buffer_option:
            print("rin buffer has been enabled.")


class _LinearBase(FunctionalModule):
    def __init__(
        self,
        neuron_s: NeuDyn | InputProj,
        out_features: Shape,
        weights: np.ndarray,
        bias: DataType = 0,
        bit_trunc: int = 8,
        *,
        keep_shape: bool = False,
        name: str | None = None,
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
class _SpikingPoolNd(FunctionalModule):
    inherent_delay = 0
    tfm: _PoolNdForward

    def __init__(
        self,
        neuron: NeuDyn | InputProj,
        shape_out: SizeAnyType,
        keep_shape: bool,
        name: str | None,
        **kwargs,
    ) -> None:
        """Basic Nd pooling."""
        _pool_ksize_check(self.tfm.ksize, self.tfm.in_shape, self.tfm.padding)
        super().__init__(
            neuron, shape_out=shape_out, keep_shape=keep_shape, name=name, **kwargs
        )


class _SpikingPool1d(_SpikingPoolNd):
    tfm: _Pool1dForward

    def __init__(
        self,
        neuron: NeuDyn | InputProj,
        kernel_size: _Size1Type,
        pool_type: Literal["avg", "max"],
        stride: _Size1Type | None = None,
        padding: _Size1Type = 0,
        threshold: int | None = None,
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 1d spiking pooling."""
        _pool_type_check(pool_type)
        ci, il = fm_ndim1_check(neuron.shape_out, "CL")

        _ksize = _single(kernel_size)
        _stride = _single(stride) if stride is not None else _ksize
        _padding = _single(padding)

        ol = (il + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1

        if keep_shape:
            shape_out = (ci, ol)
        else:
            shape_out = (ci * ol,)

        self.tfm = _Pool1dForward(
            ci, (il,), (ol,), _ksize, _stride, _padding, pool_type, threshold
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
            n1_p1d = OfflineNeuron(
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
            weights=self.tfm.connectivity.astype(bool),
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
        neuron: NeuDyn | InputProj,
        kernel_size: _Size1Type,
        stride: _Size1Type | None = None,
        padding: _Size1Type = 0,
        pos_thres: int | None = None,
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 1d spiking pooling with voltage at the previous timestep."""
        ci, il = fm_ndim1_check(neuron.shape_out, "CL")

        _ksize = _single(kernel_size)
        _kernel = np.ones((ci, ci, *_ksize), dtype=WEIGHT_DTYPE)
        _stride = _single(stride) if stride is not None else _ksize
        _padding = _single(padding)

        ol = (il + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1

        if keep_shape:
            shape_out = (ci, ol)
        else:
            shape_out = (ci * ol,)

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
        return v_overflow(vjt_pre + self.tfm(x1).ravel())

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
            weights=self.tfm.connectivity.astype(bool),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p1d, syn1]
        self._rebuild_out_intf(network, n1_p1d, *generated, **build_options)

        return generated


class _SpikingPool2d(_SpikingPoolNd):
    tfm: _Pool2dForward

    def __init__(
        self,
        neuron: NeuDyn | InputProj,
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: _Size2Type | None = None,
        padding: _Size2Type = 0,
        threshold: int | None = None,
        # fm_order: _Order3d = "CHW",
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling."""
        _pool_type_check(pool_type)
        ci, hi, wi = fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        ho, wo = _conv2d_oshape((hi, wi), _ksize, _stride, _padding)

        if keep_shape:
            shape_out = (ci, ho, wo)
        else:
            shape_out = (ci * ho * wo,)

        self.tfm = _Pool2dForward(
            ci, (hi, wi), (ho, wo), _ksize, _stride, _padding, pool_type, threshold
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
            n1_p2d = OfflineNeuron(
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
            weights=self.tfm.connectivity.astype(bool),
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
        neuron: NeuDyn | InputProj,
        kernel_size: _Size2Type,
        stride: _Size2Type | None = None,
        padding: _Size2Type = 0,
        pos_thres: int | None = None,
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 2d spiking pooling with voltage at the previous timestep.

        NOTE: This is not a regular average pooling operator. It is just to correspond to the operators \
            that appear in PAIFLOW.
        """
        ci, hi, wi = fm_ndim2_check(neuron.shape_out, "CHW")

        _ksize = _pair(kernel_size)
        _kernel = np.ones((ci, ci, *_ksize), dtype=WEIGHT_DTYPE)
        _stride = _pair(stride) if stride is not None else _ksize
        _padding = _pair(padding)

        ho = (hi + 2 * _padding[0] - _ksize[0]) // _stride[0] + 1
        wo = (wi + 2 * _padding[1] - _ksize[1]) // _stride[1] + 1

        if keep_shape:
            shape_out = (ci, ho, wo)
        else:
            shape_out = (ci * ho * wo,)

        if isinstance(pos_thres, int):
            self.pos_thres = arg_check_non_neg(pos_thres, "positive threshold")
        else:
            self.pos_thres = typical_round(shape2num(_ksize) / 2)

        self.tfm = Conv2dForward((hi, wi), (ho, wo), _kernel, _stride, _padding)

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
        return v_overflow(vjt_pre + self.tfm(x1).ravel())

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
            weights=self.tfm.connectivity.astype(bool),
            conn_type=ConnType.All2All,
            name=f"s0_{self.name}",
        )

        generated = [n1_p2d, syn1]
        self._rebuild_out_intf(network, n1_p2d, *generated, **build_options)

        return generated


@set_rt_mode_ann()
class _PoolNd(FunctionalModule):
    inherent_delay = 0
    tfm: _PoolNdForward
    bit_trunc: int

    def __init__(
        self,
        neuron_s: NeuDyn | InputProj,
        shape_out: SizeAnyType,
        keep_shape: bool,
        name: str | None,
        **kwargs,
    ) -> None:
        """Basic Nd pooling."""
        _pool_ksize_check(self.tfm.ksize, self.tfm.in_shape, self.tfm.padding)
        super().__init__(
            neuron_s, shape_out=shape_out, keep_shape=keep_shape, name=name, **kwargs
        )


class _Pool1d(_PoolNd):
    tfm: _Pool1dForward

    def __init__(
        self,
        neuron_s: NeuDyn | InputProj,
        kernel_size: _Size1Type,
        pool_type: Literal["avg", "max"],
        stride: _Size1Type | None = None,
        padding: _Size1Type = 0,
        bit_trunc: int | None = None,
        keep_shape: bool = False,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 1d ANN pooling."""
        _pool_type_check(pool_type)
        ci, li = fm_ndim1_check(neuron_s.shape_out, "CL")

        ksize = _single(kernel_size)
        s = _single(stride) if stride is not None else ksize
        p = _single(padding)

        # NOTE: Division is achieved with the help of output truncation.
        # See comments in `AvgPool2dSemiFolded` in functional.py for more details.
        n_ksize = shape2num(ksize)
        self.bit_trunc = (
            8 + n_ksize.bit_length() - 1 if bit_trunc is None else bit_trunc
        )

        (lo,) = _conv1d_oshape((li,), ksize, s, p)
        k = ksize[0]
        assert 0 <= p[0] <= k

        self.tfm = _Pool1dForward(ci, (li,), (lo,), ksize, s, p, pool_type)
        super().__init__(
            neuron_s,
            shape_out=(ci, lo),
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class _Pool2d(_PoolNd):
    tfm: _Pool2dForward

    def __init__(
        self,
        neuron_s: NeuDyn | InputProj,
        kernel_size: _Size2Type,
        pool_type: Literal["avg", "max"],
        stride: _Size2Type | None = None,
        padding: _Size2Type = 0,
        bit_trunc: int | None = None,
        keep_shape: bool = False,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Basic 2d ANN pooling."""
        _pool_type_check(pool_type)
        ci, hi, wi = fm_ndim2_check(neuron_s.shape_out, "CHW")

        ksize = _pair(kernel_size)
        s = _pair(stride) if stride is not None else ksize
        p = _pair(padding)

        # NOTE: Division is achieved with the help of output truncation.
        # See comments in `AvgPool2dSemiFolded` in functional.py for more details.
        n_ksize = shape2num(ksize)
        self.bit_trunc = (
            8 + n_ksize.bit_length() - 1 if bit_trunc is None else bit_trunc
        )

        ho, wo = _conv2d_oshape((hi, wi), ksize, s, p)
        kh, kw = ksize
        ph, pw = p
        assert 0 <= ph <= kh and 0 <= pw <= kw

        self.tfm = _Pool2dForward(ci, (hi, wi), (ho, wo), ksize, s, p, pool_type)
        super().__init__(
            neuron_s, shape_out=(ci, ho, wo), keep_shape=keep_shape, name=name, **kwargs
        )


def _spike_func_avg_pool(
    vjt: VoltageType, pos_thres: int
) -> tuple[NeuOutType, VoltageType]:
    # Fire
    thres_mode = np.where(
        vjt >= pos_thres,
        NeuFireState.FIRING_POS,
        np.where(vjt < 0, NeuFireState.FIRING_NEG, NeuFireState.NOT_FIRING),
    )
    spike = thres_mode == NeuFireState.FIRING_POS
    # Reset
    v_reset = np.where(thres_mode == NeuFireState.FIRING_POS, 0, vjt)

    return spike.astype(NEUOUT_U8_DTYPE), v_reset


def _pool_type_check(pool_type: str) -> None:
    if pool_type not in ("avg", "max"):
        raise ValueError("type of pooling must be 'avg' or 'max'.")


def _pool_ksize_check(
    ksize: SizeAnyType, ifm_shape: SizeAnyType, padding: SizeAnyType
) -> None:
    eff_i = [i + 2 * p for i, p in zip(ifm_shape, padding)]
    if any(k > ei for k, ei in zip(ksize, eff_i)):
        raise ShapeError(
            f"Kernel size {ksize} > effective input size {tuple(eff_i)}, (p={padding})"
        )
