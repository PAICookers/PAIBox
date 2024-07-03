import warnings
from collections.abc import Iterable
from typing import Any, Literal, NoReturn, Optional, Union

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    TM,
    HwConfig,
    MaxPoolingEnable,
    SpikeWidthFormat,
)

from paibox.base import NeuDyn
from paibox.exceptions import PAIBoxWarning, ShapeError
from paibox.types import LeakVType, Shape, SpikeType, VoltageType
from paibox.utils import (
    arg_check_non_neg,
    arg_check_non_pos,
    arg_check_pos,
    as_shape,
    shape2num,
)

from .utils import NEG_THRES_MIN, _is_leak_v_overflow, _mask, vjt_overflow

__all__ = ["Neuron"]

L = Literal


class MetaNeuron:
    """Meta neuron"""

    def __init__(
        self,
        shape: Shape,
        reset_mode: RM,
        reset_v: int,
        leak_comparison: LCM,
        threshold_mask_bits: int,
        neg_thres_mode: NTM,
        neg_threshold: int,
        pos_threshold: int,
        leak_direction: LDM,
        leak_integr: LIM,
        leak_v: Union[int, LeakVType],
        synaptic_integr: SIM,
        bit_truncation: int,
        overflow_strict: bool,
        keep_shape: bool = False,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._shape = as_shape(shape)
        self._n_neuron = shape2num(self._shape)

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self.reset_mode = reset_mode
        self.reset_v = reset_v  # Signed 30-bit
        self.leak_comparison = leak_comparison
        self.threshold_mask_bits = threshold_mask_bits
        self.neg_thres_mode = neg_thres_mode
        self.neg_threshold = (-1) * neg_threshold  # Unsigned 29-bit
        self.pos_threshold = pos_threshold  # Unsigned 29-bit
        self.leak_direction = leak_direction
        self.leak_integr = leak_integr
        self.synaptic_integr = synaptic_integr
        self.bit_truncation = bit_truncation  # Unsigned 5-bit

        if isinstance(leak_v, int) or leak_v.size == 1:
            # np.array([x]) is treated as a scalar.
            self.leak_v = int(leak_v)
        elif np.prod(leak_v.shape) == np.prod(self._shape):
            # leak with shape (32,32) == (1,32,32) is allowed.
            self.leak_v = leak_v.ravel()
        elif leak_v.ndim == 1 and leak_v.shape[0] == self._shape[0]:
            self.leak_v = np.repeat(leak_v, shape2num(self._shape[1:])).ravel()
        else:
            raise ShapeError(
                f"'leak' is either a scalar or have shape (output channels, ), but got ({self._shape[0]},)."
            )

        _is_leak_v_overflow(self.leak_v)

        # TODO These two config below are parameters of CORE.
        self._spike_width_format: SpikeWidthFormat
        self._pool_max_en: MaxPoolingEnable

        # Auxiliary attributes or variables.
        self._thres_mask = _mask(threshold_mask_bits)
        self.thres_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.uint8)
        self._v_th_rand = self.init_param(0).astype(np.int32)
        self.overflow_strict = overflow_strict

        if self.synaptic_integr is SIM.MODE_STOCHASTIC:
            warnings.warn(
                f"mode {SIM.MODE_STOCHASTIC.name} is configurated "
                f"but will not be simulated.",
                PAIBoxWarning,
            )

        if self.leak_integr is LIM.MODE_STOCHASTIC:
            warnings.warn(
                f"mode {LIM.MODE_STOCHASTIC.name} is configurated "
                f"but will not be simulated.",
                PAIBoxWarning,
            )

        if threshold_mask_bits > 0:
            warnings.warn(
                "random threshold is configurated but will not be simulated.",
                PAIBoxWarning,
            )

    def _neuronal_charge(
        self, incoming_v: VoltageType, vjt_pre: VoltageType, strict: bool = False
    ) -> VoltageType:
        r"""1. Synaptic integration.

        Description:
            _rho_w_ij: Random synaptic integration enable, 0 or 1.

            If synaptic integration mode is deterministic, then
                `vjt` = `vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j} (incoming_v)
            else (stochastic)
                `vjt` = `vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        """
        if incoming_v.ndim == 2:
            _v = incoming_v.sum(axis=1, dtype=np.int32)
        else:
            _v = incoming_v

        v_charged = np.add(vjt_pre, _v, dtype=np.int32)

        return vjt_overflow(v_charged, strict)  # Handle with overflow here

    def _neuronal_leak(self, vjt: VoltageType) -> VoltageType:
        r"""2. Leak integration.

        2.1 Leak direction, forward or reversal.
            If leak direction is `MODE_FORWARD`, the `_ld` is 1, else is \sgn{`vjt`}.

        2.2 Random leak.
            If leak integration is `MODE_DETERMINISTIC`, then
                `vjt` = `vjt` + `_ld` * `leak_v`
            else (`MODE_STOCHASTIC`)
                if abs(`leak_v`) >= `_rho_j_lambda`, then
                    `_F` = 1
                else
                    `_F` = 0

                `vjt` = `vjt` + \sgn{`leak_v`}* `_ld` * `_F`
        """
        if self.leak_direction is LDM.MODE_FORWARD:
            _ld = np.ones((self._n_neuron,), dtype=np.bool_)
        else:
            _ld = np.sign(vjt)

        v_leaked = np.add(vjt, _ld * self.leak_v, dtype=np.int32)

        return v_leaked

    def _neuronal_fire(self, vjt: VoltageType) -> SpikeType:
        r"""3. Threshold comparison.

        3.1 Random threshold.
            `_v_th_rand` = `_rho_j_T` & `_thres_mask`

        3.2 Fire.
            If negative threshold mode is `MODE_RESET`, then
                `_v_th_neg` = `neg_threshold` + `_v_th_rand`
            else
                `_v_th_neg` = `neg_threshold`

            If `vjt` >= `_pos_threshold` + `_v_th_rand`, then
                `spike` = 1
            else if `vjt` < -`_v_th_neg`, then
                `spike` = 0
            else
                `spike` = 0
        """
        # fixed at 0 since we won't simulate random threshold
        _v_th_rand = 0 & self._thres_mask
        self._v_th_rand = self.init_param(_v_th_rand).astype(np.int32)

        if self.neg_thres_mode is NTM.MODE_RESET:
            _v_th_neg = self.neg_threshold + _v_th_rand
        else:
            _v_th_neg = self.neg_threshold

        """Fire"""
        self.thres_mode = np.where(
            vjt >= self.pos_threshold + _v_th_rand,
            TM.EXCEED_POSITIVE,
            np.where(vjt < -_v_th_neg, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
        ).astype(np.uint8)

        spike = np.equal(self.thres_mode, TM.EXCEED_POSITIVE)

        return spike

    def _neuronal_reset(self, vjt: VoltageType) -> VoltageType:
        r"""4. Reset.

        If `thres_mode` is `EXCEED_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `vjt` = `reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `vjt` = `vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `vjt` = `vjt`

        else if `thres_mode` is `EXCEED_NEGATIVE`
            If negative threshold mode is `MODE_RESET`, then
                If reset mode is `MODE_NORMAL`, then
                    `vjt` = -`reset_v`
                else if reset mode is `MODE_LINEAR`, then
                    `vjt` = `vjt` + (`neg_threshold` + `_v_th_rand`)
                else
                    `vjt` = `vjt`
            else (`MODE_SATURATION`)
                `vjt` = `neg_threshold`

        else (not beyond the threshold)
            `vjt` = `vjt`
        """

        def _when_exceed_pos() -> VoltageType:
            if self.reset_mode is RM.MODE_NORMAL:
                return np.full((self._n_neuron,), self.reset_v, dtype=np.int32)

            elif self.reset_mode is RM.MODE_LINEAR:
                return np.subtract(
                    vjt, self.pos_threshold + self._v_th_rand, dtype=np.int32
                )
            else:  # RM.MODE_NONRESET
                return vjt

        def _when_exceed_neg() -> VoltageType:
            if self.neg_thres_mode is NTM.MODE_RESET:
                if self.reset_mode is RM.MODE_NORMAL:
                    return np.full((self._n_neuron,), -self.reset_v, dtype=np.int32)
                elif self.reset_mode is RM.MODE_LINEAR:
                    return np.add(
                        vjt,
                        self.neg_threshold + self._v_th_rand,
                        dtype=np.int32,
                    )
                else:  # RM.MODE_NONRESET
                    return vjt

            else:
                return np.full((self._n_neuron,), -self.neg_threshold, dtype=np.int32)

        # USE "=="!
        v_reset = np.where(
            self.thres_mode == TM.EXCEED_POSITIVE,
            _when_exceed_pos(),
            np.where(
                self.thres_mode == TM.EXCEED_NEGATIVE,
                _when_exceed_neg(),
                vjt,
            ),
        ).astype(np.int32)

        self._aux_post_hook()

        return v_reset

    def _relu(self, vj: VoltageType) -> VoltageType:
        r"""ReLU(ANN mode ONLY)

        If spiking width format is `WIDTH_1BIT`, then
            if `vj` >= `_pos_threshold`, then
                `_yj` = 1
            else
                `_yj` = 0
        else (`WIDTH_8BIT`)
            `vj` >= `_pos_threshold`, then
                `_yj` = `y_truncated`
            else
                `_yj` = 0

        NOTE: Truncation of membrane potential
            _bit_truncation   Position of truncation
                0                  8'd0
                1               [0], 7'd0
                2              [1:0], 6'd0
                X            [X-1:0], {8-X}'d0
                7              [6:0], 1'd0
                8                 [7:0]
               ...                 ...
                X               [X-1:X-8]
        """

        def _when_exceed_pos() -> VoltageType:
            if self._spike_width_format is SpikeWidthFormat.WIDTH_1BIT:
                return np.ones((self._n_neuron,), dtype=np.int32)

            if self.bit_truncation >= 8:
                return np.full(
                    (self._n_neuron,),
                    ((vj >> self.bit_truncation) - 8) & ((1 << 8) - 1),
                    dtype=np.int32,
                )
            elif self.bit_truncation > 0:
                _mask = (1 << self.bit_truncation) - 1
                _truncated_vj = vj & _mask
                return np.full(
                    (self._n_neuron,),
                    _truncated_vj << (8 - self.bit_truncation),
                    dtype=np.int32,
                )
            else:
                return np.zeros((self._n_neuron,), dtype=np.int32)

        y = np.where(
            vj >= self.pos_threshold,
            _when_exceed_pos(),
            np.zeros((self._n_neuron,), dtype=np.int32),
        ).astype(np.int32)

        return y

    def _max_pooling(self, x: np.ndarray) -> None:
        # TODO
        pass

    def _aux_pre_hook(self) -> None:
        """Pre-hook before the entire activation."""
        pass

    def _aux_post_hook(self) -> None:
        """Post-hook after the entire activation."""
        # Reset the auxiliary threshold mode.
        self.thres_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.uint8)

    def update(
        self, incoming_v: VoltageType, vjt_pre: VoltageType
    ) -> tuple[SpikeType, VoltageType, NDArray[np.uint8]]:
        """Update at one time step."""
        # 1. Charge
        v_charged = self._neuronal_charge(incoming_v, vjt_pre, self.overflow_strict)

        # 2. Leak & fire
        if self.leak_comparison is LCM.LEAK_BEFORE_COMP:
            v_leaked = self._neuronal_leak(v_charged)
            spike = self._neuronal_fire(v_leaked)
        else:
            spike = self._neuronal_fire(v_charged)
            v_leaked = self._neuronal_leak(v_charged)

        # Store the intermediate threshold mode & return
        _debug_thres_mode = self.thres_mode

        # 3. Reset
        v_reset = self._neuronal_reset(v_leaked)

        return spike, v_reset, _debug_thres_mode

    def init_param(self, param: Any) -> np.ndarray:
        return np.full((self._n_neuron,), param)

    @property
    def varshape(self) -> tuple[int, ...]:
        return self._shape if self.keep_shape else (self._n_neuron,)

    @property
    def bias(self) -> Union[int, LeakVType]:
        return self.leak_v


class Neuron(MetaNeuron, NeuDyn):
    _n_copied = 0
    """Counter of copies."""

    def __init__(
        self,
        shape: Shape,
        reset_mode: RM = RM.MODE_NORMAL,
        reset_v: int = 0,
        leak_comparison: LCM = LCM.LEAK_BEFORE_COMP,
        threshold_mask_bits: int = 0,
        neg_thres_mode: NTM = NTM.MODE_RESET,
        neg_threshold: int = NEG_THRES_MIN,
        pos_threshold: int = 1,
        leak_direction: LDM = LDM.MODE_FORWARD,
        leak_integration_mode: Union[L[0, 1], bool, LIM] = LIM.MODE_DETERMINISTIC,
        leak_v: Union[int, LeakVType] = 0,
        synaptic_integration_mode: Union[L[0, 1], bool, SIM] = SIM.MODE_DETERMINISTIC,
        bit_truncation: int = 0,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if neg_threshold > 0:
            # XXX *(-1) if passing a negative threshold > 0
            neg_threshold = (-1) * neg_threshold

        super().__init__(
            shape,
            reset_mode,
            reset_v,
            leak_comparison,
            threshold_mask_bits,
            neg_thres_mode,
            arg_check_non_pos(neg_threshold, "negative threshold"),
            arg_check_non_neg(pos_threshold, "positive threshold"),
            leak_direction,
            LIM(leak_integration_mode),
            leak_v,
            SIM(synaptic_integration_mode),
            arg_check_non_neg(bit_truncation, "bit of tuncation"),
            overflow_strict,
            keep_shape,
        )
        super(MetaNeuron, self).__init__(name)

        """Stateful attributes. Vector."""
        # Initial vjt is fixed at 0.
        self.set_memory("_vjt", self.init_param(0).astype(np.int32))
        self.set_memory("_inner_spike", self.init_param(0).astype(np.bool_))

        # Not supported for attributes in ANN mode
        self.set_memory("vj", self.init_param(0).astype(np.int32))
        self.set_memory("y", self.init_param(0).astype(np.int32))

        """Auxiliary internal stateful attributes for debugging"""
        self.set_memory(
            "_debug_thres_mode", self.init_param(TM.NOT_EXCEEDED).astype(np.uint8)
        )

        # Delay registers
        self.set_memory(
            "delay_registers",
            np.zeros(
                (HwConfig.N_TIMESLOT_MAX,) + self._inner_spike.shape, dtype=np.bool_
            ),
        )

        self._delay = arg_check_pos(delay, "'delay'")
        self._tws = arg_check_non_neg(tick_wait_start, "'tick_wait_start'")
        self._twe = arg_check_non_neg(tick_wait_end, "'tick_wait_end'")
        self._uf = arg_check_pos(unrolling_factor, "'unrolling_factor'")

    def __len__(self) -> int:
        return self._n_neuron

    def __call__(
        self, x: Optional[np.ndarray] = None, *args, **kwargs
    ) -> Optional[SpikeType]:
        return self.update(x, *args, **kwargs)

    def update(
        self, x: Optional[np.ndarray] = None, *args, **kwargs
    ) -> Optional[SpikeType]:
        # Priority order is a must.
        # The neuron doesn't work if `tws = 0` & done working
        # until `t - tws + 1 > twe` under the condition `twe > 0`.
        if not self.is_working():
            self._inner_spike = self.init_param(0).astype(np.bool_)
            return None

        # The neuron is going to work.
        if x is None:
            x = self.sum_inputs()

        self._inner_spike, self._vjt, self._debug_thres_mode = super().update(
            x, self._vjt
        )

        idx = (self.timestamp + self.delay_relative - 1) % HwConfig.N_TIMESLOT_MAX
        self.delay_registers[idx] = self._inner_spike.copy()

        return self._inner_spike

    def reset_state(self, *args, **kwargs) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def __copy__(self) -> "Neuron":
        """Same as `__deepcopy__`."""
        return self.__deepcopy__()

    def __deepcopy__(self) -> "Neuron":
        """Deepcopy a neuron.

        NOTE: It simply reinitializes a neuron with the parameters of the original neuron.
            Two neurons are not related.
        """
        self._n_copied += 1

        return Neuron(
            **self.attrs(all=True),
            name=f"{self.name}_copied_{self._n_copied}",
        )

    def copy(self) -> "Neuron":
        return self.__deepcopy__()

    def attrs(self, all: bool) -> dict[str, Any]:
        attrs = {
            "reset_mode": self.reset_mode,
            "reset_v": self.reset_v,
            "leak_comparison": self.leak_comparison,
            "threshold_mask_bits": self.threshold_mask_bits,
            "neg_thres_mode": self.neg_thres_mode,
            "neg_threshold": self.neg_threshold,
            "pos_threshold": self.pos_threshold,
            "leak_direction": self.leak_direction,
            "leak_integration_mode": self.leak_integr,
            "leak_v": self.leak_v,
            "synaptic_integration_mode": self.synaptic_integr,
            "bit_truncation": self.bit_truncation,
        }

        if all:
            attrs |= {
                "shape": self._shape,
                "keep_shape": self.keep_shape,
                "delay": self.delay_relative,
                "tick_wait_start": self.tick_wait_start,
                "tick_wait_end": self.tick_wait_end,
                "unrolling_factor": self.unrolling_factor,
                "overflow_strict": self.overflow_strict,
            }

        return attrs

    def _slice_attrs(
        self,
        index: Union[int, slice, tuple[Union[int, slice]]],
        all: bool = False,
        with_shape: bool = False,
    ) -> dict[str, Any]:
        """Slice the vector variables in the target.

        NOTE: since it does not participate in the simulation, all stateful attributes can be left \
            unrecorded.
        """
        attrs = self.attrs(all)

        for k, v in attrs.items():
            # Flatten the array-like attributes
            if isinstance(v, np.ndarray):
                if with_shape:
                    attrs[k] = v.reshape(self.varshape)[index]
                else:
                    attrs[k] = v.ravel()[index]

        return attrs

    def __getitem__(self, index) -> "NeuronSubView":
        return NeuronSubView(self, index)

    @property
    def shape_in(self) -> tuple[int, ...]:
        return self._shape

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape

    @property
    def num_in(self) -> int:
        return self._n_neuron

    @property
    def num_out(self) -> int:
        return self._n_neuron

    @property
    def output(self) -> SpikeType:
        return self.delay_registers

    @property
    def spike(self) -> SpikeType:
        return self._inner_spike

    @property
    def feature_map(self) -> SpikeType:
        return self._inner_spike.reshape(self.varshape)

    @property
    def voltage(self) -> VoltageType:
        return self._vjt.reshape(self.varshape)


class NeuronSubView(Neuron):
    __gh_build_ignore__ = True

    def __init__(
        self,
        target: Neuron,
        index: Union[int, slice, tuple[Union[int, slice]]],
        name: Optional[str] = None,
    ) -> None:
        if isinstance(index, (int, slice)):
            index = (index,)

        if len(index) > len(target.varshape):
            raise ValueError(
                f"index {index} is too long for target's shape {target.varshape}."
            )

        self.target = target
        self.index = index

        shape = []
        for i, idx in enumerate(self.index):
            if isinstance(idx, int):
                shape.append(1)
            elif isinstance(idx, slice):
                shape.append(len(range(target.varshape[i])[idx]))
            elif not isinstance(idx, Iterable):
                raise TypeError(
                    f"the index should be an iterable, but got {type(idx)}."
                )
            else:
                shape.append(len(idx))

        shape += list(target.varshape[len(self.index) :])

        super().__init__(
            shape,
            **target._slice_attrs(self.index, with_shape=True),
            keep_shape=target.keep_shape,
            name=name,
        )

    def update(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            f"{NeuronSubView.__name__} {self.name} cannot be updated."
        )

    def reset_state(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            f"{NeuronSubView.__name__} {self.name} cannot be reset."
        )
