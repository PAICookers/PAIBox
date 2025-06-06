import warnings
from collections.abc import Iterable
from typing import Any, Literal, NoReturn, Optional, Union

import numpy as np
from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    TM,
    CoreMode,
    HwConfig,
    InputWidthFormat,
    MaxPoolingEnable,
    SNNModeEnable,
    SpikeWidthFormat,
    get_core_mode,
)

from paibox.base import DataFlowFormat, NeuDyn
from paibox.exceptions import ConfigInvalidError, PAIBoxWarning, ShapeError
from paibox.types import (
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    LeakVType,
    NeuOutType,
    Shape,
    VoltageType,
)
from paibox.utils import (
    arg_check_non_neg,
    arg_check_non_pos,
    arg_check_pos,
    as_shape,
    shape2num,
)

from .utils import (
    BIT_TRUNCATE_MAX,
    NEG_THRES_MIN,
    RTModeKwds,
    _input_width_format,
    _leak_v_check,
    _mask,
    _spike_width_format,
    vjt_overflow,
)

__all__ = ["Neuron"]

L = Literal
NEU_TARGET_CHIP_NOT_SET = -1


class MetaNeuron:
    """Meta neuron"""

    rt_mode_kwds: RTModeKwds
    mode: CoreMode

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
        input_width: InputWidthFormat,
        spike_width: SpikeWidthFormat,
        snn_en: SNNModeEnable,
        pool_max: MaxPoolingEnable,
        overflow_strict: bool,
        keep_shape: bool = False,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._shape = as_shape(shape)
        self._n_neuron = shape2num(self._shape)

        self.rt_mode_kwds = {
            "input_width": input_width,
            "spike_width": spike_width,
            "snn_en": snn_en,
        }
        # check whether the mode is valid
        self.mode = get_core_mode(input_width, spike_width, snn_en)

        if pool_max and self.mode != CoreMode.MODE_ANN:
            raise ConfigInvalidError(
                f"max pooling is only supported in {CoreMode.MODE_ANN.name}, "
                f"but got {self.mode.name}."
            )

        self.pool_max = pool_max

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

        # Auxiliary attributes or variables.
        self._thres_mask = _mask(threshold_mask_bits)
        self.thres_mode = self.init_param(TM.NOT_EXCEEDED)
        self.overflow_strict = overflow_strict

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

        _leak_v_check(self.leak_v)

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

        if bit_truncation > BIT_TRUNCATE_MAX:
            raise ValueError(
                f"'bit_truncation' should be less than or equal to {BIT_TRUNCATE_MAX}, but got {bit_truncation}."
            )

    def _neuronal_charge(
        self, incoming_v: VoltageType, vjt_pre: VoltageType
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
            _v = np.sum(incoming_v, axis=1)
        else:
            _v = incoming_v

        if self.rt_mode_kwds["snn_en"]:
            v_charged = vjt_pre + _v
        else:
            # SNN_EN=0, the previous voltage is unused
            v_charged = _v

        return vjt_overflow(v_charged, self.overflow_strict)

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
        if self.rt_mode_kwds["snn_en"]:
            if self.leak_direction is LDM.MODE_FORWARD:
                _ld = 1
            else:
                _ld = np.sign(vjt)

            v_leaked = vjt + _ld * self.leak_v
        else:
            v_leaked = vjt + self.bias

        return vjt_overflow(v_leaked, self.overflow_strict)

    def _neuronal_fire(self, vjt: VoltageType) -> NeuOutType:
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
        self.thres_mode = np.where(
            vjt >= self.pos_threshold,
            TM.EXCEED_POSITIVE,
            np.where(vjt + self.neg_threshold < 0, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
        )

        spike = self.thres_mode == TM.EXCEED_POSITIVE
        return spike.astype(NEUOUT_U8_DTYPE)

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
                return np.full_like(vjt, self.reset_v)
            elif self.reset_mode is RM.MODE_LINEAR:
                return vjt - self.pos_threshold
            else:  # RM.MODE_NONRESET
                return vjt

        def _when_exceed_neg() -> VoltageType:
            if self.neg_thres_mode is NTM.MODE_RESET:
                if self.reset_mode is RM.MODE_NORMAL:
                    return np.full_like(vjt, -self.reset_v)
                elif self.reset_mode is RM.MODE_LINEAR:
                    return vjt + self.neg_threshold
                else:  # RM.MODE_NONRESET
                    return vjt
            else:
                return np.full_like(vjt, -self.neg_threshold)

        # USE "=="!
        v_reset = np.where(
            self.thres_mode == TM.EXCEED_POSITIVE,
            _when_exceed_pos(),
            np.where(self.thres_mode == TM.EXCEED_NEGATIVE, _when_exceed_neg(), vjt),
        )

        return v_reset.astype(VOLTAGE_DTYPE)

    def _bit_truncate(self, vj: VoltageType) -> NeuOutType:
        r"""Bit Truncation.

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

        NOTE: output under x-bit truncation
            _bit_truncation  Position of truncation
                0                   8'd0
                1                [0], 7'd0
                2               [1:0], 6'd0
                X             [X-1:0], {8-X}'d0
                7               [6:0], 1'd0
                8                  [7:0]
               ...                  ...
                X                [X-1:X-8]

            If the MSB of voltage is greater than the truncation bit, return 8'd255.
        """
        v_truncated = np.where(
            self.thres_mode == TM.EXCEED_POSITIVE,
            self._truncate(vj, self.bit_truncation),
            self._vjt0,
        )
        return v_truncated.astype(NEUOUT_U8_DTYPE)

    def _aux_pre_hook(self) -> None:
        """Pre-hook before the entire update."""
        pass

    def _aux_post_hook(self) -> None:
        """Post-hook after the entire update."""
        # Reset the auxiliary threshold mode
        self.thres_mode = self.init_param(TM.NOT_EXCEEDED)

    def update(
        self, incoming_v: VoltageType, vjt_pre: VoltageType
    ) -> tuple[NeuOutType, VoltageType]:
        """Update at one timestep."""
        self._aux_pre_hook()

        # 1. Charge
        v_charged = self._neuronal_charge(incoming_v, vjt_pre)

        # 2. Leak & fire
        if self.leak_comparison is LCM.LEAK_BEFORE_COMP:
            v_leaked = self._neuronal_leak(v_charged)
            spike = self._neuronal_fire(v_leaked)
        else:
            spike = self._neuronal_fire(v_charged)
            v_leaked = self._neuronal_leak(v_charged)

        # 3. Reset. Reset is performed in all modes.
        v_reset = self._neuronal_reset(v_leaked)

        if self.rt_mode_kwds["spike_width"] is SpikeWidthFormat.WIDTH_8BIT:
            # Althought the truncated voltage is of type VOLTAGE_DTYPE, its value <= uint8.
            # The voltage to truncate is the one before neuronal reset.
            v_truncated = self._bit_truncate(v_leaked)

        self._aux_post_hook()

        if self.rt_mode_kwds["spike_width"] is SpikeWidthFormat.WIDTH_1BIT:
            # When output width is 1 bit, bit truncation is not performed.
            return spike, v_reset
        else:
            return v_truncated, v_reset

    def init_param(self, param: Any) -> np.ndarray:
        return np.full((self._n_neuron,), param)

    @staticmethod
    def _truncate(v: VoltageType, bit_trunc: int) -> VoltageType:
        def _truncate_below_u8(vt):
            if bit_trunc == 0:
                return 0
            elif bit_trunc < 8:
                return (vt << (8 - bit_trunc)) & _mask(8)
            else:
                return (vt >> (bit_trunc - 8)) & _mask(8)

        # Saturate truncation
        return np.where((v >> bit_trunc) > 0, _mask(8), _truncate_below_u8(v))

    @property
    def _vjt0(self) -> VoltageType:
        return self.init_param(0).astype(VOLTAGE_DTYPE)

    @property
    def _neu_out0(self) -> NeuOutType:
        return self.init_param(0).astype(NEUOUT_U8_DTYPE)

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
        neg_threshold: Optional[int] = None,
        pos_threshold: int = 1,
        leak_direction: LDM = LDM.MODE_FORWARD,
        leak_integration_mode: Union[L[0, 1], bool, LIM] = LIM.MODE_DETERMINISTIC,
        leak_v: Union[int, LeakVType] = 0,
        synaptic_integration_mode: Union[L[0, 1], bool, SIM] = SIM.MODE_DETERMINISTIC,
        bit_truncation: int = 8,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        input_width: Union[L[1, 8], InputWidthFormat] = InputWidthFormat.WIDTH_1BIT,
        spike_width: Union[L[1, 8], SpikeWidthFormat] = SpikeWidthFormat.WIDTH_1BIT,
        snn_en: Union[bool, SNNModeEnable] = True,
        pool_max: Union[bool, MaxPoolingEnable] = False,
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        target_chip: int = NEU_TARGET_CHIP_NOT_SET,
        name: Optional[str] = None,
    ) -> None:
        if neg_threshold is None:
            neg_threshold = NEG_THRES_MIN

        if neg_threshold > 0:
            # XXX *(-1) if passing a negative threshold > 0
            neg_threshold = (-1) * neg_threshold

        if bit_truncation > BIT_TRUNCATE_MAX:
            raise ValueError(
                f"'bit_truncation' should be less than or equal to {BIT_TRUNCATE_MAX}, "
                f"but got {bit_truncation}."
            )

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
            _input_width_format(input_width),
            _spike_width_format(spike_width),
            SNNModeEnable(snn_en),
            MaxPoolingEnable(pool_max),
            overflow_strict,
            keep_shape,
        )
        super(MetaNeuron, self).__init__(name)

        """Stateful attributes. Vector."""
        self.set_memory("_vjt", self._vjt0)  # Initial vjt is fixed at 0.
        self.set_memory("_neu_out", self._neu_out0)
        self.set_memory(
            "delay_registers",
            np.zeros(
                (HwConfig.N_TIMESLOT_MAX,) + self._neu_out.shape, dtype=NEUOUT_U8_DTYPE
            ),
        )

        """Non-stateful attributes."""
        self._delay = arg_check_pos(delay, "'delay'")
        self._tws = arg_check_non_neg(tick_wait_start, "'tick_wait_start'")
        self._twe = arg_check_non_neg(tick_wait_end, "'tick_wait_end'")
        self._uf = arg_check_pos(unrolling_factor, "'unrolling_factor'")
        self.target_chip_idx = target_chip
        # Default dataflow is infinite and continuous, starting at tws+0.
        self._oflow_format = DataFlowFormat(0, is_local_time=True)

    def __len__(self) -> int:
        return self._n_neuron

    def __call__(
        self, x: Optional[np.ndarray] = None, *args, **kwargs
    ) -> Optional[NeuOutType]:
        return self.update(x, *args, **kwargs)

    def update(
        self, x: Optional[np.ndarray] = None, *args, **kwargs
    ) -> Optional[NeuOutType]:
        # Priority order is a must.
        # The neuron doesn't work if `tws = 0` & done working
        # until `t - tws + 1 > twe` under the condition `twe > 0`.
        if not self.is_working():
            self._neu_out.fill(0)
            return None

        if x is None:
            if self.pool_max:
                x = self.max_inputs()
            else:
                x = self.sum_inputs()
        else:
            x = np.atleast_1d(x)

        self._neu_out, self._vjt = super().update(x, self._vjt)

        idx = (self.timestamp + self.delay_relative - 1) % HwConfig.N_TIMESLOT_MAX
        self.delay_registers[idx] = self._neu_out.copy()

        return self._neu_out

    def reset_state(self, *args, **kwargs) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def set_oflow_format(
        self,
        t_1st_vld: Optional[int] = None,
        interval: Optional[int] = None,
        n_vld: Optional[int] = None,
        *,
        format_type: type[DataFlowFormat] = DataFlowFormat,
    ) -> None:
        """Set the attributes of output dataflow format by given arguments."""
        if hasattr(self, "_oflow_format"):
            _t_1st_vld = (
                t_1st_vld
                if isinstance(t_1st_vld, int)
                else self._oflow_format.t_1st_vld
            )
            _interval = (
                arg_check_pos(interval, "interval")
                if isinstance(interval, int)
                else self._oflow_format.interval
            )
            _n_vld = (
                arg_check_non_neg(n_vld, "n_vld")
                if isinstance(n_vld, int)
                else self._oflow_format.n_vld
            )
            self._assign_flow_format(_t_1st_vld, _interval, _n_vld)
        else:
            if not (
                isinstance(interval, int)
                and isinstance(n_vld, int)
                and isinstance(t_1st_vld, int)
            ):
                raise ValueError(
                    "if '_oflow_format' is not set, 't_1st_vld', 'interval' & 'n_vld' must be set."
                )

            self._oflow_format = format_type(t_1st_vld, interval, n_vld)
            self._oflow_format._check_after_assign(self.tick_wait_start, self.end_tick)

    def _assign_flow_format(self, t_1st_vld: int, intv: int, n_vld: int) -> None:
        self._oflow_format.t_1st_vld = t_1st_vld
        self._oflow_format.interval = intv
        self._oflow_format.n_vld = n_vld
        self._oflow_format._check_after_assign(self.tick_wait_start, self.end_tick)

    def __copy__(self) -> "Neuron":
        """Same as `__deepcopy__`."""
        return self.__deepcopy__()

    def __deepcopy__(self, memo=None) -> "Neuron":
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

    def shape_change(self, new_shape: Shape) -> None:
        # print(self.name,"shape change")
        self._n_neuron = shape2num(new_shape)
        self._shape = as_shape(new_shape)
        self._vjt = self.init_param(0).astype(np.int32)
        self.set_reset_value("_vjt", self._vjt)
        self._inner_spike = self.init_param(0).astype(np.bool_)
        self.set_reset_value("_inner_spike", self._inner_spike)
        self.vj = self.init_param(0).astype(np.int32)
        self.set_reset_value("vj", self.vj)
        self.y = self.init_param(0).astype(np.int32)
        self.set_reset_value("y", self.y)
        self.delay_registers = np.zeros(
            (HwConfig.N_TIMESLOT_MAX,) + self._inner_spike.shape, dtype=np.bool_
        )
        self.set_reset_value("delay_registers", self.delay_registers)

        return

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
    def output(self) -> NeuOutType:
        return self._neu_out

    @property
    def spike(self) -> NeuOutType:
        return self._neu_out

    @property
    def feature_map(self) -> NeuOutType:
        return self._neu_out.reshape(self.varshape)

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
