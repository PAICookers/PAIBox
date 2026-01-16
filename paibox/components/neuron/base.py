import sys
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    CoreMode,
    DecayRandomEnable,
    InputWidthFormat,
    LeakOrder,
    LUTDataType,
    MaxPoolingEnable,
    OnlineModeEnable,
    SNNModeEnable,
    SpikeWidthFormat,
    get_core_mode,
)

from paibox.base import DataFlowFormat, NeuDyn, is_learnable
from paibox.exceptions import ConfigInvalidError, ParamNotSimulatedWarning, ShapeError
from paibox.types import (
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    WEIGHT_DTYPE,
    LeakVType,
    NeuOutType,
    Shape,
    VoltageType,
)
from paibox.utils import arg_check_non_neg, arg_check_pos, as_shape, shape2num

from .utils import (
    BIT_TRUNC_MAX,
    NEG_THRES_MAX,
    NeuFireState,
    RTModeKwds,
    _input_width_format,
    _leak_v_check,
    _mask,
    _spike_width_format,
    get_delay_reg_len,
    v_overflow,
)

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

if sys.version_info >= (3, 14):
    from annotationlib import get_annotations
else:
    from typing_extensions import get_annotations

if TYPE_CHECKING:
    from ..synapses.learning import STDPSynAttrKwds

__all__ = ["Neuron", "OfflineNeuron", "OnlineNeuron"]

L = Literal
NEU_TARGET_CHIP_UNSET = -1


def _neg_thres_check(th: int | None, signed: bool) -> int:
    if th is None:
        return -NEG_THRES_MAX
    elif signed:
        return th
    else:
        return th if th < 0 else -th


class Neuron(NeuDyn):
    _n_copied = 0
    """Counter of copies."""

    rt_mode_kwds: RTModeKwds
    mode: CoreMode
    online: ClassVar[bool]

    def __init__(
        self,
        shape: Shape,
        reset_v: int = 0,
        leak_v: int | LeakVType = 0,
        pos_threshold: int = 1,
        leak_comparison: LCM = LCM.LEAK_BEFORE_COMP,
        init_v: int | np.ndarray = 0,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        target_chip: int | None = None,
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._shape = as_shape(shape)
        self._n_neuron = shape2num(self._shape)

        # Core parameters, common to all neuron types
        self.reset_v = reset_v
        self.pos_threshold = arg_check_non_neg(pos_threshold, "'pos_threshold'")
        self.leak_comparison = leak_comparison
        self.overflow_strict = overflow_strict
        self.v0 = (
            self.init_voltage(init_v)
            if isinstance(init_v, int)
            else init_v.astype(VOLTAGE_DTYPE).ravel()
        )

        # Handle leak_v parameter
        if isinstance(leak_v, int) or (hasattr(leak_v, "size") and leak_v.size == 1):
            self.leak_v = int(leak_v)
        elif np.prod(leak_v.shape) == np.prod(self._shape):
            self.leak_v = leak_v.ravel()
        elif (
            hasattr(leak_v, "ndim")
            and leak_v.ndim == 1
            and leak_v.shape[0] == self._shape[0]
        ):
            self.leak_v = np.repeat(leak_v, shape2num(self._shape[1:])).ravel()
        else:
            raise ShapeError(
                f"'leak' is either a scalar or have shape (output channels, ), but got ({self._shape[0]},)."
            )

        _leak_v_check(self.leak_v)

        # Common stateful attributes
        self.set_memory("v", self.v0)
        self.set_memory("_neu_out", self.init_neu_out())

        # Non-stateful attributes.
        self._delay = arg_check_pos(delay, "'delay'")
        self._tws = arg_check_non_neg(tick_wait_start, "'tick_wait_start'")
        self._twe = arg_check_non_neg(tick_wait_end, "'tick_wait_end'")
        self._uf = arg_check_pos(unrolling_factor, "'unrolling_factor'")
        self.target_chip_idx = (
            NEU_TARGET_CHIP_UNSET
            if target_chip is None
            else arg_check_pos(target_chip, "'target_chip'")
        )
        # Default dataflow is infinite and continuous, starting at tws+0.
        self._oflow_format = DataFlowFormat(0, is_local_time=True)

    def __call__(
        self, x: np.ndarray | None = None, *args, **kwargs
    ) -> NeuOutType | None:
        return self.update(x, *args, **kwargs)

    def update(self, x: np.ndarray | None = None, *args, **kwargs) -> NeuOutType | None:
        raise NotImplementedError("Subclasses must implement this method")

    def step(
        self, incoming_v: VoltageType, v_pre: VoltageType, *args, **kwargs
    ) -> tuple[NeuOutType, VoltageType]:
        raise NotImplementedError("Subclasses must implement this method")

    def put_out_in_delay_reg(self, neu_out: NeuOutType) -> None:
        idx = (self.timestamp + self.delay_relative - 1) % self.delay_reg_len
        self.delay_registers[idx] = neu_out.copy()

    def init_param(self, val: Any) -> np.ndarray:
        return np.full((self._n_neuron,), val)

    def init_voltage(self, v: int) -> VoltageType:
        return self.init_param(v).astype(VOLTAGE_DTYPE)

    def init_neu_out(self) -> NeuOutType:
        return self.init_param(0).astype(NEUOUT_U8_DTYPE)

    def init_delay_registers(self) -> None:
        self.set_memory(
            "delay_registers",
            np.zeros((self.delay_reg_len,) + self.output.shape, dtype=NEUOUT_U8_DTYPE),
        )

    def reset_state(self, *args, **kwargs) -> None:
        self.reset_memory()  # Call reset of `StatusMemory`.

    def set_oflow_format(
        self,
        t_1st_vld: int | None = None,
        interval: int | None = None,
        n_vld: int | None = None,
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

    def __len__(self) -> int:
        return self._n_neuron

    def __copy__(self):
        """Same as `__deepcopy__`."""
        return self.__deepcopy__()

    def __deepcopy__(self, memo=None):
        """Deepcopy a neuron.

        NOTE: It simply reinitializes a neuron with the parameters of the original neuron.
            Two neurons are not related.
        """
        self._n_copied += 1

        neu_type = OnlineNeuron if is_learnable(self) else OfflineNeuron

        return neu_type(
            **self.attrs(for_copy=True),
            name=f"{self.name}_copied_{self._n_copied}",
        )

    def copy(self):
        return self.__deepcopy__()

    def attrs(self, for_copy: bool = False) -> dict[str, Any]:
        attrs = {
            "reset_v": self.reset_v,
            "leak_v": self.leak_v,
            "pos_threshold": self.pos_threshold,
            "leak_comparison": self.leak_comparison,
            "init_v": self.v0,
        }
        if for_copy:
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
        index: int | slice | list[int] | tuple[int | slice],
        for_copy: bool = False,
    ) -> dict[str, Any]:
        """Slice the vector variables in the target.

        NOTE: since it does not participate in the simulation, all stateful attributes can be left \
            unrecorded.
        """
        attrs = self.attrs(for_copy)

        for k, v in attrs.items():
            # Flatten the array-like attributes
            if isinstance(v, np.ndarray):
                attrs[k] = v.ravel()[index]

        return attrs

    def has_spike(self) -> bool:
        return bool(np.any(self.spike > 0))

    @property
    def shape_in(self) -> tuple[int, ...]:
        return self._shape

    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape

    @property
    def varshape(self) -> tuple[int, ...]:
        return self._shape if self.keep_shape else (self._n_neuron,)

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
        return self.v.reshape(self.varshape)

    @property
    def bias(self) -> int | LeakVType:
        return self.leak_v

    @property
    def delay_reg_len(self) -> int:
        return get_delay_reg_len(self)


def bit_truncate(v: VoltageType, bit: int = 8) -> VoltageType:
    def _truncate_below_u8(vt):
        if bit == 0:
            return 0
        elif bit < 8:
            return (vt << (8 - bit)) & _mask(8)
        else:
            return (vt >> (bit - 8)) & _mask(8)

    # Saturate truncation
    return np.where((v >> bit) > 0, _mask(8), _truncate_below_u8(v))


class OfflineNeuron(Neuron):
    online: ClassVar[bool] = False

    def __init__(
        self,
        shape: Shape,
        reset_mode: RM = RM.MODE_NORMAL,
        reset_v: int = 0,
        leak_comparison: LCM = LCM.LEAK_BEFORE_COMP,
        thres_mask_bits: int = 0,
        neg_thres_mode: NTM = NTM.MODE_RESET,
        neg_threshold: int | None = None,
        pos_threshold: int = 1,
        leak_direction: LDM = LDM.MODE_FORWARD,
        leak_integration_mode: L[0, 1] | bool | LIM = LIM.MODE_DETERMINISTIC,
        leak_v: int | LeakVType = 0,
        syn_integration_mode: L[0, 1] | bool | SIM = SIM.MODE_DETERMINISTIC,
        bit_trunc: int = 8,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        input_width: L[1, 8] | InputWidthFormat = InputWidthFormat.WIDTH_1BIT,
        spike_width: L[1, 8] | SpikeWidthFormat = SpikeWidthFormat.WIDTH_1BIT,
        snn_en: bool | SNNModeEnable = True,
        pool_max: bool | MaxPoolingEnable = False,
        target_chip: int | None = None,
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape,
            reset_v,
            leak_v,
            pos_threshold,
            leak_comparison,
            0,  # Fixed to 0 for offline neuron
            delay,
            tick_wait_start,
            tick_wait_end,
            target_chip,
            unrolling_factor,
            overflow_strict,
            keep_shape,
            name,
        )

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self.reset_mode = reset_mode
        # u29, but here always a negative int
        self.neg_threshold = _neg_thres_check(neg_threshold, signed=False)
        self.threshold_mask_bits = thres_mask_bits
        self.neg_thres_mode = neg_thres_mode
        self.leak_direction = leak_direction
        self.leak_integr = LIM(leak_integration_mode)
        self.synaptic_integr = SIM(syn_integration_mode)
        self.bit_trunc = arg_check_non_neg(bit_trunc, "bit truncation")  # u5
        self.pool_max = MaxPoolingEnable(pool_max)

        iw = _input_width_format(input_width)
        sw = _spike_width_format(spike_width)
        snn_en = SNNModeEnable(snn_en)

        self.rt_mode_kwds = {"input_width": iw, "spike_width": sw, "snn_en": snn_en}
        self.mode = get_core_mode(iw, sw, snn_en)

        if pool_max and self.mode != CoreMode.MODE_ANN:
            raise ConfigInvalidError(
                f"max pooling is only supported in {CoreMode.MODE_ANN.name}, "
                f"but got {self.mode.name}."
            )

        if self.synaptic_integr is SIM.MODE_STOCHASTIC:
            warnings.warn(
                f"mode {SIM.MODE_STOCHASTIC.name} is configurated "
                f"but will not be simulated.",
                ParamNotSimulatedWarning,
            )

        if self.leak_integr is LIM.MODE_STOCHASTIC:
            warnings.warn(
                f"mode {LIM.MODE_STOCHASTIC.name} is configurated "
                f"but will not be simulated.",
                ParamNotSimulatedWarning,
            )

        if self.threshold_mask_bits > 0:
            warnings.warn(
                "random threshold is configurated but will not be simulated.",
                ParamNotSimulatedWarning,
            )

        if self.bit_trunc > BIT_TRUNC_MAX:
            raise ValueError(
                f"'bit_trunc' should be less than or equal to {BIT_TRUNC_MAX}, but got {self.bit_trunc}."
            )

        self.init_delay_registers()

        # Auxiliary attributes or variables.
        self.thres_mode = self.init_param(NeuFireState.NOT_FIRING)
        self.overflow_strict = overflow_strict

    def _neuronal_charge(
        self, incoming_v: VoltageType, v_pre: VoltageType
    ) -> VoltageType:
        if incoming_v.ndim == 2:
            _v = np.sum(incoming_v, axis=1)
        else:
            _v = incoming_v

        if self.rt_mode_kwds["snn_en"]:
            v_charged = v_pre + _v
        else:
            # SNN_EN=0, the previous voltage is unused
            v_charged = _v

        return v_overflow(v_charged, self.overflow_strict)

    def _neuronal_leak(self, v: VoltageType) -> VoltageType:
        if self.rt_mode_kwds["snn_en"]:
            if self.leak_direction is LDM.MODE_FORWARD:
                _ld = 1
            else:
                _ld = np.sign(v)

            v_leaked = v + _ld * self.leak_v
        else:
            v_leaked = v + self.bias

        return v_overflow(v_leaked, self.overflow_strict)

    def _neuronal_fire(self, v: VoltageType) -> NeuOutType:
        self.thres_mode = np.where(
            v >= self.pos_threshold,
            NeuFireState.FIRING_POS,
            np.where(
                v < self.neg_threshold,
                NeuFireState.FIRING_NEG,
                NeuFireState.NOT_FIRING,
            ),
        )

        return (v >= self.pos_threshold).astype(NEUOUT_U8_DTYPE)

    def _neuronal_reset(self, v: VoltageType) -> VoltageType:
        def _when_exceed_pos() -> VoltageType:
            if self.reset_mode is RM.MODE_NORMAL:
                return np.full_like(v, self.reset_v)
            elif self.reset_mode is RM.MODE_LINEAR:
                return v - self.pos_threshold
            else:  # RM.MODE_NONRESET
                return v

        def _when_exceed_neg() -> VoltageType:
            if self.neg_thres_mode is NTM.MODE_RESET:
                if self.reset_mode is RM.MODE_NORMAL:
                    return np.full_like(v, -self.reset_v)
                elif self.reset_mode is RM.MODE_LINEAR:
                    return v - self.neg_threshold
                else:  # RM.MODE_NONRESET
                    return v
            else:  # NTM.MODE_SATURATION
                return np.full_like(v, self.neg_threshold)

        # USE "=="!
        v_reset = np.where(
            self.thres_mode == NeuFireState.FIRING_POS,
            _when_exceed_pos(),
            np.where(self.thres_mode == NeuFireState.FIRING_NEG, _when_exceed_neg(), v),
        )

        return v_reset.astype(VOLTAGE_DTYPE)

    def _bit_truncate(self, v: VoltageType) -> NeuOutType:
        v_truncated = np.where(
            self.thres_mode == NeuFireState.FIRING_POS,
            bit_truncate(v, self.bit_trunc),
            self.v0,
        )
        return v_truncated.astype(NEUOUT_U8_DTYPE)

    def _aux_pre_hook(self, *args, **kwargs) -> None:
        """Pre-hook before the entire update."""
        pass

    def _aux_post_hook(self, *args, **kwargs) -> None:
        """Post-hook after the entire update."""
        # Reset the auxiliary threshold mode
        self.thres_mode.fill(NeuFireState.NOT_FIRING)

    def step(
        self, incoming_v: VoltageType, v_pre: VoltageType, *args, **kwargs
    ) -> tuple[NeuOutType, VoltageType]:
        """Update at one timestep."""
        self._aux_pre_hook(*args, **kwargs)

        # 1. Charge
        v_charged = self._neuronal_charge(incoming_v, v_pre)

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

        self._aux_post_hook(*args, **kwargs)

        if self.rt_mode_kwds["spike_width"] is SpikeWidthFormat.WIDTH_1BIT:
            return spike, v_reset
        else:
            return v_truncated, v_reset

    def update(self, x: np.ndarray | None = None, *args, **kwargs) -> NeuOutType | None:
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

        self._neu_out, self.v = self.step(x, self.v)
        self.put_out_in_delay_reg(self._neu_out)
        return self._neu_out

    def attrs(self, for_copy: bool = False) -> dict[str, Any]:
        attrs = {
            "reset_mode": self.reset_mode,
            "neg_threshold": (-1) * self.neg_threshold,  # negative -> unsigned
            "thres_mask_bits": self.threshold_mask_bits,
            "neg_thres_mode": self.neg_thres_mode,
            "leak_direction": self.leak_direction,
            "leak_integration_mode": self.leak_integr,
            "syn_integration_mode": self.synaptic_integr,
            "bit_trunc": self.bit_trunc,
        }
        attrs |= super().attrs(for_copy)
        return attrs


class OnlineNeuron(Neuron):
    rt_mode_kwds = {
        "input_width": InputWidthFormat.WIDTH_1BIT,
        "spike_width": SpikeWidthFormat.WIDTH_1BIT,
        "snn_en": SNNModeEnable.ENABLE,
    }
    mode = CoreMode.MODE_SNN
    online: ClassVar[bool] = True

    # STDP synapse's attributes
    weight_decay_value: WEIGHT_DTYPE
    upper_weight: int
    lower_weight: int
    lut: LUTDataType
    lut_random_en: NDArray[np.uint8]
    decay_random_en: DecayRandomEnable
    random_seed: int
    online_mode_en: OnlineModeEnable
    plasticity_start: int
    plasticity_end: int

    def __init__(
        self,
        shape: Shape,
        reset_v: int = 0,
        leak_v: int | LeakVType = 0,
        neg_threshold: int | None = None,
        pos_threshold: int = 1,
        leak_comparison: LeakOrder = LeakOrder.LEAK_BEFORE_COMP,
        lateral_inhi_value: int = 0,
        init_v: int | np.ndarray = 0,
        *,
        lateral_inhi_target: "OnlineNeuron | Sequence[OnlineNeuron] | None" = None,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        target_chip: int | None = None,
        unrolling_factor: int = 1,
        overflow_strict: bool = False,
        keep_shape: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape,
            reset_v,
            leak_v,
            pos_threshold,
            leak_comparison,
            init_v,
            delay,
            tick_wait_start,
            tick_wait_end,
            target_chip,
            unrolling_factor,
            overflow_strict,
            keep_shape,
            name,
        )

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self.neg_threshold = _neg_thres_check(neg_threshold, signed=True)  # s32
        self.lateral_inhi_value = lateral_inhi_value  # s32

        self.init_delay_registers()

        # Common stateful variables
        # NOTE: Lateral inhibition can be performed in both inference & learning mode.
        # NOTE: The latertal inhibition will reset when receiving type I-2 work frame.
        self.set_memory("need_lateral_inhi", False)
        self.set_memory("source_lateral_inhi_flag", 0)  # controlled by the source

        # Auxiliary attributes or variables.
        self.lateral_inhi_source: set[OnlineNeuron] = set()
        self.lateral_inhi_target: set[OnlineNeuron] = set()

        # NOTE: Self is always a lateral inhibition source & target. However, when `lateral_inhi_value` == 0 & the only
        # target is the layer itself, no need to multicast the lateral inhibition.
        self.lateral_inhi_source.add(self)
        self.lateral_inhi_target.add(self)

        if lateral_inhi_target is not None:
            self.set_lateral_inhi_target(lateral_inhi_target)

        # Wether `_set_syn_attrs` is called by source STDP synapse for at least one time.
        self.syn_attrs_set = False

    def set_lateral_inhi_target(
        self, target: "OnlineNeuron | Sequence[OnlineNeuron]"
    ) -> None:
        """Set the lateral inhibition targets of the current layer. In order to support recursive lateral inhibition,   \
            it should be called after all the target neurons are created.
        """
        if isinstance(target, OnlineNeuron):
            self.lateral_inhi_target.add(target)
            target.lateral_inhi_source.add(self)
        else:
            for t in target:
                t.lateral_inhi_source.add(t)

            self.lateral_inhi_target.update(target)

    def _set_syn_attrs(self, **kwargs: Unpack["STDPSynAttrKwds"]) -> None:
        """Set the synapse attributes called by the source STDP synapse only."""
        annotations = get_annotations(OnlineNeuron)

        for k, v in kwargs.items():
            if k not in annotations:
                raise ValueError(f"'{k}' is not a valid annotation.")
            elif hasattr(self, k):
                if (cur_v := getattr(self, k)) != v:
                    raise ValueError(
                        f"Synapse's attribute '{k}' already exists, but with a different value: {cur_v} != {v}"
                    )
            else:
                setattr(self, k, v)

        self.syn_attrs_set = True

    def _aux_pre_hook(self, *args, **kwargs) -> None:
        """Pre-hook before the entire update."""
        # Before the entire update, update the lateral inhibition status
        self._update_lateral_inhi_status()

    def _aux_post_hook(self, *args, **kwargs) -> None:
        """Post-hook after the entire update."""
        self._update_target_lateral_inhi_status(*args)

    def _neuronal_charge(
        self, incoming_v: VoltageType, v_pre: VoltageType
    ) -> VoltageType:
        v = v_pre + incoming_v
        return v_overflow(v, self.overflow_strict)

    def _neuronal_leak(self, v: VoltageType) -> VoltageType:
        v += self.leak_v
        if self.need_lateral_inhi:
            v += self.lateral_inhi_value

        return v_overflow(v, self.overflow_strict)

    def _neuronal_fire(self, v: VoltageType) -> NeuOutType:
        # NOTE: The negative threshold is **signed** int
        v[v < self.neg_threshold] = self.neg_threshold
        spike = (v >= self.pos_threshold).astype(NEUOUT_U8_DTYPE)
        return spike

    def _neuronal_reset(self, v: VoltageType, spike: NeuOutType) -> VoltageType:
        v[spike] = self.reset_v
        return v

    def _update_lateral_inhi_status(self) -> None:
        """Update lateral inhibition status of the current neuron."""
        # NOTE: As long as the online cores receive any type I-4 work frames, do lateral inhibition.
        self.need_lateral_inhi = self.source_lateral_inhi_flag > 0
        self.source_lateral_inhi_flag = 0

    def _update_target_lateral_inhi_status(self, spike: NeuOutType) -> None:
        """Update lateral inhibition status of the current neuron & its targets."""
        # NOTE: If the online cores generate a spike, type I-4 work frames will be sent to the targets.
        # NOTE: Use the spike at **current** timestep, instead of `self.spike` which is the spike at last timestep.
        has_spike = int(np.any(spike > 0))
        for t in self.lateral_inhi_target:
            t.source_lateral_inhi_flag += has_spike

    def step(
        self, incoming_v: VoltageType, v_pre: VoltageType, *args, **kwargs
    ) -> tuple[NeuOutType, VoltageType]:
        """Update at one timestep."""
        self._aux_pre_hook(*args, **kwargs)

        # 1. Charge
        v_charged = self._neuronal_charge(incoming_v, v_pre)

        # 2. Leak & fire
        if self.leak_comparison is LeakOrder.LEAK_BEFORE_COMP:
            v_leaked = self._neuronal_leak(v_charged)
            spike = self._neuronal_fire(v_leaked)
        else:
            spike = self._neuronal_fire(v_charged)
            v_leaked = self._neuronal_leak(v_charged)

        # 3. Reset
        v_reset = self._neuronal_reset(v_leaked, spike)

        self._aux_post_hook(spike)

        return spike, v_reset

    def update(self, x: np.ndarray | None = None, *args, **kwargs) -> NeuOutType | None:
        if not self.is_working():
            self._neu_out.fill(0)
            return None

        if x is None:
            x = self.sum_inputs()
        else:
            x = np.atleast_1d(x)

        self._neu_out, self.v = self.step(x, self.v)
        self.put_out_in_delay_reg(self._neu_out)
        return self._neu_out

    def attrs(self, for_copy: bool = False) -> dict[str, Any]:
        attrs = {
            "lateral_inhi_value": self.lateral_inhi_value,
            "neg_threshold": self.neg_threshold,  # signed int
        }
        if self.syn_attrs_set:
            attrs |= {
                # Attributes of source STDP synapse `STDPSynAttrKwds`
                "weight_decay_value": self.weight_decay_value,
                "upper_weight": self.upper_weight,
                "lower_weight": self.lower_weight,
                "lut": self.lut,
                "lut_random_en": self.lut_random_en,
                "decay_random_en": self.decay_random_en,
                "random_seed": self.random_seed,
                "online_mode_en": self.online_mode_en,
            }
        attrs |= super().attrs(for_copy)
        return attrs
