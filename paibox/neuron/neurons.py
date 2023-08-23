from typing import Optional
import numpy as np

from paibox.base import DynamicSys
from paibox.core.reg_types import (
    LCNExtensionType,
    MaxPoolingEnableType,
    SpikeWidthFormatType,
)
from paibox.neuron.ram_types import LeakingComparisonMode as LCM
from paibox.neuron.ram_types import LeakingDirectionMode as LDM
from paibox.neuron.ram_types import LeakingIntegrationMode as LIM
from paibox.neuron.ram_types import NegativeThresholdMode as NTM
from paibox.neuron.ram_types import ResetMode as RM
from paibox.neuron.ram_types import SynapticIntegrationMode as SIM
from paibox.neuron.ram_types import ThresholdMode as TM
from paibox.utils import fn_sgn


class MetaNeuron:
    """Meta neuron"""

    state = {}

    def __init__(
        self,
        reset_mode: RM,
        reset_v: int,
        leaking_comparison: LCM,
        threshold_mask_bits: int,
        neg_thres_mode: NTM,
        neg_threshold: int,
        pos_threshold: int,
        leaking_direction: LDM,
        leaking_integration_mode: LIM,
        leak_v: int,
        synaptic_integration_mode: SIM,
        bit_truncate: int,
        vjt_init: int,
    ) -> None:
        # SNN
        self._reset_mode: RM = reset_mode
        self._leaking_comparison: LCM = leaking_comparison
        self._neg_thres_mode: NTM = neg_thres_mode
        self._leaking_direction: LDM = leaking_direction
        self._synaptic_integration_mode: SIM = synaptic_integration_mode
        self._leaking_integration_mode: LIM = leaking_integration_mode
        self._threshold_mask_bits: int = threshold_mask_bits
        self._threshold_mask: int = (1 << threshold_mask_bits) - 1
        self._neg_threshold: int = neg_threshold  # Unsigned 29-bit
        self._pos_threshold: int = pos_threshold  # Unsigned 29-bit

        # ANN
        self._bit_truncate: int = bit_truncate

        """Inherent attributes"""

        # SNN
        self._vjt_init: int = vjt_init
        self._vjt_pre: int = vjt_init  # Membrane potential at Last time step.
        self._vjt: int = 0  # Membrane potential.
        self._spike: int = 0  # TODO Maybe need related to `tick_relative` & time slot?

        # ANN
        self._vj: int = vjt_init
        self._y: int = 0

        """Attributes below FIXED once reigistered."""
        # SNN
        self._reset_v: int = reset_v  # Signed 30-bit
        self._leak_v: int = leak_v  # Signed 30-bit

        # ANN
        self._bias: int = leak_v  # Signed 30-bit(ANN mode ONLY)

        """Inherited attributes from the core"""
        # SNN
        self._lcn_extension: LCNExtensionType
        self._spike_width_format: SpikeWidthFormatType

        # ANN
        self._pool_max_en: MaxPoolingEnableType

        """Auxiliary variables"""
        self._threshold_mode: TM = TM.NOT_EXCEEDED
        self._v_th_rand = 0

    def _neuronal_charge(self, x: np.ndarray) -> None:
        r"""1. Synaptic integration.

        Argument:
            - x: input to the neuron.

        Description:
            _rho_w_ij: Random synaptic integration enable, 0 or 1.

            If synaptic integration mode is deterministic, then
                `_vjt` = `_vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
            else (stochastic)
                `_vjt` = `_vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        """
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1
        xt = 0

        if self._synaptic_integration_mode is SIM.MODE_STOCHASTIC:
            xt = _rho_w_ij * x
        else:
            xt = x

        self._vjt = self._vjt_pre + int(xt)

    def _neuronal_leak(self) -> None:
        r"""2. Leaking integration.

        2.1 Leaking direction, forward or reversal.
            If leaking direction is `MODE_FORWARD`, the `_ld` is 1, else is \sgn{`_vjt`}.

        2.2 Random leaking.
            If leaking integration is `MODE_DETERMINISTIC`, then
                `_vjt` = `_vjt` + `_ld` * `_leak_v`
            else (`MODE_STOCHASTIC`)
                if abs(`_leak_v`) >= `_rho_j_lambda`, then
                    `_F` = 1
                else
                    `_F` = 0

                `_vjt` = `_vjt` + `_ld` * `_F` * \sgn{`_leak_v`}
        """
        _rho_j_lambda = 2  # Random leaking, unsigned 29-bit.
        _ld: int = 0

        if self._leaking_direction is LDM.MODE_FORWARD:
            _ld = 1
        else:
            _ld = fn_sgn(self._vjt, 0)

        if self._leaking_integration_mode is LIM.MODE_DETERMINISTIC:
            self._vjt = self._vjt + _ld * self._leak_v
        else:
            if self._leak_v >= _rho_j_lambda or self._leak_v <= -_rho_j_lambda:
                _F = 1
            else:
                _F = 0

            _sgn_leak_v = fn_sgn(self._leak_v, 0)

            self._vjt = self._vjt + _ld * _F * _sgn_leak_v

    def _neuronal_fire(self) -> None:
        r"""3. Threshold comparison

        3.1 Random threshold
            `_v_th_rand` = `_rho_j_T` & `_threshold_mask`

        3.2 Fire
            If negative threshold mode is `MODE_RESET`, then
                `_v_th_neg` = `_neg_threshold` + `_v_th_rand`
            else
                `_v_th_neg` = `_neg_threshold`

            If `_vjt` >= `_pos_threshold` + `_v_th_rand`, then
                `_spike` = 1
            else if `_vjt` < -`_v_th_neg`, then
                `_spike` = 0
            else
                `_spike` = 0
        """
        _rho_j_T = 3  # Random threshold, unsigned 29-bit.

        # TODO Is _rho_j_T permanent?
        _v_th_rand = _rho_j_T & self._threshold_mask
        self._v_th_rand = _v_th_rand

        if self._neg_thres_mode is NTM.MODE_RESET:
            _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self._neg_threshold

        """Fire"""
        if self._vjt >= self._pos_threshold + _v_th_rand:
            self._threshold_mode = TM.EXCEED_POSITIVE
            yt = 1
        elif self._vjt < -_v_th_neg:
            self._threshold_mode = TM.EXCEED_NEGATIVE
            yt = 0
        else:
            self._threshold_mode = TM.NOT_EXCEEDED
            yt = 0

        self._spike = yt

    def _neuronal_reset(self) -> None:
        r"""4. Reset

        If `_threshold_mode` is `EXCEED_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `_vjt` = `_reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `_vjt` = `_vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `_vjt` = `_vjt`

        else if `_threshold_mode` is `EXCEED_NEGATIVE`
            If negative threshold mode is `MODE_RESET`, then
                If reset mode is `MODE_NORMAL`, then
                    `_vjt` = -`_reset_v`
                else if reset mode is `MODE_LINEAR`, then
                    `_vjt` = `_vjt` + (`_neg_threshold` + `_v_th_rand`)
                else
                    `_vjt` = `_vjt`
            else (`MODE_SATURATION`)
                `_vjt` = `_neg_threshold`

        else (not beyond the threshold)
            `_vjt` = `_vjt`
        """
        if self._threshold_mode is TM.EXCEED_POSITIVE:
            if self._reset_mode is RM.MODE_NORMAL:
                self._vjt = self._reset_v
            elif self._reset_mode is RM.MODE_LINEAR:
                self._vjt = self._vjt - (self._pos_threshold + self._v_th_rand)
            else:
                self._vjt = self._vjt
        elif self._threshold_mode is TM.EXCEED_NEGATIVE:
            if self._neg_thres_mode is NTM.MODE_RESET:
                if self._reset_mode is RM.MODE_NORMAL:
                    self._vjt = -self._reset_v
                elif self._reset_mode is RM.MODE_LINEAR:
                    self._vjt = self._vjt + (self._neg_threshold + self._v_th_rand)
                else:
                    self._vjt = self._vjt
            else:
                self._vjt = -self._neg_threshold
        else:
            self._vjt = self._vjt

    def _relu(self) -> None:
        r"""ReLU(ANN mode ONLY)

        If spiking width format is `WIDTH_1BIT`, then
            if `_vj` >= `_pos_threshold`, then
                `_yj` = 1
            else
                `_yj` = 0
        else (`WIDTH_8BIT`)
            `_vj` >= `_pos_threshold`, then
                `_yj` = `y_truncated`
            else
                `_yj` = 0

        NOTE: Truncation of membrane potential
            _bit_truncate   Position of truncation
                0                  8'd0
                1               [0], 7'd0
                2              [1:0], 6'd0
                X            [X-1:0], {8-X}'d0
                7              [6:0], 1'd0
                8                 [7:0]
               ...                 ...
                X               [X-1:X-8]
        """
        _yj = 0

        if self._vj < self._pos_threshold:
            _yj = 0
        else:
            if self._spike_width_format is SpikeWidthFormatType.WIDTH_1BIT:
                _yj = 1
            else:
                if self._bit_truncate == 0:
                    _yj = 0
                elif self._bit_truncate < 8:
                    _mask = (1 << self._bit_truncate) - 1
                    _truncated_vj: int = self._vj & _mask
                    _yj = _truncated_vj << (8 - self._bit_truncate)
                else:
                    _yj = (self._vj >> self._bit_truncate - 8) & ((1 << 8) - 1)

        self._y = _yj

    def _max_pooling(self, x: np.ndarray) -> int:
        return int(x.max())

    def _post_hook(self) -> None:
        """Post-hook after the entire activation."""
        # Update the vjt_pre, and reset the threshold mode.
        self._vjt_pre = self._vjt
        self._threshold_mode = TM.NOT_EXCEEDED

    def update(self, x: np.ndarray) -> int:
        """Single-step update.

        TODO type of x may be considered as np.integer.
        """

        """1. Charge"""
        self._neuronal_charge(x)

        """2. Leak & fire"""
        if self._leaking_comparison is LCM.LEAK_BEFORE_COMP:
            self._neuronal_leak()
            self._neuronal_fire()
        else:
            self._neuronal_fire()
            self._neuronal_leak()

        """3. Reset"""
        self._neuronal_reset()

        # State update
        self._post_hook()

        return self._spike


class BaseNeuron(MetaNeuron):
    state = {}

    @property
    def shape_in(self) -> int:
        return 1

    @property
    def shape_out(self) -> int:
        return 1

    @property
    def detectable(self):
        return ("output",) + tuple(self.state)

    def reset(self):
        self._vjt = self._vjt_init
        self._vjt_pre = self._vjt


class TonicSpikingNeuron(BaseNeuron, DynamicSys):
    """Tonic spiking neuron"""

    def __init__(self, fire_step: int, vjt_init: int = 0, name: Optional[str] = None):
        """
        Arguments:
            - fire_step: every `N` spike, the neuron will fire positively.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            The neuron receives `N` spikes and fires, then resets to 0.
            `N` stands for firing steps.
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = 0
        _pos_thres = fire_step
        _neg_thres = 0
        _mask = 0
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _reset_mode,
            _reset_v,
            _lc,
            _mask,
            _ntm,
            _neg_thres,
            _pos_thres,
            _ld,
            _lim,
            _leak_v,
            _sim,
            _bt,
            vjt_init,
        )
        super(MetaNeuron, self).__init__(name)


class PhasicSpikingNeuron(MetaNeuron):
    """Phasic spiking neuron"""

    def __init__(
        self,
        time_to_fire: int,
        neg_floor: int = 10,
        vjt_init: int = 0,
    ):
        """
        Arguments:
            - time_to_fire: after `time_to_fire` spikes, the neuron will fire positively.
            - neg_floor: the negative floor that the neuron stays once firing. Default is 10 (unsigned).
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            The neuron receives `N` spikes and fires, then resets the membrane potential to 0,
            and never fires again.

            `N` stands for `time_to_fire`.
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_REVERSAL
        _lc = LCM.LEAK_BEFORE_COMP
        _leak_v = 1
        _pos_thres = (1 + _leak_v) * time_to_fire
        _neg_thres = neg_floor
        _mask = 0
        _reset_v = -1 - _neg_thres
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _reset_mode,
            _reset_v,
            _lc,
            _mask,
            _ntm,
            _neg_thres,
            _pos_thres,
            _ld,
            _lim,
            _leak_v,
            _sim,
            _bt,
            vjt_init,
        )
