from typing import Optional, Tuple

import numpy as np

from paibox._types import Shape
from paibox.base import NeuDyn
from paibox.core.reg_types import MaxPoolingEnableType, SpikeWidthFormatType
from paibox.neuron.ram_types import LeakingComparisonMode as LCM
from paibox.neuron.ram_types import LeakingDirectionMode as LDM
from paibox.neuron.ram_types import LeakingIntegrationMode as LIM
from paibox.neuron.ram_types import NegativeThresholdMode as NTM
from paibox.neuron.ram_types import ResetMode as RM
from paibox.neuron.ram_types import SynapticIntegrationMode as SIM
from paibox.neuron.ram_types import ThresholdMode as TM
from paibox.utils import as_shape, fn_sgn, shape2num


class MetaNeuron:
    """Meta neuron"""

    def __init__(
        self,
        shape: Shape,
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
        *,
        keep_size: bool = False,
        **kwargs,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic
        self.keep_size = keep_size
        self.n_neurons = shape2num(shape)
        self._shape = as_shape(shape)

        # Configurations in SNN mode
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

        self._reset_v: int = reset_v  # Signed 30-bit
        self._leak_v: int = leak_v  # Signed 30-bit
        self._bias: int = leak_v  # Signed 30-bit(ANN mode ONLY)

        # Configurations in ANN mode
        self._bit_truncate: int = bit_truncate

        # TODO These two config below are parameters of CORE.
        self._spike_width_format: SpikeWidthFormatType
        self._pool_max_en: MaxPoolingEnableType

        """Stateful attributes. Vector."""
        # Attributes about SNN
        self._vjt_init = self.init_param(vjt_init)
        self._vjt_pre = self.init_param(
            vjt_init
        )  # Membrane potential at last timestep.
        self._vjt = self.init_param(vjt_init)  # Membrane potential.
        self._spike = self.init_param(0).astype(np.bool_)

        # Attributes about ANN
        self._vj = self.init_param(vjt_init)
        self._y = self.init_param(0)

        # Auxiliary attributes
        self._threshold_mode = self.init_param(TM.NOT_EXCEEDED)
        self._v_th_rand = self.init_param(0)

    def _neuronal_charge(self, x: np.ndarray) -> None:
        r"""1. Synaptic integration.

        Argument:
            - x: input to the neuron(s). [1*N]

        Description:
            _rho_w_ij: Random synaptic integration enable, 0 or 1.

            If synaptic integration mode is deterministic, then
                `_vjt` = `_vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
            else (stochastic)
                `_vjt` = `_vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        """
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1
        xt = np.zeros(self.varshape)

        if self._synaptic_integration_mode is SIM.MODE_STOCHASTIC:
            raise NotImplementedError
        else:
            if x.ndim == 2:
                xt = x.sum(axis=1)
            else:
                xt = x

        self._vjt = np.add(self._vjt_pre, xt)

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

        if self._leaking_direction is LDM.MODE_FORWARD:
            _ld = np.ones(self.varshape)
        else:
            _ld = np.vectorize(fn_sgn)(self._vjt, 0)

        if self._leaking_integration_mode is LIM.MODE_DETERMINISTIC:
            self._vjt = np.add(self._vjt, _ld * self._leak_v)
        else:
            _F = 1 if abs(self._leak_v) >= _rho_j_lambda else 0
            _sgn_leak_v = np.vectorize(fn_sgn)(self._leak_v, 0)

            self._vjt = np.add(self._vjt, _F * _ld @ _sgn_leak_v)

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

        # TODO Is _rho_j_T and _v_th_rand for all neurons or for each neuron?
        _v_th_rand = _rho_j_T & self._threshold_mask
        self._v_th_rand = np.full(self.varshape, _v_th_rand)

        if self._neg_thres_mode is NTM.MODE_RESET:
            _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self._neg_threshold

        """Fire"""
        self._threshold_mode = np.where(
            self._vjt >= self._pos_threshold + _v_th_rand,
            TM.EXCEED_POSITIVE,
            np.where(self._vjt < -_v_th_neg, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
        )

        self._spike = np.where(self._threshold_mode == TM.EXCEED_POSITIVE, 1, 0).astype(
            np.bool_
        )

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

        def _when_exceed_pos():
            if self._reset_mode is RM.MODE_NORMAL:
                return np.full(self.varshape, self._reset_v)

            elif self._reset_mode is RM.MODE_LINEAR:
                return np.subtract(self._vjt, (self._pos_threshold + self._v_th_rand))
            else:
                return self._vjt

        def _when_exceed_neg():
            if self._neg_thres_mode is NTM.MODE_RESET:
                if self._reset_mode is RM.MODE_NORMAL:
                    return np.full(self.varshape, -self._reset_v)
                elif self._reset_mode is RM.MODE_LINEAR:
                    return np.add(self._vjt, (self._neg_threshold + self._v_th_rand))
                else:
                    return self._vjt

            else:
                return np.full(self.varshape, -self._neg_threshold)

        self._vjt_pre = self._vjt = np.where(
            self._threshold_mode == TM.EXCEED_POSITIVE,
            _when_exceed_pos(),
            np.where(
                self._threshold_mode == TM.EXCEED_NEGATIVE,
                _when_exceed_neg(),
                self._vjt,
            ),
        )

        self._aux_post_hook()

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

        def _when_exceed_pos():
            if self._spike_width_format is SpikeWidthFormatType.WIDTH_1BIT:
                return np.ones(self.varshape)

            if self._bit_truncate >= 8:
                return ((self._vj >> self._bit_truncate) - 8) & ((1 << 8) - 1)
            elif self._bit_truncate > 0:
                _mask = (1 << self._bit_truncate) - 1
                _truncated_vj = self._vj & _mask
                return _truncated_vj << (8 - self._bit_truncate)
            else:
                return np.zeros(self.varshape)

        self._y = np.where(
            self._vj >= self._pos_threshold, _when_exceed_pos(), np.zeros(self.varshape)
        )

    def _max_pooling(self, x: np.ndarray):
        # TODO
        pass

    def _aux_pre_hook(self):
        """Pre-hook before the entire activation."""
        pass

    def _aux_post_hook(self) -> None:
        """Post-hook after the entire activation."""
        # Reset the auxiliary threshold mode.
        self._threshold_mode = np.full(self.varshape, TM.NOT_EXCEEDED)

    def update(self, x: np.ndarray) -> np.ndarray:
        """Update at one time step."""

        """1. Charge.
            Sum the membrane potential from the previous synapses.
            Don't care the /rho right now.
        """
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

        return self._spike

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self._shape if self.keep_size else (self.n_neurons,)

    def init_param(self, param) -> np.ndarray:
        return np.full(self.varshape, param)


class Neuron(MetaNeuron, NeuDyn):
    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self.varshape

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.varshape

    @property
    def num_in(self) -> int:
        return self.n_neurons

    @property
    def num_out(self) -> int:
        return self.n_neurons

    @property
    def output(self) -> np.ndarray:
        return self._spike

    @property
    def spike(self) -> np.ndarray:
        return self._spike

    @property
    def state(self) -> np.ndarray:
        return self._spike

    def __len__(self) -> int:
        return self.n_neurons

    def __call__(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.update(x)

    def update(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self.sum_inputs()

        return super().update(x)

    def reset(self) -> None:
        """Initialization, not the neuronal reset."""
        self._vjt = self._vjt_pre = self._vjt_init


class TonicSpikingNeuron(Neuron):
    """Tonic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        vjt_init: int = 0,
        *,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
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
            shape,
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
            keep_size=keep_size,
        )
        super(MetaNeuron, self).__init__(name)


class PhasicSpikingNeuron(Neuron):
    """Phasic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        time_to_fire: int,
        neg_floor: int = 10,
        vjt_init: int = 0,
        *,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
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
            shape,
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
            keep_size=keep_size,
        )
        super(MetaNeuron, self).__init__(name)
