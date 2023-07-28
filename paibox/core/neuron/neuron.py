from abc import ABC, abstractmethod
from typing import ClassVar, List

from ..params_types import *
from ._type import ThresholdMode


class _AbstractNeuron(ABC):
    """Abstract neuron."""

    @abstractmethod
    def neuronal_charge(self, x, w):
        raise NotImplementedError

    def neuronal_fire(self, x):
        raise NotImplementedError

    def neuronal_reset(self, x):
        raise NotImplementedError

    @abstractmethod
    def single_step_forward(self, x):
        raise NotImplementedError


class _MetaNeuron(_AbstractNeuron):
    """Naked neuron"""

    __runtime_mode: ClassVar[bool] = False

    def __init__(
        self,
        tick_relative: int,
        addr_axon: int,
        addr_core_x: int,
        addr_core_y: int,
        addr_core_x_ex: int,
        addr_core_y_ex: int,
        addr_chip_x: int,
        addr_chip_y: int,
        chip_x: int,
        chip_y: int,
        core_x: int,
        core_y: int,
        id: int,
        reset_mode: ResetMode,
        reset_v: int,
        leaking_comparison: LeakingComparisonMode,
        threshold_mask_bits: int,
        neg_thres_mode: NegativeThresholdMode,
        neg_threshold: int,
        pos_threshold: int,
        leaking_direction: LeakingDirectionMode,
        leaking_integration_mode: LeakingIntegrationMode,
        leak_v: int,
        synaptic_integration_mode: SynapticIntegrationMode,
        bit_truncate: int,
        vjt_pre: int,
        *args,
        **kwargs,
    ) -> None:
        # Common
        self._tick_relative: int
        self._addr_axon: int

        # SNN
        self._reset_mode: ResetMode = reset_mode
        self._leaking_comparison: LeakingComparisonMode = leaking_comparison
        self._neg_thres_mode: NegativeThresholdMode = neg_thres_mode
        self._leaking_direction: LeakingDirectionMode = leaking_direction
        self._synaptic_integration_mode: SynapticIntegrationMode = (
            synaptic_integration_mode
        )
        self._leaking_integration_mode: LeakingIntegrationMode = (
            leaking_integration_mode
        )
        self._threshold_mask: int = (1 << threshold_mask_bits) - 1
        self._neg_threshold: int = neg_threshold  # Unsigned 29-bit
        self._pos_threshold: int = pos_threshold  # Unsigned 29-bit

        # ANN
        self._bit_truncate: int = bit_truncate

        """Inherent attributes"""
        # Common
        self._chip_x: int = chip_x
        self._chip_y: int = chip_y
        self._core_x: int = core_x
        self._core_y: int = core_y
        self._id: int = id

        # SNN
        self._time_step = 0  # As an global class variable?
        self._vjt_pre: int = vjt_pre  # Membrane potential at Last time step.
        self._vjt: int = vjt_pre  # Membrane potential.
        self._spike: int = 0

        # ANN
        self._vj: int = vjt_pre
        self._y: int = 0

        """Attributes below FIXED once reigistered."""
        # Indicate the neuron is registered into a core or not.
        self._is_registered: bool = False

        # SNN
        self._reset_v: int = reset_v  # Signed 30-bit
        self._leak_v: int = leak_v  # Signed 30-bit

        # ANN
        self._bias: int = leak_v  # Signed 30-bit(ANN mode ONLY)

        """Inherited attributes from the core"""
        # SNN
        self._lcn_extension: LCNExtensionType = LCNExtensionType.LCN_1X
        self._spike_width_format: SpikeWidthFormatType = SpikeWidthFormatType.WIDTH_1BIT

        # ANN
        self._pool_max_en: MaxPoolingEnableType = MaxPoolingEnableType.ENABLE

        """Auxiliary variables"""
        self._threshold_mode: ThresholdMode = ThresholdMode.MODE_UNSET
        self._v_th_rand = 0

    def neuronal_charge(self, input_spikes: List[int], weights: List[int]) -> None:
        """1. Synaptic integration.

        _rho_w_ij: Random synaptic integration enable, 0 or 1.

        If synaptic integration mode is deterministic, then
            `_vjt` = `_vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        else (stochastic)
            `_vjt` = `_vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}

        Arguments:
        1. input_spikes, a spiking train width of `N` axons.
            | |
        1 - x x
        0 - x ·
        1 - · x, here length of x is 3.

        2. weights, length of `N` and of `WEIGHT_WIDTH_XBIT`-bit.
        """
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1
        xt = 0
        _vjt_pre = self._vjt

        for i in range(len(input_spikes)):
            _is = input_spikes[i]
            _w = weights[i]
            if (
                self._synaptic_integration_mode
                is SynapticIntegrationMode.MODE_DETERMINISTIC
            ):
                xt += _is * _w
            else:
                xt += _rho_w_ij * _is * _w

        self._vjt = _vjt_pre + xt

    def neuronal_leak(self) -> None:
        """2. Leaking integration.

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

        if self._leaking_direction is LeakingDirectionMode.MODE_FORWARD:
            _ld = 1
        else:
            _ld = 1 if self._vjt >= 0 else -1

        if self._leaking_integration_mode is LeakingIntegrationMode.MODE_DETERMINISTIC:
            self._vjt = self._vjt + _ld * self._leak_v
        else:
            if self._leak_v >= _rho_j_lambda or self._leak_v <= -_rho_j_lambda:
                _F = 1
            else:
                _F = 0

            if self._leak_v > 0:
                _sgn_leak_v = 1
            elif self._leak_v < 0:
                _sgn_leak_v = -1
            else:
                _sgn_leak_v = 0

            self._vjt = self._vjt + _ld * _F * _sgn_leak_v

    def neuronal_fire(self) -> None:
        """3. Threshold comparison

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

        if self._neg_thres_mode is NegativeThresholdMode.MODE_RESET:
            _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self._neg_threshold

        """Fire"""
        if self._vjt >= self._pos_threshold + _v_th_rand:
            self._threshold_mode = ThresholdMode.MODE_POSITIVE
            yt = 1
        elif self._vjt < -_v_th_neg:
            self._threshold_mode = ThresholdMode.MODE_NEGATIVE
            yt = 0
        else:
            yt = 0

        self._spike = yt

    def neuronal_reset(self) -> None:
        """4. Reset

        If `_threshold_mode` is `MODE_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `_vjt` = `_reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `_vjt` = `_vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `_vjt` = `_vjt`

        else (`MODE_NEGATIVE`)
            If negative threshold mode is `MODE_RESET`, then
                If reset mode is `MODE_NORMAL`, then
                    `_vjt` = -`_reset_v`
                else if reset mode is `MODE_LINEAR`, then
                    `_vjt` = `_vjt` + (`_neg_threshold` + `_v_th_rand`)
                else
                    `_vjt` = `_vjt`
            else (`MODE_SATURATION`)
                `_vjt` = `_neg_threshold`
        """
        if self._threshold_mode is ThresholdMode.MODE_POSITIVE:
            if self._reset_mode is ResetMode.MODE_NORMAL:
                self._vjt = self._reset_v
            elif self._reset_mode is ResetMode.MODE_LINEAR:
                self._vjt = self._vjt - (self._pos_threshold + self._v_th_rand)
            else:
                self._vjt = self._vjt
        else:
            if self._neg_thres_mode is NegativeThresholdMode.MODE_RESET:
                if self._reset_mode is ResetMode.MODE_NORMAL:
                    self._vjt = -self._reset_v
                elif self._reset_mode is ResetMode.MODE_LINEAR:
                    self._vjt = self._vjt + (self._neg_threshold + self._v_th_rand)
                else:
                    self._vjt = self._vjt
            else:
                self._vjt = -self._neg_threshold

    def _relu(self) -> None:
        """ReLU(ANN mode ONLY)

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

    def _max_pooling(self, input_spikes: List[int]) -> int:
        return max(input_spikes)

    def _pre_hook(self):
        pass

    def _post_hook(self):
        self._threshold_mode = ThresholdMode.MODE_UNSET

    def single_step_forward(self, x, w):
        self._pre_hook()

        """1. Charge"""
        self.neuronal_charge(x, w)

        """2. Leak & fire"""
        if self._leaking_comparison is LeakingComparisonMode.LEAK_BEFORE_COMP:
            self.neuronal_leak()
            self.neuronal_fire()
        else:
            self.neuronal_fire()
            self.neuronal_leak()

        """3. Reset"""
        self.neuronal_reset()

        self._post_hook()
