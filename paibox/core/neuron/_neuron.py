from abc import ABC, abstractmethod
from .ram_types import (
    ResetMode as RM,
    LeakingComparisonMode as LCM,
    NegativeThresholdMode as NTM,
    LeakingDirectionMode as LDM,
    LeakingIntegrationMode as LIM,
    SynapticIntegrationMode as SIM,
    ThresholdMode as TM,
)
from ..reg_types import *
from .ram_model import ParamsRAM
from typing import List, Dict, ClassVar, TypeVar, Union


__all__ = ["Neuron"]


T = TypeVar("T")
Spike = Union[int, List[int]]
Spike_t = List[Spike]


class _AbstractNeuron(ABC):
    """Abstract neuron."""

    @abstractmethod
    def neuronal_charge(self, x):
        raise NotImplementedError

    def neuronal_leak(self):
        raise NotImplementedError

    def neuronal_fire(self):
        raise NotImplementedError

    def neuronal_reset(self):
        raise NotImplementedError

    @abstractmethod
    def single_step_forward(self, x):
        raise NotImplementedError


class _MetaNeuron(_AbstractNeuron):
    """Meta neuron"""

    __runtime_mode: ClassVar[bool] = False
    __snn_mode: ClassVar[bool] = True  # Default value

    def __init__(
        self,
        # TODO Maybe don't need so much parameters
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
        nid: int,
        weights: List[int],
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
        # Common
        self._tick_relative: int = tick_relative
        self._addr_axon: int = addr_axon
        self._addr_core_x: int = addr_core_x
        self._addr_core_y: int = addr_core_y
        self._addr_core_x_ex: int = addr_core_x_ex
        self._addr_core_y_ex: int = addr_core_y_ex
        self._addr_chip_x: int = addr_chip_x
        self._addr_chip_y: int = addr_chip_y

        # SNN
        self._reset_mode: RM = reset_mode
        self._leaking_comparison: LCM = leaking_comparison
        self._neg_thres_mode: NTM = neg_thres_mode
        self._leaking_direction: LDM = leaking_direction
        self._synaptic_integration_mode: SIM = synaptic_integration_mode
        self._leaking_integration_mode: LIM = leaking_integration_mode
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
        self._nid: int = nid
        self._weights = weights

        # SNN
        self._timestep = 0  # As an global class variable?
        self._vjt_pre: int = vjt_init  # Membrane potential at Last time step.
        self._vjt: int = 0  # Membrane potential.
        self._spike: int = 0  # TODO Related to `tick_relative` ans time slot.

        # ANN
        self._vj: int = vjt_init
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
        self._lcn_extension: LCNExtensionType
        self._spike_width_format: SpikeWidthFormatType

        # ANN
        self._pool_max_en: MaxPoolingEnableType

        """Auxiliary variables"""
        self._threshold_mode: TM = TM.MODE_UNSET
        self._v_th_rand = 0
        self._input_axon_num = len(weights)

    def neuronal_charge(self, input_spikes: Spike) -> None:
        """1. Synaptic integration.

        ## Arguments:
        - `input_spikes`: one spike width of `N` axons.
            | |
        1 - x x
        0 - x ·
        1 - · x, here the width of x is 3.

        NOTE: `N` axons correspond to `N` weights.

        ## Description
        _rho_w_ij: Random synaptic integration enable, 0 or 1.

        If synaptic integration mode is deterministic, then
            `_vjt` = `_vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        else (stochastic)
            `_vjt` = `_vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        """
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1
        xt = 0

        if isinstance(input_spikes, int):
            if self._input_axon_num > 1:
                raise ValueError(
                    f"width of weights({self._input_axon_num}) > width of input axon(1)"
                )

            _is = [input_spikes]

        else:
            if len(input_spikes) != self._input_axon_num:
                raise ValueError(
                    f"width of weights({self._input_axon_num}) != width of input axon({len(input_spikes)})"
                )

            _is = input_spikes

        for i in range(self._input_axon_num):
            # xt = xt + spikes[i_of_axon] * weights[i_of_axon]
            if self._synaptic_integration_mode is SIM.MODE_DETERMINISTIC:
                xt += _is[i] * self._weights[i]
            else:
                xt += _rho_w_ij * _is[i] * self._weights[i]

        self._vjt = self._vjt_pre + xt

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

        if self._leaking_direction is LDM.MODE_FORWARD:
            _ld = 1
        else:
            _ld = 1 if self._vjt >= 0 else -1

        if self._leaking_integration_mode is LIM.MODE_DETERMINISTIC:
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

        # print(f"leak: vjt = {self._vjt}")

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

        if self._neg_thres_mode is NTM.MODE_RESET:
            _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self._neg_threshold

        """Fire"""
        if self._vjt >= self._pos_threshold + _v_th_rand:
            self._threshold_mode = TM.MODE_POSITIVE
            yt = 1
        elif self._vjt < -_v_th_neg:
            self._threshold_mode = TM.MODE_NEGATIVE
            yt = 0
        else:
            self._threshold_mode = TM.MODE_UNSET
            yt = 0

        self._spike = yt

        # print(f"fire: vjt = {self._vjt}")

    def neuronal_reset(self) -> None:
        """4. Reset

        If `_threshold_mode` is `MODE_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `_vjt` = `_reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `_vjt` = `_vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `_vjt` = `_vjt`

        else if `_threshold_mode` is `MODE_NEGATIVE`
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
        if self._threshold_mode is TM.MODE_POSITIVE:
            if self._reset_mode is RM.MODE_NORMAL:
                self._vjt = self._reset_v
            elif self._reset_mode is RM.MODE_LINEAR:
                self._vjt = self._vjt - (self._pos_threshold + self._v_th_rand)
            else:
                self._vjt = self._vjt
        elif self._threshold_mode is TM.MODE_NEGATIVE:
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

        # print(f"reset: vjt = {self._vjt}")

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
        print(f"{self._threshold_mode}")
        self._vjt_pre = self._vjt
        self._threshold_mode = TM.MODE_UNSET

        print(f"vjt = {self._vjt}, spike = {self._spike}")

    def single_step_forward(self, x: List[int]):
        self._pre_hook()

        """1. Charge"""
        self.neuronal_charge(x)

        """2. Leak & fire"""
        if self._leaking_comparison is LCM.LEAK_BEFORE_COMP:
            self.neuronal_leak()
            self.neuronal_fire()
        else:
            self.neuronal_fire()
            self.neuronal_leak()

        """3. Reset"""
        self.neuronal_reset()

        self._post_hook()

    def multi_step_forward(self, x: List[List[int]]):
        pass

    """Properties"""

    @property
    def weights(self) -> List[int]:
        """Weights"""
        return self._weights

    @property
    def nid(self) -> int:
        return self._nid


class Neuron(_MetaNeuron):
    """Father class of wrapped neurons.

    The parameters are always legal.
    """

    __neuron_num = 1

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
        nid: int,
        weights: List[int],
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
    ):
        super().__init__(
            tick_relative,
            addr_axon,
            addr_core_x,
            addr_core_y,
            addr_core_x_ex,
            addr_core_y_ex,
            addr_chip_x,
            addr_chip_y,
            chip_x,
            chip_y,
            core_x,
            core_y,
            nid,
            weights,
            reset_mode,
            reset_v,
            leaking_comparison,
            threshold_mask_bits,
            neg_thres_mode,
            neg_threshold,
            pos_threshold,
            leaking_direction,
            leaking_integration_mode,
            leak_v,
            synaptic_integration_mode,
            bit_truncate,
            vjt_init,
        )

        self._threshold_mask_bits = threshold_mask_bits
        self._vjt_init = vjt_init

    def export_params_model(self) -> ParamsRAM:
        model = ParamsRAM(
            tick_relative=self._tick_relative,
            addr_axon=self._addr_axon,
            addr_core_x=self._addr_core_x,
            addr_core_y=self._addr_core_y,
            addr_core_x_ex=self._addr_core_x_ex,
            addr_core_y_ex=self._addr_core_y_ex,
            addr_chip_x=self._addr_chip_x,
            addr_chip_y=self._addr_chip_y,
            reset_mode=self._reset_mode,
            reset_v=self._reset_v,
            leaking_comparison=self._leaking_comparison,
            threshold_mask_bits=self._threshold_mask_bits,
            neg_thres_mode=self._neg_thres_mode,
            neg_threshold=self._neg_threshold,
            pos_threshold=self._pos_threshold,
            leaking_direction=self._leaking_direction,
            leaking_integration_mode=self._leaking_integration_mode,
            leak_v=self._leak_v,
            synaptic_integration_mode=self._synaptic_integration_mode,
            bit_truncate=self._bit_truncate,
            vjt_init=self._vjt_init,
        )

        return model

    def export_params_dict(self) -> Dict[str, int]:
        model = self.export_params_model()
        return model.model_dump(by_alias=True)


class NeuronGroup(_MetaNeuron):
    __neuron_num: int

    def __init__(
        self,
        n: int,
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
        nid: int,
        weights: List[int],
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
    ):
        if n < 2:
            raise ValueError("If number of neurons < 2, use subclass of Neuron please.")

        self.__neuron_num = n
