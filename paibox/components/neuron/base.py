from typing import Any, Optional, Tuple

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
from paibox.types import Shape, SpikeType, VoltageType
from paibox.utils import as_shape, shape2num

from .utils import _vjt_overflow

__all__ = ["Neuron"]


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
        leak_integration_mode: LIM,
        leak_v: int,
        synaptic_integration_mode: SIM,
        bit_truncation: int,
        keep_shape: bool = False,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._n_neuron = shape2num(shape)
        self._shape = as_shape(shape)

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self.reset_mode: RM = reset_mode
        self.reset_v: int = reset_v  # Signed 30-bit
        self.leak_comparison: LCM = leak_comparison
        self.threshold_mask_bits: int = threshold_mask_bits
        self.neg_thres_mode: NTM = neg_thres_mode
        self.neg_threshold: int = neg_threshold  # Unsigned 29-bit
        self.pos_threshold: int = pos_threshold  # Unsigned 29-bit
        self.leak_direction: LDM = leak_direction
        self.leak_integration_mode: LIM = leak_integration_mode
        self.leak_v: int = leak_v  # Signed 30-bit
        self.synaptic_integration_mode: SIM = synaptic_integration_mode
        self.bit_truncation: int = bit_truncation  # Unsigned 5-bit

        # TODO These two config below are parameters of CORE.
        self._spike_width_format: SpikeWidthFormat
        self._pool_max_en: MaxPoolingEnable

        # Auxiliary attributes or variables.
        self._thres_mask: int = (1 << threshold_mask_bits) - 1
        self.thres_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.uint8)
        self._v_th_rand = self.init_param(0).astype(np.int32)

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
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1

        if self.synaptic_integration_mode is SIM.MODE_STOCHASTIC:
            raise NotImplementedError(
                f"mode {SIM.MODE_STOCHASTIC.name} is not implemented."
            )
        else:
            if incoming_v.ndim == 2:
                _v = incoming_v.sum(axis=1).astype(np.int32)
            else:
                _v = incoming_v

        v_charged = np.add(vjt_pre, _v).astype(np.int32)

        return _vjt_overflow(v_charged)  # Handle with overflow here

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
        _rho_j_lambda = 2  # Random leak, unsigned 29-bit.

        if self.leak_direction is LDM.MODE_FORWARD:
            _ld = np.ones((self._n_neuron,), dtype=np.bool_)
        else:
            _ld = np.sign(vjt)

        if self.leak_integration_mode is LIM.MODE_DETERMINISTIC:
            v_leaked = np.add(vjt, _ld * self.leak_v).astype(np.int32)
        else:
            raise NotImplementedError(
                f"mode {LIM.MODE_STOCHASTIC.name} is not implemented."
            )
            # _F = 1 if abs(self.leak_v) >= _rho_j_lambda else 0
            # sgn_leak_v = fn_sgn(self.leak_v, 0)
            # self.vjt = np.add(self.vjt, sgn_leak_v * _F * _ld).astype(np.int32)

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
        # TODO Is _rho_j_T and _v_th_rand for all neurons or for each neuron?
        _rho_j_T = 3  # Random threshold, unsigned 29-bit.
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
    ) -> Tuple[SpikeType, VoltageType, NDArray[np.uint8]]:
        """Update at one time step."""
        # 1. Charge
        v_charged = self._neuronal_charge(incoming_v, vjt_pre)

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
    def varshape(self) -> Tuple[int, ...]:
        return self._shape if self.keep_shape else (self._n_neuron,)

    @property
    def bias(self) -> int:
        """Signed 30-bit. ANN mode only."""
        return self.leak_v


class Neuron(MetaNeuron, NeuDyn):
    _excluded_vars = (
        "vjt",
        "vj",
        "y",
        "threshold_mode",
        "spike",
        "v_th_rand",
        "_spike_width_format",
        "_pool_max_en",
        "master_nodes",
    )

    def __init__(
        self,
        shape: Shape,
        reset_mode: RM = RM.MODE_NORMAL,
        reset_v: int = 0,
        leak_comparison: LCM = LCM.LEAK_AFTER_COMP,
        threshold_mask_bits: int = 0,
        neg_thres_mode: NTM = NTM.MODE_RESET,
        neg_threshold: int = 0,
        pos_threshold: int = 1,
        leak_direction: LDM = LDM.MODE_FORWARD,
        leak_integration_mode: LIM = LIM.MODE_DETERMINISTIC,
        leak_v: int = 0,
        synaptic_integration_mode: SIM = SIM.MODE_DETERMINISTIC,
        bit_truncation: int = 0,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if neg_threshold > 0:
            raise ValueError(
                f"negative threshold must be non-positive, but got {neg_threshold}."
            )

        if pos_threshold < 0:
            raise ValueError(
                f"positive threshold must be non-negative, but got {pos_threshold}."
            )

        if bit_truncation < 0:
            raise ValueError(
                f"bit of tuncation must be non-negative, but got {bit_truncation}."
            )

        if delay < 1:
            raise ValueError(f"'delay' must be positive, but got {delay}.")

        if tick_wait_start < 0:
            raise ValueError(
                f"'tick_wait_start' must be non-negative, but got {tick_wait_start}."
            )

        if tick_wait_end < 0:
            raise ValueError(
                f"'tick_wait_end' must be non-negative, but got {tick_wait_end}."
            )

        if unrolling_factor < 1:
            raise ValueError(
                f"'unrolling_factor' must be positive, but got {unrolling_factor}."
            )

        super().__init__(
            shape,
            reset_mode,
            reset_v,
            leak_comparison,
            threshold_mask_bits,
            neg_thres_mode,
            (-neg_threshold),  # In `MetaNeuron`, it is unsgined.
            pos_threshold,
            leak_direction,
            leak_integration_mode,
            leak_v,
            synaptic_integration_mode,
            bit_truncation,
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

        self._delay = delay
        self._tws = tick_wait_start
        self._twe = tick_wait_end
        self._uf = unrolling_factor

        """Counter of copies."""
        self._n_copied = 0

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
            self._shape,
            self.reset_mode,
            self.reset_v,
            self.leak_comparison,
            self.threshold_mask_bits,
            self.neg_thres_mode,
            self.neg_threshold,
            self.pos_threshold,
            self.leak_direction,
            self.leak_integration_mode,
            self.leak_v,
            self.synaptic_integration_mode,
            self.bit_truncation,
            delay=self.delay_relative,
            tick_wait_start=self.tick_wait_start,
            tick_wait_end=self.tick_wait_end,
            unrolling_factor=self.unrolling_factor,
            keep_shape=self.keep_shape,
            name=f"{self.name}_copied_{self._n_copied}",
        )

    def copy(self) -> "Neuron":
        return self.__deepcopy__()

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def shape_out(self) -> Tuple[int, ...]:
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
