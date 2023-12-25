from typing import Optional, Tuple

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
    MaxPoolingEnable,
    SpikeWidthFormat,
)

from paibox._types import Shape
from paibox.base import NeuDyn
from paibox.context import _FRONTEND_CONTEXT
from paibox.utils import as_shape, shape2num

__all__ = ["Neuron"]


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
        self.leaking_comparison: LCM = leaking_comparison
        self.threshold_mask_bits: int = threshold_mask_bits
        self.neg_thres_mode: NTM = neg_thres_mode
        self.neg_threshold: int = neg_threshold  # Unsigned 29-bit
        self.pos_threshold: int = pos_threshold  # Unsigned 29-bit
        self.leaking_direction: LDM = leaking_direction
        self.leaking_integration_mode: LIM = leaking_integration_mode
        self.leak_v: int = leak_v  # Signed 30-bit
        self.synaptic_integration_mode: SIM = synaptic_integration_mode
        self.bit_truncate: int = bit_truncate
        self.vjt_init = vjt_init

        # TODO These two config below are parameters of CORE.
        self._spike_width_format: SpikeWidthFormat
        self._pool_max_en: MaxPoolingEnable

        # Auxiliary attributes or variables.
        self._thres_mask: int = (1 << threshold_mask_bits) - 1
        self._threshold_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.int8)
        self._v_th_rand = self.init_param(0).astype(np.int32)

    def _neuronal_charge(
        self, x: NDArray[np.int32], vjt_pre: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        r"""1. Synaptic integration.

        Argument:
            - x: input to the neuron(s). [1*N]

        Description:
            _rho_w_ij: Random synaptic integration enable, 0 or 1.

            If synaptic integration mode is deterministic, then
                `vjt` = `vjt_pre` + \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
            else (stochastic)
                `vjt` = `vjt_pre` + `_rho_w_ij` * \sum^{N-1}_{i=0} * x_i(t) * w_{i,j}
        """
        _rho_w_ij = 1  # Random synaptic integration enable, 0/1
        xt = self.init_param(0).astype(np.int32)

        if self.synaptic_integration_mode is SIM.MODE_STOCHASTIC:
            raise NotImplementedError
        else:
            if x.ndim == 2:
                xt = x.sum(axis=1).astype(np.int32)
            else:
                xt = x

        v_charged = np.add(vjt_pre, xt).astype(np.int32)

        return v_charged

    def _neuronal_leak(self, vjt: NDArray[np.int32]) -> NDArray[np.int32]:
        r"""2. Leaking integration.

        2.1 Leaking direction, forward or reversal.
            If leaking direction is `MODE_FORWARD`, the `_ld` is 1, else is \sgn{`vjt`}.

        2.2 Random leaking.
            If leaking integration is `MODE_DETERMINISTIC`, then
                `vjt` = `vjt` + `_ld` * `leak_v`
            else (`MODE_STOCHASTIC`)
                if abs(`leak_v`) >= `_rho_j_lambda`, then
                    `_F` = 1
                else
                    `_F` = 0

                `vjt` = `vjt` + \sgn{`leak_v`}* `_ld` * `_F`
        """
        _rho_j_lambda = 2  # Random leaking, unsigned 29-bit.

        if self.leaking_direction is LDM.MODE_FORWARD:
            _ld = np.ones(self.varshape, dtype=np.bool_)
        else:
            _ld = np.sign(vjt)

        if self.leaking_integration_mode is LIM.MODE_DETERMINISTIC:
            v_leaked = np.add(vjt, _ld * self.leak_v).astype(np.int32)
        else:
            raise NotImplementedError
            # _F = 1 if abs(self.leak_v) >= _rho_j_lambda else 0
            # sgn_leak_v = fn_sgn(self.leak_v, 0)
            # self.vjt = np.add(self.vjt, sgn_leak_v * _F * _ld).astype(np.int32)

        return v_leaked

    def _neuronal_fire(self, vjt: NDArray[np.int32]) -> NDArray[np.bool_]:
        r"""3. Threshold comparison.

        3.1 Random threshold.
            `_v_th_rand` = `_rho_j_T` & `_thres_mask`

        3.2 Fire.
            If negative threshold mode is `MODE_RESET`, then
                `_v_th_neg` = `_neg_threshold` + `_v_th_rand`
            else
                `_v_th_neg` = `_neg_threshold`

            If `vjt` >= `_pos_threshold` + `_v_th_rand`, then
                `spike` = 1
            else if `vjt` < -`_v_th_neg`, then
                `spike` = 0
            else
                `spike` = 0
        """
        _rho_j_T = 3  # Random threshold, unsigned 29-bit.

        # TODO Is _rho_j_T and _v_th_rand for all neurons or for each neuron?
        _v_th_rand = _rho_j_T & self._thres_mask
        self._v_th_rand = self.init_param(_v_th_rand).astype(np.int32)

        if self.neg_thres_mode is NTM.MODE_RESET:
            raise NotImplementedError
            # _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self.neg_threshold

        """Fire"""
        self._threshold_mode = np.where(
            vjt >= self.pos_threshold + _v_th_rand,
            TM.EXCEED_POSITIVE,
            np.where(vjt < -_v_th_neg, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
        ).astype(np.int8)

        spike = np.equal(self._threshold_mode, TM.EXCEED_POSITIVE)

        return spike

    def _neuronal_reset(self, vjt: NDArray[np.int32]) -> NDArray[np.int32]:
        r"""4. Reset.

        If `_threshold_mode` is `EXCEED_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `vjt` = `reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `vjt` = `vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `vjt` = `vjt`

        else if `_threshold_mode` is `EXCEED_NEGATIVE`
            If negative threshold mode is `MODE_RESET`, then
                If reset mode is `MODE_NORMAL`, then
                    `vjt` = -`reset_v`
                else if reset mode is `MODE_LINEAR`, then
                    `vjt` = `vjt` + (`_neg_threshold` + `_v_th_rand`)
                else
                    `vjt` = `vjt`
            else (`MODE_SATURATION`)
                `vjt` = `_neg_threshold`

        else (not beyond the threshold)
            `vjt` = `vjt`
        """

        def _when_exceed_pos() -> NDArray[np.int32]:
            if self.reset_mode is RM.MODE_NORMAL:
                return np.full(self.varshape, self.reset_v, dtype=np.int32)

            elif self.reset_mode is RM.MODE_LINEAR:
                raise NotImplementedError
                # return np.subtract(
                #     self.vjt, (self._pos_threshold + self._v_th_rand), dtype=np.int32
                # )
            else:
                return vjt

        def _when_exceed_neg() -> NDArray[np.int32]:
            if self.neg_thres_mode is NTM.MODE_RESET:
                if self.reset_mode is RM.MODE_NORMAL:
                    return np.full(self.varshape, -self.reset_v, dtype=np.int32)
                elif self.reset_mode is RM.MODE_LINEAR:
                    raise NotImplementedError
                    # return np.add(
                    #     self.vjt, (self._neg_threshold + self._v_th_rand), dtype=np.int32
                    # )
                else:
                    return vjt

            else:
                return np.full(self.varshape, -self.neg_threshold, dtype=np.int32)

        # USE "=="!
        v_reseted = np.where(
            self._threshold_mode == TM.EXCEED_POSITIVE,
            _when_exceed_pos(),
            np.where(
                self._threshold_mode == TM.EXCEED_NEGATIVE,
                _when_exceed_neg(),
                vjt,
            ),
        ).astype(np.int32)

        self._aux_post_hook()

        return v_reseted

    def _relu(self, vj: NDArray[np.int32]) -> NDArray[np.int32]:
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

        def _when_exceed_pos() -> NDArray[np.int32]:
            if self._spike_width_format is SpikeWidthFormat.WIDTH_1BIT:
                return np.ones(self.varshape, dtype=np.int32)

            if self.bit_truncate >= 8:
                return np.full(
                    self.varshape,
                    ((vj >> self.bit_truncate) - 8) & ((1 << 8) - 1),
                    dtype=np.int32,
                )
            elif self.bit_truncate > 0:
                _mask = (1 << self.bit_truncate) - 1
                _truncated_vj = vj & _mask
                return np.full(
                    self.varshape,
                    _truncated_vj << (8 - self.bit_truncate),
                    dtype=np.int32,
                )
            else:
                return np.zeros(self.varshape, dtype=np.int32)

        y = np.where(
            vj >= self.pos_threshold,
            _when_exceed_pos(),
            np.zeros(self.varshape, dtype=np.int32),
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
        self._threshold_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.int8)

    def update(
        self, x: NDArray[np.int32], vjt_pre: NDArray[np.int32]
    ) -> Tuple[NDArray[np.bool_], NDArray[np.int32]]:
        """Update at one time step."""

        """1. Charge.
            Sum the membrane potential from the previous synapses.
            Don't care the /rho right now.
        """
        v_charged = self._neuronal_charge(x, vjt_pre)

        """2. Leak & fire"""
        if self.leaking_comparison is LCM.LEAK_BEFORE_COMP:
            v_leaked = self._neuronal_leak(v_charged)
            spike = self._neuronal_fire(v_leaked)
        else:
            spike = self._neuronal_fire(v_charged)
            v_leaked = self._neuronal_leak(v_charged)

        """3. Reset"""
        v_reseted = self._neuronal_reset(v_leaked)

        return spike, v_reseted

    def init_param(self, param) -> np.ndarray:
        return np.full(self.varshape, param)

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self._shape if self.keep_shape else (self._n_neuron,)

    @property
    def bias(self) -> int:
        """Signed 30-bit. ANN mode only."""
        return self.leak_v


class Neuron(MetaNeuron, NeuDyn):
    _excluded_vars = (
        "vjt_pre",
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
        keep_shape: bool = False,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape,
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
            keep_shape,
        )
        super(MetaNeuron, self).__init__(name)

        """Stateful attributes. Vector."""
        self.set_memory("vjt", self.init_param(vjt_init).astype(np.int32))
        self.set_memory("vjt_pre", self.init_param(vjt_init).astype(np.int32))
        self.set_memory("spike", self.init_param(0).astype(np.bool_))

        # Attributes about ANN
        self.set_memory("vj", self.init_param(vjt_init).astype(np.int32))
        self.set_memory("y", self.init_param(0).astype(np.int32))

        self._tws = tick_wait_start
        self._twe = tick_wait_end

    def __len__(self) -> int:
        return self._n_neuron

    def __call__(self, x: Optional[np.ndarray] = None, **kwargs) -> NDArray[np.bool_]:
        return self.update(x, **kwargs)

    def update(
        self, x: Optional[np.ndarray] = None, **kwargs
    ) -> Optional[NDArray[np.bool_]]:
        if _FRONTEND_CONTEXT["t"] < self.tick_wait_start:
            # The neuron start working until `t >= tws`.
            return None

        # Priority order is a must.
        if self.tick_wait_end > 0:
            if _FRONTEND_CONTEXT["t"] > self.tick_wait_start + self.tick_wait_end:
                # The neuron is done working until `t > tws + twe` under the condition `twe > 0`.
                return None

        if x is None:
            x = self.sum_inputs()

        self.spike, self.vjt = super().update(x, self.vjt)

        return self.spike

    def reset_state(self) -> None:
        """Initialization, not the neuronal reset."""
        self.reset()  # Call reset of `StatusMemory`.

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self.varshape

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.varshape

    @property
    def num_in(self) -> int:
        return self._n_neuron

    @property
    def num_out(self) -> int:
        return self._n_neuron

    @property
    def output(self) -> NDArray[np.bool_]:
        return self.spike

    @property
    def feature_map(self) -> NDArray[np.bool_]:
        return self.output.reshape(self.varshape)

    @property
    def state(self) -> NDArray[np.bool_]:
        return self.spike

    @property
    def voltage(self) -> NDArray[np.int32]:
        return self.vjt.reshape(self.varshape)

    @property
    def tick_wait_start(self) -> int:
        return self._tws

    @property
    def tick_wait_end(self) -> int:
        return self._twe
