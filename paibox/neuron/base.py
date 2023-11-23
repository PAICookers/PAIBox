from typing import Optional, Tuple

import numpy as np

from paibox._types import Shape
from paibox.base import NeuDyn
from paibox.libpaicore.v2 import (
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
        *,
        keep_shape: bool = False,
        **kwargs,
    ) -> None:
        """Stateless attributes. Scalar."""
        # Basic attributes.
        self.keep_shape = keep_shape
        self._n_neuron = shape2num(shape)
        self._shape = as_shape(shape)

        # DO NOT modify the names of the following variables.
        # They will be exported to the parameter verification model.
        self._reset_mode: RM = reset_mode
        self._reset_v: int = reset_v  # Signed 30-bit
        self._leaking_comparison: LCM = leaking_comparison
        self._threshold_mask_bits: int = threshold_mask_bits
        self._neg_thres_mode: NTM = neg_thres_mode
        self._neg_threshold: int = neg_threshold  # Unsigned 29-bit
        self._pos_threshold: int = pos_threshold  # Unsigned 29-bit
        self._leaking_direction: LDM = leaking_direction
        self._leaking_integration_mode: LIM = leaking_integration_mode
        self._leak_v: int = leak_v  # Signed 30-bit
        self._synaptic_integration_mode: SIM = synaptic_integration_mode
        self._bit_truncate: int = bit_truncate
        self._vjt_init = vjt_init

        # TODO These two config below are parameters of CORE.
        self._spike_width_format: SpikeWidthFormat
        self._pool_max_en: MaxPoolingEnable

        """Stateful attributes. Vector."""
        # Membrane potential at last timestep.
        self._vjt_pre = self.init_param(vjt_init).astype(np.int32)
        # Membrane potential.
        self._vjt = self.init_param(vjt_init).astype(np.int32)
        self._spike = self.init_param(0).astype(np.bool_)

        # Attributes about ANN.
        self._vj = self.init_param(vjt_init).astype(np.int32)
        self._y = self.init_param(0).astype(np.int32)

        # Auxiliary attributes/variables.
        self._thres_mask: int = (1 << threshold_mask_bits) - 1
        self._threshold_mode = self.init_param(TM.NOT_EXCEEDED).astype(np.int8)
        self._v_th_rand = self.init_param(0).astype(np.int32)

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
        xt = np.zeros(self.varshape, dtype=np.int32)

        if self._synaptic_integration_mode is SIM.MODE_STOCHASTIC:
            raise NotImplementedError
        else:
            if x.ndim == 2:
                xt = x.sum(axis=1).astype(np.int32)
            else:
                xt = x

        self._vjt = np.add(self._vjt_pre, xt).astype(np.int32)

    def _neuronal_leak(self) -> None:
        r"""2. Leaking integration.

        2.1 Leaking direction, forward or reversal.
            If leaking direction is `MODE_FORWARD`, the `_ld` is 1, else is \sgn{`_vjt`}.

        2.2 Random leaking.
            If leaking integration is `MODE_DETERMINISTIC`, then
                `_vjt` = `_vjt` + `_ld` * `leak_v`
            else (`MODE_STOCHASTIC`)
                if abs(`leak_v`) >= `_rho_j_lambda`, then
                    `_F` = 1
                else
                    `_F` = 0

                `_vjt` = `_vjt` + \sgn{`leak_v`}* `_ld` * `_F`
        """
        _rho_j_lambda = 2  # Random leaking, unsigned 29-bit.

        if self._leaking_direction is LDM.MODE_FORWARD:
            _ld = np.ones(self.varshape, dtype=np.bool_)
        else:
            _ld = np.sign(self._vjt)

        if self._leaking_integration_mode is LIM.MODE_DETERMINISTIC:
            self._vjt = np.add(self._vjt, _ld * self.leak_v).astype(np.int32)
        else:
            raise NotImplementedError
            # _F = 1 if abs(self.leak_v) >= _rho_j_lambda else 0
            # sgn_leak_v = fn_sgn(self.leak_v, 0)

            # self._vjt = np.add(self._vjt, sgn_leak_v * _F * _ld).astype(np.int32)

    def _neuronal_fire(self) -> None:
        r"""3. Threshold comparison.

        3.1 Random threshold.
            `_v_th_rand` = `_rho_j_T` & `_thres_mask`

        3.2 Fire.
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
        _v_th_rand = _rho_j_T & self._thres_mask
        self._v_th_rand = np.full(self.varshape, _v_th_rand, dtype=np.int32)

        if self._neg_thres_mode is NTM.MODE_RESET:
            raise NotImplementedError
            # _v_th_neg = self._neg_threshold + _v_th_rand
        else:
            _v_th_neg = self._neg_threshold

        """Fire"""
        self._threshold_mode = np.where(
            self._vjt >= self._pos_threshold + _v_th_rand,
            TM.EXCEED_POSITIVE,
            np.where(self._vjt < -_v_th_neg, TM.EXCEED_NEGATIVE, TM.NOT_EXCEEDED),
        ).astype(np.int8)

        self._spike = np.equal(self._threshold_mode, TM.EXCEED_POSITIVE).astype(
            np.bool_
        )

    def _neuronal_reset(self) -> None:
        r"""4. Reset.

        If `_threshold_mode` is `EXCEED_POSITIVE`
            If reset mode is `MODE_NORMAL`, then
                `_vjt` = `reset_v`
            else if reset mode is `MODE_LINEAR`, then
                `_vjt` = `_vjt` - `_pos_threshold` - `_v_th_rand`
            else (`MODE_NONRESET`)
                `_vjt` = `_vjt`

        else if `_threshold_mode` is `EXCEED_NEGATIVE`
            If negative threshold mode is `MODE_RESET`, then
                If reset mode is `MODE_NORMAL`, then
                    `_vjt` = -`reset_v`
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
                return np.full(self.varshape, self.reset_v, dtype=np.int32)

            elif self._reset_mode is RM.MODE_LINEAR:
                raise NotImplementedError
                # return np.subtract(
                #     self._vjt, (self._pos_threshold + self._v_th_rand), dtype=np.int32
                # )
            else:
                return self._vjt

        def _when_exceed_neg():
            if self._neg_thres_mode is NTM.MODE_RESET:
                if self._reset_mode is RM.MODE_NORMAL:
                    return np.full(self.varshape, -self.reset_v, dtype=np.int32)
                elif self._reset_mode is RM.MODE_LINEAR:
                    raise NotImplementedError
                    # return np.add(
                    #     self._vjt, (self._neg_threshold + self._v_th_rand), dtype=np.int32
                    # )
                else:
                    return self._vjt

            else:
                return np.full(self.varshape, -self._neg_threshold, dtype=np.int32)

        # USE "=="!
        self._vjt_pre = self._vjt = np.where(
            self._threshold_mode == TM.EXCEED_POSITIVE,
            _when_exceed_pos(),
            np.where(
                self._threshold_mode == TM.EXCEED_NEGATIVE,
                _when_exceed_neg(),
                self._vjt,
            ),
        ).astype(np.int32)

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
            if self._spike_width_format is SpikeWidthFormat.WIDTH_1BIT:
                return np.ones(self.varshape, dtype=np.int32)

            if self._bit_truncate >= 8:
                return np.full(
                    self.varshape,
                    ((self._vj >> self._bit_truncate) - 8) & ((1 << 8) - 1),
                    dtype=np.int32,
                )
            elif self._bit_truncate > 0:
                _mask = (1 << self._bit_truncate) - 1
                _truncated_vj = self._vj & _mask
                return np.full(
                    self.varshape,
                    _truncated_vj << (8 - self._bit_truncate),
                    dtype=np.int32,
                )
            else:
                return np.zeros(self.varshape, dtype=np.int32)

        self._y = np.where(
            self._vj >= self._pos_threshold,
            _when_exceed_pos(),
            np.zeros(self.varshape, dtype=np.int32),
        ).astype(np.int32)

    def _max_pooling(self, x: np.ndarray) -> None:
        # TODO
        pass

    def _aux_pre_hook(self) -> None:
        """Pre-hook before the entire activation."""
        pass

    def _aux_post_hook(self) -> None:
        """Post-hook after the entire activation."""
        # Reset the auxiliary threshold mode.
        self._threshold_mode = np.full(self.varshape, TM.NOT_EXCEEDED, dtype=np.int8)

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
        return self._shape if self.keep_shape else (self._n_neuron,)

    @property
    def reset_v(self) -> int:
        return self._reset_v

    @property
    def leak_v(self) -> int:
        return self._leak_v

    @property
    def bias(self) -> int:
        return self._leak_v  # Signed 30-bit(ANN mode ONLY)

    @property
    def vjt_init(self) -> int:
        return self._vjt_init

    @property
    def voltage(self) -> np.ndarray:
        return self._vjt.reshape(self.varshape)

    @property
    def neg_threshold(self) -> int:
        return self._neg_threshold

    @property
    def pos_threshold(self) -> int:
        return self._pos_threshold

    def init_param(self, param) -> np.ndarray:
        return np.full(self.varshape, param)


class Neuron(MetaNeuron, NeuDyn):
    _excluded_vars = (
        "_vjt_pre",
        "_vjt",
        "_vj",
        "_y",
        "_threshold_mode",
        "_spike",
        "_v_th_rand",
        "_spike_width_format",
        "_pool_max_en",
        "master_nodes",
    )

    def __len__(self) -> int:
        return self._n_neuron

    def __call__(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.update(x)

    def update(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        if x is None:
            x = self.sum_inputs()

        return super().update(x)

    def reset_state(self) -> None:
        """Initialization, not the neuronal reset."""
        self._vjt = self._vjt_pre = self.init_param(self.vjt_init).astype(np.int32)
        self._spike = self.init_param(0).astype(np.bool_)

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
    def output(self) -> np.ndarray:
        return self._spike

    @property
    def spike(self) -> np.ndarray:
        return self._spike

    @property
    def feature_map(self) -> np.ndarray:
        return self.output.reshape(self.varshape)

    @property
    def state(self) -> np.ndarray:
        return self._spike
