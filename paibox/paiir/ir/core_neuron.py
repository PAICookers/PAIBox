"""Unified chip-accurate neuron model for the v2.5 offline core.

The v2.5 offline core has a single shared datapath for both SNN and ANN modes::

    charge -> leak -> range_clamp -> {threshold_fire(SNN) | lut_lookup(ANN)} -> reset -> output

:class:`CoreNeuronV25` unifies SNN (spike) and ANN (LUT) activation into a single
class that reflects the chip architecture. When ``lut`` is ``None`` the neuron
operates in SNN mode (threshold comparison -> spike); when ``lut`` is provided
it operates in ANN mode (LUT lookup replaces the spike step).

Example::

    from paibox.paiir import ANNNodeV25, IFNodeV25, LIFNodeV25, LutReLU

    # SNN mode (same as before)
    snn_act = IFNodeV25(v_threshold=1.0)

    # ANN mode: neuron with LUT activation
    ann_act = ANNNodeV25(LutReLU())

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        ANNNodeV25(LutReLU()),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10),
        IFNodeV25(v_threshold=0.5),
    )
"""

import copy
import math
import warnings
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import torch
from paicorelib import (
    RM,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
)
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.base import MemoryModule
from torch import Tensor

from ...exceptions import AutoOptimizationWarning
from .calc_params import DEFAULT_NEG_THRESHOLD, NeuronParams
from .lut_activation import LutActivation

__all__ = [
    "CoreNeuronV25",
    "ANNNodeV25",
    "IFNodeV25",
    "LIFNodeV25",
    "LeakyBeta0NodeV25",
]

_T = TypeVar("_T", bound="CoreNeuronV25")
_FIXED_DYNAMICS_ATTRS = (
    "leak_tau_shift",
    "leak_multi_input",
    "leak_multi_mode",
    "leak_multi_sequence",
    "lut",
)


class CoreNeuronV25(MemoryModule):
    __vectorized_attrs__: ClassVar[tuple[str, ...]] = NeuronParams.__vectorized_attrs__
    __fixed_attrs__: ClassVar[tuple[str, ...]] = ()
    __export_attrs__: tuple[str, ...] = (
        "reset_mode",
        "reset_v",
        "thres_neg_mode",
        "thres_pos_mode",
        "thres_neg",
        "thres_pos",
        "lateral_inhi",
        "leak_multi_sequence",
        "leak_multi_input",
        "leak_multi_mode",
        "leak_add_mode",
        "leak_tau",
        "leak_v",
        "init_v",
    )

    @classmethod
    def _reject_fixed_attr_overrides(cls, kwargs: dict[str, Any]) -> None:
        overridden = tuple(name for name in cls.__fixed_attrs__ if name in kwargs)
        if overridden:
            raise ValueError(f"{cls.__name__} fixes " + ", ".join(overridden))

    def __init__(
        self,
        reset_mode: RM = RM.MODE_NORMAL,
        reset_v: float | Tensor = 0,
        thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE,
        thres_neg_mode: ThresholdNegMode | None = None,
        thres_pos: float | Tensor = 1,
        thres_neg: float | Tensor | None = None,
        lateral_inhi: LateralInhibitionMode | bool = LateralInhibitionMode.DISABLE,
        leak_multi_sequence: LeakMultiComparisonOrder = LeakMultiComparisonOrder.AFTER_COMPARE,
        leak_multi_input: LeakMultiInputMode | bool = LeakMultiInputMode.DISABLE,
        leak_multi_mode: LeakMultiMode | bool = LeakMultiMode.DISABLE,
        leak_add_mode: LeakAddMode = LeakAddMode.FORWARD,
        tau: float | Tensor = 1,
        leak_v: float | Tensor = 0,
        init_v: float | Tensor = 0,
        *,
        lut: LutActivation | None = None,
        leak_tau_shift: int | Tensor | None = None,
        # For SNN training with surrogate gradients
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
    ) -> None:
        """Unified chip-accurate neuron for the v2.5 offline core.

        Implements the full charge -> leak -> range_clamp -> fire/lookup -> reset
        pipeline. In SNN mode (``lut=None``), threshold comparison produces spikes.
        In ANN mode (``lut`` provided), the LUT replaces the threshold-fire step.

        End users should prefer :class:`IFNodeV25` or :class:`LIFNodeV25` for SNN,
        or ``ANNNodeV25(LutReLU())`` for ANN.

        Args:
            reset_mode: Reset mode.
            reset_v: Hard-reset voltage.
            thres_pos_mode: Positive threshold mode.
            thres_neg_mode: Negative threshold mode.
            thres_pos: Positive threshold. A scalar applies to the whole neuron
                instance; a 1D tensor is interpreted as per-output-channel
                threshold and is explicitly broadcast against ``N,C,...`` state.
            thres_neg: Negative threshold. ``None`` uses a very small default.
            lateral_inhi: Lateral inhibition mode.
            leak_multi_sequence: Order of multiplicative leak relative to
                threshold comparison.
            leak_multi_input: Whether input participates in multiplicative leak.
            leak_multi_mode: Multiplicative leak mode.
            leak_add_mode: Additive leak direction.
            tau: Time constant (must be a power of 2).
            leak_v: Additive leak voltage.
            init_v: Initial membrane potential.
            lut: Optional LUT activation module for ANN mode.
        """
        super().__init__()

        self.reset_mode = reset_mode
        self.reset_v = _normalize_numeric_param(reset_v, name="reset_v")
        self.thres_pos_mode = thres_pos_mode
        # Default thres_neg_mode:
        #   SNN (lut=None) -> FLOOR (negative V clamped, single-sided firing)
        #   ANN unsigned LUT -> FLOOR (negative region unused)
        #   ANN signed LUT   -> FIRE  (both sides active)
        if thres_neg_mode is not None:
            self.thres_neg_mode = thres_neg_mode
        elif lut is not None and lut.output_signed:
            self.thres_neg_mode = ThresholdNegMode.FIRE
        else:
            self.thres_neg_mode = ThresholdNegMode.FLOOR

        self.thres_pos = _normalize_numeric_param(thres_pos, name="thres_pos")
        self.thres_neg = _normalize_numeric_param(
            thres_neg if thres_neg is not None else DEFAULT_NEG_THRESHOLD,
            name="thres_neg",
        )
        self.lut = lut
        self.lateral_inhi = LateralInhibitionMode(lateral_inhi)
        self.leak_multi_sequence = leak_multi_sequence
        self.leak_multi_input = LeakMultiInputMode(leak_multi_input)
        self.leak_multi_mode = LeakMultiMode(leak_multi_mode)
        self.leak_add_mode = leak_add_mode

        # tau -> leak_tau (right-shift exponent)
        # Keep the original tau for compensation passes that need the
        # precise value (e.g. AvgPool threshold compensation).
        self.tau = _normalize_numeric_param(tau, name="tau")
        if leak_tau_shift is not None:
            self.leak_tau = _normalize_leak_tau_shift(leak_tau_shift)
        else:
            self.leak_tau = self._tau_to_shift(self.tau)

        self.leak_v = _normalize_numeric_param(leak_v, name="leak_v")
        self.init_v = _normalize_numeric_param(init_v, name="init_v")

        self._validate_threshold_bounds()
        self._validate_dynamics_homogeneity()

        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset

        self.register_memory("v", init_v)
        self._any_pos_spike_at_last_ts = False

    def reset(self) -> None:
        self._any_pos_spike_at_last_ts = False
        return super().reset()

    @staticmethod
    def _deepcopy_state_value(value, memo: dict[int, Any]):
        if torch.is_tensor(value):
            return value.detach().clone()
        return copy.deepcopy(value, memo)

    def clone(self):
        cloned = self.__deepcopy__({})
        cloned.reset()
        return cloned

    def __deepcopy__(self: _T, memo: dict[int, Any]) -> _T:
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]

        cls = type(self)
        cloned = cls.__new__(cls)
        memo[id(self)] = cloned

        for name, value in self.__dict__.items():
            if name in ("_memories", "_memories_rv"):
                # MemoryModule stores runtime state in plain dicts rather than
                # nn.Module buffers/parameters, so copy them entry-by-entry and
                # bypass torch's unsupported non-leaf tensor deepcopy path.
                copied = {
                    key: self._deepcopy_state_value(mem_value, memo)
                    for key, mem_value in value.items()
                }
            else:
                copied = self._deepcopy_state_value(value, memo)

            setattr(cloned, name, copied)

        return cloned

    @property
    def snn_mode(self) -> SNNMode:
        """Return the chip SNN/ANN mode based on LUT presence."""
        return SNNMode.SNN if self.lut is None else SNNMode.ANN

    @property
    def is_snn(self) -> bool:
        return self.snn_mode == SNNMode.SNN

    @property
    def has_if_dynamics(self) -> bool:
        """Return True for spike neurons without leak dynamics."""
        if not self.is_snn or self.leak_multi_mode != LeakMultiMode.DISABLE:
            return False
        if torch.is_tensor(self.leak_tau):
            return bool(torch.all(self.leak_tau == 0).item())
        return self.leak_tau == 0

    @property
    def has_lif_dynamics(self) -> bool:
        """Return True for spike neurons with leak dynamics."""
        return self.is_snn and not self.has_if_dynamics

    @property
    def output_sign(self) -> bool:
        """Return the output sign for data format inference.

        ANN mode: delegates to ``self.lut.output_signed``.
        SNN mode: True if negative threshold fires, False otherwise.
        """
        if self.lut is not None:
            return self.lut.output_signed
        return self.thres_neg_mode == ThresholdNegMode.FIRE

    def extra_repr(self) -> str:
        parts = [
            f"v_threshold={self.thres_pos}, v_reset={self.reset_v}, "
            f"reset_mode={self.reset_mode}, thres_pos_mode={self.thres_pos_mode}",
        ]
        if self.lut is not None:
            parts.append(f"lut={self.lut.__class__.__name__}")
        return ", ".join(parts)

    def forward(self, x: Tensor) -> Tensor:
        return self.single_step_forward(x)

    def single_step_forward(self, x: Tensor) -> Tensor:
        """Single time-step forward pass reproducing exact chip behaviour."""
        self._validate_simulation_params()
        if self.training and self.lut is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} with LUT activation (ANN mode) does not "
                f"support training: torch.bucketize in LUT lookup is not differentiable. "
                f"Train with the original float activation (e.g. nn.ReLU) and convert "
                f"to ANNNodeV25(lut=...) for deployment."
            )

        # At t=0, initialize v to init_v
        if isinstance(self.v, (int, float)):
            self.v = torch.full_like(x.data, self.v)

        # Lateral inhibition: reset v if a positive spike occurred last timestep
        if (
            self.lateral_inhi == LateralInhibitionMode.ENABLE
            and self._any_pos_spike_at_last_ts
        ):
            self.v.fill_(self.reset_v)

        self._neuronal_charge(x)
        self._v_range_clamp()

        # Divergent step: lut_lookup + reset (ANN) or threshold_fire (SNN).
        # In ANN mode, CEILING/FLOOR only clamp V (done above); LUT lookup
        # and reset always execute on the clamped V.
        if self.lut is not None:
            output, indices = self.lut.lookup(self.v)
            self._ann_reset(output, indices)
        else:
            output = torch.zeros_like(self.v)
            pos_mask = torch.zeros_like(self.v, dtype=torch.bool)
            neg_mask = torch.zeros_like(self.v, dtype=torch.bool)

            # Fire: compare V against threshold (only for FIRE sides)
            if self.thres_pos_mode == ThresholdPosMode.FIRE:
                pos_thres = self._thres_pos_for_v()
                if self.training:
                    pos_spike = self.surrogate_function(self.v - pos_thres)
                else:
                    pos_spike = (self.v >= pos_thres).to(torch.int8)
                output = output + pos_spike
                pos_mask = (
                    pos_spike.detach().bool()
                    if self.training and self.detach_reset
                    else pos_spike.bool()
                )

            if self.thres_neg_mode == ThresholdNegMode.FIRE:
                if self.training:
                    neg_spike = self.surrogate_function(self.thres_neg - self.v)
                else:
                    neg_spike = (self.v <= self.thres_neg).to(torch.int8)
                output = output - neg_spike
                neg_mask = (
                    neg_spike.detach().bool()
                    if self.training and self.detach_reset
                    else neg_spike.bool()
                )

            # Build per-element threshold for soft reset (only fired elements matter)
            thres = torch.zeros_like(self.v)
            if self.thres_pos_mode == ThresholdPosMode.FIRE:
                pos_thres = self._thres_pos_for_v()
                if torch.is_tensor(pos_thres):
                    thres = torch.where(pos_mask, pos_thres.expand_as(self.v), thres)
                else:
                    thres[pos_mask] = pos_thres
            thres[neg_mask] = self.thres_neg
            self.v = self._reset_v(self.v, pos_mask, neg_mask, thres)

            if not self.training:
                output = output.to(torch.int8)

        self._any_pos_spike_at_last_ts = torch.any(output > 0).item()

        # Post-comparison multiplicative leak (shared), SJ LIF
        if self.leak_multi_sequence == LeakMultiComparisonOrder.AFTER_COMPARE:
            self.v = self._multiplicative_leak()

        return output

    def _v_range_clamp(self) -> None:
        """Clamp membrane potential based on threshold modes.

        CEILING mode: clamp V to thres_pos from above.
        FLOOR mode: clamp V to thres_neg from below.

        In both SNN and ANN modes, clamping happens before the comparison /
        LUT lookup step.
        """
        if self.thres_pos_mode == ThresholdPosMode.CEILING:
            pos_thres = self._thres_pos_for_v()
            if torch.is_tensor(pos_thres):
                self.v = torch.minimum(self.v, pos_thres.expand_as(self.v))
            else:
                self.v.clamp_max_(pos_thres)
        if self.thres_neg_mode == ThresholdNegMode.FLOOR:
            self.v.clamp_min_(self.thres_neg)

    def _ann_reset(self, output: Tensor, indices: Tensor) -> None:
        """ANN reset based on LUT output values.

        The LUT output sign determines which side fired:
            output > 0 -> positive fire,  output < 0 -> negative fire.

        For LINEAR (soft) reset, the threshold is the lower bound of the
        bin that V falls into (``lut.thresholds[indices]``), used for both
        positive and negative fires.
        """
        assert self.lut is not None
        pos_mask = output > 0
        neg_mask = output < 0
        thres = self.lut.thresholds[indices]
        self.v = self._reset_v(self.v, pos_mask, neg_mask, thres)

    def _neuronal_charge(self, x: Tensor) -> None:
        """Charge phase: accumulate input + additive leak + optional pre-comparison
        multiplicative leak."""
        if self.leak_multi_input == LeakMultiInputMode.ENABLE:
            self.v += self._apply_tau_shift(x)
        else:
            self.v += x

        # Additive leak
        if self.leak_add_mode == LeakAddMode.FORWARD:
            self.v += self.leak_v
        else:
            self.v += torch.sign(self.v) * self.leak_v

        # Pre-comparison multiplicative leak
        if self.leak_multi_sequence == LeakMultiComparisonOrder.BEFORE_COMPARE:
            self.v = self._multiplicative_leak()

    def _reset_v(
        self, v: Tensor, pos_mask: Tensor, neg_mask: Tensor, threshold: Tensor
    ) -> Tensor:
        """Reset membrane potential after threshold comparison.

        After comparison, each element falls into one of three groups:

        - **pos_mask**: crossed positive threshold (v >= thres_pos)
        - **neg_mask**: crossed negative threshold (v <= thres_neg)
        - **neither**: stayed between thresholds, v unchanged

        Args:
            v: Membrane potential (pre-reset).
            pos_mask: Elements that crossed the positive threshold.
            neg_mask: Elements that crossed the negative threshold.
            threshold: Per-element threshold used by LINEAR mode.
        """
        if self.reset_mode == RM.MODE_NORMAL:
            v[pos_mask] = self.reset_v
            v[neg_mask] = -self.reset_v
        elif self.reset_mode == RM.MODE_LINEAR:
            fired = pos_mask | neg_mask
            v[fired] -= threshold[fired]
        # MODE_NONRESET: no reset, v unchanged

        return v

    def _apply_tau_shift(self, v: Tensor) -> Tensor:
        """Apply bit-shift scaling derived from ``leak_tau`` to *v*."""
        if v.is_floating_point():
            if self.leak_tau >= 0:
                return v * (2.0**self.leak_tau)
            else:
                return v / (2.0 ** (-self.leak_tau))

        if self.leak_tau >= 0:
            return v << self.leak_tau
        else:
            return v >> (-self.leak_tau)

    def _multiplicative_leak(self) -> Tensor:
        """Multiplicative leak: ``v -= shift(v - offset)``."""
        if self.leak_multi_mode == LeakMultiMode.ENABLE:
            return self.v - self._apply_tau_shift(self.v - self.reset_v)
        if self.leak_tau == 0:
            return self.v
        return self.v - self._apply_tau_shift(self.v)

    def _validate_dynamics_homogeneity(self) -> None:
        if not self.is_snn or not torch.is_tensor(self.leak_tau):
            return
        if self.leak_multi_mode == LeakMultiMode.ENABLE:
            return

        zero_shift = self.leak_tau == 0
        if torch.any(zero_shift).item() and not torch.all(zero_shift).item():
            raise ValueError(
                "leak_tau mixes IF (zero shift) and LIF (non-zero shift) dynamics "
                "while leak_multi_mode is DISABLE"
            )

    def _validate_simulation_params(self) -> None:
        unsupported = [
            name
            for name in ("reset_v", "thres_neg", "leak_tau", "init_v")
            if torch.is_tensor(getattr(self, name))
        ]
        if unsupported:
            raise NotImplementedError(
                "vector neuron simulation is not implemented for "
                + ", ".join(unsupported)
                + "; compile the model for deployment instead"
            )

    def _validate_threshold_bounds(self) -> None:
        """Validate bounds that do not depend on the eventual output shape."""
        pos = self.thres_pos
        neg = self.thres_neg

        if torch.is_tensor(pos) and torch.is_tensor(neg):
            if pos.shape != neg.shape:
                return
            neg = neg.to(pos.device)

        invalid = pos < neg
        if torch.is_tensor(invalid):
            invalid = bool(torch.any(invalid).item())

        if invalid:
            raise ValueError(f"'thres_pos' ({pos}) must be >= 'thres_neg' ({neg})")

    def _thres_pos_for_v(self) -> float | Tensor:
        """Return scalar threshold or a channel-broadcast tensor for ``self.v``."""
        if not torch.is_tensor(self.thres_pos):
            return self.thres_pos
        if not torch.is_tensor(self.v):
            raise RuntimeError("per-channel 'thres_pos' requires tensor membrane state")
        if self.v.ndim < 2:
            raise ValueError(
                "per-channel 'thres_pos' requires neuron state with a channel dimension"
            )
        if self.v.shape[1] != self.thres_pos.numel():
            raise ValueError(
                f"per-channel 'thres_pos' has {self.thres_pos.numel()} element(s) "
                f"but neuron state channel count is {self.v.shape[1]}"
            )

        threshold = self.thres_pos.to(device=self.v.device)
        return threshold.reshape(
            _channel_broadcast_shape(self.v.ndim, threshold.numel())
        )

    def _tau_to_shift(self, tau: float | Tensor) -> int | Tensor:
        """Convert time constant *tau* to a right-shift exponent."""
        if torch.is_tensor(tau):
            shifts: list[int] = []
            approximated = 0
            for value in tau.reshape(-1).tolist():
                shift, is_approximate = _tau_scalar_to_shift(float(value))
                shifts.append(shift)
                approximated += int(is_approximate)
            if approximated:
                warnings.warn(
                    f"{approximated} tau value(s) are not powers of 2; "
                    "using ceil(log2(tau)) shift approximation",
                    AutoOptimizationWarning,
                )
            return torch.tensor(shifts, dtype=torch.int64, device=tau.device).reshape(
                tau.shape
            )

        shift, is_approximate = _tau_scalar_to_shift(float(tau))
        if is_approximate:
            exponent = -shift
            warnings.warn(
                f"tau={tau} is not a power of 2, using nearest shift bit {exponent}",
                AutoOptimizationWarning,
            )
        return shift

    def to_neuron_params(self, bias: Tensor | None = None) -> NeuronParams:
        """Export neuron parameters for the current pre-materialization API."""
        values = {attr: getattr(self, attr) for attr in self.__export_attrs__}
        if bias is not None:
            if self.leak_add_mode == LeakAddMode.BACKWARD:
                raise ValueError(
                    "'bias' cannot be fused when 'leak_add_mode' is BACKWARD"
                )
            values["leak_v"] += bias
        return NeuronParams(**values)


def _tau_scalar_to_shift(tau: float) -> tuple[int, bool]:
    """Return the hardware shift and whether *tau* required approximation."""
    if tau <= 1:
        return 0, False

    log2_tau = math.log2(tau)
    if log2_tau.is_integer():
        return -int(log2_tau), False
    return -math.ceil(log2_tau), True


def _normalize_numeric_param(value: int | float | Tensor, *, name: str):
    if not torch.is_tensor(value):
        return value
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if value.ndim == 0:
        return value.item()
    return value.detach().clone()


def _normalize_leak_tau_shift(value: int | Tensor) -> int | Tensor:
    if isinstance(value, bool):
        raise TypeError("leak_tau_shift must be an integer or integer Tensor")
    if not torch.is_tensor(value):
        if not isinstance(value, int):
            raise TypeError("leak_tau_shift must be an integer or integer Tensor")
        return value
    if value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("leak_tau_shift Tensor must have an integer dtype")
    return _normalize_numeric_param(value, name="leak_tau_shift")  # type: ignore


def _resolve_reset(v_reset: float | Tensor | None) -> tuple[float | Tensor, RM]:
    """Determine reset mode from *v_reset*.

    - ``None`` -> soft reset (MODE_LINEAR), reset_v = 0
    - ``float`` -> hard reset (MODE_NORMAL), reset_v = v_reset
    """
    if v_reset is None:
        return 0.0, RM.MODE_LINEAR  # soft reset
    return v_reset, RM.MODE_NORMAL  # hard reset


def _channel_broadcast_shape(ndim: int, channel_count: int) -> tuple[int, ...]:
    """Broadcast shape for applying a `(C,)` tensor across an `N,C,...` state."""
    return (1, channel_count, *(1 for _ in range(ndim - 2)))


class IFNodeV25(CoreNeuronV25):
    __fixed_attrs__: ClassVar[tuple[str, ...]] = ("tau", *_FIXED_DYNAMICS_ATTRS)

    def __init__(
        self,
        v_threshold: float | Tensor = 1.0,
        v_reset: float | Tensor | None = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        **kwargs,
    ) -> None:
        """Integrate-and-Fire neuron.

        Args:
            v_threshold: Firing threshold.
            v_reset: Reset voltage. ``None`` for soft reset.
            surrogate_function: Surrogate gradient function.
            detach_reset: If ``True``, detach the reset operation.
            **kwargs: Forwarded to :class:`CoreNeuronV25`.
        """
        self._reject_fixed_attr_overrides(kwargs)

        reset_v, reset_mode = _resolve_reset(v_reset)
        super().__init__(
            reset_mode=reset_mode,
            reset_v=reset_v,
            thres_pos=v_threshold,
            tau=1,  # leak_tau_shift=0,
            leak_multi_input=LeakMultiInputMode.DISABLE,
            leak_multi_mode=LeakMultiMode.DISABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            init_v=reset_v,  # Match SpikingJelly: init_v = v_reset
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            **kwargs,
        )


class LIFNodeV25(CoreNeuronV25):
    __fixed_attrs__: ClassVar[tuple[str, ...]] = _FIXED_DYNAMICS_ATTRS

    def __init__(
        self,
        tau: float | Tensor = 2.0,
        decay_input: bool = True,
        v_threshold: float | Tensor = 1.0,
        v_reset: float | Tensor | None = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        **kwargs,
    ) -> None:
        """Leaky Integrate-and-Fire neuron.

        Args:
            tau: Membrane time constant. Must be > 1 and a power of 2.
            decay_input: Whether input current is scaled by tau.
            v_threshold: Firing threshold.
            v_reset: Reset voltage. ``None`` for soft reset.
            surrogate_function: Surrogate gradient function.
            detach_reset: If ``True``, detach the reset operation.
            **kwargs: Forwarded to :class:`CoreNeuronV25`.
        """
        self._reject_fixed_attr_overrides(kwargs)

        invalid_tau = torch.any(tau <= 1).item() if torch.is_tensor(tau) else tau <= 1
        if invalid_tau:
            raise ValueError(f"tau must be > 1, got {tau}")

        reset_v, reset_mode = _resolve_reset(v_reset)
        has_nonzero_reset = (
            torch.any(reset_v != 0).item() if torch.is_tensor(reset_v) else reset_v != 0
        )
        leak_multi_mode = LeakMultiMode(has_nonzero_reset)

        super().__init__(
            reset_mode=reset_mode,
            reset_v=reset_v,
            thres_pos=v_threshold,
            tau=tau,
            leak_multi_input=decay_input,
            leak_multi_mode=leak_multi_mode,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            init_v=reset_v,  # Match SpikingJelly: init_v = v_reset
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"leak_tau={self.leak_tau}, " + super().extra_repr()


class LeakyBeta0NodeV25(CoreNeuronV25):
    """Memoryless leaky neuron at the exact ``beta=0``."""

    __fixed_attrs__: ClassVar[tuple[str, ...]] = ("tau", *_FIXED_DYNAMICS_ATTRS)

    def __init__(
        self,
        v_threshold: float | Tensor = 1.0,
        v_reset: float | Tensor | None = 0.0,
        **kwargs,
    ) -> None:
        self._reject_fixed_attr_overrides(kwargs)

        reset_v, reset_mode = _resolve_reset(v_reset)
        super().__init__(
            reset_mode=reset_mode,
            reset_v=reset_v,
            thres_pos=v_threshold,
            tau=1,
            leak_tau_shift=0,
            leak_multi_input=LeakMultiInputMode.DISABLE,
            leak_multi_mode=LeakMultiMode.ENABLE,
            leak_multi_sequence=LeakMultiComparisonOrder.AFTER_COMPARE,
            init_v=reset_v,
            **kwargs,
        )


class ANNNodeV25(CoreNeuronV25):
    def __init__(self, lut: LutActivation, **kwargs) -> None:
        """ANN activation node using LUT lookup.

        Convenience wrapper for ANN mode that takes only a
        :class:`~paibox.paiir.ir.lut_activation.LutActivation` and hides
        SNN-specific parameters.

        Args:
            lut: LUT activation module (required).
            **kwargs: Forwarded to :class:`CoreNeuronV25` for advanced overrides.
        """
        super().__init__(lut=lut, **kwargs)
