import math
import warnings
from typing import Any

import torch
from paicorelib import (
    RM,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    ThresholdNegMode,
    ThresholdPosMode,
)
from spikingjelly.activation_based.base import MemoryModule
from torch import Tensor

from ..exceptions import NotSupportedError
from .ir_base import PAIIR


class NeuronV2(MemoryModule, PAIIR):
    _attrs_to_save: tuple[str, ...] = (
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
        "tick_start",
        "tick_duration",
        "tick_initial",
    )

    def __init__(
        self,
        reset_mode: RM = RM.MODE_NORMAL,
        reset_v: float = 0,
        thres_neg_mode: ThresholdNegMode = ThresholdNegMode.FIRE,
        thres_pos_mode: ThresholdPosMode = ThresholdPosMode.FIRE,
        thres_neg: float | None = None,
        thres_pos: float = 0,
        lateral_inhi: LateralInhibitionMode | bool = LateralInhibitionMode.DISABLE,
        leak_multi_sequence: LeakMultiComparisonOrder = LeakMultiComparisonOrder.AFTER_COMPARE,
        leak_multi_input: LeakMultiInputMode | bool = LeakMultiInputMode.DISABLE,
        leak_multi_mode: LeakMultiMode | bool = LeakMultiMode.DISABLE,
        leak_add_mode: LeakAddMode = LeakAddMode.FORWARD,
        tau: float = 2,
        leak_v: float | Tensor = 0,
        init_v: float = 0,
        *,
        leak_tau_shift: int | None = None,
        delay: int = 1,
        tick_start: int = 1,
        tick_duration: int = 0,
        tick_initial: int = 0,
    ) -> None:
        """Offline neuron model for chip v2.5."""
        super().__init__()
        _thres_neg = (
            thres_neg
            if thres_neg is not None
            else -9999
            # OfflineNeuRegLimV2.THRES_NEG_MIN
        )
        self.reset_mode = reset_mode
        self.reset_v = reset_v
        self.thres_neg_mode = thres_neg_mode
        self.thres_pos_mode = thres_pos_mode
        self.thres_neg = _thres_neg
        self.thres_pos = thres_pos
        self.lateral_inhi = LateralInhibitionMode(lateral_inhi)
        self.leak_multi_sequence = leak_multi_sequence
        self.leak_multi_input = LeakMultiInputMode(leak_multi_input)
        self.leak_multi_mode = LeakMultiMode(leak_multi_mode)
        self.leak_add_mode = leak_add_mode

        if leak_tau_shift is None:
            self.leak_tau = self._get_tau_exponent(tau)
        else:
            self.leak_tau = leak_tau_shift

        self.leak_v = self._reduce_leak_v_tensor(leak_v)
        self.init_v = init_v
        self.delay = delay
        self.tick_start = tick_start
        self.tick_duration = tick_duration
        self.tick_initial = tick_initial

        self.register_memory("v", init_v)
        self._any_spike = False  # whether any spike has been fired at last timestep

    def forward(self, x: Tensor) -> Tensor:
        return self.single_step_forward(x)

    def single_step_forward(self, x: Tensor) -> Tensor:
        if isinstance(self.v, int):
            init_v = self.v
            self.v = torch.full_like(x.data, init_v)

        if self.lateral_inhi == LateralInhibitionMode.ENABLE and self._any_spike:
            # Lateral inhibition. Voltage starts accumulating at reset_v
            self.v = torch.full_like(x.data, self.reset_v)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()

        self.neuronal_reset(spike)

        if self.leak_multi_sequence == LeakMultiComparisonOrder.AFTER_COMPARE:
            self.v = self._leak_multi()

        self._any_spike = torch.any(spike)
        return spike

    def neuronal_charge(self, x: Tensor) -> None:
        if self.leak_multi_input == LeakMultiInputMode.ENABLE:
            self.v += self._leak_tau_shift(x)
        else:
            self.v += x

        # Leak add
        if self.leak_add_mode == LeakAddMode.FORWARD:
            self.v += self.leak_v
        else:
            self.v += torch.sign(self.v) * self.leak_v

        if self.leak_multi_sequence == LeakMultiComparisonOrder.BEFORE_COMPARE:
            self.v = self._leak_multi()

    def neuronal_fire(self) -> Tensor:
        return ((self.v - self.thres_pos) >= 0).to(torch.int8)

    def neuronal_reset(self, spike: Tensor) -> None:
        self.v = self._pos_thres_reset(self.v, spike)
        neg_spike = (self.v - self.thres_neg) <= 0
        self.v = self._neg_thres_reset(self.v, neg_spike)

    def _leak_tau_shift(self, v: Tensor) -> Tensor:
        if self.leak_tau >= 0:
            return v << self.leak_tau
        else:
            return v >> -self.leak_tau

    def _leak_multi(self) -> Tensor:
        if self.leak_multi_mode == LeakMultiMode.ENABLE:
            return self.v - self._leak_tau_shift(self.v - self.reset_v)
        else:
            return self.v - self._leak_tau_shift(self.v)

    def _pos_thres_reset(self, v: Tensor, spike: Tensor) -> Tensor:
        if self.thres_pos_mode == ThresholdPosMode.FIRE:
            if self.reset_mode == RM.MODE_NORMAL:
                return (1 - spike) * v + spike * self.reset_v  # hard reset
            elif self.reset_mode == RM.MODE_LINEAR:
                return v - spike * self.thres_pos  # soft reset
            else:
                return v
        else:  # CEILING
            return (1 - spike) * v + spike * self.thres_pos

    def _neg_thres_reset(self, v: Tensor, spike: Tensor) -> Tensor:
        if self.thres_neg_mode == ThresholdNegMode.FIRE:
            if self.reset_mode == RM.MODE_NORMAL:
                return (1 - spike) * v - spike * self.reset_v  # hard reset
            elif self.reset_mode == RM.MODE_LINEAR:
                return v - spike * self.thres_neg  # soft reset
            else:
                return v
        else:  # FLOOR
            return (1 - spike) * v - spike * self.thres_neg

    @staticmethod
    def _get_tau_exponent(tau: float) -> int:
        assert tau > 1
        RIGHT = -1

        log2_tau = math.log2(tau)
        if log2_tau == int(log2_tau):
            exponent = int(log2_tau)
        else:
            exponent = math.ceil(log2_tau)
            warnings.warn(
                f"tau={tau} is not a power of 2, find the nearest right shift bit {exponent}"
            )
        return RIGHT * exponent

    @staticmethod
    def _reduce_leak_v_tensor(
        leak_v: float | Tensor,
    ) -> float | Tensor:
        """If the 'leak_v' is a scalar or a tensor with a single value, it can be reduced to a scalar."""
        if isinstance(leak_v, (int, float)):
            return leak_v
        elif (u_val := torch.unique(leak_v)).numel() == 1:
            return u_val.item()  # Tensor with a single value
        else:
            return leak_v

    def get_attrs(self, bias: Tensor | None = None) -> dict[str, Any]:
        """Get the attributes of the neuron. Bias must be added to the `leak_v`."""
        if bias is not None:
            if self.leak_add_mode == LeakAddMode.BACKWARD:
                raise NotSupportedError(
                    "when bias is added to the 'leak_v', 'leak_add_mode' must be forward"
                )

        attrs = self.get_extra_state()
        attrs["leak_v"] += bias
        return attrs

    def get_extra_state(self) -> dict[str, Any]:
        state = {}
        for name in self._attrs_to_save:
            state[name] = getattr(self, name)

        return state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)


def _get_soft_hard_reset(v_reset: float | None) -> tuple[float, RM]:
    if v_reset is None:
        return 0, RM.MODE_LINEAR  # soft reset
    else:
        return v_reset, RM.MODE_NORMAL  # hard reset


class SJIFNode(NeuronV2):
    def __init__(
        self, v_threshold: float = 1, v_reset: float | None = 0, **kwargs
    ) -> None:
        v_reset, reset_mode = _get_soft_hard_reset(v_reset)
        super().__init__(
            reset_mode=reset_mode, reset_v=v_reset, thres_pos=v_threshold, **kwargs
        )


class SJLIFNode(NeuronV2):
    def __init__(
        self,
        tau: float = 2,
        decay_inut: bool = True,
        v_threshold: float = 1,
        v_reset: float | None = 0,
        **kwargs,
    ) -> None:
        assert tau > 1
        v_reset, reset_mode = _get_soft_hard_reset(v_reset)
        leak_multi_mode = (
            LeakMultiMode.ENABLE if v_reset != 0 else LeakMultiMode.DISABLE
        )

        super().__init__(
            reset_mode=reset_mode,
            reset_v=v_reset,
            thres_pos=v_threshold,
            tau=tau,
            leak_multi_input=decay_inut,
            leak_multi_mode=leak_multi_mode,
            **kwargs,
        )


class OfflineNeuronV2(NeuronV2):
    pass


class OnlineNeuronV2(NeuronV2):
    pass


# class _NeuV2CalcParamsKwds(TypedDict, total=False):
#     reset_mode: RM
#     reset_v: float
#     thres_neg_mode: ThresholdNegMode
#     thres_pos_mode: ThresholdPosMode
#     thres_neg: float
#     thres_pos: float
#     lateral_inhi: bool | LateralInhibitionMode
#     leak_multi_sequence: LeakMultiComparisonOrder
#     leak_multi_input: bool | LeakMultiInputMode
#     leak_multi_mode: bool | LeakMultiMode
#     leak_add_mode: LeakAddMode
#     leak_tau: int
#     leak_v: float
#     init_v: float


# def convert_sj_neuron(
#     n: LIFNode | IFNode, **kwargs: Unpack[_NeuV2CalcParamsKwds]
# ) -> NeuronV2:
#     """Convert a SpikingJelly neuron to an `NeuronV2` instance.

#     Args:
#         n (IFNode, LIFNode): A SpikingJelly neuron instance.
#         **kwargs: Additional arguments to pass to `NeuronV2` constructor. If a keyword argument is specified,
#             the parameter will be overridden by the corresponding keyword argument.
#     """
#     if not isinstance(n, tuple(SUPPORTED_NEU_OPS)):
#         supported = ", ".join([n._get_name() for n in SUPPORTED_NEU_OPS])
#         raise TypeError(
#             f"unsupported neuron type: {type(n)}, only {supported} are supported"
#         )

#     default = NeuV2ClacParams.default()

#     thres_pos = kwargs.get("thres_pos", n.v_threshold)
#     thres_neg = kwargs.get("thres_neg", None)

#     thres_neg_mode = kwargs.get("thres_neg_mode", ThresholdNegMode.FIRE)
#     thres_pos_mode = kwargs.get("thres_pos_mode", ThresholdPosMode.FIRE)

#     if n.v_reset is None:
#         # hard reset
#         _reset_v = 0
#         reset_mode = RM.MODE_NORMAL
#     else:
#         # soft reset
#         _reset_v = n.v_reset
#         _reset_mode = RM.MODE_LINEAR

#     reset_v = kwargs.get("reset_v", _reset_v)
#     reset_mode = kwargs.get("reset_mode", _reset_mode)

#     if isinstance(n, LIFNode):
#         _tau = int(n.tau)  # In LIF, tau > 1
#         _leak_multi_input = n.decay_input
#     else:
#         _tau = 0
#         _leak_multi_input = False

#     tau = int(kwargs.get("leak_tau", _tau))
#     # NOTE: for LIF, multiplication leaking must implemented after theshold comparison
#     leak_multi_sequence = kwargs.get("leak_multi_sequence", default.leak_multi_sequence)
#     leak_multi_input = kwargs.get("leak_multi_input", _leak_multi_input)

#     _leak_multi_mode = LeakMultiMode.ENABLE if reset_v != 0 else LeakMultiMode.DISABLE
#     leak_multi_mode = kwargs.get("leak_multi_mode", _leak_multi_mode)

#     lateral_inhi = kwargs.get("lateral_inhi", default.lateral_inhi)
#     leak_add_mode = kwargs.get("leak_add_mode", default.leak_add_mode)
#     leak_v = kwargs.get("leak_v", default.leak_v)
#     init_v = kwargs.get("init_v", reset_v)

#     return NeuronV2(
#         reset_mode,
#         reset_v,
#         thres_neg_mode,
#         thres_pos_mode,
#         thres_neg,
#         thres_pos,
#         lateral_inhi,
#         leak_multi_sequence,
#         leak_multi_input,
#         leak_multi_mode,
#         leak_add_mode,
#         leak_tau,
#         leak_v,
#         init_v,
#     )
