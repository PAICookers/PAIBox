"""Quantized deployment IR nodes.

These nodes are an adapter layer between project-specific quantized PyTorch
modules and the backend-ready PAIIR offline-core subset. They should not reach
backendv2 directly. The compile pipeline materializes them into existing
``StandaloneCompOp`` / ``PotentialAddOp`` / ``StandaloneActOp`` nodes before
fusion and validation.
"""

import torch
from torch import Tensor, nn

from paicorelib import (
    LeakMultiInputMode,
    OutputType,
    ThresholdNegMode,
    ThresholdPosMode,
)

from .calc_params import NeuronParams
from .core_neuron import CoreNeuronV25
from .op_node import OpNode, _run_comp

__all__ = [
    "IdentityScale",
    "PotentialPassthroughNodeV25",
    "QuantizedConvAddReLU2dOp",
]


class IdentityScale(nn.Module):
    """Identity compute path with an integer gain.

    This module represents the shortcut branch scale factor ``M`` in
    ``ratio ~= M * 2^n``. Backend weight extraction recognizes it and creates
    an identity matrix multiplied by ``M`` without materializing a large
    PyTorch parameter tensor.
    """

    _is_leaf_module = True

    def __init__(self, gain: int) -> None:
        super().__init__()
        if gain <= 0:
            raise ValueError(f"IdentityScale gain must be positive, got {gain}")
        self.gain = int(gain)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gain

    def extra_repr(self) -> str:
        return f"gain={self.gain}"


class PotentialPassthroughNodeV25(CoreNeuronV25):
    """Neuron config for a core that writes membrane potential as output.

    用途是把量化残差分支的 ``M * 2^n`` 写进现有硬件字段：

    - ``M`` 放在前面的 ``IdentityScale`` 权重里
    - ``n`` 放在 ``leak_tau``，并打开 ``leak_multi_input``
    - ``output_type`` 强制为 ``POTENTIAL``，让后端导出膜电平输出

    这个类不引入新的后端协议，只是把现有寄存器字段组合成一个明确的
    PAIIR 语义。
    """

    def __init__(self, leak_tau_shift: int = 0) -> None:
        super().__init__(
            thres_pos_mode=ThresholdPosMode.FIRE,
            thres_neg_mode=ThresholdNegMode.FLOOR,
            thres_pos=0,
            leak_multi_input=LeakMultiInputMode.ENABLE,
            leak_tau_shift=int(leak_tau_shift),
        )

    def single_step_forward(self, x: Tensor) -> Tensor:
        # 仿真时直接返回缩放后的膜电平；部署时以后端导出的
        # NeuronParams(output_type=POTENTIAL) 为准。
        return self._apply_tau_shift(x)

    def to_neuron_params(self, bias: Tensor | None = None) -> NeuronParams:
        params = super().to_neuron_params(bias)
        params.output_type = OutputType.POTENTIAL
        return params


class QuantizedConvAddReLU2dOp(OpNode):
    """Quantized residual ``conv(y) + shortcut(x) -> ReLU``.

    The shortcut branch is represented by ``shortcut_m`` and ``shortcut_n``:

    ``x_scale / (conv_input_scale * weight_scale) ~= shortcut_m * 2^shortcut_n``.

    The materialize pass expands this node into:

    1. convolution core that emits membrane potential
    2. shortcut identity-scale core that emits membrane potential
    3. potential add plus LUT ReLU activation

    This keeps all hardware-facing details in normal PAIIR fields instead of
    teaching backendv2 about a new fused operator.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        act: CoreNeuronV25,
        shortcut_m: int,
        shortcut_n: int,
    ) -> None:
        super().__init__()
        if shortcut_m <= 0:
            raise ValueError(f"shortcut_m must be positive, got {shortcut_m}")
        self.conv = conv
        self.act = act
        self.shortcut_m = int(shortcut_m)
        self.shortcut_n = int(shortcut_n)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        conv_acc = _run_comp(self.conv, y)
        shortcut = x * self.shortcut_m * (2.0 ** self.shortcut_n)
        return self.act(conv_acc + shortcut)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, conv={type(self.conv).__name__}, "
            f"act={type(self.act).__name__}, shortcut_m={self.shortcut_m}, "
            f"shortcut_n={self.shortcut_n}"
        )
