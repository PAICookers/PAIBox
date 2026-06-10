"""PAIIR integration for manual quantized modules.

这个模块把 ``quantize_tools.ops`` 里的 Manual 算子注册到 PAIIR。长期路径
应该是 ``ManualQuant* -> PAIIR quantized IR -> backendv2``，而不是先把
Manual 算子改写成普通 deploy PyTorch 模块。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from paibox.paiir import ANNNodeV25, register_ir_module
from paibox.paiir.ir.lut_activation import LutLinear, LutReLU, LutReLUSymmetric
from paibox.paiir.ir.quantized_ops import QuantizedConvAddReLU2dOp
from paibox.paiir.ir.op_node import SequentialOp

from .ops import (
    ManualConvAddReLU2d,
    ManualConv2d,
    ManualConvReLU2d,
    ManualLinear,
    ManualLinearReLU,
)
from .utils import approximate_scale_ratio

__all__ = ["register_manual_quantized_paiir"]


def _bind_quantized_params(
    module: nn.Conv2d | nn.Linear,
    weight_q: torch.Tensor,
    bias_fp32: torch.Tensor | None,
    accum_scale: float,
) -> None:
    """Copy quantized weights/bias into canonical modules used by PAIIR.

    PAIIR/backendv2 现在从 ``nn.Conv2d`` / ``nn.Linear`` 的参数直接取权重，
    所以这里把 int8/int32 参数既保留成 buffer，也复制到原参数里。
    """

    weight_q = weight_q.detach()
    weight_param = cast(torch.Tensor, module.weight)
    module.register_buffer(
        "weight_int8", weight_q.to(weight_param.device).clone())

    with torch.no_grad():
        weight_param.copy_(weight_q.to(
            weight_param.device, weight_param.dtype))

    if bias_fp32 is None:
        return

    bias_q = torch.round(bias_fp32.detach() / accum_scale).to(torch.int32)
    module.register_buffer(
        "bias_int32", bias_q.to(weight_param.device).clone())

    if module.bias is not None:
        bias_param = cast(torch.Tensor, module.bias)
        with torch.no_grad():
            bias_param.copy_(bias_q.to(bias_param.device, bias_param.dtype))


def _make_conv2d_like(module: ManualConv2d | ManualConvReLU2d) -> nn.Conv2d:
    conv = nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=getattr(module, "dilation", 1),
        groups=module.groups,
        bias=module.bias_val is not None,
        padding_mode=getattr(module, "padding_mode", "zeros"),
    )
    _bind_quantized_params(conv, module.weight_q,
                           module.bias_val, module.s_in * module.s_w)
    return conv


def _make_linear_like(module: ManualLinear | ManualLinearReLU) -> nn.Linear:
    out_features, in_features = module.weight_q.shape
    linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=module.bias_val is not None,
    )
    _bind_quantized_params(
        linear, module.weight_q, module.bias_val, module.s_in * module.s_w
    )
    return linear


def _relu_act(
    s_in: float,
    s_w: float,
    s_out: float,
    activation_symmetric: bool,
) -> ANNNodeV25:
    s_accum = s_in * s_w
    lut_scale = s_out / s_accum if s_accum != 0 else 0.0
    if activation_symmetric:
        return ANNNodeV25(
            LutReLUSymmetric(
                min_val=-lut_scale * 128.0,
                max_val=lut_scale * 127.0,
                output_sign=1,
            )
        )
    return ANNNodeV25(
        LutReLU(min_val=-5.0, max_val=lut_scale * 255.0, output_sign=0)
    )


def _linear_act(s_in: float, s_w: float, s_out: float) -> ANNNodeV25:
    s_accum = s_in * s_w
    lut_scale = s_out / s_accum if s_accum != 0 else 0.0
    return ANNNodeV25(
        LutLinear(
            min_val=-lut_scale * 128.0,
            max_val=lut_scale * 127.0,
            output_sign=1,
        )
    )


def _map_manual_conv_relu(module: ManualConvReLU2d) -> SequentialOp:
    return SequentialOp(
        _make_conv2d_like(module),
        _relu_act(module.s_in, module.s_w, module.s_out,
                  module.activation_symmetric),
    )


def _map_manual_conv(module: ManualConv2d) -> SequentialOp:
    return SequentialOp(
        _make_conv2d_like(module),
        _linear_act(module.s_in, module.s_w, module.s_out),
    )


def _map_manual_linear_relu(module: ManualLinearReLU) -> SequentialOp:
    return SequentialOp(
        _make_linear_like(module),
        _relu_act(module.s_in, module.s_w, module.s_out,
                  module.activation_symmetric),
    )


def _map_manual_linear(module: ManualLinear) -> SequentialOp:
    return SequentialOp(
        _make_linear_like(module),
        _linear_act(module.s_in, module.s_w, module.s_out),
    )


def _map_manual_conv_add_relu(
    module: ManualConvAddReLU2d,
) -> QuantizedConvAddReLU2dOp:
    target_scale = module.s_in * module.s_w
    exact_ratio = module.x_scale / target_scale if target_scale != 0 else 0.0
    shortcut_m, shortcut_n = approximate_scale_ratio(exact_ratio)

    return QuantizedConvAddReLU2dOp(
        conv=_make_conv2d_like(module),
        act=_relu_act(
            module.s_in,
            module.s_w,
            module.out_scale,
            module.activation_symmetric,
        ),
        shortcut_m=shortcut_m,
        shortcut_n=shortcut_n,
    )


def _mark_leaf_modules() -> None:
    # FX tracing 必须把 Manual 算子当作叶子，否则它会展开 Python forward，
    # 量化参数和 fused 语义就会丢失。
    for cls in (
        ManualConvReLU2d,
        ManualConv2d,
        ManualLinearReLU,
        ManualLinear,
        ManualConvAddReLU2d,
    ):
        setattr(cls, "_is_leaf_module", True)


def _register_once(module_type: type[nn.Module], mapper) -> None:
    try:
        register_ir_module(module_type, mapper)
    except ValueError:
        # 测试或交互式脚本可能重复 import。本函数保持幂等。
        pass


def register_manual_quantized_paiir() -> None:
    """Register all ManualQuant modules with PAIIR lowering."""

    _mark_leaf_modules()
    _register_once(ManualConvReLU2d, _map_manual_conv_relu)
    _register_once(ManualConv2d, _map_manual_conv)
    _register_once(ManualLinearReLU, _map_manual_linear_relu)
    _register_once(ManualLinear, _map_manual_linear)
    _register_once(ManualConvAddReLU2d, _map_manual_conv_add_relu)
