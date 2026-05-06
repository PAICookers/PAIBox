"""Helpers for turning manual quantized FX models into PAIIR-ready models."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn

from paibox.paiir import ANNNodeV25, register_neuron
from paibox.paiir.ir.lut_activation import LutLinear, LutReLU, LutReLUSymmetric

from .converter import convert_fx_to_manual
from .ops import (
    ManualIntAddResidual,
    ManualQuantConv2d,
    ManualQuantConvReLU2d,
    ManualQuantLinear,
    ManualQuantLinearReLU,
)

__all__ = [
    "DeployLutReLU",
    "convert_ready_paiir"
]


class DeployLutReLU(nn.Module):
    """Leaf activation that lowers to a calibrated LUT-based ReLU."""

    _is_leaf_module = True

    def __init__(
        self,
        min_val: float,
        max_val: float,
        output_sign: int,
        is_symmetric: bool,
    ) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.output_sign = int(output_sign)
        self.is_symmetric = bool(is_symmetric)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class DeployLutLinear(nn.Module):
    """Leaf activation that lowers to a calibrated LUT-based linear map."""

    _is_leaf_module = True

    def __init__(
        self,
        min_val: float,
        max_val: float,
        output_sign: int,
    ) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.output_sign = int(output_sign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _register_deploy_lut_relu() -> None:
    def _convert_deploy_lut_relu(mod: nn.Module) -> ANNNodeV25:
        deploy_mod = cast(DeployLutReLU, mod)
        min_val = float(deploy_mod.min_val)
        max_val = float(deploy_mod.max_val)
        output_sign = int(deploy_mod.output_sign)

        if deploy_mod.is_symmetric:
            return ANNNodeV25(
                LutReLUSymmetric(
                    min_val=min_val,
                    max_val=max_val,
                    output_sign=output_sign,
                )
            )

        return ANNNodeV25(
            LutReLU(
                min_val=min_val,
                max_val=max_val,
                output_sign=output_sign,
            )
        )

    try:
        register_neuron(DeployLutReLU, _convert_deploy_lut_relu)
    except ValueError:
        pass


_register_deploy_lut_relu()


def _register_deploy_lut_linear() -> None:
    def _convert_deploy_lut_linear(mod: nn.Module) -> ANNNodeV25:
        deploy_mod = cast(DeployLutLinear, mod)
        return ANNNodeV25(
            LutLinear(
                min_val=float(deploy_mod.min_val),
                max_val=float(deploy_mod.max_val),
                output_sign=int(deploy_mod.output_sign),
            )
        )

    try:
        register_neuron(DeployLutLinear, _convert_deploy_lut_linear)
    except ValueError:
        pass


_register_deploy_lut_linear()


def _build_deploy_lut_relu(
    s_in: float,
    s_w: float,
    s_out: float,
    activation_symmetric: bool,
) -> DeployLutReLU:
    s_accum = s_in * s_w
    lut_scale = s_out / s_accum if s_accum != 0 else 0.0

    if activation_symmetric:
        return DeployLutReLU(
            min_val=-lut_scale * 128.0,
            max_val=lut_scale * 127.0,
            output_sign=1,
            is_symmetric=True,
        )

    return DeployLutReLU(
        min_val=-5.0,
        max_val=lut_scale * 255.0,
        output_sign=0,
        is_symmetric=False,
    )


def _build_deploy_lut_linear(
    s_in: float,
    s_w: float,
    s_out: float,
) -> DeployLutLinear:
    s_accum = s_in * s_w
    lut_scale = s_out / s_accum if s_accum != 0 else 0.0

    return DeployLutLinear(
        min_val=-lut_scale * 128.0,
        max_val=lut_scale * 127.0,
        output_sign=1,
    )


def _bind_quantized_linear_like_params(
    module: nn.Module,
    weight_q: torch.Tensor,
    bias_fp32: torch.Tensor | None,
    accum_scale: float,
) -> None:
    weight_q = weight_q.detach()
    weight_param = cast(torch.Tensor, module.weight)
    module.register_buffer(
        "weight_int8", weight_q.to(device=weight_param.device).clone()
    )

    with torch.no_grad():
        weight_param.copy_(weight_q.to(
            device=weight_param.device, dtype=weight_param.dtype))

    if bias_fp32 is None:
        return

    bias_q = torch.round(bias_fp32.detach() / accum_scale).to(torch.int32)
    module.register_buffer(
        "bias_int32", bias_q.to(device=weight_param.device).clone()
    )

    if module.bias is not None:
        bias_param = cast(torch.Tensor, module.bias)
        with torch.no_grad():
            bias_param.copy_(
                bias_q.to(device=bias_param.device, dtype=bias_param.dtype))


def _make_conv2d_like(module: ManualQuantConv2d | ManualQuantConvReLU2d) -> nn.Conv2d:
    bias_enabled = module.bias_val is not None
    conv = nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=getattr(module, "dilation", 1),
        groups=module.groups,
        bias=bias_enabled,
        padding_mode=getattr(module, "padding_mode", "zeros"),
    )

    _bind_quantized_linear_like_params(
        conv,
        module.weight_q,
        module.bias_val,
        module.s_in * module.s_w,
    )
    return conv


def _make_linear_like(module: ManualQuantLinear | ManualQuantLinearReLU) -> nn.Linear:
    out_features, in_features = module.weight_q.shape
    bias_enabled = module.bias_val is not None
    linear = nn.Linear(in_features=in_features,
                       out_features=out_features, bias=bias_enabled)

    _bind_quantized_linear_like_params(
        linear,
        module.weight_q,
        module.bias_val,
        module.s_in * module.s_w,
    )
    return linear


def _make_conv_relu_block(
    module: ManualQuantConvReLU2d,
) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", _make_conv2d_like(module)),
                (
                    "act",
                    _build_deploy_lut_relu(
                        module.s_in,
                        module.s_w,
                        module.s_out,
                        module.activation_symmetric,
                    ),
                ),
            ]
        )
    )


def _make_conv_linear_block(
    module: ManualQuantConv2d,
) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", _make_conv2d_like(module)),
                (
                    "act",
                    _build_deploy_lut_linear(
                        module.s_in,
                        module.s_w,
                        module.s_out,
                    ),
                ),
            ]
        )
    )


def _make_linear_relu_block(
    module: ManualQuantLinearReLU,
) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("linear", _make_linear_like(module)),
                (
                    "act",
                    _build_deploy_lut_relu(
                        module.s_in,
                        module.s_w,
                        module.s_out,
                        module.activation_symmetric,
                    ),
                ),
            ]
        )
    )


class DeployResidualAdd(nn.Module):
    """Standard residual block that keeps the add path traceable for PAIIR."""

    def __init__(self, manual_module: ManualIntAddResidual) -> None:
        super().__init__()
        self.conv = _make_conv2d_like(manual_module.conv)
        self.act = _build_deploy_lut_relu(
            manual_module.conv.s_in,
            manual_module.conv.s_w,
            manual_module.out_scale,
            manual_module.activation_symmetric,
        )
        self.activation_symmetric = manual_module.activation_symmetric
        self.conv2_out_scale = manual_module.conv2_out_scale
        self.out_scale = manual_module.out_scale
        self.out_zp = manual_module.out_zp
        self.x_scale = manual_module.x_scale
        self.x_zp = manual_module.x_zp

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(y))


def _convert_manual_module(module: nn.Module) -> nn.Module:
    if isinstance(module, ManualIntAddResidual):
        return DeployResidualAdd(module)

    if isinstance(module, ManualQuantConvReLU2d):
        return _make_conv_relu_block(module)

    if isinstance(module, ManualQuantConv2d):
        return _make_conv_linear_block(module)

    if isinstance(module, ManualQuantLinearReLU):
        return _make_linear_relu_block(module)

    if isinstance(module, ManualQuantLinear):
        return _make_linear_like(module)

    for child_name, child_module in list(module.named_children()):
        converted_child = _convert_manual_module(child_module)
        if converted_child is not child_module:
            setattr(module, child_name, converted_child)

    return module


def convert_ready_paiir(
    manual_model: nn.Module,
) -> nn.Module:
    """Rewrite a manual quantized model into a standard PyTorch deploy model.

    The returned model keeps quantized weights and biases as float-valued
    parameters so :func:`paibox.paiir.compile_to_paiir` can trace it using
    regular ``nn.Conv2d`` / ``nn.Linear`` layers and leaf LUT activations.
    """

    deploy_model = copy.deepcopy(manual_model)
    deploy_model = _convert_manual_module(deploy_model)
    deploy_model.eval()
    return deploy_model
