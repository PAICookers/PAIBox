from __future__ import annotations

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn


class _ObservedManualBase(nn.Module):
    float_module: nn.Module

    @classmethod
    def from_float(cls, float_module: nn.Module):
        observed = cls(float_module)
        observed.qconfig = float_module.qconfig
        return observed

    def __init__(self, float_module: nn.Module) -> None:
        super().__init__()
        self.float_module = float_module
        self.qconfig = float_module.qconfig
        self.input_activation_post_process = self.qconfig.activation()
        self.activation_post_process = self.qconfig.activation()


class _ObservedSingleInput(_ObservedManualBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_activation_post_process(x)
        out = self.float_module(x)
        return self.activation_post_process(out)


class ObservedManualConv2d(_ObservedSingleInput):
    pass


class ObservedManualConvReLU2d(_ObservedSingleInput):
    pass


class ObservedManualLinear(_ObservedSingleInput):
    pass


class ObservedManualLinearReLU(_ObservedSingleInput):
    pass


class ObservedManualConvAddReLU2d(_ObservedManualBase):
    def __init__(self, float_module: nni.ConvAddReLU2d) -> None:
        super().__init__(float_module)
        self.y_activation_post_process = self.input_activation_post_process
        self.x_activation_post_process = self.qconfig.activation()

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = self.y_activation_post_process(y)
        x = self.x_activation_post_process(x)
        out = self.float_module(y, x)
        return self.activation_post_process(out)
