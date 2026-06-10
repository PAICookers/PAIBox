from __future__ import annotations

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn

from torch.ao.quantization.fx.custom_config import ConvertCustomConfig, PrepareCustomConfig

from .observed_ops import (
    ObservedManualConv2d,
    ObservedManualConvAddReLU2d,
    ObservedManualConvReLU2d,
    ObservedManualLinear,
    ObservedManualLinearReLU,
)
from .ops import (
    ManualConvAddReLU2d,
    ManualConv2d,
    ManualConvReLU2d,
    ManualLinear,
    ManualLinearReLU,
)

MANUAL_MODULE_TYPES = (
    ManualConv2d,
    ManualConvReLU2d,
    ManualLinear,
    ManualLinearReLU,
    ManualConvAddReLU2d,
)


def build_manual_prepare_custom_config() -> PrepareCustomConfig:
    return (
        PrepareCustomConfig()
        .set_float_to_observed_mapping(nn.Conv2d, ObservedManualConv2d)
        .set_float_to_observed_mapping(nni.ConvReLU2d, ObservedManualConvReLU2d)
        .set_float_to_observed_mapping(nn.Linear, ObservedManualLinear)
        .set_float_to_observed_mapping(nni.LinearReLU, ObservedManualLinearReLU)
        .set_float_to_observed_mapping(nni.ConvAddReLU2d, ObservedManualConvAddReLU2d)
    )


def build_manual_convert_custom_config() -> ConvertCustomConfig:
    return (
        ConvertCustomConfig()
        .set_observed_to_quantized_mapping(ObservedManualConv2d, ManualConv2d)
        .set_observed_to_quantized_mapping(
            ObservedManualConvReLU2d, ManualConvReLU2d
        )
        .set_observed_to_quantized_mapping(ObservedManualLinear, ManualLinear)
        .set_observed_to_quantized_mapping(
            ObservedManualLinearReLU, ManualLinearReLU
        )
        .set_observed_to_quantized_mapping(
            ObservedManualConvAddReLU2d, ManualConvAddReLU2d
        )
    )
