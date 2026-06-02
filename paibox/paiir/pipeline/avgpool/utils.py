"""Shared helpers for AvgPool-related deployment logic."""

import math
from typing import TypeAlias, TypeGuard

import torch
import torch.nn as nn

from ...ir.lut_activation import LutCustom
from ...nn import SumPool1d, SumPool2d

__all__ = [
    "ValueCodeRange",
    "build_range_identity_lut",
    "build_sum_pool",
    "get_avgpool_divisor",
    "get_pool_window_size",
    "is_avgpool",
    "is_value_avgpool",
]


PoolWindowModule: TypeAlias = nn.AvgPool1d | nn.AvgPool2d | SumPool1d | SumPool2d
ValueCodeRange: TypeAlias = tuple[int, int]


def build_range_identity_lut(code_range: ValueCodeRange) -> LutCustom:
    """Build a 256-entry identity LUT over a known integer code range."""
    code_min, code_max = code_range
    codes = torch.arange(code_min, code_max + 1, dtype=torch.int32)
    pad_count = 256 - codes.numel()
    if pad_count < 0:
        raise ValueError(f"identity LUT code range too large: {code_range}")
    if pad_count:
        pad = torch.full((pad_count,), code_max, dtype=torch.int32)
        codes = torch.cat((codes, pad))

    return LutCustom(codes, codes, output_sign=1 if code_min < 0 else 0, is_float=False)


def is_avgpool(comp: nn.Module) -> TypeGuard[nn.AvgPool1d | nn.AvgPool2d]:
    """Check if a compute module is an average-pooling op."""
    return isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d))


def is_value_avgpool(
    comp: nn.Module,
) -> TypeGuard[
    nn.AvgPool1d | nn.AvgPool2d | nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d
]:
    """Check if a compute module emits VALUE-domain average-pooling data."""
    return isinstance(
        comp,
        (
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
        ),
    )


def build_sum_pool(comp: nn.AvgPool1d | nn.AvgPool2d) -> SumPool1d | SumPool2d:
    """Build the shape-equivalent SumPool carrier for one AvgPool module."""
    if isinstance(comp, nn.AvgPool1d):
        return SumPool1d(
            comp.kernel_size,
            comp.stride,
            comp.padding,
            1,
            comp.ceil_mode,
        )
    if isinstance(comp, nn.AvgPool2d):
        return SumPool2d(
            comp.kernel_size,
            comp.stride,
            comp.padding,
            1,
            comp.ceil_mode,
        )

    raise TypeError("SumPool conversion only supports AvgPool1d/2d")


def get_pool_window_size(comp: PoolWindowModule) -> int:
    """Return the number of elements in the pooling window."""
    ks = comp.kernel_size
    if isinstance(ks, int):
        if isinstance(comp, (nn.AvgPool2d, SumPool2d)):
            ks = (ks, ks)
        else:
            ks = (ks,)
    return math.prod(ks)


def get_avgpool_divisor(comp: nn.AvgPool1d | nn.AvgPool2d) -> int:
    """Return the effective AvgPool divisor used by the frontend module."""
    divisor_override = comp.divisor_override if isinstance(comp, nn.AvgPool2d) else None
    if divisor_override is not None:
        return int(divisor_override)

    return get_pool_window_size(comp)
