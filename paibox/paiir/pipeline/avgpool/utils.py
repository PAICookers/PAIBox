"""Shared helpers for AvgPool-related deployment logic."""

import math
from typing import TypeAlias, TypeGuard

import torch
import torch.nn as nn

from ...ir.calc_params import LUT_TABLE_SIZE
from ...ir.lut_activation import LutCustom
from ...nn import SumPool1d, SumPool2d

__all__ = [
    "ValueCodeRange",
    "build_integer_identity_lut",
    "build_integer_interval_lut",
    "build_sum_pool",
    "get_avgpool_divisor",
    "get_pool_window_size",
    "is_avgpool",
    "is_value_avgpool",
]


PoolWindowModule: TypeAlias = nn.AvgPool1d | nn.AvgPool2d | SumPool1d | SumPool2d
ValueCodeRange: TypeAlias = tuple[int, int]


_INT32_MAX = torch.iinfo(torch.int32).max
_INT32_MIN = torch.iinfo(torch.int32).min


def build_integer_interval_lut(
    thres: list[int], values: list[int], output_signed: bool | None = None
) -> LutCustom:
    """Build a logical integer interval LUT for AvgPool deployment rewrites."""
    if len(thres) != len(values):
        raise ValueError(
            "integer interval LUT thres & values must have the same length"
        )
    if len(thres) > LUT_TABLE_SIZE:
        raise ValueError(
            f"integer interval LUT supports at most {LUT_TABLE_SIZE} intervals, got "
            f"{len(thres)}"
        )
    if min(thres) < _INT32_MIN or max(thres) > _INT32_MAX:
        raise ValueError("integer interval LUT thres must fit int32")
    if any(b < a for a, b in zip(thres, thres[1:])):
        raise ValueError("integer interval LUT thres must be monotonic")

    resolved_output_signed = min(values) < 0 if output_signed is None else output_signed
    thresholds = torch.tensor(thres, dtype=torch.int32)
    lut_values = torch.tensor(values, dtype=torch.int32)
    if len(thres) < LUT_TABLE_SIZE:
        pad_count = LUT_TABLE_SIZE - len(thres)
        thresholds = torch.cat(
            (
                thresholds,
                torch.full((pad_count,), thres[-1], dtype=torch.int32),
            )
        )
        lut_values = torch.cat(
            (
                lut_values,
                torch.full((pad_count,), values[-1], dtype=torch.int32),
            )
        )

    return LutCustom(thresholds, lut_values, resolved_output_signed, is_float=False)


def build_integer_identity_lut(code_min: int, code_max: int) -> LutCustom:
    """Build a logical integer identity LUT for AvgPool deployment rewrites."""
    if code_min > code_max:
        raise ValueError(
            f"identity LUT requires code_min <= code_max, got "
            f"{code_min} > {code_max}"
        )
    if code_max - code_min + 1 > LUT_TABLE_SIZE:
        raise ValueError(f"identity LUT code range too large: ({code_min}, {code_max})")

    codes = list(range(code_min, code_max + 1))
    return build_integer_interval_lut(codes, codes, output_signed=code_min < 0)


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
