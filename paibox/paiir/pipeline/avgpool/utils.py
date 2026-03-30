"""Internal helpers for AvgPool-related deployment logic."""

import math
from typing import TypeAlias, TypeGuard

import torch.nn as nn

from ...nn import SumPool1d, SumPool2d

__all__ = ["_get_avgpool_divisor", "_get_pool_window_size", "_is_avgpool"]


PoolWindowModule: TypeAlias = nn.AvgPool1d | nn.AvgPool2d | SumPool1d | SumPool2d


def _is_avgpool(comp: nn.Module) -> TypeGuard[nn.AvgPool1d | nn.AvgPool2d]:
    """Check if a compute module is an average-pooling op."""
    return isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d))


def _get_pool_window_size(comp: PoolWindowModule) -> int:
    """Return the number of elements in the pooling window."""
    ks = comp.kernel_size
    if isinstance(ks, int):
        if isinstance(comp, (nn.AvgPool2d, SumPool2d)):
            ks = (ks, ks)
        else:
            ks = (ks,)
    return math.prod(ks)


def _get_avgpool_divisor(comp: nn.AvgPool1d | nn.AvgPool2d) -> int:
    """Return the effective AvgPool divisor used by the frontend module."""
    divisor_override = comp.divisor_override if isinstance(comp, nn.AvgPool2d) else None
    if divisor_override is not None:
        return int(divisor_override)

    return _get_pool_window_size(comp)
