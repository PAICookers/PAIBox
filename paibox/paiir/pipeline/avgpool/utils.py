"""Internal helpers for AvgPool-related deployment logic."""

import math

import torch.nn as nn

from ...nn import SumPool1d, SumPool2d

__all__ = ["_get_pool_window_size", "_is_avgpool"]


def _is_avgpool(comp: nn.Module) -> bool:
    """Check if a compute module is an average-pooling op."""
    return isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d))


def _get_pool_window_size(comp: nn.Module) -> int:
    """Return the number of elements in the pooling window."""
    assert isinstance(comp, (nn.AvgPool1d, nn.AvgPool2d, SumPool1d, SumPool2d))
    ks = comp.kernel_size
    if isinstance(ks, int):
        if isinstance(comp, (nn.AvgPool2d, SumPool2d)):
            ks = (ks, ks)
        else:
            ks = (ks,)
    return math.prod(ks)
