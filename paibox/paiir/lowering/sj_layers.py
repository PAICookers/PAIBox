"""Frontend policy for ``spikingjelly.activation_based.layer`` modules."""

from spikingjelly.activation_based import layer
from torch import nn

SJ_LAYER_ERASE_MODULE_TYPES = (layer.Dropout, layer.Dropout2d)

# SJ layer types that lower correctly through Python MRO; no canonicalization needed.
#
# Conv1d/2d, Linear, MaxPool1d/2d, AvgPool1d/2d, and AdaptiveAvgPool1d/2d lower
# through explicit module_map registration.
# Flatten lowers via reshape-sink handling.
_SUPPORTED_SJ_LAYER_COMP_TYPES = (
    layer.Conv1d,
    layer.Conv2d,
    layer.Linear,
    layer.MaxPool1d,
    layer.MaxPool2d,
    layer.AvgPool1d,
    layer.AvgPool2d,
    layer.AdaptiveAvgPool1d,
    layer.AdaptiveAvgPool2d,
)

_SUPPORTED_SJ_LAYER_TYPES = (
    *_SUPPORTED_SJ_LAYER_COMP_TYPES,
    layer.Flatten,
)


def is_sj_layer_module(m: nn.Module) -> bool:
    return type(m).__module__ == layer.__name__


def is_supported_sj_layer_module(m: nn.Module) -> bool:
    """Return True if this SJ layer module lowers correctly without canonicalization."""
    return isinstance(m, _SUPPORTED_SJ_LAYER_TYPES)


def describe_sj_layer_module(m: nn.Module) -> str:
    return f"spikingjelly.activation_based.layer.{type(m).__name__}"
