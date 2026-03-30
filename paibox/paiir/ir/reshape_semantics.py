"""Shared reshape-like semantics used across lowering, IR, and backend code.

This module intentionally stays small and function-oriented. It centralizes
target classification and pure shape/dims helpers for reshape-like operators
without introducing another object layer.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "FLATTEN_FUNCTION_TARGETS",
    "RESHAPE_FUNCTION_TARGETS",
    "RESHAPE_LEAF_MODULE_TYPES",
    "RESHAPE_METHOD_NAMES",
    "SQUEEZE_FUNCTION_TARGETS",
    "UNSQUEEZE_FUNCTION_TARGETS",
    "flatten_nested_values",
    "is_dims_reset_target",
    "is_identity_dims",
    "is_identity_repeat_values",
    "is_squeeze_target",
    "is_unsqueeze_target",
    "materialize_logical_layout",
    "shape_after_dims",
]

FLATTEN_FUNCTION_TARGETS = (torch.flatten,)
SQUEEZE_FUNCTION_TARGETS = (torch.squeeze,)
UNSQUEEZE_FUNCTION_TARGETS = (torch.unsqueeze,)
RESHAPE_FUNCTION_TARGETS = (torch.reshape,)
RESHAPE_METHOD_NAMES = frozenset({"flatten", "reshape", "view", "view_as"})
RESHAPE_LEAF_MODULE_TYPES = (nn.Flatten,)


def is_unsqueeze_target(target: Any) -> bool:
    return target == "unsqueeze" or target in UNSQUEEZE_FUNCTION_TARGETS


def is_squeeze_target(target: Any) -> bool:
    return target == "squeeze" or target in SQUEEZE_FUNCTION_TARGETS


def is_dims_reset_target(target: Any) -> bool:
    return (
        target in RESHAPE_METHOD_NAMES
        or is_unsqueeze_target(target)
        or is_squeeze_target(target)
    )


def flatten_nested_values(values: Iterable[Any]) -> tuple[Any, ...]:
    flattened: list[Any] = []

    def _flatten(value: Any) -> None:
        if isinstance(value, (tuple, list, torch.Size)):
            for item in value:
                _flatten(item)
            return
        flattened.append(value)

    for value in values:
        _flatten(value)

    return tuple(flattened)


def is_identity_repeat_values(values: Iterable[Any]) -> bool:
    flattened = flatten_nested_values(values)
    return bool(flattened) and all(
        isinstance(value, int) and value == 1 for value in flattened
    )


def is_identity_dims(dims: tuple[int, ...]) -> bool:
    return not dims or dims == tuple(range(len(dims)))


def shape_after_dims(
    shape: tuple[int, ...] | torch.Size, dims: tuple[int, ...]
) -> tuple[int, ...] | None:
    if not shape or not dims or len(shape) != len(dims):
        return None
    if sorted(dims) != list(range(len(dims))):
        return None
    return tuple(shape[axis] for axis in dims)


def materialize_logical_layout(x: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Apply a pending logical axis permutation when metadata requires it."""
    if is_identity_dims(dims) or x.dim() != len(dims):
        return x
    return x.permute(*dims)
