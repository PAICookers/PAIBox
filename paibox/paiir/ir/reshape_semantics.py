"""Shared reshape-like semantics used across lowering, IR, and backend code.

This module intentionally stays small and function-oriented. It centralizes
target classification and pure shape/dims helpers for reshape-like operators
without introducing another object layer.
"""

from collections.abc import Callable, Iterable
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
    "is_layout_invisible_reshape",
    "is_layout_invisible_dims",
    "is_dims_reset_target",
    "is_identity_repeat_values",
    "materialize_logical_layout",
    "reshape_output_shape",
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
        or target in FLATTEN_FUNCTION_TARGETS
        or target in RESHAPE_FUNCTION_TARGETS
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


def is_layout_invisible_dims(shape: torch.Size, dims: tuple[int, ...]) -> bool:
    """Return whether ``dims`` only changes logical grouping, not chip-visible order.

    A permutation is considered layout-invisible when every non-singleton axis
    stays in the same relative order and only singleton axes move around it.
    """
    if not dims:
        return True
    if not shape or len(shape) != len(dims):
        return False
    if sorted(dims) != list(range(len(dims))):
        return False
    if is_identity_dims(dims):
        return True

    non_singleton_axes = [axis for axis, size in enumerate(shape) if size != 1]
    permuted_non_singleton_axes = [axis for axis in dims if shape[axis] != 1]
    return permuted_non_singleton_axes == non_singleton_axes


def canonicalize_layout_view(
    shape: torch.Size, dims: tuple[int, ...]
) -> tuple[torch.Size, tuple[int, ...]]:
    """Normalize a layout-invisible view into ``(logical_shape, identity_dims)``.

    This lets later simulation/layout code treat singleton-axis-only
    permutations as a reshape-equivalent view instead of a real permute.
    """
    if not dims:
        return shape, dims

    logical_shape = shape_after_dims(shape, dims)
    if logical_shape is None or not is_layout_invisible_dims(shape, dims):
        return shape, dims

    return logical_shape, tuple(range(len(logical_shape)))


def shape_after_dims(shape: torch.Size, dims: tuple[int, ...]) -> torch.Size | None:
    if not shape or not dims or len(shape) != len(dims):
        return None
    if sorted(dims) != list(range(len(dims))):
        return None

    return torch.Size(shape[axis] for axis in dims)


def reshape_output_shape(
    input_shape: torch.Size,
    input_dims: tuple[int, ...],
    shape_fn: Callable[[torch.Size], torch.Size] | None,
) -> torch.Size:
    logical_shape = shape_after_dims(input_shape, input_dims)
    if logical_shape is None:
        logical_shape = input_shape

    if shape_fn is None:
        return torch.Size((logical_shape.numel(),))

    return torch.Size(shape_fn(logical_shape))


def is_layout_invisible_reshape(
    input_shape: torch.Size,
    output_shape: torch.Size,
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
) -> bool:
    if not input_shape or not output_shape:
        return False
    if input_shape.numel() != output_shape.numel():
        return False

    return is_layout_invisible_dims(
        input_shape, input_dims
    ) and is_layout_invisible_dims(output_shape, output_dims)


def materialize_logical_layout(x: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Apply a pending logical axis permutation when metadata requires it."""
    if is_identity_dims(dims) or x.dim() != len(dims):
        return x

    logical_shape, canonical_dims = canonicalize_layout_view(x.shape, dims)
    if is_identity_dims(canonical_dims):
        # Singleton-axis-only dims changes are chip-invisible, so a view-like
        # reshape is sufficient and keeps the simulation closer to deployment.
        if logical_shape == x.shape:
            return x
        return x.reshape(logical_shape)

    return x.permute(*canonical_dims)
