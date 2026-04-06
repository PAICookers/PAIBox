"""Shared FX-node helper utilities for lowering."""

from typing import Any

import torch
from torch import fx

from .dims_prop import DimsProp, DimsType

__all__ = [
    "get_call_arg",
    "get_fx_call_target_name",
    "get_input_dims",
    "get_input_shapes",
    "get_output_dims",
    "get_output_shape",
]


def get_call_arg(node: fx.Node, index: int, name: str, default: Any = None) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def get_fx_call_target_name(node: fx.Node) -> str:
    """Return a stable display name for an FX call target."""
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    return str(node.target)


def get_output_shape(node: fx.Node) -> torch.Size:
    """Extract output shape from an FX node's tensor metadata."""
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return torch.Size()
    if hasattr(meta, "shape"):
        return torch.Size(meta.shape)
    return torch.Size()


def get_input_shapes(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[torch.Size]:
    """Extract input shapes from predecessor tensor metadata."""
    source_nodes = (
        input_nodes if input_nodes is not None else tuple(node.all_input_nodes)
    )
    return [get_output_shape(inp) for inp in source_nodes]


def get_output_dims(node: fx.Node) -> DimsType:
    """Get output dims from an FX node's propagated layout metadata."""
    return node.meta.get(DimsProp.KEY, ())


def get_input_dims(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[DimsType]:
    """Get input dims from predecessor propagated layout metadata."""
    source_nodes = (
        input_nodes if input_nodes is not None else tuple(node.all_input_nodes)
    )
    return [get_output_dims(inp) for inp in source_nodes]
