"""Shared FX-node helper utilities for lowering."""

from typing import Any

import torch
from torch import fx

from ..ir.op_node import TensorLayout
from .dims_prop import DimsProp, DimsType

__all__ = [
    "get_call_arg",
    "get_fx_call_target_name",
    "get_input_layouts",
    "get_input_shapes",
    "get_input_dims",
    "get_output_layouts",
    "get_output_shape",
    "get_output_dims",
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


def get_output_dims(node: fx.Node) -> DimsType:
    """Get output dims from an FX node's propagated layout metadata."""
    return node.meta.get(DimsProp.KEY, ())


def get_output_layouts(node: fx.Node) -> tuple[TensorLayout, ...]:
    """Extract output layouts from an FX node.

    Most supported nodes are single-output and return one layout entry.
    Tuple-valued nodes such as ``torch.split`` are handled by dedicated lowering
    paths and therefore typically do not call this helper.
    """
    return (TensorLayout(shape=get_output_shape(node), dims=get_output_dims(node)),)


def get_input_layouts(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> tuple[TensorLayout, ...]:
    """Extract input layouts from predecessor tensor metadata."""
    source_nodes = (
        input_nodes if input_nodes is not None else tuple(node.all_input_nodes)
    )
    return tuple(
        TensorLayout(shape=get_output_shape(inp), dims=get_output_dims(inp))
        for inp in source_nodes
    )


def get_input_shapes(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[torch.Size]:
    return [layout.shape for layout in get_input_layouts(node, input_nodes)]


def get_input_dims(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[DimsType]:
    return [layout.dims for layout in get_input_layouts(node, input_nodes)]
