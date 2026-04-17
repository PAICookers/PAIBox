"""Shared graph-query helpers for compile-time pipeline passes."""

from collections.abc import Callable
from typing import TypeGuard, TypeVar

import torch
from torch import nn

from ..ir.graph import PAIIRGraph
from ..ir.ir_base import PAIIRNode
from ..ir.op_node import ConcatOp, SplitOp, StandaloneCompOp, TransformOp

__all__ = [
    "collect_effective_predecessor_values",
    "collect_effective_node_values",
    "is_order_preserving_transform_node",
    "is_transform_node",
    "is_format_transparent_routing_node",
    "match_pre_activation_transform",
    "is_standalone_maxpool",
]

_T = TypeVar("_T")


def is_transform_node(node: PAIIRNode | None) -> TypeGuard[TransformOp]:
    """Return whether *node* is a transform-like routing op."""
    return isinstance(node, TransformOp)


def is_order_preserving_transform_node(
    node: PAIIRNode,
) -> TypeGuard[TransformOp]:
    """Return whether *node* preserves flattened element ordering."""
    if not is_transform_node(node):
        return False
    if node.num_inputs != 1 or node.num_outputs != 1:
        return False

    input_layout = node.input_layouts[0]
    output_layout = node.output_layouts[0]
    if not input_layout.shape or not output_layout.shape:
        return False
    if input_layout.shape.numel() != output_layout.shape.numel():
        return False

    flat_indices = torch.arange(input_layout.shape.numel(), dtype=torch.int32).reshape(
        input_layout.shape
    )
    reordered = node(flat_indices).reshape(-1)
    return torch.equal(reordered, torch.arange(reordered.numel(), dtype=torch.int32))


def is_standalone_maxpool(node: PAIIRNode) -> TypeGuard[StandaloneCompOp]:
    """Return whether *node* is a standalone MaxPool compute op."""
    return isinstance(node, StandaloneCompOp) and isinstance(
        node.comp, (nn.MaxPool1d, nn.MaxPool2d)
    )


def is_format_transparent_routing_node(
    node: PAIIRNode,
) -> TypeGuard[ConcatOp | TransformOp | SplitOp]:
    """Return whether *node* preserves scalar encoding across routing."""
    return isinstance(node, (ConcatOp, TransformOp, SplitOp))


def match_pre_activation_transform(
    graph: PAIIRGraph, consumer_name: str
) -> tuple[str, str, TransformOp] | None:
    """Match one transform node directly feeding an activation consumer."""
    preds = graph.predecessors(consumer_name)
    if len(preds) != 1:
        return None

    transform_name = preds[0]
    transform = graph.nodes[transform_name]
    if not is_transform_node(transform):
        return None
    if len(graph.successors(transform_name)) != 1:
        return None

    transform_preds = graph.predecessors(transform_name)
    if len(transform_preds) != 1:
        return None

    return transform_preds[0], transform_name, transform


def collect_effective_predecessor_values(
    graph: PAIIRGraph,
    node_name: str,
    resolve: Callable[[PAIIRNode, str], list[_T] | None],
    passthrough: Callable[[PAIIRNode], bool],
) -> list[_T]:
    """Collect effective predecessor payloads for *node_name*.

    Each predecessor is recursively traced through nodes that satisfy
    ``passthrough`` until ``resolve`` returns a concrete payload list or the
    traversal reaches a dead end.
    """

    values: list[_T] = []
    for pred_name in graph.predecessors(node_name):
        values.extend(
            collect_effective_node_values(graph, pred_name, resolve, passthrough)
        )
    return values


def collect_effective_node_values(
    graph: PAIIRGraph,
    node_name: str,
    resolve: Callable[[PAIIRNode, str], list[_T] | None],
    passthrough: Callable[[PAIIRNode], bool],
) -> list[_T]:
    """Collect effective payloads for one node through transparent routing.

    ``resolve`` should return:
    - a payload list when the current node is a terminal source of interest
    - ``None`` when traversal should continue or stop
    """

    node = graph.nodes[node_name]
    resolved = resolve(node, node_name)
    if resolved is not None:
        return resolved

    if passthrough(node):
        values: list[_T] = []
        for pred_name in graph.predecessors(node_name):
            values.extend(
                collect_effective_node_values(graph, pred_name, resolve, passthrough)
            )
        return values

    return []
