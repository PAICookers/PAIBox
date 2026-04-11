"""Shared graph-query helpers for compile-time pipeline passes."""

from collections.abc import Callable
from typing import TypeGuard, TypeVar

from torch import nn

from ..ir.graph import PAIIRGraph
from ..ir.ir_base import PAIIRNode
from ..ir.op_node import ConcatOp, ReshapeOp, SplitOp, StandaloneCompOp

__all__ = [
    "collect_effective_predecessor_values",
    "collect_effective_node_values",
    "is_format_transparent_routing_node",
    "is_standalone_maxpool",
]

_T = TypeVar("_T")


def is_standalone_maxpool(node: PAIIRNode) -> TypeGuard[StandaloneCompOp]:
    """Return whether *node* is a standalone MaxPool compute op."""

    return isinstance(node, StandaloneCompOp) and isinstance(
        node.comp, (nn.MaxPool1d, nn.MaxPool2d)
    )


def is_format_transparent_routing_node(
    node: PAIIRNode,
) -> TypeGuard[ConcatOp | ReshapeOp | SplitOp]:
    """Return whether *node* preserves scalar encoding across routing."""

    return isinstance(node, (ConcatOp, ReshapeOp, SplitOp))


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
