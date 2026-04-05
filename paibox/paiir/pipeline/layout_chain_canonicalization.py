"""Canonicalize consecutive layout-only routing chains before fusion."""

from collections.abc import Callable

import torch

from ..ir.graph import PAIIRGraph
from ..ir.op_node import ReshapeOp
from ..ir.reshape_semantics import is_layout_invisible_dims, reshape_output_shape

__all__ = ["canonicalize_layout_chains"]


def canonicalize_layout_chains(graph: PAIIRGraph) -> PAIIRGraph:
    """Collapse local reshape-only chains until no further rewrite applies."""
    changed = True
    while changed:
        changed = False
        for name in graph.topo_sort():
            node = graph.nodes.get(name)
            if not isinstance(node, ReshapeOp):
                continue

            if _try_remove_identity_reshape(graph, name):
                changed = True
                break

            if _try_collapse_adjacent_reshapes(graph, name):
                changed = True
                break

    return graph


def _try_remove_identity_reshape(graph: PAIIRGraph, name: str) -> bool:
    """Delete a reshape when both shape and effective layout are unchanged."""
    node = graph.nodes.get(name)
    if not isinstance(node, ReshapeOp):
        return False

    preds = graph.predecessors(name)
    if len(preds) != 1:
        return False

    if len(node.input_shapes) != 1 or not node.input_shapes[0] or not node.output_shape:
        return False

    if node.input_shapes[0] != node.output_shape:
        return False

    input_dims = node.input_dims[0] if len(node.input_dims) == 1 else ()
    if not is_layout_invisible_dims(node.input_shapes[0], input_dims):
        return False
    if not is_layout_invisible_dims(node.output_shape, node.output_dims):
        return False

    graph.remove_node_and_reconnect(name)
    return True


def _try_collapse_adjacent_reshapes(graph: PAIIRGraph, first_name: str) -> bool:
    first = graph.nodes.get(first_name)
    if not isinstance(first, ReshapeOp):
        return False

    succs = graph.successors(first_name)
    if len(succs) != 1:
        return False

    second_name = succs[0]
    second = graph.nodes.get(second_name)
    if not isinstance(second, ReshapeOp):
        return False

    if len(graph.predecessors(second_name)) != 1:
        return False

    if (
        len(first.input_shapes) != 1
        or not first.input_shapes[0]
        or not second.output_shape
    ):
        return False

    if len(second.input_shapes) != 1 or not second.input_shapes[0]:
        return False

    if len(first.input_dims) != 1 or len(second.input_dims) != 1:
        return False

    composed = ReshapeOp(shape_fn=_compose_shape_fns([first, second]))
    composed.input_shapes = list(first.input_shapes)
    composed.output_shape = second.output_shape
    composed.input_dims = list(first.input_dims)
    composed.output_dims = second.output_dims

    graph.add_node(composed)
    graph.add_edge(graph.predecessors(first_name)[0], composed.name, dst_port=0)
    graph.replace_all_uses_with(second_name, composed.name, delete_old=True)

    graph.remove_node(first_name)
    return True


def _compose_shape_fns(chain: list[ReshapeOp]) -> Callable[[torch.Size], torch.Size]:
    """Compose reshape functions using each op's logical input view."""

    def _composed(input_shape: torch.Size) -> torch.Size:
        current = input_shape
        for op in chain:
            dims = op.input_dims[0] if len(op.input_dims) == 1 else ()
            current = reshape_output_shape(current, dims, op.shape_fn)
        return current

    return _composed
