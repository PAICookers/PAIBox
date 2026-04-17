"""Canonicalize consecutive transform chains before fusion."""

from ..ir.graph import PAIIRGraph
from ..ir.op_node import TransformOp
from .graph_utils import is_order_preserving_transform_node, is_transform_node

__all__ = ["canonicalize_layout_chains"]


def canonicalize_layout_chains(graph: PAIIRGraph) -> PAIIRGraph:
    """Collapse local transform chains until no further rewrite applies."""
    changed = True
    while changed:
        changed = False
        for name in graph.topo_sort():
            node = graph.nodes.get(name)
            if not is_transform_node(node):
                continue

            if _try_remove_identity_transform(graph, name):
                changed = True
                break

            if _try_collapse_adjacent_transforms(graph, name):
                changed = True
                break

    return graph


def _try_remove_identity_transform(graph: PAIIRGraph, name: str) -> bool:
    """Delete a transform when it preserves both shape and element ordering."""
    node = graph.nodes.get(name)
    if not is_transform_node(node):
        return False

    preds = graph.predecessors(name)
    if len(preds) != 1:
        return False

    input_layout = node.input_layouts[0]
    output_layout = node.output_layouts[0]
    if not input_layout.shape or not output_layout.shape:
        return False

    if input_layout.shape != output_layout.shape:
        return False

    if not is_order_preserving_transform_node(node):
        return False

    graph.remove_node_and_reconnect(name)
    return True


def _try_collapse_adjacent_transforms(graph: PAIIRGraph, first_name: str) -> bool:
    first = graph.nodes.get(first_name)
    if not is_transform_node(first):
        return False

    succs = graph.successors(first_name)
    if len(succs) != 1:
        return False

    second_name = succs[0]
    second = graph.nodes.get(second_name)
    if not is_transform_node(second):
        return False

    if len(graph.predecessors(second_name)) != 1:
        return False

    if first.num_inputs != 1 or first.num_outputs != 1:
        return False

    if second.num_inputs != 1 or second.num_outputs != 1:
        return False

    if not first.input_layouts[0].shape or not second.output_layouts[0].shape:
        return False

    composed = TransformOp(first.stages + second.stages)
    composed.input_layouts = first.input_layouts
    composed.output_layouts = second.output_layouts

    graph.add_node(composed)
    graph.add_edge(graph.predecessors(first_name)[0], composed.name, dst_port=0)
    graph.replace_all_uses_with(second_name, composed.name, delete_old=True)

    graph.remove_node(first_name)
    return True
