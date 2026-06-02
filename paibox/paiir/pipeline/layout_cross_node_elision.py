"""Commute transparent pre-activation transforms to expose fusion opportunities."""

from ..ir.graph import PAIIRGraph
from ..ir.op_node import StandaloneActOp, StandaloneCompOp, TransformOp
from .graph_utils import (
    is_order_preserving_transform_node,
    match_pre_activation_transform,
)

__all__ = ["commute_pre_activation_transforms"]


def commute_pre_activation_transforms(graph: PAIIRGraph) -> PAIIRGraph:
    """Move transparent transform prefixes from ``comp -> transform -> act`` to after ``act``."""
    changed = True
    changed_any = False
    while changed:
        changed = False
        for name in graph.topo_sort():
            node = graph.nodes.get(name)
            if not isinstance(node, StandaloneActOp):
                continue

            if _try_commute_pre_activation_transform(graph, name):
                changed = True
                changed_any = True
                break

    return graph.clone_shallow() if changed_any else graph


def _try_commute_pre_activation_transform(graph: PAIIRGraph, act_name: str) -> bool:
    act = graph.nodes.get(act_name)
    if not isinstance(act, StandaloneActOp):
        return False

    match = match_pre_activation_transform(graph, act_name)
    if match is None:
        return False

    comp_name, pre_name, pre = match
    comp = graph.nodes.get(comp_name)
    if not isinstance(comp, StandaloneCompOp):
        return False

    if not is_order_preserving_transform_node(pre):
        return False

    if comp.num_outputs != 1 or act.num_inputs != 1 or act.num_outputs != 1:
        return False

    comp_output_layout = comp.output_layouts[0]
    if not comp_output_layout.shape:
        return False
    if pre.input_layouts != (comp_output_layout,):
        return False
    if pre.output_layouts != act.input_layouts:
        return False

    post = TransformOp(pre.stages)
    post.input_layouts = (comp_output_layout,)
    post.output_layouts = pre.output_layouts
    graph.add_node(post)

    graph.replace_all_uses_with(act_name, post.name, delete_old=False)
    graph.remove_node(pre_name)
    act.input_layouts = (comp_output_layout,)
    act.output_layouts = (comp_output_layout,)
    graph.add_edge(comp_name, act_name, dst_port=0)
    graph.add_edge(act_name, post.name, dst_port=0)
    return True
