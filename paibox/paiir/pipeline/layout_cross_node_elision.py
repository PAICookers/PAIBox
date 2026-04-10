"""Elide layout-invisible reshape wrappers around deployable operators."""

from ..ir.graph import PAIIRGraph
from ..ir.op_node import ReshapeOp, StandaloneActOp, StandaloneCompOp
from ..ir.reshape_semantics import is_layout_invisible_reshape

__all__ = ["elide_layout_invisible_reshapes"]


def elide_layout_invisible_reshapes(graph: PAIIRGraph) -> PAIIRGraph:
    """Remove chip-invisible reshape wrappers around standalone activations."""
    changed = True
    while changed:
        changed = False
        for name in graph.topo_sort():
            node = graph.nodes.get(name)
            if not isinstance(node, StandaloneActOp):
                continue

            if _try_elide_comp_act_reshape_sandwich(graph, name):
                changed = True
                break

    return graph


def _try_elide_comp_act_reshape_sandwich(graph: PAIIRGraph, act_name: str) -> bool:
    """Elide a balanced ``Comp -> Reshape -> Act -> Reshape`` sandwich.

    The trailing reshape matters: we only remove the pair when the post-reshape
    restores the compute node's visible output contract, so downstream users
    continue to observe the same shape after the rewrite.
    """
    act = graph.nodes.get(act_name)
    if not isinstance(act, StandaloneActOp):
        return False

    act_preds = graph.predecessors(act_name)
    act_succs = graph.successors(act_name)
    if len(act_preds) != 1 or len(act_succs) != 1:
        return False

    pre_name = act_preds[0]
    post_name = act_succs[0]
    pre = graph.nodes.get(pre_name)
    post = graph.nodes.get(post_name)
    if not isinstance(pre, ReshapeOp) or not isinstance(post, ReshapeOp):
        return False

    pre_preds = graph.predecessors(pre_name)
    if len(pre_preds) != 1:
        return False

    if graph.successors(pre_name) != [act_name]:
        return False

    comp_name = pre_preds[0]
    comp = graph.nodes.get(comp_name)
    if not isinstance(comp, StandaloneCompOp):
        return False

    if graph.successors(comp_name) != [pre_name]:
        return False
    if graph.predecessors(post_name) != [act_name]:
        return False

    if not _is_layout_invisible(pre) or not _is_layout_invisible(post):
        return False

    if comp.num_outputs != 1 or not comp.output_layouts[0].shape:
        return False
    if pre.input_layouts != (comp.output_layouts[0],):
        return False
    if pre.num_outputs != 1 or pre.output_layouts != act.input_layouts:
        return False
    if act.num_outputs != 1 or act.output_layouts != post.input_layouts:
        return False
    if post.num_outputs != 1 or post.output_layouts[0] != comp.output_layouts[0]:
        return False

    act.input_layouts = (comp.output_layouts[0],)
    act.output_layouts = (comp.output_layouts[0],)

    graph.replace_all_uses_with(post_name, act_name, delete_old=True)
    graph.remove_node(pre_name)
    graph.add_edge(comp_name, act_name, dst_port=0)

    return True


def _is_layout_invisible(node: ReshapeOp) -> bool:
    if node.num_inputs != 1 or node.num_outputs != 1:
        return False

    input_layout = node.input_layouts[0]
    output_layout = node.output_layouts[0]
    if not input_layout.shape or not output_layout.shape:
        return False

    return is_layout_invisible_reshape(
        input_layout.shape, output_layout.shape, input_layout.dims, output_layout.dims
    )
