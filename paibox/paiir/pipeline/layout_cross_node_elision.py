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

    if not comp.output_shape or pre.input_shapes != [comp.output_shape]:
        return False
    if not pre.output_shape or act.input_shapes != [pre.output_shape]:
        return False
    if not act.output_shape or post.input_shapes != [act.output_shape]:
        return False
    if post.output_shape != comp.output_shape:
        return False

    act.input_shapes = [comp.output_shape]
    act.output_shape = comp.output_shape
    act.input_dims = [comp.output_dims]
    act.output_dims = comp.output_dims

    graph.replace_all_uses_with(post_name, act_name, delete_old=True)
    graph.remove_node(pre_name)
    graph.add_edge(comp_name, act_name, dst_port=0)

    return True


def _is_layout_invisible(node: ReshapeOp) -> bool:
    if len(node.input_shapes) != 1 or not node.input_shapes[0] or not node.output_shape:
        return False

    input_dims = node.input_dims[0] if len(node.input_dims) == 1 else ()
    return is_layout_invisible_reshape(
        node.input_shapes[0], node.output_shape, input_dims, node.output_dims
    )
