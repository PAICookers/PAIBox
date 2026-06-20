"""Semantic online-training graph expansion for minimal Phase 1 support."""

import copy

from torch import nn

from ..ir.calc_params import (
    OnlineCoreSemanticMode,
    OnlineCoreType,
    OnlineGradientRole,
)
from ..ir.graph import PAIIRGraph
from ..ir.ir_base import OutputNode, TensorLayout
from ..ir.op_node import OnlineCoreOp, OpNode

__all__ = ["expand_online_training_graph"]


def expand_online_training_graph(graph: PAIIRGraph) -> PAIIRGraph:
    """Expand online-forward IR into a minimal semantic training graph.

    Phase 1 keeps the supported surface intentionally small:

    - online nodes must currently be forward-semantic ``OnlineCoreOp``
    - the user graph must have a single model output
    - training expansion only models the semantic stage order, not backend
      work-mode refinement or online backend lowering
    """
    online_names = [
        name
        for name in graph.topo_sort()
        if isinstance(graph.nodes[name], OnlineCoreOp)
    ]
    if not online_names:
        return graph

    if len(graph.output_nodes()) != 1:
        raise NotImplementedError(
            "Phase 1 online training expansion currently supports single-output graphs only."
        )

    for name in online_names:
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        if node.core_params.semantic_mode is not OnlineCoreSemanticMode.FORWARD:
            raise NotImplementedError(
                "Phase 1 online training expansion expects forward-semantic "
                "OnlineCoreOp nodes only."
            )
    _validate_online_forward_path(graph, online_names)

    expanded = graph.clone_shallow()
    model_output = graph.output_nodes()[0]
    preds = graph.predecessors(model_output.name)
    if len(preds) != 1:
        raise NotImplementedError(
            "Phase 1 online training expansion expects each model output to have "
            "exactly one predecessor."
        )

    tail_name = preds[0]
    last_forward = graph.nodes[online_names[-1]]
    assert isinstance(last_forward, OnlineCoreOp)

    loss_node = _build_online_stage_node(
        last_forward,
        semantic_mode=OnlineCoreSemanticMode.LOSS,
        comp=None,
        input_layouts=(graph.get_node_output_layout(tail_name),),
        output_layouts=(graph.get_node_output_layout(tail_name),),
    )
    expanded.add_node(loss_node)
    expanded.add_edge(tail_name, loss_node.name)
    tail_name = loss_node.name

    for idx, forward_name in enumerate(reversed(online_names)):
        forward_node = graph.nodes[forward_name]
        assert isinstance(forward_node, OnlineCoreOp)
        role = (
            OnlineGradientRole.OUTPUT if idx == 0 else OnlineGradientRole.HIDDEN
        )
        tail_layout = _single_output_layout(expanded.nodes[tail_name])
        grad_output_layouts = (
            forward_node.input_layouts if forward_node.input_layouts else (tail_layout,)
        )
        grad_node = _build_online_stage_node(
            forward_node,
            semantic_mode=OnlineCoreSemanticMode.GRADIENT,
            gradient_role=role,
            input_layouts=(tail_layout,),
            output_layouts=grad_output_layouts,
        )
        expanded.add_node(grad_node)
        expanded.add_edge(tail_name, grad_node.name)
        tail_name = grad_node.name

    for forward_name in reversed(online_names):
        forward_node = graph.nodes[forward_name]
        assert isinstance(forward_node, OnlineCoreOp)
        tail_layout = _single_output_layout(expanded.nodes[tail_name])
        update_node = _build_online_stage_node(
            forward_node,
            semantic_mode=OnlineCoreSemanticMode.UPDATE,
            input_layouts=(tail_layout,),
            output_layouts=(tail_layout,),
        )
        expanded.add_node(update_node)
        expanded.add_edge(tail_name, update_node.name)
        tail_name = update_node.name

    final_layout = _single_output_layout(expanded.nodes[tail_name])
    training_output = OutputNode(shape=final_layout.shape, dims=final_layout.dims)
    expanded.add_node(training_output)
    expanded.add_edge(tail_name, training_output.name)

    return expanded


def _validate_online_forward_path(
    graph: PAIIRGraph, online_names: list[str]
) -> None:
    if len(online_names) <= 1:
        return

    for upstream, downstream in zip(online_names, online_names[1:]):
        if graph.predecessors(downstream) != [upstream]:
            raise NotImplementedError(
                "Phase 1 online training expansion currently supports a single "
                "serial online path only."
            )


def _build_online_stage_node(
    source: OnlineCoreOp,
    *,
    semantic_mode: OnlineCoreSemanticMode,
    comp: nn.Module | None | object = ...,
    gradient_role: OnlineGradientRole | None = None,
    input_layouts: tuple[TensorLayout, ...],
    output_layouts: tuple[TensorLayout, ...],
) -> OnlineCoreOp:
    params = copy.deepcopy(source.core_params)
    params.semantic_mode = semantic_mode
    params.gradient_role = gradient_role
    params.update_direction = (
        source.core_params.update_direction
        if semantic_mode is OnlineCoreSemanticMode.UPDATE
        else None
    )
    params.work_mode = None
    if semantic_mode is not OnlineCoreSemanticMode.FORWARD:
        params.input_core = OnlineCoreType.ONLINE
        params.output_core = OnlineCoreType.ONLINE

    if comp is ...:
        comp = source.comp

    node = OnlineCoreOp(comp=comp, core_params=params)
    node.input_layouts = input_layouts
    node.output_layouts = output_layouts
    return node


def _single_output_layout(node: object) -> TensorLayout:
    if isinstance(node, OpNode) and node.output_layouts:
        return node.output_layouts[0]
    return TensorLayout()
