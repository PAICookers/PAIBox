import torch
from torch import nn

from paibox.paiir.ir.core_neuron import ANNNodeV25
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode, TensorLayout
from paibox.paiir.ir.lut_activation import LutReLU
from paibox.paiir.ir.op_node import (
    LayoutStage,
    SequentialOp,
    ShapeStage,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.pipeline.layout_cross_node_elision import (
    commute_pre_activation_transforms,
)
from paibox.paiir.pipeline.passes import fuse_to_offline_cores
from tests.paiir.conftest import find_transform_nodes, make_transform


def _transform(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
):
    node = make_transform(input_shape, output_shape, input_dims)
    node.input_layouts = (TensorLayout(torch.Size(input_shape), input_dims),)
    node.output_layouts = (TensorLayout(torch.Size(output_shape), output_dims),)
    return node


def _build_conv_reshape_act_graph(
    *, nonidentity_dims: tuple[int, ...] | None = None, shared_predecessor: bool = False
) -> PAIIRGraph:
    graph = PAIIRGraph("conv_reshape_act")
    inp = InputNode(shape=torch.Size((1, 3, 8, 8)))
    comp = StandaloneCompOp(nn.Conv2d(3, 4, 1))
    comp.input_layouts = (TensorLayout(inp.shape, (0, 1, 2, 3)),)
    comp.output_layouts = (TensorLayout(torch.Size((1, 4, 8, 8)), (0, 1, 2, 3)),)

    pre = _transform(
        (1, 4, 8, 8),
        (1, 1, 4, 8, 8),
        nonidentity_dims or (0, 1, 2, 3),
        (0, 1, 2, 3, 4),
    )
    act = StandaloneActOp(ANNNodeV25(LutReLU()))
    act.input_layouts = pre.output_layouts
    act.output_layouts = pre.output_layouts
    out = OutputNode(shape=torch.Size((1, 1, 4, 8, 8)))

    for node in (inp, comp, pre, act, out):
        graph.add_node(node)
    graph.add_edge(inp.name, comp.name)
    graph.add_edge(comp.name, pre.name)
    graph.add_edge(pre.name, act.name)
    graph.add_edge(act.name, out.name)

    if shared_predecessor:
        extra = OutputNode(shape=torch.Size((1, 1, 4, 8, 8)))
        graph.add_node(extra)
        graph.add_edge(pre.name, extra.name)

    return graph


def _stage_types(node) -> tuple[type[object], ...]:
    return tuple(type(stage) for stage in node.stages)


class TestLayoutCrossNodeElision:
    def test_commutes_pre_activation_transform_to_after_activation(self):
        graph = _build_conv_reshape_act_graph()

        commute_pre_activation_transforms(graph)
        fused = fuse_to_offline_cores(graph)

        transform_nodes = find_transform_nodes(fused)
        seq_nodes = [n for n in fused.nodes.values() if isinstance(n, SequentialOp)]

        assert len(seq_nodes) == 1
        assert len(transform_nodes) == 1
        assert isinstance(seq_nodes[0].comp, nn.Conv2d)
        assert _stage_types(transform_nodes[0]) == (LayoutStage, ShapeStage)
        assert fused.predecessors(transform_nodes[0].name) == [seq_nodes[0].name]
        assert transform_nodes[0].input_layouts == seq_nodes[0].output_layouts
        assert transform_nodes[0].output_layouts == (
            TensorLayout(torch.Size((1, 1, 4, 8, 8)), (0, 1, 2, 3, 4)),
        )

    def test_keeps_nontransparent_pre_activation_transform(self):
        graph = _build_conv_reshape_act_graph(nonidentity_dims=(0, 2, 3, 1))

        commute_pre_activation_transforms(graph)

        comp_nodes = [
            n for n in graph.nodes.values() if isinstance(n, StandaloneCompOp)
        ]
        transform_nodes = find_transform_nodes(graph)
        act_nodes = [n for n in graph.nodes.values() if isinstance(n, StandaloneActOp)]
        assert len(comp_nodes) == 1
        assert len(transform_nodes) == 1
        assert len(act_nodes) == 1
        assert _stage_types(transform_nodes[0]) == (LayoutStage, ShapeStage)
        assert graph.predecessors(transform_nodes[0].name) == [comp_nodes[0].name]
        assert graph.predecessors(act_nodes[0].name) == [transform_nodes[0].name]

    def test_keeps_shared_pre_activation_transform(self):
        graph = _build_conv_reshape_act_graph(shared_predecessor=True)

        commute_pre_activation_transforms(graph)

        comp_nodes = [
            n for n in graph.nodes.values() if isinstance(n, StandaloneCompOp)
        ]
        transform_nodes = find_transform_nodes(graph)
        act_nodes = [n for n in graph.nodes.values() if isinstance(n, StandaloneActOp)]
        assert len(comp_nodes) == 1
        assert len(transform_nodes) == 1
        assert len(act_nodes) == 1
        assert _stage_types(transform_nodes[0]) == (LayoutStage, ShapeStage)
        assert graph.predecessors(transform_nodes[0].name) == [comp_nodes[0].name]
        assert graph.predecessors(act_nodes[0].name) == [transform_nodes[0].name]
