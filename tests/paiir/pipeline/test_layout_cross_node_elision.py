import torch
from torch import nn

from paibox.paiir.ir.core_neuron import ANNNodeV25
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.lut_activation import LutReLU
from paibox.paiir.ir.op_node import (
    ReshapeOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.pipeline.layout_cross_node_elision import (
    elide_layout_invisible_reshapes,
)
from paibox.paiir.pipeline.passes import fuse_to_offline_cores


def _reshape(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
) -> ReshapeOp:
    target = torch.Size(output_shape)
    node = ReshapeOp(shape_fn=lambda _shape, bound=target: bound)
    node.input_shapes = [torch.Size(input_shape)]
    node.output_shape = target
    node.input_dims = [input_dims]
    node.output_dims = output_dims
    return node


def _build_conv_reshape_act_reshape_graph(
    *, nonidentity_dims: tuple[int, ...] | None = None, shared_predecessor: bool = False
) -> PAIIRGraph:
    graph = PAIIRGraph("conv_reshape_act_reshape")
    inp = InputNode(shape=torch.Size((1, 3, 8, 8)))
    comp = StandaloneCompOp(nn.Conv2d(3, 4, 1))
    comp.input_shapes = [inp.shape]
    comp.output_shape = torch.Size((1, 4, 8, 8))
    comp.input_dims = [(0, 1, 2, 3)]
    comp.output_dims = (0, 1, 2, 3)

    pre = _reshape(
        (1, 4, 8, 8),
        (1, 1, 4, 8, 8),
        nonidentity_dims or (0, 1, 2, 3),
        (0, 1, 2, 3, 4),
    )
    act = StandaloneActOp(ANNNodeV25(LutReLU()))
    act.input_shapes = [pre.output_shape]
    act.output_shape = pre.output_shape
    act.input_dims = [pre.output_dims]
    act.output_dims = pre.output_dims
    post = _reshape((1, 1, 4, 8, 8), (1, 4, 8, 8), (0, 1, 2, 3, 4), (0, 1, 2, 3))
    out = OutputNode(shape=torch.Size((1, 4, 8, 8)))

    for node in (inp, comp, pre, act, post, out):
        graph.add_node(node)
    graph.add_edge(inp.name, comp.name)
    graph.add_edge(comp.name, pre.name)
    graph.add_edge(pre.name, act.name)
    graph.add_edge(act.name, post.name)
    graph.add_edge(post.name, out.name)

    if shared_predecessor:
        extra = OutputNode(shape=torch.Size((1, 1, 4, 8, 8)))
        graph.add_node(extra)
        graph.add_edge(pre.name, extra.name)

    return graph


class TestLayoutCrossNodeElision:
    def test_elides_conv_reshape_act_reshape_sandwich(self):
        graph = _build_conv_reshape_act_reshape_graph()

        elide_layout_invisible_reshapes(graph)
        fused = fuse_to_offline_cores(graph)

        reshape_nodes = [n for n in fused.nodes.values() if isinstance(n, ReshapeOp)]
        seq_nodes = [n for n in fused.nodes.values() if isinstance(n, SequentialOp)]

        assert reshape_nodes == []
        assert len(seq_nodes) == 1
        assert isinstance(seq_nodes[0].comp, nn.Conv2d)

    def test_keeps_nonidentity_dims_sandwich(self):
        graph = _build_conv_reshape_act_reshape_graph(nonidentity_dims=(0, 2, 3, 1))

        elide_layout_invisible_reshapes(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 2

    def test_keeps_shared_predecessor_reshape(self):
        graph = _build_conv_reshape_act_reshape_graph(shared_predecessor=True)

        elide_layout_invisible_reshapes(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 2
