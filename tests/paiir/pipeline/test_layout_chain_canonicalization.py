import torch

from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import ReshapeOp, TensorLayout
from paibox.paiir.ir.reshape_semantics import is_layout_invisible_dims
from paibox.paiir.pipeline.layout_chain_canonicalization import (
    canonicalize_layout_chains,
)


def _fixed_shape(shape: tuple[int, ...]):
    target = torch.Size(shape)
    return lambda _input_shape, bound=target: bound


def _reshape(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
) -> ReshapeOp:
    node = ReshapeOp(shape_fn=_fixed_shape(output_shape))
    node.input_layouts = (TensorLayout(torch.Size(input_shape), input_dims),)
    node.output_layouts = (TensorLayout(torch.Size(output_shape), output_dims),)
    return node


class TestLayoutChainCanonicalization:
    def test_singleton_axis_dims_permutation_is_layout_invisible(self):
        assert is_layout_invisible_dims(torch.Size((1, 3, 4)), (1, 0, 2))
        assert is_layout_invisible_dims(torch.Size((1, 1, 64, 40, 40)), (1, 0, 2, 3, 4))
        assert not is_layout_invisible_dims(torch.Size((2, 3, 4)), (1, 0, 2))

    def test_collapses_three_identity_reshape_nodes(self):
        graph = PAIIRGraph("reshape_chain")
        inp = InputNode(shape=torch.Size((1, 3, 4)))
        r1 = _reshape((1, 3, 4), (1, 1, 3, 4), (0, 1, 2), (0, 1, 2, 3))
        r2 = _reshape((1, 1, 3, 4), (1, 12), (0, 1, 2, 3), (0, 1))
        r3 = _reshape((1, 12), (1, 3, 4), (0, 1), (0, 1, 2))
        out = OutputNode(shape=torch.Size((1, 3, 4)))

        for node in (inp, r1, r2, r3, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, r3.name)
        graph.add_edge(r3.name, out.name)

        canonicalize_layout_chains(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert reshape_nodes == []
        assert graph.predecessors(out.name) == [inp.name]

    def test_collapses_reshape_pair_with_leading_layout_metadata(self):
        graph = PAIIRGraph("reshape_pair")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _reshape((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _reshape((1, 3, 2), (1, 6), (0, 1, 2), (0, 1))
        out = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out.name)

        canonicalize_layout_chains(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1
        reshape = reshape_nodes[0]
        assert reshape.input_layouts == (
            TensorLayout(torch.Size((1, 2, 3)), (0, 2, 1)),
        )
        assert reshape.output_layouts == (TensorLayout(torch.Size((1, 6)), (0, 1)),)

    def test_removes_identity_shape_when_dims_only_swap_singleton_axes(self):
        graph = PAIIRGraph("reshape_singleton_dims")
        inp = InputNode(shape=torch.Size((1, 3, 4)))
        reshape = _reshape((1, 3, 4), (1, 3, 4), (1, 0, 2), (1, 0, 2))
        out = OutputNode(shape=torch.Size((1, 3, 4)))

        for node in (inp, reshape, out):
            graph.add_node(node)
        graph.add_edge(inp.name, reshape.name)
        graph.add_edge(reshape.name, out.name)

        canonicalize_layout_chains(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert reshape_nodes == []
        assert graph.predecessors(out.name) == [inp.name]

    def test_keeps_pair_when_second_requires_nonidentity_input_dims(self):
        graph = PAIIRGraph("reshape_pair_nonidentity")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _reshape((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _reshape((1, 3, 2), (1, 6), (0, 2, 1), (0, 1))
        out = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out.name)

        canonicalize_layout_chains(graph)

        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1
        reshape = reshape_nodes[0]
        assert reshape.input_layouts == (
            TensorLayout(torch.Size((1, 2, 3)), (0, 2, 1)),
        )
        assert reshape.output_layouts == (TensorLayout(torch.Size((1, 6)), (0, 1)),)
