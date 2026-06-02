import torch

from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode, TensorLayout
from paibox.paiir.ir.op_node import LayoutStage, ShapeStage
from paibox.paiir.ir.reshape_semantics import is_layout_invisible_dims
from paibox.paiir.pipeline.layout_chain_canonicalization import (
    canonicalize_layout_chains,
)
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


def _stage_types(node) -> tuple[type[object], ...]:
    return tuple(type(stage) for stage in node.stages)


class TestLayoutChainCanonicalization:
    def test_singleton_axis_dims_permutation_is_layout_invisible(self):
        assert is_layout_invisible_dims(torch.Size((1, 3, 4)), (1, 0, 2))
        assert is_layout_invisible_dims(torch.Size((1, 1, 64, 40, 40)), (1, 0, 2, 3, 4))
        assert not is_layout_invisible_dims(torch.Size((2, 3, 4)), (1, 0, 2))

    def test_collapses_three_identity_transform_nodes(self):
        graph = PAIIRGraph("reshape_chain")
        inp = InputNode(shape=torch.Size((1, 3, 4)))
        r1 = _transform((1, 3, 4), (1, 1, 3, 4), (0, 1, 2), (0, 1, 2, 3))
        r2 = _transform((1, 1, 3, 4), (1, 12), (0, 1, 2, 3), (0, 1))
        r3 = _transform((1, 12), (1, 3, 4), (0, 1), (0, 1, 2))
        out = OutputNode(shape=torch.Size((1, 3, 4)))

        for node in (inp, r1, r2, r3, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, r3.name)
        graph.add_edge(r3.name, out.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert transform_nodes == []
        assert graph.predecessors(out.name) == [inp.name]

    def test_collapses_transform_pair_with_leading_layout_metadata(self):
        graph = PAIIRGraph("reshape_pair")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _transform((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _transform((1, 3, 2), (1, 6), (0, 1, 2), (0, 1))
        out = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 1
        transform = transform_nodes[0]
        assert _stage_types(transform) == (
            LayoutStage,
            ShapeStage,
            LayoutStage,
            ShapeStage,
        )
        assert transform.input_layouts == (
            TensorLayout(torch.Size((1, 2, 3)), (0, 2, 1)),
        )
        assert transform.output_layouts == (TensorLayout(torch.Size((1, 6)), (0, 1)),)

    def test_collapses_entire_nonidentity_transform_chain_in_one_pass(self):
        graph = PAIIRGraph("transform_chain_nonidentity")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _transform((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _transform((1, 3, 2), (1, 1, 3, 2), (0, 1, 2), (0, 1, 2, 3))
        r3 = _transform((1, 1, 3, 2), (1, 6), (0, 1, 2, 3), (0, 1))
        out = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, r3, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, r3.name)
        graph.add_edge(r3.name, out.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 1
        transform = transform_nodes[0]
        assert _stage_types(transform) == (
            LayoutStage,
            ShapeStage,
            LayoutStage,
            ShapeStage,
            LayoutStage,
            ShapeStage,
        )
        assert transform.input_layouts == (
            TensorLayout(torch.Size((1, 2, 3)), (0, 2, 1)),
        )
        assert transform.output_layouts == (TensorLayout(torch.Size((1, 6)), (0, 1)),)

    def test_removes_identity_shape_when_dims_only_swap_singleton_axes(self):
        graph = PAIIRGraph("reshape_singleton_dims")
        inp = InputNode(shape=torch.Size((1, 3, 4)))
        transform = _transform((1, 3, 4), (1, 3, 4), (1, 0, 2), (1, 0, 2))
        out = OutputNode(shape=torch.Size((1, 3, 4)))

        for node in (inp, transform, out):
            graph.add_node(node)
        graph.add_edge(inp.name, transform.name)
        graph.add_edge(transform.name, out.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert transform_nodes == []
        assert graph.predecessors(out.name) == [inp.name]

    def test_keeps_pair_when_second_requires_nonidentity_input_dims(self):
        graph = PAIIRGraph("reshape_pair_nonidentity")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _transform((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _transform((1, 3, 2), (1, 6), (0, 2, 1), (0, 1))
        out = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, out):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 1
        transform = transform_nodes[0]
        assert _stage_types(transform) == (
            LayoutStage,
            ShapeStage,
            LayoutStage,
            ShapeStage,
        )
        assert transform.input_layouts == (
            TensorLayout(torch.Size((1, 2, 3)), (0, 2, 1)),
        )
        assert transform.output_layouts == (TensorLayout(torch.Size((1, 6)), (0, 1)),)

    def test_keeps_identity_endpoints_chain_when_middle_layout_semantics_change(self):
        graph = PAIIRGraph("transform_identity_false_positive")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        t1 = _transform((1, 2, 3), (1, 3, 2), (0, 1, 2), (0, 1, 2))
        t2 = _transform((1, 3, 2), (1, 2, 3), (0, 2, 1), (0, 1, 2))
        out = OutputNode(shape=torch.Size((1, 2, 3)))

        for node in (inp, t1, t2, out):
            graph.add_node(node)
        graph.add_edge(inp.name, t1.name)
        graph.add_edge(t1.name, t2.name)
        graph.add_edge(t2.name, out.name)

        x = torch.arange(6, dtype=torch.int32).reshape(1, 2, 3)
        expected_before = graph.forward(x)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert transform_nodes
        graph.reset()
        actual_after = graph.forward(x)

        assert torch.is_tensor(actual_after)
        assert torch.is_tensor(expected_before)
        assert torch.equal(actual_after, expected_before)

    def test_collapses_suffix_slice_when_upstream_transform_is_shared(self):
        graph = PAIIRGraph("transform_shared_prefix")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        shared = _transform((1, 2, 3), (1, 3, 2), (0, 1, 2), (0, 1, 2))
        r1 = _transform((1, 3, 2), (1, 1, 3, 2), (0, 1, 2), (0, 1, 2, 3))
        r2 = _transform((1, 1, 3, 2), (1, 6), (0, 1, 2, 3), (0, 1))
        out_main = OutputNode(shape=torch.Size((1, 6)))
        out_side = OutputNode(shape=torch.Size((1, 3, 2)))

        for node in (inp, shared, r1, r2, out_main, out_side):
            graph.add_node(node)
        graph.add_edge(inp.name, shared.name)
        graph.add_edge(shared.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out_main.name)
        graph.add_edge(shared.name, out_side.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 2
        assert graph.predecessors(out_side.name) == [shared.name]
        main_pred = graph.predecessors(out_main.name)[0]
        assert main_pred != shared.name

    def test_collapses_prefix_slice_when_tail_transform_has_multiple_successors(self):
        graph = PAIIRGraph("transform_shared_tail")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _transform((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _transform((1, 3, 2), (1, 6), (0, 1, 2), (0, 1))
        out_a = OutputNode(shape=torch.Size((1, 6)))
        out_b = OutputNode(shape=torch.Size((1, 6)))

        for node in (inp, r1, r2, out_a, out_b):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, out_a.name)
        graph.add_edge(r2.name, out_b.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 1
        transform = transform_nodes[0]
        assert graph.predecessors(out_a.name) == [transform.name]
        assert graph.predecessors(out_b.name) == [transform.name]

    def test_does_not_cross_middle_observer_when_collapsing_chain(self):
        graph = PAIIRGraph("transform_middle_observer")
        inp = InputNode(shape=torch.Size((1, 2, 3)))
        r1 = _transform((1, 2, 3), (1, 3, 2), (0, 2, 1), (0, 1, 2))
        r2 = _transform((1, 3, 2), (1, 1, 3, 2), (0, 1, 2), (0, 1, 2, 3))
        r3 = _transform((1, 1, 3, 2), (1, 6), (0, 1, 2, 3), (0, 1))
        out_main = OutputNode(shape=torch.Size((1, 6)))
        out_obs = OutputNode(shape=torch.Size((1, 1, 3, 2)))

        for node in (inp, r1, r2, r3, out_main, out_obs):
            graph.add_node(node)
        graph.add_edge(inp.name, r1.name)
        graph.add_edge(r1.name, r2.name)
        graph.add_edge(r2.name, r3.name)
        graph.add_edge(r3.name, out_main.name)
        graph.add_edge(r2.name, out_obs.name)

        canonicalize_layout_chains(graph)

        transform_nodes = find_transform_nodes(graph)
        assert len(transform_nodes) == 2
        obs_pred = graph.predecessors(out_obs.name)[0]
        main_pred = graph.predecessors(out_main.name)[0]
        assert obs_pred != main_pred
