import pytest
import torch
from torch import nn

from paibox.paiir.exceptions import GraphValidationError
from paibox.paiir.ir.add_ops import PotentialAddOp
from paibox.paiir.ir.core_neuron import IFNodeV25
from paibox.paiir.ir.graph import Edge, PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import (
    RoutingOp,
    SequentialOp,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TensorLayout,
)


class TestPAIIRGraph:
    def _build_simple_graph(self):
        """Build Input -> Conv+IF -> Output."""
        graph = PAIIRGraph("test")
        inp = InputNode(shape=torch.Size((1, 3, 8, 8)))
        seq = SequentialOp(nn.Conv2d(3, 8, 3, padding=1), IFNodeV25())
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(seq)
        graph.add_node(out)
        graph.add_edge(inp.name, seq.name)
        graph.add_edge(seq.name, out.name)
        return graph, inp, seq, out

    def test_duplicate_node_raises(self):
        graph = PAIIRGraph()
        inp = InputNode()
        graph.add_node(inp)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(inp)

    def test_edge_invalid_node(self):
        graph = PAIIRGraph()
        inp = InputNode()
        graph.add_node(inp)
        with pytest.raises(KeyError):
            graph.add_edge(inp.name, "nonexistent")

    def test_topo_sort(self):
        graph, inp, seq, out = self._build_simple_graph()
        order = graph.topo_sort()
        assert order.index(inp.name) < order.index(seq.name)
        assert order.index(seq.name) < order.index(out.name)

    def test_predecessors_successors(self):
        graph, inp, seq, out = self._build_simple_graph()
        assert graph.predecessors(seq.name) == [inp.name]
        assert graph.successors(seq.name) == [out.name]

    def test_add_edge_invalidates_cached_predecessors(self):
        graph = PAIIRGraph("port_order_cache")
        inp = InputNode(shape=torch.Size((1, 4)))
        a = StandaloneCompOp(nn.Linear(4, 8))
        b = StandaloneCompOp(nn.Linear(4, 8))
        add = PotentialAddOp(op_signs=(1, 1))

        for node in (inp, a, b, add):
            graph.add_node(node)

        graph.add_edge(inp.name, a.name)
        graph.add_edge(inp.name, b.name)
        graph.add_edge(a.name, add.name, dst_port=1)

        assert graph.predecessors(add.name) == [a.name]

        graph.add_edge(b.name, add.name, dst_port=0)

        assert graph.predecessors(add.name) == [b.name, a.name]

    def test_remove_node_removes_incident_edges(self):
        graph, inp, seq, out = self._build_simple_graph()

        graph.remove_node(seq.name)

        assert seq.name not in graph.nodes
        assert graph.successors(inp.name) == []
        assert graph.predecessors(out.name) == []

    def test_replace_all_uses_with(self):
        graph = PAIIRGraph("replace_uses")
        inp = InputNode(shape=torch.Size((1, 4)))
        a = StandaloneCompOp(nn.Linear(4, 8))
        b = StandaloneCompOp(nn.Linear(4, 8))
        out1 = OutputNode()
        out2 = OutputNode()

        for node in (inp, a, b, out1, out2):
            graph.add_node(node)

        graph.add_edge(inp.name, a.name)
        graph.add_edge(inp.name, b.name)
        graph.add_edge(a.name, out1.name, src_port=1)
        graph.add_edge(a.name, out2.name, src_port=1)
        graph.add_edge(b.name, out2.name, src_port=1)

        graph.replace_all_uses_with(a.name, b.name)

        assert a.name in graph.nodes
        assert graph.outgoing_edges(a.name) == []
        assert graph.predecessors(out1.name) == [b.name]
        assert graph.predecessors(out2.name) == [b.name]
        assert len(graph.outgoing_edges(b.name)) == 2
        assert any(
            edge.dst == out1.name and edge.src_port == 1
            for edge in graph.outgoing_edges(b.name)
        )

    def test_remove_node_and_reconnect(self):
        graph = PAIIRGraph("remove_reconnect")
        inp = InputNode(shape=torch.Size((1, 4)))
        reshape = StandaloneActOp(IFNodeV25())
        out1 = OutputNode()
        out2 = OutputNode()

        for node in (inp, reshape, out1, out2):
            graph.add_node(node)

        graph.add_edge(inp.name, reshape.name, src_port=1)
        graph.add_edge(reshape.name, out1.name)
        graph.add_edge(reshape.name, out2.name)

        graph.remove_node_and_reconnect(reshape.name)

        assert reshape.name not in graph.nodes
        assert graph.predecessors(out1.name) == [inp.name]
        assert graph.predecessors(out2.name) == [inp.name]
        assert all(edge.src_port == 1 for edge in graph.outgoing_edges(inp.name))

    def test_remove_node_and_reconnect_rejects_non_predecessor_source(self):
        graph = PAIIRGraph("remove_reconnect_source_validation")
        inp = InputNode(shape=torch.Size((1, 4)))
        other = StandaloneCompOp(nn.Linear(4, 4))
        reshape = StandaloneActOp(IFNodeV25())
        out = OutputNode()

        for node in (inp, other, reshape, out):
            graph.add_node(node)

        graph.add_edge(inp.name, reshape.name)
        graph.add_edge(reshape.name, out.name)

        with pytest.raises(ValueError, match="is not a predecessor"):
            graph.remove_node_and_reconnect(reshape.name, source_name=other.name)

    def test_input_output_nodes(self):
        graph, inp, seq, out = self._build_simple_graph()
        assert len(graph.input_nodes()) == 1
        assert len(graph.output_nodes()) == 1

    def test_lint_rejects_edge_with_missing_endpoint(self):
        graph, _, _, out = self._build_simple_graph()
        graph.edges.append(Edge(src="missing_node", dst=out.name))

        with pytest.raises(GraphValidationError, match="missing source node"):
            graph.lint()

    def test_lint_rejects_disconnected_node(self):
        graph, _, _, _ = self._build_simple_graph()
        dead_end = StandaloneCompOp(nn.Linear(4, 8))
        graph.add_node(dead_end)

        with pytest.raises(
            GraphValidationError, match="nodes not on any input-to-output path"
        ):
            graph.lint()

    def test_verify_before_sim_reports_structural_lint_errors(self):
        graph, _, seq, _ = self._build_simple_graph()
        seq.output_layouts = (TensorLayout(torch.Size((1, 8, 8, 8)), (0, 1, 2, 3)),)
        seq.core_params.tick_start = 1
        seq.core_params.tick_duration = 0
        seq.core_params.tick_initial = 0

        disconnected_input = InputNode(shape=torch.Size((1, 3, 8, 8)))
        graph.add_node(disconnected_input)

        with pytest.raises(RuntimeError, match="nodes not on any input-to-output path"):
            graph.verify_before_sim()

    def test_diamond_graph(self):
        """Diamond: Input -> [A, B] -> Add -> Output."""
        graph = PAIIRGraph("diamond")
        inp = InputNode(shape=torch.Size((1, 4)))
        a = StandaloneCompOp(nn.Linear(4, 8))
        b = StandaloneCompOp(nn.Linear(4, 8))
        add = PotentialAddOp(op_signs=(1, 1))
        out = OutputNode()

        for n in [inp, a, b, add, out]:
            graph.add_node(n)

        graph.add_edge(inp.name, a.name)
        graph.add_edge(inp.name, b.name)
        graph.add_edge(a.name, add.name, dst_port=0)
        graph.add_edge(b.name, add.name, dst_port=1)
        graph.add_edge(add.name, out.name)

        order = graph.topo_sort()
        assert order.index(inp.name) < order.index(a.name)
        assert order.index(inp.name) < order.index(b.name)
        assert order.index(a.name) < order.index(add.name)
        assert order.index(b.name) < order.index(add.name)
        assert order.index(add.name) < order.index(out.name)

    def test_summary_prefers_comp_and_act_type_names(self, capsys):
        graph = PAIIRGraph("summary_labels")
        inp = InputNode(shape=torch.Size((1, 4)))
        comp = StandaloneCompOp(nn.Linear(4, 8))
        comp.output_layouts = (TensorLayout(torch.Size((1, 8)), (0, 1)),)
        act = StandaloneActOp(IFNodeV25())
        act.output_layouts = (TensorLayout(torch.Size((1, 8)), (0, 1)),)
        out = OutputNode(shape=torch.Size((1, 8)))

        for node in (inp, comp, act, out):
            graph.add_node(node)

        graph.add_edge(inp.name, comp.name)
        graph.add_edge(comp.name, act.name)
        graph.add_edge(act.name, out.name)

        graph.summary()
        captured = capsys.readouterr().out

        assert f"{inp.name} (InputNode) (4,)" in captured
        assert f"{comp.name} (Linear) (8,)" in captured
        assert f"{act.name} (IFNodeV25) (8,)" in captured
        assert f"{out.name} (OutputNode) (8,)" in captured

    def test_boundary_layout_is_explicit_and_not_synthesized(self):
        graph = PAIIRGraph("boundary_layout")
        inp = InputNode(shape=torch.Size((1, 2, 3)), dims=(0, 2, 1))
        out = OutputNode(shape=torch.Size((1, 2, 3)), dims=(0, 2, 1))
        graph.add_node(inp)
        graph.add_node(out)
        graph.add_edge(inp.name, out.name)

        assert graph.get_node_output_layout(inp.name) == inp.layout
        assert graph.get_node_output_layout(out.name) == out.layout
        assert graph.get_edge_output_layout(graph.edges[0]) == inp.layout

    def test_summary_verbose_shows_multi_output_layouts_and_ports(self, capsys):
        graph = PAIIRGraph("multi_output_summary")
        inp = InputNode(shape=torch.Size((1, 4)), dims=(0, 1))
        split = SplitOp(sections=2, dim=1)
        split.input_layouts = (TensorLayout(torch.Size((1, 4)), (0, 1)),)
        split.output_layouts = (
            TensorLayout(torch.Size((1, 2)), (0, 1)),
            TensorLayout(torch.Size((1, 2)), (0, 1)),
        )
        out_a = OutputNode(shape=torch.Size((1, 2)))
        out_b = OutputNode(shape=torch.Size((1, 2)))

        for node in (inp, split, out_a, out_b):
            graph.add_node(node)
        graph.add_edge(inp.name, split.name)
        graph.add_edge(split.name, out_a.name, src_port=0)
        graph.add_edge(split.name, out_b.name, src_port=1)

        graph.summary(verbose=True)
        summary = capsys.readouterr().out
        assert "out[0] shape=(1, 2), dims=(0, 1)" in summary
        assert "out[1] shape=(1, 2), dims=(0, 1)" in summary
        assert "src_port=0 ->" in summary
        assert "src_port=1 ->" in summary

    def test_runtime_supports_generic_multi_output_routing_node(self):
        class DuplicateOp(RoutingOp):
            def __init__(self):
                super().__init__()
                self.input_layouts = (TensorLayout(torch.Size((1, 3)), (0, 1)),)
                self.output_layouts = (
                    TensorLayout(torch.Size((1, 3)), (0, 1)),
                    TensorLayout(torch.Size((1, 3)), (0, 1)),
                )

            def forward(self, x: torch.Tensor):
                return (x, x + 1)

        graph = PAIIRGraph("generic_multi_output")
        inp = InputNode(shape=torch.Size((1, 3)))
        dup = DuplicateOp()
        out_a = OutputNode(shape=torch.Size((1, 3)))
        out_b = OutputNode(shape=torch.Size((1, 3)))

        for node in (inp, dup, out_a, out_b):
            graph.add_node(node)
        graph.add_edge(inp.name, dup.name)
        graph.add_edge(dup.name, out_a.name, src_port=0)
        graph.add_edge(dup.name, out_b.name, src_port=1)

        x = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = graph.step(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        torch.testing.assert_close(outputs[0], x)
        torch.testing.assert_close(outputs[1], x + 1)
