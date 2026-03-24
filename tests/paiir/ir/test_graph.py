
import pytest
from torch import nn

from paibox.paiir.ir.add_ops import PotentialAddOp
from paibox.paiir.ir.core_neuron import IFNodeV25
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.op_node import (
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)


class TestPAIIRGraph:
    def _build_simple_graph(self):
        """Build Input -> Conv+IF -> Output."""
        graph = PAIIRGraph("test")
        inp = InputNode(shape=(1, 3, 8, 8))
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
        inp = InputNode(shape=(1, 4))
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

    def test_input_output_nodes(self):
        graph, inp, seq, out = self._build_simple_graph()
        assert len(graph.input_nodes()) == 1
        assert len(graph.output_nodes()) == 1

    def test_diamond_graph(self):
        """Diamond: Input -> [A, B] -> Add -> Output."""
        graph = PAIIRGraph("diamond")
        inp = InputNode(shape=(1, 4))
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
        inp = InputNode(shape=(1, 4))
        comp = StandaloneCompOp(nn.Linear(4, 8))
        act = StandaloneActOp(IFNodeV25())
        out = OutputNode()

        for node in (inp, comp, act, out):
            graph.add_node(node)

        graph.add_edge(inp.name, comp.name)
        graph.add_edge(comp.name, act.name)
        graph.add_edge(act.name, out.name)

        graph.summary()
        captured = capsys.readouterr().out

        assert f"{inp.name} (InputNode)" in captured
        assert f"{comp.name} (Linear)" in captured
        assert f"{act.name} (IFNodeV25)" in captured
        assert f"{out.name} (OutputNode)" in captured


