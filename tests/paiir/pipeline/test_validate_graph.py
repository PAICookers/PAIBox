"""Tests for the validate_graph pass."""

import pytest
import torch
from paicorelib import DataSign, DataWidth, SNNMode
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir.ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.lut_activation import LutReLU
from paibox.paiir.ir.op_node import SequentialOp
from paibox.paiir.lowering.converter import torch_to_paiir
from paibox.paiir.pipeline.passes import (
    GraphCleanupWarning,
    GraphValidationError,
    fuse_to_offline_cores,
    validate_compiled_graph,
    validate_graph,
)


class TestValidGraphs:
    """Well-formed graphs should pass validation without errors."""

    def test_simple_snn(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.conv(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(unfused)
        validate_graph(fused)

    def test_two_layer_ann(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.linear = nn.Linear(16 * 8 * 8, 10)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.conv(x))
                x = x.flatten(1)
                return self.sigmoid(self.linear(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(unfused)
        validate_graph(fused)

    def test_residual_snn(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = nn.Conv2d(3, 16, 3, padding=1)
                self.conv_b = nn.Conv2d(3, 16, 3, padding=1)
                self.lif = sj.LIFNode(tau=2.0)

            def forward(self, x):
                return self.lif(self.conv_a(x) + self.conv_b(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(unfused)
        validate_graph(fused)


class TestUnrecoverableErrors:
    def test_no_input_node(self):
        graph = PAIIRGraph("no_input")
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(op.name, out.name)

        with pytest.raises(GraphValidationError, match="no input node"):
            validate_graph(graph)

    def test_no_output_node(self):
        graph = PAIIRGraph("no_output")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_edge(inp.name, op.name)

        with pytest.raises(GraphValidationError, match="no output node"):
            validate_graph(graph)

    def test_input_has_predecessors(self):
        """InputNode with predecessors is a structural error."""
        graph = PAIIRGraph("bad_input")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        # Manually add an invalid edge pointing into the InputNode
        graph.edges.append(type(graph.edges[0])(src=op.name, dst=inp.name))

        with pytest.raises(GraphValidationError, match="has predecessors"):
            validate_graph(graph)

    def test_output_has_successors(self):
        """OutputNode with successors is a structural error."""
        graph = PAIIRGraph("bad_output")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        # Manually add an invalid edge going out of the OutputNode
        graph.edges.append(type(graph.edges[0])(src=out.name, dst=op.name))

        with pytest.raises(GraphValidationError, match="has successors"):
            validate_graph(graph)

    def test_collects_all_errors(self):
        """Multiple unrecoverable errors reported at once."""
        graph = PAIIRGraph("multiple_errors")
        # No InputNode, no OutputNode
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        graph.add_node(op)

        with pytest.raises(GraphValidationError) as exc_info:
            validate_graph(graph)

        err = exc_info.value
        assert len(err.errors) >= 2  # no input + no output


class TestAutoCleanup:
    def test_orphan_op_removed_with_warning(self):
        """Orphan OpNode is removed and a warning is emitted."""
        graph = PAIIRGraph("orphan_op")
        inp = InputNode(shape=(1, 8))
        op1 = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op1.input_shapes = [(1, 8)]
        op1.output_shape = (1, 4)
        op2 = SequentialOp(nn.Linear(8, 4), IFNodeV25())  # orphan
        op2.input_shapes = [(1, 8)]
        op2.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_node(out)
        graph.add_edge(inp.name, op1.name)
        graph.add_edge(op1.name, out.name)

        with pytest.warns(GraphCleanupWarning, match="disconnected"):
            validate_graph(graph)

        assert op2.name not in graph.nodes
        assert (
            len([n for n in graph.nodes.values() if isinstance(n, SequentialOp)]) == 1
        )

    def test_dead_end_op_removed(self):
        """OpNode with input but no output is removed."""
        graph = PAIIRGraph("dead_end")
        inp = InputNode(shape=(1, 8))
        op1 = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op1.input_shapes = [(1, 8)]
        op1.output_shape = (1, 4)
        op2 = SequentialOp(nn.Linear(4, 2), IFNodeV25())  # dead end
        op2.input_shapes = [(1, 4)]
        op2.output_shape = (1, 2)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_node(out)
        graph.add_edge(inp.name, op1.name)
        graph.add_edge(op1.name, op2.name)
        graph.add_edge(op1.name, out.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert op2.name not in graph.nodes
        # Edge from op1 -> op2 should also be removed
        assert all(e.dst != op2.name for e in graph.edges)

    def test_disconnected_input_removed(self):
        """Disconnected InputNode is removed with warning."""
        graph = PAIIRGraph("disconnected_input")
        inp1 = InputNode(shape=(1, 8))
        inp2 = InputNode(shape=(1, 8))  # disconnected
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp1)
        graph.add_node(inp2)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp1.name, op.name)
        graph.add_edge(op.name, out.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert inp2.name not in graph.nodes
        assert len(graph.input_nodes()) == 1

    def test_disconnected_output_removed(self):
        """Disconnected OutputNode is removed with warning."""
        graph = PAIIRGraph("disconnected_output")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out1 = OutputNode()
        out2 = OutputNode()  # disconnected
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out1)
        graph.add_node(out2)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out1.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert out2.name not in graph.nodes
        assert len(graph.output_nodes()) == 1

    def test_all_inputs_disconnected_raises_after_cleanup(self):
        """If all InputNodes are disconnected, cleanup leaves no inputs -> error."""
        graph = PAIIRGraph("all_disconnected")
        inp = InputNode(shape=(1, 8))  # disconnected
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(op.name, out.name)
        # inp has no edges, op has no predecessors -> both will be removed

        # First the orphan nodes are removed, then post-cleanup check fails
        with pytest.raises(GraphValidationError, match="no input node after cleanup"):
            with pytest.warns(GraphCleanupWarning):
                validate_graph(graph)


class TestSNNModeLUTConsistency:
    """Validate that snn_mode matches LUT presence on OfflineCoreOp nodes."""

    def _build_graph(self, act: CoreNeuronV25) -> PAIIRGraph:
        """Build a minimal valid graph with a single SequentialOp."""
        graph = PAIIRGraph("test")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), act)
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        return graph

    def test_snn_mode_correct(self):
        """SNN neuron with snn_mode=SNN passes validation."""
        graph = self._build_graph(IFNodeV25())
        validate_graph(graph)

    def test_ann_mode_correct(self):
        """ANN neuron (LUT) with snn_mode=ANN passes validation."""
        graph = self._build_graph(ANNNodeV25(lut=LutReLU()))
        validate_graph(graph)

    def test_lut_with_snn_mode_raises(self):
        """LUT activation but snn_mode=SNN raises GraphValidationError."""
        act = ANNNodeV25(lut=LutReLU())
        graph = self._build_graph(act)

        # Force snn_mode mismatch
        op = next(n for n in graph.nodes.values() if isinstance(n, SequentialOp))
        op.core_params.snn_mode = SNNMode.SNN

        with pytest.raises(GraphValidationError, match="LUT activation.*SNNMode.ANN"):
            validate_graph(graph)

    def test_no_lut_with_ann_mode_raises(self):
        """No LUT but snn_mode=ANN raises GraphValidationError."""
        act = IFNodeV25()
        graph = self._build_graph(act)

        # Force snn_mode mismatch
        op = next(n for n in graph.nodes.values() if isinstance(n, SequentialOp))
        op.core_params.snn_mode = SNNMode.ANN

        with pytest.raises(
            GraphValidationError, match="no LUT activation.*SNNMode.SNN"
        ):
            validate_graph(graph)


class TestValidateCompiledGraph:
    def _build_compiled_graph(self) -> tuple[PAIIRGraph, SequentialOp]:
        graph = PAIIRGraph("compiled")
        inp = InputNode(shape=(1, 8))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        op.input_shapes = [(1, 8)]
        op.output_shape = (1, 4)
        op.input_dims = [(0, 1)]
        op.output_dims = (0, 1)
        op.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        op.core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))
        op.core_params.set_weight_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        op.core_params.tick_start = 1
        op.core_params.tick_duration = 0
        op.core_params.tick_initial = 0
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        return graph, op

    def test_valid_compiled_graph_passes(self):
        graph, _ = self._build_compiled_graph()
        validate_compiled_graph(graph)

    def test_missing_data_format_assignment_raises(self):
        graph, op = self._build_compiled_graph()
        op.core_params._input_format_assigned = False

        with pytest.raises(GraphValidationError, match="missing propagated data format"):
            validate_compiled_graph(graph)

    def test_catches_nodes_not_on_any_input_to_output_path(self):
        graph, _ = self._build_compiled_graph()
        branch = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        dead_end = SequentialOp(nn.Linear(4, 2), IFNodeV25())

        branch.input_shapes = [(1, 8)]
        branch.output_shape = (1, 4)
        branch.input_dims = [(0, 1)]
        branch.output_dims = (0, 1)
        dead_end.input_shapes = [(1, 4)]
        dead_end.output_shape = (1, 2)
        dead_end.input_dims = [(0, 1)]
        dead_end.output_dims = (0, 1)

        for node in (branch, dead_end):
            node.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
            node.core_params.set_output_format(
                (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)
            )
            node.core_params.set_weight_format(
                (DataSign.SIGNED, DataWidth.WIDTH_8BIT)
            )
            node.core_params.tick_start = 1
            node.core_params.tick_duration = 0
            node.core_params.tick_initial = 0

        inp = graph.input_nodes()[0]
        graph.add_node(branch)
        graph.add_node(dead_end)
        graph.add_edge(inp.name, branch.name)
        graph.add_edge(branch.name, dead_end.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        with pytest.raises(GraphValidationError, match="nodes not on any input-to-output path"):
            validate_compiled_graph(graph)
