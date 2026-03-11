import pytest
import torch
from paicorelib import RM, LeakMultiInputMode, OutputType, PoolingMode, SNNMode
from torch import nn

from paibox.paiir.calc_params import NeuronParams
from paibox.paiir.core_neuron import ANNNodeV25, IFNodeV25, LIFNodeV25
from paibox.paiir.graph import PAIIRGraph
from paibox.paiir.ir_base import InputNode, OutputNode
from paibox.paiir.lut_activation import LutReLU, LutSigmoid
from paibox.paiir.op_node import (
    AccumulateOp,
    AddOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)


class TestSequentialOp:
    def test_conv_neuron(self):
        op = SequentialOp(comp=nn.Conv2d(3, 8, 3, padding=1), act=IFNodeV25(1))
        assert op.core_params.snn_mode == SNNMode.SNN
        x = torch.randn(1, 3, 8, 8)
        out = op(x)
        assert out.shape == (1, 8, 8, 8)

    def test_linear_lut(self):
        op = SequentialOp(comp=nn.Linear(16, 10), act=ANNNodeV25(lut=LutReLU()))
        op.eval()
        assert op.core_params.snn_mode == SNNMode.ANN
        x = torch.randn(1, 16)
        out = op(x)
        assert out.shape == (1, 10)

    def test_maxpool_neuron(self):
        op = SequentialOp(comp=nn.MaxPool2d(2), act=LIFNodeV25(tau=2))
        assert op.core_params.pooling_mode == PoolingMode.MAX
        assert op.core_params.snn_mode == SNNMode.SNN


class TestAccumulateOp:
    def test_two_path_add(self):
        op = AccumulateOp(
            comps=[nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(3, 8, 3, padding=1)],
            act=IFNodeV25(1),
            op_signs=(1, 1),
        )
        x1 = torch.randn(1, 3, 8, 8)
        x2 = torch.randn(1, 3, 8, 8)
        out = op(x1, x2)
        assert out.shape == (1, 8, 8, 8)

    def test_two_path_sub(self):
        op = AccumulateOp(
            comps=[nn.Linear(4, 8), nn.Linear(4, 8)],
            act=ANNNodeV25(lut=LutReLU()),
            op_signs=(1, -1),
        )
        op.eval()
        x1 = torch.randn(1, 4)
        x2 = torch.randn(1, 4)
        out = op(x1, x2)
        assert out.shape == (1, 8)

    def test_sign_length_mismatch(self):
        with pytest.raises(ValueError, match="op_signs"):
            AccumulateOp(comps=[nn.Linear(4, 8)], act=IFNodeV25(), op_signs=(1, -1))

    def test_bias_fusion(self):
        """Bias from conv layers is fused into neuron params."""
        conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        conv2 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        op = AccumulateOp(comps=[conv1, conv2], act=IFNodeV25(1), op_signs=(1, 1))
        params = op.neuron_params
        assert params.leak_v is not None


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
        add = AddOp(op_signs=(1, 1))
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


class TestWeights:
    """Test weights property on OpNode subclasses."""

    def test_sequential_conv_returns_weight(self):
        conv = nn.Conv2d(3, 8, 3, padding=1)
        op = SequentialOp(comp=conv, act=ANNNodeV25(lut=LutReLU()))
        op.output_shape = (1, 8, 8, 8)
        ws = op.weights
        assert len(ws) == 1
        assert ws[0].dtype == torch.int8
        assert ws[0].shape == conv.weight.shape

    def test_sequential_pool_returns_none(self):
        """Weightless ops (pool etc.) return None, not identity matrices."""
        op = SequentialOp(comp=nn.MaxPool2d(2), act=IFNodeV25())
        op.output_shape = (1, 4, 4, 4)
        assert op.weights is None

    def test_accumulate_returns_per_path_weights(self):
        conv1 = nn.Conv2d(3, 8, 3, padding=1)
        conv2 = nn.Conv2d(3, 8, 3, padding=1)
        op = AccumulateOp(comps=[conv1, conv2], act=IFNodeV25(), op_signs=(1, 1))
        op.output_shape = (1, 8, 8, 8)
        ws = op.weights
        assert len(ws) == 2
        assert ws[0].shape == conv1.weight.shape
        assert ws[1].shape == conv2.weight.shape
        assert all(w.dtype == torch.int8 for w in ws)

    def test_standalone_act_returns_identity(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutReLU()))
        op.output_shape = (1, 16)
        ws = op.weights
        assert len(ws) == 1
        assert ws[0].shape == (16, 16)
        assert torch.equal(ws[0], torch.eye(16, dtype=torch.int8))

    def test_standalone_comp_linear_returns_weight(self):
        linear = nn.Linear(4, 8, bias=False)
        op = StandaloneCompOp(comp=linear)
        op.output_shape = (1, 8)
        ws = op.weights
        assert len(ws) == 1
        assert ws[0].dtype == torch.int8
        assert ws[0].shape == linear.weight.shape

    def test_standalone_comp_pool_returns_none(self):
        """Weightless ops (pool etc.) return None, not identity matrices."""
        op = StandaloneCompOp(comp=nn.AvgPool2d(2))
        op.output_shape = (1, 3, 4, 4)
        assert op.weights is None

    def test_add_op_returns_identity_per_path(self):
        op = AddOp(op_signs=(1, -1))
        op.output_shape = (1, 8)
        ws = op.weights
        assert len(ws) == 2
        expected = torch.eye(8, dtype=torch.int8)
        assert torch.equal(ws[0], expected)
        assert torch.equal(ws[1], expected)

    def test_weights_before_output_shape_raises(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutReLU()))
        with pytest.raises(AssertionError, match="output_shape"):
            _ = op.weights


class TestNeuronParams:
    """Test neuron_params property on OpNode subclasses."""

    def test_sequential_snn_neuron_params(self):
        op = SequentialOp(comp=nn.Conv2d(3, 8, 3, padding=1), act=IFNodeV25(1, 0))
        params = op.neuron_params
        assert isinstance(params, NeuronParams)
        assert params.thres_pos == 1
        assert params.reset_mode == RM.MODE_NORMAL
        assert params.reset_v == 0
        assert params.output_type == OutputType.VALUE

    def test_sequential_lut_neuron_params(self):
        op = SequentialOp(comp=nn.Linear(16, 10), act=ANNNodeV25(lut=LutReLU()))
        params = op.neuron_params
        assert isinstance(params, NeuronParams)
        assert params.output_type == OutputType.VALUE

    def test_accumulate_neuron_params(self):
        op = AccumulateOp(
            comps=[nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(3, 8, 3, padding=1)],
            act=LIFNodeV25(tau=2, v_threshold=2),
            op_signs=(1, 1),
        )
        params = op.neuron_params
        assert params.thres_pos == 2
        assert params.leak_tau == -1

    def test_add_op_pass_through(self):
        op = AddOp(op_signs=(1, -1))
        params = op.neuron_params
        assert params.output_type == OutputType.POTENTIAL

    def test_standalone_comp_pass_through(self):
        op = StandaloneCompOp(comp=nn.Conv2d(3, 8, 3))
        params = op.neuron_params
        assert params.output_type == OutputType.POTENTIAL

    def test_standalone_activation_neuron_params(self):
        op = StandaloneActOp(act=IFNodeV25(1))
        params = op.neuron_params
        assert params.thres_pos == 1
        assert params.output_type == OutputType.VALUE


class TestLutData:
    def test_sequential_lut_exports_data(self):
        op = SequentialOp(
            comp=nn.Conv2d(3, 8, 3, padding=1), act=ANNNodeV25(lut=LutReLU())
        )
        data = op.lut_data
        assert data is not None
        assert data.thresholds.shape == (256,)
        assert data.values.shape == (256,)

    def test_sequential_neuron_returns_none(self):
        op = SequentialOp(comp=nn.Conv2d(3, 8, 3, padding=1), act=IFNodeV25())
        assert op.lut_data is None

    def test_standalone_act_lut_exports_data(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutSigmoid()))
        data = op.lut_data
        assert data is not None

    def test_add_op_returns_none(self):
        op = AddOp(op_signs=(1, 1))
        assert op.lut_data is None

    def test_accumulate_lut_exports_data(self):
        op = AccumulateOp(
            comps=[nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(3, 8, 3, padding=1)],
            act=ANNNodeV25(lut=LutReLU()),
            op_signs=(1, 1),
        )
        data = op.lut_data
        assert data is not None
        assert data.thresholds.shape == (256,)

    def test_accumulate_neuron_returns_none(self):
        op = AccumulateOp(comps=[nn.Conv2d(3, 8, 3, padding=1)], act=IFNodeV25())
        assert op.lut_data is None

    def test_standalone_comp_returns_none(self):
        op = StandaloneCompOp(comp=nn.Conv2d(3, 8, 3))
        assert op.lut_data is None


class TestAvgPoolCompensation:
    def test_sequential_avgpool_lut_compensates(self):
        """AvgPool + LutReLU: lut_data thresholds are scaled."""
        op = SequentialOp(comp=nn.AvgPool2d(2), act=ANNNodeV25(lut=LutReLU()))
        data = op.lut_data
        assert data is not None
        # k=2, window_size=4=2^2, scale=1.0: thresholds should be unchanged
        baseline = LutReLU().export_lut()
        assert torch.equal(data.thresholds, baseline.thresholds)

    def test_sequential_avgpool_lif_compensates_threshold(self):
        """AvgPool + LIF: neuron threshold is compensated."""
        op = SequentialOp(comp=nn.AvgPool2d(3), act=LIFNodeV25(tau=2, v_threshold=1.0))
        params = op.neuron_params
        # LIFNodeV25 default decay_input=True -> leak_multi_input=ENABLE
        # k=3, window_size=9, factor=9 (decay_input=True)
        # theta' = 0 + (1.0 - 0) * 9 = 9.0
        assert params.thres_pos == 9.0
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE

    def test_sequential_avgpool_neuron_params_leak_set(self):
        """AvgPool sets leak_tau on neuron_params."""
        op = SequentialOp(comp=nn.AvgPool2d(2), act=ANNNodeV25(lut=LutReLU()))
        params = op.neuron_params
        assert params.leak_tau == -2
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
