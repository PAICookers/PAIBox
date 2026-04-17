"""Tests for data format inference and propagation pass."""

import pytest
import torch
from paicorelib import (
    AddPotentialMode,
    DataSign,
    DataWidth,
    OutputType,
    ThresholdNegMode,
)
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir.ir.add_ops import PotentialAddOp
from paibox.paiir.ir.calc_params import NeuronParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.lut_activation import LutCustom, LutReLU, LutSigmoid, LutTanh
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    OfflineCoreOp,
    SequentialOp,
    ShapeStage,
    StandaloneActOp,
    StandaloneCompOp,
    TransformOp,
)
from paibox.paiir.lowering.converter import torch_to_paiir
from paibox.paiir.pipeline.data_format import (
    infer_output_format,
    infer_weight_format,
    merge_data_formats,
)
from paibox.paiir.pipeline.passes import (
    fuse_to_offline_cores,
    propagate_data_format,
    specialize_general_adds,
)
from tests.paiir.conftest import (
    convert_fuse_propagate,
    find_nodes,
    make_multispike4_lut,
)


def _make_custom_lut(values: torch.Tensor, *, output_sign: int) -> LutCustom:
    repeats = (256 + values.numel() - 1) // values.numel()
    tiled = values.repeat(repeats)[:256].to(torch.int32)
    thresholds = torch.arange(256, dtype=torch.int32)
    return LutCustom(thresholds, tiled, output_sign=output_sign)


class TestInferOutputFormat:
    @pytest.mark.parametrize(
        "act, expected_sign, expected_width",
        [
            (IFNodeV25(v_threshold=1.0), DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
            (LIFNodeV25(tau=2.0), DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
            (
                CoreNeuronV25(
                    thres_pos=1.0,
                    thres_neg=-1.0,
                    thres_neg_mode=ThresholdNegMode.FIRE,
                ),
                DataSign.SIGNED,
                DataWidth.WIDTH_2BIT,
            ),
            (ANNNodeV25(lut=LutReLU()), DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
            (ANNNodeV25(lut=LutSigmoid()), DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
            (
                ANNNodeV25(lut=LutTanh(output_sign=1)),
                DataSign.SIGNED,
                DataWidth.WIDTH_8BIT,
            ),
        ],
        ids=[
            "if_unsigned_1bit",
            "lif_unsigned_1bit",
            "neuron_neg_fire_signed_2bit",
            "lut_relu_unsigned_8bit",
            "lut_sigmoid_unsigned_8bit",
            "lut_tanh_signed_8bit",
        ],
    )
    def test_output_format(self, act, expected_sign, expected_width):
        sign, width = infer_output_format(act)
        assert sign == expected_sign
        assert width == expected_width

    @pytest.mark.parametrize(
        "lut, expected_sign, expected_width",
        [
            (
                _make_custom_lut(torch.tensor([0, 1]), output_sign=0),
                DataSign.UNSIGNED,
                DataWidth.WIDTH_1BIT,
            ),
            (
                _make_custom_lut(torch.tensor([0, 1, 2, 3]), output_sign=0),
                DataSign.UNSIGNED,
                DataWidth.WIDTH_2BIT,
            ),
            (
                make_multispike4_lut(),
                DataSign.UNSIGNED,
                DataWidth.WIDTH_4BIT,
            ),
            (
                _make_custom_lut(torch.tensor([-1, 0]), output_sign=1),
                DataSign.SIGNED,
                DataWidth.WIDTH_1BIT,
            ),
            (
                _make_custom_lut(torch.tensor([-2, -1, 0, 1]), output_sign=1),
                DataSign.SIGNED,
                DataWidth.WIDTH_2BIT,
            ),
        ],
        ids=[
            "lut_unsigned_binary",
            "lut_unsigned_quaternary",
            "lut_unsigned_five_level",
            "lut_signed_negative_zero",
            "lut_signed_quaternary",
        ],
    )
    def test_ann_output_format_uses_lut_value_range(
        self, lut, expected_sign, expected_width
    ):
        sign, width = infer_output_format(ANNNodeV25(lut=lut))
        assert sign == expected_sign
        assert width == expected_width


class TestInferWeightFormat:
    @pytest.mark.parametrize(
        "w_min, w_max, expected_sign, expected_width",
        [
            (0, 1, DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
            (-1, 0, DataSign.SIGNED, DataWidth.WIDTH_1BIT),
            (-1, 1, DataSign.SIGNED, DataWidth.WIDTH_2BIT),
            (-8, 7, DataSign.SIGNED, DataWidth.WIDTH_4BIT),
            (-3, 3, DataSign.SIGNED, DataWidth.WIDTH_4BIT),
            (0, 255, DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
            (-128, 127, DataSign.SIGNED, DataWidth.WIDTH_8BIT),
            (-50, 50, DataSign.SIGNED, DataWidth.WIDTH_8BIT),
            (0, 0, DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
        ],
        ids=[
            "binary_unsigned",
            "binary_signed",
            "ternary_signed_2bit",
            "int4_signed",
            "int4_partial",
            "uint8",
            "int8",
            "int8_partial",
            "zero_weights",
        ],
    )
    def test_weight_format(self, w_min, w_max, expected_sign, expected_width):
        sign, width = infer_weight_format(w_min, w_max)
        assert sign == expected_sign
        assert width == expected_width

    def test_overflow_raises(self):
        """Values exceeding 8-bit range should raise."""
        with pytest.raises(ValueError, match="exceeds .*8-bit"):
            infer_weight_format(-200, 200)


class TestMergeDataFormats:
    @pytest.mark.parametrize(
        "formats, expected",
        [
            (
                [(DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)],
                (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
            ),
            (
                [
                    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
                    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
                ],
                (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
            ),
            (
                [
                    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
                    (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
                ],
                (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
            ),
            (
                [
                    (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
                    (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
                ],
                (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
            ),
            (
                [
                    (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
                    (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
                ],
                (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
            ),
        ],
        ids=[
            "single",
            "same_formats",
            "mixed_width",
            "mixed_sign",
            "mixed_sign_and_width",
        ],
    )
    def test_merge(self, formats, expected):
        assert merge_data_formats(formats) == expected

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_data_formats([])


class TestPropagateDataFormatSNN:
    """SNN networks: spike-based data flow."""

    def test_two_layer_snn(self):
        """Conv-LIF -> Conv-IF: output 1BIT, second layer input 1BIT."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.lif = sj.LIFNode(tau=2.0)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.conv2(self.lif(self.conv1(x))))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))
        ops = find_nodes(fused, OfflineCoreOp)
        assert len(ops) == 2

        topo = fused.topo_sort()
        ops_sorted = sorted(ops, key=lambda o: topo.index(o.name))
        first, second = ops_sorted

        assert first.core_params.input_sign == DataSign.UNSIGNED
        assert first.core_params.input_width == DataWidth.WIDTH_1BIT
        assert first.core_params.output_sign == DataSign.UNSIGNED
        assert first.core_params.output_width == DataWidth.WIDTH_1BIT

        assert second.core_params.input_sign == DataSign.UNSIGNED
        assert second.core_params.input_width == DataWidth.WIDTH_1BIT
        assert second.core_params.output_sign == DataSign.UNSIGNED
        assert second.core_params.output_width == DataWidth.WIDTH_1BIT

    def test_snn_with_custom_input_format(self):
        """User override: 8BIT input to SNN network."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.conv(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        inp_name = fused.input_nodes()[0].name
        custom_fmt = {inp_name: (DataSign.SIGNED, DataWidth.WIDTH_8BIT)}
        propagate_data_format(fused, input_formats=custom_fmt)

        ops = find_nodes(fused, OfflineCoreOp)
        assert len(ops) == 1
        op = ops[0]

        assert op.core_params.input_sign == DataSign.SIGNED
        assert op.core_params.input_width == DataWidth.WIDTH_8BIT
        assert op.core_params.output_sign == DataSign.UNSIGNED
        assert op.core_params.output_width == DataWidth.WIDTH_1BIT


class TestPropagateDataFormatANN:
    """ANN networks: activation-based data flow."""

    def test_conv_relu_linear_sigmoid(self):
        """Conv-ReLU -> Linear-Sigmoid: 8BIT data flow."""

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

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))
        ops = find_nodes(fused, OfflineCoreOp)

        topo = fused.topo_sort()
        ops_sorted = sorted(ops, key=lambda o: topo.index(o.name))

        first = ops_sorted[0]
        assert first.core_params.input_sign == DataSign.SIGNED
        assert first.core_params.input_width == DataWidth.WIDTH_8BIT
        assert first.core_params.output_sign == DataSign.UNSIGNED
        assert first.core_params.output_width == DataWidth.WIDTH_8BIT

        second = ops_sorted[1]
        assert second.core_params.input_sign == DataSign.UNSIGNED
        assert second.core_params.input_width == DataWidth.WIDTH_8BIT
        assert second.core_params.output_sign == DataSign.UNSIGNED
        assert second.core_params.output_width == DataWidth.WIDTH_8BIT

    @pytest.mark.parametrize(
        "input_format",
        [
            (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT),
            (DataSign.SIGNED, DataWidth.WIDTH_8BIT),
            (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT),
        ],
        ids=["u8", "i8", "u1"],
    )
    def test_standalone_maxpool_preserves_predecessor_format(self, input_format):
        class MaxPoolConvRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2, 2)
                self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(self.pool(x)))

        unfused = torch_to_paiir(MaxPoolConvRelu(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        inp_name = fused.input_nodes()[0].name
        propagate_data_format(fused, input_formats={inp_name: input_format})

        pool = next(
            node
            for node in fused.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.MaxPool2d)
        )
        conv = next(
            node
            for node in fused.nodes.values()
            if isinstance(node, SequentialOp) and isinstance(node.comp, nn.Conv2d)
        )

        assert (
            pool.core_params.input_sign,
            pool.core_params.input_width,
        ) == input_format
        assert (
            pool.core_params.output_sign,
            pool.core_params.output_width,
        ) == input_format
        assert (
            conv.core_params.input_sign,
            conv.core_params.input_width,
        ) == input_format

    def test_subtract_tanh(self):
        """Two linear branches with subtraction -> tanh: UNSIGNED 8BIT."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = nn.Linear(8, 16)
                self.linear_b = nn.Linear(8, 16)
                self.tanh = nn.Tanh()

            def forward(self, x):
                return self.tanh(self.linear_a(x) - self.linear_b(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 8))
        accum = find_nodes(fused, AccumulateOp)
        assert len(accum) == 1
        op = accum[0]
        assert op.core_params.output_sign == DataSign.UNSIGNED
        assert op.core_params.output_width == DataWidth.WIDTH_8BIT

    def test_custom_lut_width_propagates_to_successor_input(self):
        from paibox.paiir.lowering.converter import register_neuron

        class Quant4Like(nn.Module):
            def forward(self, x):
                return torch.round(torch.clamp(x, min=0, max=4))

        register_neuron(
            Quant4Like, converter=lambda _: ANNNodeV25(lut=make_multispike4_lut())
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 8)
                self.act = Quant4Like()
                self.linear2 = nn.Linear(8, 4)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.act(self.linear1(x))
                return self.sigmoid(self.linear2(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 8))
        ops = find_nodes(fused, OfflineCoreOp)

        topo = fused.topo_sort()
        ops_sorted = sorted(ops, key=lambda o: topo.index(o.name))
        first, second = ops_sorted

        assert first.core_params.output_sign == DataSign.UNSIGNED
        assert first.core_params.output_width == DataWidth.WIDTH_4BIT
        assert second.core_params.input_sign == DataSign.UNSIGNED
        assert second.core_params.input_width == DataWidth.WIDTH_4BIT

    def test_potential_width_propagates_through_routing_nodes(self):
        graph = PAIIRGraph("potential_routing")
        inp = InputNode(shape=torch.Size((1, 1, 4, 4)))
        comp_main = StandaloneCompOp(nn.Conv2d(1, 1, 1, bias=False))
        reshape_main = TransformOp((ShapeStage(lambda _: torch.Size((1, 1, 1, 4, 4))),))
        act = StandaloneActOp(ANNNodeV25(lut=LutReLU()))
        comp_skip = StandaloneCompOp(nn.Conv2d(1, 1, 1, bias=False))
        reshape_skip = TransformOp((ShapeStage(lambda _: torch.Size((1, 1, 1, 4, 4))),))
        add = PotentialAddOp((1, 1))
        out = OutputNode(shape=torch.Size((1, 1, 1, 4, 4)))

        for node in (
            inp,
            comp_main,
            reshape_main,
            act,
            comp_skip,
            reshape_skip,
            add,
            out,
        ):
            graph.add_node(node)

        graph.add_edge(inp.name, comp_main.name)
        graph.add_edge(comp_main.name, reshape_main.name)
        graph.add_edge(reshape_main.name, act.name)
        graph.add_edge(reshape_main.name, add.name)
        graph.add_edge(inp.name, comp_skip.name)
        graph.add_edge(comp_skip.name, reshape_skip.name)
        graph.add_edge(reshape_skip.name, add.name)
        graph.add_edge(add.name, out.name)

        propagate_data_format(
            graph, input_formats={inp.name: (DataSign.SIGNED, DataWidth.WIDTH_8BIT)}
        )

        assert comp_main.core_params.add_potential == AddPotentialMode.NORMAL
        assert comp_main.core_params.output_sign == DataSign.SIGNED
        assert comp_main.core_params.output_width == DataWidth.WIDTH_32BIT
        assert act.core_params.add_potential == AddPotentialMode.DIRECT_ADD
        assert act.core_params.input_sign == DataSign.SIGNED
        assert act.core_params.input_width == DataWidth.WIDTH_32BIT
        assert act.core_params.output_width == DataWidth.WIDTH_8BIT
        assert add.core_params.add_potential == AddPotentialMode.DIRECT_ADD
        assert add.core_params.input_sign == DataSign.SIGNED
        assert add.core_params.input_width == DataWidth.WIDTH_32BIT
        assert add.core_params.output_width == DataWidth.WIDTH_32BIT

    def test_value_output_without_activation_raises(self):
        class ValueWithoutActOp(OfflineCoreOp):
            @property
            def neuron_params(self) -> NeuronParams:
                return NeuronParams(output_type=OutputType.VALUE)

        graph = PAIIRGraph("value_without_activation")
        inp = InputNode(shape=torch.Size((1, 1, 4, 4)))
        op = ValueWithoutActOp()
        out = OutputNode(shape=torch.Size((1, 1, 4, 4)))

        for node in (inp, op, out):
            graph.add_node(node)

        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)

        with pytest.raises(
            ValueError,
            match="declares VALUE output but has no activation",
        ):
            propagate_data_format(
                graph, input_formats={inp.name: (DataSign.SIGNED, DataWidth.WIDTH_8BIT)}
            )


class TestPropagateDataFormatResidual:
    """Residual/multi-input patterns."""

    def test_residual_add_snn(self):
        """Two conv branches + add + LIF: both inputs same format."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = nn.Conv2d(3, 16, 3, padding=1)
                self.conv_b = nn.Conv2d(3, 16, 3, padding=1)
                self.lif = sj.LIFNode(tau=2.0)

            def forward(self, x):
                return self.lif(self.conv_a(x) + self.conv_b(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))

        accum = find_nodes(fused, AccumulateOp)
        assert len(accum) == 1
        op = accum[0]

        assert op.core_params.input_sign == DataSign.UNSIGNED
        assert op.core_params.input_width == DataWidth.WIDTH_1BIT
        assert op.core_params.output_sign == DataSign.UNSIGNED
        assert op.core_params.output_width == DataWidth.WIDTH_1BIT


class TestPropagateWeightFormat:
    """Weight format inference."""

    def test_default_float_weights_as_int8(self):
        """Default float weights interpreted as int8 range."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.conv(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))
        ops = find_nodes(fused, OfflineCoreOp)
        assert len(ops) == 1
        op = ops[0]

        assert op.core_params.weight_sign in (DataSign.SIGNED, DataSign.UNSIGNED)
        assert op.core_params.weight_width.value <= DataWidth.WIDTH_8BIT.value

    def test_quantized_int8_weights(self):
        """Explicitly quantized int8 weights -> SIGNED 8BIT."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.conv(x))

        model = Model()
        with torch.no_grad():
            model.conv.weight.copy_(
                torch.randint(-128, 128, model.conv.weight.shape, dtype=torch.float32)
            )

        fused = convert_fuse_propagate(model, torch.randn(1, 3, 8, 8))
        ops = find_nodes(fused, OfflineCoreOp)
        op = ops[0]

        assert op.core_params.weight_sign == DataSign.SIGNED
        assert op.core_params.weight_width == DataWidth.WIDTH_8BIT

    def test_binary_weights(self):
        """Binary {0, 1} weights -> UNSIGNED 1BIT."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4, bias=False)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.linear(x))

        model = Model()
        with torch.no_grad():
            model.linear.weight.copy_(
                torch.randint(0, 2, model.linear.weight.shape, dtype=torch.float32)
            )

        fused = convert_fuse_propagate(model, torch.randn(1, 8))
        ops = find_nodes(fused, OfflineCoreOp)
        op = ops[0]

        assert op.core_params.weight_sign == DataSign.UNSIGNED
        assert op.core_params.weight_width == DataWidth.WIDTH_1BIT

    def test_standalone_act_weightless(self):
        """StandaloneActOp: no weights -> UNSIGNED 1BIT (most compact)."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(x)

        fused = convert_fuse_propagate(Model(), torch.randn(1, 8))
        act_ops = find_nodes(fused, StandaloneActOp)
        assert len(act_ops) == 1
        op = act_ops[0]

        assert op.core_params.weight_sign == DataSign.UNSIGNED
        assert op.core_params.weight_width == DataWidth.WIDTH_1BIT


class TestPropagateEdgeConsistency:
    """Verify predecessor output matches successor input at every edge."""

    def _check_edge_consistency(self, fused):
        """Assert output format matches input format at every inter-op edge."""
        for edge in fused.edges:
            src = fused.nodes[edge.src]
            dst = fused.nodes[edge.dst]
            if isinstance(src, OfflineCoreOp) and isinstance(dst, OfflineCoreOp):
                assert (
                    src.core_params.output_sign == dst.core_params.input_sign
                ), f"Sign mismatch at edge {edge.src} -> {edge.dst}"
                assert (
                    src.core_params.output_width == dst.core_params.input_width
                ), f"Width mismatch at edge {edge.src} -> {edge.dst}"

    def test_two_layer_ann(self):
        """Conv-ReLU -> Conv-Sigmoid: edge format consistency."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                return self.sigmoid(self.conv2(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))
        self._check_edge_consistency(fused)

    def test_three_layer_snn(self):
        """Three-layer SNN: verify all edges are consistent."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.if1 = sj.IFNode()
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.if2 = sj.IFNode()
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.if3 = sj.IFNode()

            def forward(self, x):
                x = self.if1(self.conv1(x))
                x = self.if2(self.conv2(x))
                return self.if3(self.conv3(x))

        fused = convert_fuse_propagate(Model(), torch.randn(1, 3, 8, 8))
        ops = find_nodes(fused, OfflineCoreOp)
        assert len(ops) == 3
        self._check_edge_consistency(fused)
