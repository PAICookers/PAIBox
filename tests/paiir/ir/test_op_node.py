import math

import pytest
import torch
from paicorelib import (
    RM,
    AddPotentialMode,
    DataSign,
    DataWidth,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    OutputType,
    PoolingMode,
    SNNMode,
    ThresholdNegMode,
    ThresholdPosMode,
)
from torch import nn

from paibox.paiir.ir.add_ops import PotentialAddOp
from paibox.paiir.ir.calc_params import LUT_TABLE_SIZE, NeuronParams, OfflineCoreParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.ir_base import FormatFlow, OutputNode
from paibox.paiir.ir.lut_activation import LutReLU, LutSigmoid, _lookup_hw_lut_data
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    LayoutStage,
    PadOp,
    SequentialOp,
    ShapeStage,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TensorLayout,
    TransformOp,
)
from paibox.paiir.ir.signal_domain import SignalDomain
from paibox.paiir.pipeline.avgpool.compensation import (
    apply_avgpool_lut_compensation,
    apply_avgpool_snn_compensation,
)
from paibox.paiir.pipeline.passes import _infer_node_weight_format


class TestNodeCapabilities:
    def test_routing_nodes_declare_format_flow_and_zero_tick_depth(self):
        routing_nodes = [TransformOp(), PadOp((1, 1)), SplitOp(sections=2, dim=1)]

        for node in routing_nodes:
            assert node.__format_flow__ is FormatFlow.PASS_THROUGH
            assert node.__tick_depth__ == 0

    def test_concat_declares_merge_format_flow_and_zero_tick_depth(self):
        node = ConcatOp()

        assert node.__format_flow__ is FormatFlow.MERGE
        assert node.__tick_depth__ == 0

    def test_graph_output_passes_format_without_tick_depth(self):
        node = OutputNode()

        assert node.__format_flow__ is FormatFlow.PASS_THROUGH
        assert node.__tick_depth__ == 0

    def test_offline_core_node_defaults_to_no_format_flow_and_one_tick_depth(self):
        node = StandaloneCompOp(nn.Linear(4, 2))

        assert node.__format_flow__ is FormatFlow.NONE
        assert node.__tick_depth__ == 1


class TestTransformOp:
    def test_layout_stage_materializes_permuted_layout(self):
        op = TransformOp((LayoutStage((0, 2, 1)),))
        x = torch.arange(6, dtype=torch.int64).reshape(1, 2, 3)

        expected = x.permute(0, 2, 1)
        assert torch.equal(op(x), expected)

    def test_shape_stage_reshapes_without_reordering_flat_indices(self):
        op = TransformOp((ShapeStage(lambda _: torch.Size((1, 3, 2))),))
        x = torch.arange(6, dtype=torch.int64).reshape(1, 2, 3)

        expected = x.reshape(1, 3, 2)
        assert torch.equal(op(x), expected)

    def test_shape_stage_without_shape_fn_flattens_tensor(self):
        op = TransformOp((ShapeStage(None),))
        x = torch.arange(6, dtype=torch.int64).reshape(1, 2, 3)

        expected = x.flatten()
        assert torch.equal(op(x), expected)

    def test_mixed_stage_chain_preserves_declared_execution_order(self):
        op = TransformOp(
            (
                LayoutStage((0, 2, 1)),
                ShapeStage(lambda _: torch.Size((1, 2, 3))),
                LayoutStage((0, 2, 1)),
            )
        )
        x = torch.arange(6, dtype=torch.int64).reshape(1, 2, 3)

        expected = x.permute(0, 2, 1).reshape(1, 2, 3).permute(0, 2, 1)
        assert torch.equal(op(x), expected)


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

    def test_override_compile_state_preserves_prepared_compile_state(self):
        base = OfflineCoreParams(tick_start=5, tick_duration=9, tick_initial=13)
        base.set_input_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))
        base.set_output_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        base.set_weight_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))

        op = SequentialOp(comp=nn.MaxPool2d(2), act=IFNodeV25())
        op.override_compile_state(base)

        assert op.core_params.tick_start == 5
        assert op.core_params.tick_duration == 9
        assert op.core_params.tick_initial == 13
        assert op.core_params.input_sign == DataSign.UNSIGNED
        assert op.core_params.input_width == DataWidth.WIDTH_1BIT
        assert op.core_params.snn_mode == SNNMode.SNN
        assert op.core_params.pooling_mode == PoolingMode.MAX
        op.core_params.validate_data_formats()


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

    def test_without_activation_emits_potential_sum(self):
        linear_a = nn.Linear(2, 2, bias=False)
        linear_b = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear_a.weight.copy_(torch.eye(2))
            linear_b.weight.copy_(2 * torch.eye(2))

        op = AccumulateOp(comps=[linear_a, linear_b], act=None, op_signs=(1, -1))
        x1 = torch.tensor([[3.0, 4.0]])
        x2 = torch.tensor([[1.0, 2.0]])

        assert torch.equal(op(x1, x2), torch.tensor([[1.0, 0.0]]))
        assert op.hw_lut_data is None
        assert op.src_params()[0].output_type is OutputType.POTENTIAL

    def test_sign_length_mismatch(self):
        with pytest.raises(ValueError, match="op_signs"):
            AccumulateOp(comps=[nn.Linear(4, 8)], act=IFNodeV25(), op_signs=(1, -1))

    def test_bias_fusion(self):
        """Bias from conv layers is fused into neuron params."""
        conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        conv2 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        op = AccumulateOp(comps=[conv1, conv2], act=IFNodeV25(1), op_signs=(1, 1))
        params, bias = op.src_params()
        assert params.leak_v == 0
        assert bias is not None


class TestWeights:
    """Test weights property on OpNode subclasses."""

    def test_sequential_conv_returns_weight(self):
        conv = nn.Conv2d(3, 8, 3, padding=1)
        op = SequentialOp(comp=conv, act=ANNNodeV25(lut=LutReLU()))
        op.output_layouts = (TensorLayout(torch.Size((1, 8, 8, 8)), (0, 1, 2, 3)),)
        ws = op.weights
        assert ws is not None
        assert len(ws) == 1
        assert ws[0].dtype == torch.int8
        assert ws[0].shape == conv.weight.shape

    def test_sequential_pool_returns_none(self):
        """Weightless ops (pool etc.) return None, not identity matrices."""
        op = SequentialOp(comp=nn.MaxPool2d(2), act=IFNodeV25())
        op.output_layouts = (TensorLayout(torch.Size((1, 4, 4, 4)), (0, 1, 2, 3)),)
        assert op.weights is None

    def test_accumulate_returns_per_path_weights(self):
        conv1 = nn.Conv2d(3, 8, 3, padding=1)
        conv2 = nn.Conv2d(3, 8, 3, padding=1)
        op = AccumulateOp(comps=[conv1, conv2], act=IFNodeV25(), op_signs=(1, 1))
        op.output_layouts = (TensorLayout(torch.Size((1, 8, 8, 8)), (0, 1, 2, 3)),)
        ws = op.weights
        assert ws is not None
        assert len(ws) == 2
        assert ws[0].shape == conv1.weight.shape
        assert ws[1].shape == conv2.weight.shape
        assert all(w.dtype == torch.int8 for w in ws)

    def test_standalone_act_returns_none(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutReLU()))
        assert op.weights is None

    def test_standalone_comp_linear_returns_weight(self):
        linear = nn.Linear(4, 8, bias=False)
        op = StandaloneCompOp(comp=linear)
        op.output_layouts = (TensorLayout(torch.Size((1, 8)), (0, 1)),)
        ws = op.weights
        assert ws is not None
        assert len(ws) == 1
        assert ws[0].dtype == torch.int8
        assert ws[0].shape == linear.weight.shape

    def test_standalone_comp_pool_returns_none(self):
        """Weightless ops (pool etc.) return None, not identity matrices."""
        op = StandaloneCompOp(comp=nn.AvgPool2d(2))
        op.output_layouts = (TensorLayout(torch.Size((1, 3, 4, 4)), (0, 1, 2, 3)),)
        assert op.weights is None

    def test_standalone_comp_neuron_params_follow_output_domain(self):
        op = StandaloneCompOp(comp=nn.MaxPool2d(2))

        op.signal_semantics.output_domain = SignalDomain.VALUE
        assert op.src_params()[0].output_type == OutputType.VALUE

        op.signal_semantics.output_domain = SignalDomain.POTENTIAL
        assert op.src_params()[0].output_type == OutputType.POTENTIAL

    @pytest.mark.parametrize("comp", [nn.Linear(4, 2), nn.Conv2d(1, 2, 1)])
    def test_standalone_comp_preserves_current_potential(self, comp):
        params, _ = StandaloneCompOp(comp).src_params()

        assert params.reset_mode == RM.MODE_NONRESET
        assert params.thres_neg_mode == ThresholdNegMode.FIRE
        assert params.thres_pos_mode == ThresholdPosMode.FIRE
        assert params.thres_neg == 0
        assert params.thres_pos == 0

        neuron = CoreNeuronV25(
            reset_mode=params.reset_mode,
            thres_neg_mode=params.thres_neg_mode,
            thres_pos_mode=params.thres_pos_mode,
            thres_neg=params.thres_neg,
            thres_pos=params.thres_pos,
        ).eval()
        neuron(torch.tensor([[5.0, -5.0]]))
        assert torch.equal(neuron.v, torch.tensor([[5.0, -5.0]]))

    def test_add_op_returns_none(self):
        op = PotentialAddOp(op_signs=(1, -1))
        assert op.weights is None

    def test_override_compile_state_preserves_add_compile_state(self):
        base = OfflineCoreParams(tick_start=3, tick_duration=7, tick_initial=11)
        base.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        base.set_output_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        base.set_weight_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))

        op = PotentialAddOp(op_signs=(1, -1))
        op.override_compile_state(base)

        assert op.core_params.tick_start == 3
        assert op.core_params.tick_duration == 7
        assert op.core_params.tick_initial == 11
        assert op.core_params.add_potential.name == "DIRECT_ADD"
        assert op.core_params.weight_sign == DataSign.UNSIGNED
        assert op.core_params.weight_width == DataWidth.WIDTH_1BIT
        op.core_params.validate_data_formats()

    def test_override_compile_state_keeps_semantic_fields(self):
        base = OfflineCoreParams()
        base.snn_mode = SNNMode.ANN
        base.pooling_mode = PoolingMode.AVERAGE
        base.add_potential = AddPotentialMode.NORMAL

        seq = SequentialOp(comp=nn.MaxPool2d(2), act=IFNodeV25())
        seq.override_compile_state(base)
        assert seq.core_params.snn_mode == SNNMode.SNN
        assert seq.core_params.pooling_mode == PoolingMode.MAX

        add = PotentialAddOp(op_signs=(1, 1))
        add.override_compile_state(base)
        assert add.core_params.add_potential.name == "DIRECT_ADD"

    def test_weightless_ops_do_not_require_output_shape_for_weights(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutReLU()))
        assert op.weights is None

    def test_standalone_act_weight_format_uses_implicit_identity_range(self):
        op = StandaloneActOp(act=ANNNodeV25(lut=LutReLU()))
        op.output_layouts = (TensorLayout(torch.Size((1, 1024, 1024)), (0, 1, 2)),)

        assert _infer_node_weight_format(op) == (
            DataSign.UNSIGNED,
            DataWidth.WIDTH_1BIT,
        )

    def test_add_weight_format_uses_implicit_identity_range(self):
        op = PotentialAddOp(op_signs=(1, -1))
        op.output_layouts = (TensorLayout(torch.Size((1, 1024, 1024)), (0, 1, 2)),)

        assert _infer_node_weight_format(op) == (
            DataSign.UNSIGNED,
            DataWidth.WIDTH_1BIT,
        )


class TestNeuronParams:
    """Test source neuron parameters on OpNode subclasses."""

    def test_sequential_snn_neuron_params(self):
        op = SequentialOp(comp=nn.Conv2d(3, 8, 3, padding=1), act=IFNodeV25(1, 0))
        params, _ = op.src_params()
        assert isinstance(params, NeuronParams)
        assert params.thres_pos == 1
        assert params.reset_mode == RM.MODE_NORMAL
        assert params.reset_v == 0
        assert params.output_type == OutputType.VALUE

    def test_sequential_lut_neuron_params(self):
        op = SequentialOp(comp=nn.Linear(16, 10), act=ANNNodeV25(lut=LutReLU()))
        params, _ = op.src_params()
        assert isinstance(params, NeuronParams)
        assert params.output_type == OutputType.VALUE

    def test_accumulate_neuron_params(self):
        op = AccumulateOp(
            comps=[nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(3, 8, 3, padding=1)],
            act=LIFNodeV25(tau=2, v_threshold=2),
            op_signs=(1, 1),
        )
        params, _ = op.src_params()
        assert params.thres_pos == 2
        assert params.leak_tau == -1

    def test_add_op_pass_through(self):
        op = PotentialAddOp(op_signs=(1, -1))
        params, _ = op.src_params()
        assert params.output_type == OutputType.POTENTIAL

    def test_standalone_comp_pass_through(self):
        op = StandaloneCompOp(comp=nn.Conv2d(3, 8, 3))
        params, _ = op.src_params()
        assert params.output_type == OutputType.POTENTIAL

    def test_standalone_activation_neuron_params(self):
        op = StandaloneActOp(act=IFNodeV25(1))
        params, _ = op.src_params()
        assert params.thres_pos == 1
        assert params.output_type == OutputType.VALUE

    def test_standalone_maxpool_unsigned_spike_bypass_neuron_params(self):
        op = StandaloneCompOp(comp=nn.MaxPool2d(2))
        op.signal_semantics.output_domain = SignalDomain.VALUE
        op.core_params.set_input_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))

        params, _ = op.src_params()
        assert params.output_type == OutputType.VALUE
        assert params.reset_mode == RM.MODE_NORMAL
        assert params.reset_v == 0
        assert params.thres_pos_mode == ThresholdPosMode.FIRE
        assert params.thres_neg_mode == ThresholdNegMode.FLOOR
        assert params.thres_pos == 1
        assert params.thres_neg == 0
        assert op.hw_lut_data is None

        act = CoreNeuronV25(
            reset_mode=params.reset_mode,
            reset_v=params.reset_v,
            thres_pos_mode=params.thres_pos_mode,
            thres_neg_mode=params.thres_neg_mode,
            thres_pos=params.thres_pos,
            thres_neg=params.thres_neg,
            lateral_inhi=params.lateral_inhi,
            leak_multi_sequence=params.leak_multi_sequence,
            leak_multi_input=params.leak_multi_input,
            leak_multi_mode=params.leak_multi_mode,
            leak_add_mode=params.leak_add_mode,
            leak_tau_shift=params.leak_tau,
            leak_v=params.leak_v,
            init_v=params.init_v,
        )
        outputs = [
            int(act(torch.tensor([x], dtype=torch.float32)).item())
            for x in [0, 1, 0, 1]
        ]
        assert outputs == [0, 1, 0, 1]
        assert int(act.v.item()) == 0

    def test_standalone_maxpool_wider_value_exports_identity_lut(self):
        op = StandaloneCompOp(comp=nn.MaxPool2d(2))
        op.signal_semantics.output_domain = SignalDomain.VALUE
        op.core_params.set_input_format((DataSign.UNSIGNED, DataWidth.WIDTH_4BIT))
        op.core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_4BIT))

        params, _ = op.src_params()
        data = op.hw_lut_data

        assert params.output_type == OutputType.VALUE
        assert data is not None
        values, indices = _lookup_hw_lut_data(
            data, DataWidth.WIDTH_4BIT, torch.arange(16, dtype=torch.int32)
        )
        assert indices.tolist() == list(range(0, 256, 16))
        assert values.tolist() == list(range(16))


class TestLutData:
    def test_hw_lut_data_requires_output_format(self):
        op = SequentialOp(
            comp=nn.Conv2d(3, 8, 3, padding=1), act=ANNNodeV25(lut=LutReLU())
        )

        with pytest.raises(ValueError, match="requires propagated output_sign"):
            _ = op.hw_lut_data

    @pytest.mark.parametrize(
        "op_factory",
        [
            lambda: SequentialOp(
                comp=nn.Conv2d(3, 8, 3, padding=1), act=ANNNodeV25(LutReLU())
            ),
            lambda: StandaloneActOp(ANNNodeV25(LutSigmoid())),
            lambda: AccumulateOp(
                comps=[
                    nn.Conv2d(3, 8, 3, padding=1),
                    nn.Conv2d(3, 8, 3, padding=1),
                ],
                act=ANNNodeV25(LutReLU()),
                op_signs=(1, 1),
            ),
        ],
        ids=["sequential", "standalone_act", "accumulate"],
    )
    def test_ann_ops_export_hw_lut_data(self, op_factory):
        op = op_factory()
        op.core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_8BIT))
        data = op.hw_lut_data

        assert data is not None
        assert data.thresholds.shape == (LUT_TABLE_SIZE,)
        assert data.values.shape == (LUT_TABLE_SIZE,)

    @pytest.mark.parametrize(
        "op_factory",
        [
            lambda: SequentialOp(comp=nn.Conv2d(3, 8, 3, padding=1), act=IFNodeV25()),
            lambda: PotentialAddOp(op_signs=(1, 1)),
            lambda: AccumulateOp(
                comps=[nn.Conv2d(3, 8, 3, padding=1)], act=IFNodeV25()
            ),
            lambda: StandaloneCompOp(comp=nn.Conv2d(3, 8, 3)),
        ],
        ids=["sequential_snn", "add", "accumulate_snn", "standalone_comp"],
    )
    def test_non_ann_lut_ops_do_not_export_hw_lut_data(self, op_factory):
        assert op_factory().hw_lut_data is None


class TestAvgPoolCompensation:
    def test_sequential_avgpool_lut_compensates(self):
        """AvgPool + LutReLU: logical LUT thresholds are scaled."""
        op = SequentialOp(comp=nn.AvgPool2d(2), act=ANNNodeV25(LutReLU()))
        assert op.act.lut is not None
        data = op.act.lut.logical_lut_data
        assert data is not None
        # k=2, window_size=4=2^2, scale=1.0: thresholds should be unchanged
        baseline = LutReLU().logical_lut_data
        assert torch.equal(data.thresholds, baseline.thresholds)

    def test_sequential_avgpool_lif_compensates_threshold(self):
        """AvgPool + LIF: neuron threshold is compensated via apply_avgpool_snn_compensation."""
        op = SequentialOp(comp=nn.AvgPool2d(3), act=LIFNodeV25(tau=2, v_threshold=1.0))
        # Apply compensation (window_size=9 for 3x3 pool)
        apply_avgpool_snn_compensation(op.act, window_size=9, decay_input=True)
        params, _ = op.src_params()
        # LIFNodeV25 default decay_input=True -> leak_multi_input=ENABLE
        # k=3, window_size=9, factor=9 (decay_input=True)
        # theta' = 0 + (1.0 - 0) * 9 = 9.0
        assert params.thres_pos == 9.0
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE

    def test_sequential_avgpool_neuron_params_leak_set(self):
        """AvgPool + LUT sets leak_tau for division-by-shift."""
        op = SequentialOp(comp=nn.AvgPool2d(2), act=ANNNodeV25(lut=LutReLU()))
        # Apply compensation (window_size=4 for 2x2 pool)
        assert op.act.lut is not None
        apply_avgpool_lut_compensation(op.act.lut, window_size=4)
        # Set leak params for division-by-shift (as done during AvgPool fusion)
        N = round(math.log2(4))
        op.act.leak_tau = -N
        op.act.leak_multi_input = LeakMultiInputMode.ENABLE
        op.act.leak_multi_mode = LeakMultiMode.DISABLE
        op.act.leak_multi_sequence = LeakMultiComparisonOrder.AFTER_COMPARE

        params, _ = op.src_params()
        # For ANN mode with power-of-2 window_size, leak_tau = -log2(window_size) = -2
        assert params.leak_tau == -2
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
