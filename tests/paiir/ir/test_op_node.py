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
)
from torch import nn

from paibox.paiir.ir.add_ops import PotentialAddOp
from paibox.paiir.ir.calc_params import NeuronParams, OfflineCoreParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.lut_activation import LutReLU, LutSigmoid
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.pipeline.avgpool import (
    apply_avgpool_lut_compensation,
    apply_avgpool_snn_compensation,
)
from paibox.paiir.pipeline.passes import _infer_node_weight_format


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

    def test_public_ctor_does_not_accept_core_params(self):
        with pytest.raises(TypeError, match="core_params"):
            SequentialOp(
                comp=nn.Linear(16, 10), act=IFNodeV25(), core_params=OfflineCoreParams()
            )

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


class TestWeights:
    """Test weights property on OpNode subclasses."""

    def test_sequential_conv_returns_weight(self):
        conv = nn.Conv2d(3, 8, 3, padding=1)
        op = SequentialOp(comp=conv, act=ANNNodeV25(lut=LutReLU()))
        op.output_shape = (1, 8, 8, 8)
        ws = op.weights
        assert ws is not None
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
        op.output_shape = (1, 8)
        ws = op.weights
        assert ws is not None
        assert len(ws) == 1
        assert ws[0].dtype == torch.int8
        assert ws[0].shape == linear.weight.shape

    def test_standalone_comp_pool_returns_none(self):
        """Weightless ops (pool etc.) return None, not identity matrices."""
        op = StandaloneCompOp(comp=nn.AvgPool2d(2))
        op.output_shape = (1, 3, 4, 4)
        assert op.weights is None

    def test_add_op_returns_none(self):
        op = PotentialAddOp(op_signs=(1, -1))
        assert op.weights is None

    def test_public_add_ctor_does_not_accept_core_params(self):
        with pytest.raises(TypeError, match="core_params"):
            PotentialAddOp(op_signs=(1, -1), core_params=OfflineCoreParams())

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
        op.output_shape = (1, 1024, 1024)

        assert _infer_node_weight_format(op) == (
            DataSign.UNSIGNED,
            DataWidth.WIDTH_1BIT,
        )

    def test_add_weight_format_uses_implicit_identity_range(self):
        op = PotentialAddOp(op_signs=(1, -1))
        op.output_shape = (1, 1024, 1024)

        assert _infer_node_weight_format(op) == (
            DataSign.UNSIGNED,
            DataWidth.WIDTH_1BIT,
        )


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
        op = PotentialAddOp(op_signs=(1, -1))
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
        op = PotentialAddOp(op_signs=(1, 1))
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
        """AvgPool + LIF: neuron threshold is compensated via apply_avgpool_snn_compensation."""
        op = SequentialOp(comp=nn.AvgPool2d(3), act=LIFNodeV25(tau=2, v_threshold=1.0))
        # Apply compensation (window_size=9 for 3x3 pool)
        apply_avgpool_snn_compensation(op.act, window_size=9, decay_input=True)
        params = op.neuron_params
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

        params = op.neuron_params
        # For ANN mode with power-of-2 window_size, leak_tau = -log2(window_size) = -2
        assert params.leak_tau == -2
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
