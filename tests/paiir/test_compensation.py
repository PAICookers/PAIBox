import pytest
import torch
from paicorelib import LeakMultiInputMode, LeakMultiMode
from torch import nn

from paibox.paiir.avgpool_compensation import (
    apply_avgpool_leak_params,
    compensate_avgpool_lut,
    compensate_avgpool_neuron,
    compensate_splitcore_avgpool_threshold,
)
from paibox.paiir.calc_params import LutData, NeuronParams
from paibox.paiir.op_node import _get_pool_window_size


class TestAvgPoolLeakParams:
    def test_window_size_power_of_2(self):
        """k=2 -> window_size=4, N=2, no approximation error."""
        params = NeuronParams()
        params = apply_avgpool_leak_params(params, window_size=4, is_ann=True)
        assert params.leak_tau == -2
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
        assert params.leak_multi_mode == LeakMultiMode.DISABLE

    def test_window_size_not_power_of_2(self):
        """k=3 -> window_size=9, N=round(log2(9))=3."""
        params = NeuronParams()
        params = apply_avgpool_leak_params(params, window_size=9)
        assert params.leak_tau == -3

    def test_snn_mode_no_extra_flags(self):
        """SNN mode (is_ann=False) does not set leak_multi_mode."""
        params = NeuronParams()
        original_mode = params.leak_multi_mode
        params = apply_avgpool_leak_params(params, window_size=4, is_ann=False)
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
        assert params.leak_multi_mode == original_mode


class TestCompensateAvgPoolLut:
    def test_window_size_power_of_2_no_scaling(self):
        """k=2 -> scale=1.0, thresholds unchanged."""
        lut = LutData(
            thresholds=torch.arange(256, dtype=torch.float32),
            values=torch.zeros(256),
        )
        orig_thresholds = lut.thresholds.clone()
        result = compensate_avgpool_lut(lut, window_size=4)
        assert torch.equal(result.thresholds, orig_thresholds)

    def test_window_size_9_scales_and_rounds_thresholds(self):
        """k=3 -> window_size=9, N=3, scale=9/8=1.125, rounded to 112."""
        orig = torch.ones(256, dtype=torch.float32) * 100
        lut = LutData(thresholds=orig.clone(), values=torch.zeros(256))
        result = compensate_avgpool_lut(lut, window_size=9)
        # 100 * 1.125 = 112.5 -> round -> 112.0
        assert torch.equal(result.thresholds, torch.full((256,), 112.0))

    def test_float_mode_no_rounding(self):
        """Float mode skips rounding."""
        orig = torch.ones(256, dtype=torch.float32) * 100
        lut = LutData(thresholds=orig.clone(), values=torch.zeros(256), is_float=True)
        result = compensate_avgpool_lut(lut, window_size=9)
        expected = 100 * 9.0 / 8.0
        assert torch.allclose(result.thresholds.float(), torch.full((256,), expected))

    def test_values_unchanged(self):
        """Compensation only affects thresholds, not values."""
        values = torch.arange(256, dtype=torch.float32)
        lut = LutData(
            thresholds=torch.ones(256, dtype=torch.float32) * 50,
            values=values.clone(),
        )
        compensate_avgpool_lut(lut, window_size=9)
        assert torch.equal(lut.values, values)


class TestCompensateAvgPoolNeuron:
    def test_lif_decay_input_true_r0(self):
        """theta' = theta * window_size."""
        params = NeuronParams(thres_pos=1.0, reset_v=0.0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=True, tau=2.0
        )
        assert result.thres_pos == 4.0

    def test_lif_decay_input_false_r0(self):
        """theta' = theta * window_size / tau."""
        params = NeuronParams(thres_pos=1.0, reset_v=0.0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=False, tau=2.0
        )
        assert result.thres_pos == 2.0

    def test_lif_nonzero_reset(self):
        """theta' = r + (theta - r) * window_size."""
        params = NeuronParams(thres_pos=3.0, reset_v=1.0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=True, tau=2.0
        )
        assert result.thres_pos == 1.0 + (3.0 - 1.0) * 4  # = 9.0

    def test_thres_neg_also_compensated(self):
        """Both positive and negative thresholds are compensated."""
        params = NeuronParams(thres_pos=2.0, thres_neg=-4.0, reset_v=0.0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=True, tau=1.0
        )
        assert result.thres_pos == 8.0
        assert result.thres_neg == -16.0


class TestCompensateSplitcoreAvgPoolThreshold:
    def test_window_size_power_of_2_no_change(self):
        """window_size=4 -> factor=1.0, thresholds unchanged."""
        params = NeuronParams(thres_pos=1.0, thres_neg=-2.0, reset_v=0.0)
        result = compensate_splitcore_avgpool_threshold(params, window_size=4)
        assert result.thres_pos == 1.0
        assert result.thres_neg == -2.0

    def test_window_size_9_scales_threshold(self):
        """window_size=9 -> N=3, factor=9/8=1.125."""
        params = NeuronParams(thres_pos=1.0, reset_v=0.0)
        result = compensate_splitcore_avgpool_threshold(params, window_size=9)
        assert result.thres_pos == pytest.approx(1.125)

    def test_nonzero_reset(self):
        """r!=0: theta' = r + (theta - r) * factor."""
        params = NeuronParams(thres_pos=3.0, reset_v=1.0)
        result = compensate_splitcore_avgpool_threshold(params, window_size=9)
        # theta' = 1 + (3 - 1) * 9/8 = 1 + 2.25 = 3.25
        assert result.thres_pos == pytest.approx(3.25)

    def test_both_thresholds_compensated(self):
        """Both thres_pos and thres_neg are compensated."""
        params = NeuronParams(thres_pos=2.0, thres_neg=-4.0, reset_v=0.0)
        result = compensate_splitcore_avgpool_threshold(params, window_size=9)
        assert result.thres_pos == pytest.approx(2.0 * 9 / 8)
        assert result.thres_neg == pytest.approx(-4.0 * 9 / 8)


class TestGetPoolWindowSize:
    def test_avgpool2d_int_kernel(self):
        assert _get_pool_window_size(nn.AvgPool2d(2)) == 4

    def test_avgpool2d_tuple_kernel(self):
        assert _get_pool_window_size(nn.AvgPool2d((3, 5))) == 15

    def test_avgpool1d(self):
        assert _get_pool_window_size(nn.AvgPool1d(3)) == 3
