import pytest
import torch
from paicorelib import DataWidth, LeakMultiInputMode, LeakMultiMode
from torch import nn

import paibox.paiir.pipeline.avgpool.deploy_scheme as deploy_scheme_mod
from paibox.paiir.ir.calc_params import LUT_TABLE_SIZE, LutData, NeuronParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, LIFNodeV25
from paibox.paiir.ir.lut_activation import LutReLU
from paibox.paiir.pipeline import compile_to_paiir
from paibox.paiir.pipeline.avgpool import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
    CalibrationResult,
    calibrate_avgpool_threshold,
    score_avgpool_lif_candidates,
    select_avgpool_lif_candidate,
    select_avgpool_lif_deployment,
)
from paibox.paiir.pipeline.avgpool.compensation import (
    apply_avgpool_leak_params,
    compensate_avgpool_lut,
    compensate_avgpool_lut_for_sumpool,
    compensate_avgpool_neuron,
    compensate_sumpool_neuron,
)


def make_lif_node(**overrides) -> LIFNodeV25:
    params = {"tau": 4, "decay_input": True, "v_threshold": 1, "v_reset": 0}
    params.update(overrides)
    return LIFNodeV25(**params)


@pytest.mark.parametrize("ann", [False, True], ids=["snn", "ann"])
def test_avgpool_rejects_vector_neuron_parameters(ann: bool) -> None:
    act = (
        ANNNodeV25(lut=LutReLU(), reset_v=torch.tensor([0.0, 1.0]))
        if ann
        else LIFNodeV25(tau=torch.tensor([2.0, 4.0]))
    )
    model = nn.Sequential(nn.AvgPool2d(2), act)

    with pytest.raises(ValueError, match="AvgPool deployment.*vector"):
        compile_to_paiir(model, torch.zeros(1, 2, 4, 4))


class TestAvgPoolLeakParams:
    @pytest.mark.parametrize(
        ("window_size", "expected_leak_tau"),
        [(4, -2), (9, -3)],
        ids=["power_of_two", "non_power_of_two"],
    )
    def test_ann_sets_expected_leak_tau(self, window_size, expected_leak_tau):
        params = NeuronParams()
        params = apply_avgpool_leak_params(params, window_size=window_size, is_ann=True)
        assert params.leak_tau == expected_leak_tau
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE
        assert params.leak_multi_mode == LeakMultiMode.DISABLE

    def test_snn_preserves_leak_tau(self):
        """SNN mode: leak_tau is preserved from neuron's tau setting."""
        params = NeuronParams(leak_tau=-2)  # neuron's tau=4
        params = apply_avgpool_leak_params(params, window_size=9, is_ann=False)
        assert params.leak_tau == -2  # unchanged
        assert params.leak_multi_input == LeakMultiInputMode.ENABLE


class TestCompensateAvgPoolLut:
    def test_window_size_power_of_2_no_scaling(self):
        """k=2 -> scale=1.0, thresholds unchanged."""
        lut = LutData(
            torch.arange(LUT_TABLE_SIZE, dtype=torch.int32), torch.zeros(LUT_TABLE_SIZE)
        )
        orig_thresholds = lut.thresholds.clone()
        result = compensate_avgpool_lut(lut, window_size=4)
        assert torch.equal(result.thresholds, orig_thresholds)

    def test_window_size_9_scales_and_rounds_thresholds(self):
        """k=3 -> window_size=9, N=3, scale=9/8=1.125, rounded to 112."""
        orig = torch.ones(LUT_TABLE_SIZE, dtype=torch.int32) * 100
        lut = LutData(orig.clone(), torch.zeros(LUT_TABLE_SIZE))
        result = compensate_avgpool_lut(lut, window_size=9)
        # 100 * 1.125 = 112.5 -> round -> 112.0
        assert torch.equal(result.thresholds, torch.full((LUT_TABLE_SIZE,), 112.0))

    def test_float_mode_no_rounding(self):
        """Float mode skips rounding."""
        orig = torch.ones(LUT_TABLE_SIZE, dtype=torch.float32) * 100
        lut = LutData(orig.clone(), torch.zeros(LUT_TABLE_SIZE), is_float=True)
        result = compensate_avgpool_lut(lut, window_size=9)
        expected = 100 * 9 / 8
        assert torch.allclose(
            result.thresholds, torch.full((LUT_TABLE_SIZE,), expected)
        )


class TestCompensateAvgPoolNeuron:
    def test_lif_decay_input_zero_reset(self):
        """theta' = theta * window_size."""
        params = NeuronParams(thres_pos=1, reset_v=0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=True, tau=2.0
        )
        assert result.thres_pos == 4

    def test_lif_no_decay_input_zero_reset(self):
        """theta' = theta * window_size / tau."""
        params = NeuronParams(thres_pos=1, reset_v=0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=False, tau=2.0
        )
        assert result.thres_pos == 2

    def test_lif_nonzero_reset(self):
        """theta' = r + (theta - r) * window_size."""
        params = NeuronParams(thres_pos=3, reset_v=1)
        result = compensate_avgpool_neuron(
            params, window_size=9, decay_input=True, tau=2.0
        )
        assert result.thres_pos == 1 + (3 - 1) * 9  # = 19

    def test_thres_neg_also_compensated(self):
        """Both positive and negative thresholds are compensated."""
        params = NeuronParams(thres_pos=2.0, thres_neg=-4.0, reset_v=0.0)
        result = compensate_avgpool_neuron(
            params, window_size=4, decay_input=True, tau=1.0
        )
        assert result.thres_pos == 8
        assert result.thres_neg == -16

    def test_rounds_to_int_by_default(self):
        """Default is_float=False rounds thresholds to integers."""
        params = NeuronParams(thres_pos=1.0, reset_v=0.0)
        # 1 * 3 / 2 = 1.5 -> round -> 2
        result = compensate_avgpool_neuron(
            params, window_size=3, decay_input=False, tau=2.0
        )
        assert result.thres_pos == 2

    def test_float_mode_preserves_precision(self):
        """is_float=True preserves floating-point thresholds."""
        params = NeuronParams(thres_pos=1.0, reset_v=0.0)
        # 1 * 3 / 2 = 1.5 (no rounding)
        result = compensate_avgpool_neuron(
            params, window_size=3, decay_input=False, tau=2.0, is_float=True
        )
        assert result.thres_pos == 1.5


class TestCompensateSumpoolNeuron:
    def test_scales_all_voltage_domain_params(self):
        params = NeuronParams(
            thres_pos=2.0,
            thres_neg=-8.0,
            reset_v=-1.0,
            init_v=3.0,
        )
        result = compensate_sumpool_neuron(params, window_size=4)

        assert result.thres_pos == 8
        assert result.thres_neg == -32
        assert result.reset_v == -4
        assert result.init_v == 12

    def test_float_mode_keeps_fractional_values(self):
        params = NeuronParams(
            thres_pos=1.5,
            thres_neg=-2.5,
            reset_v=0.5,
            init_v=-1.25,
        )
        result = compensate_sumpool_neuron(params, window_size=3, is_float=True)

        assert result.thres_pos == 4.5
        assert result.thres_neg == -7.5
        assert result.reset_v == 1.5
        assert result.init_v == -3.75


class TestSelectAvgPoolLIFDeployment:
    @pytest.mark.parametrize(
        ("act", "pred_out_width", "window_size", "allow_split_lif"),
        [
            (make_lif_node(tau=4.0, decay_input=True), DataWidth.WIDTH_1BIT, 9, False),
            (make_lif_node(tau=4.0, decay_input=True), DataWidth.WIDTH_1BIT, 9, True),
            (make_lif_node(tau=4.0, decay_input=False), DataWidth.WIDTH_1BIT, 9, True),
            (
                make_lif_node(tau=4.0, decay_input=False),
                DataWidth.WIDTH_1BIT,
                256,
                True,
            ),
            (make_lif_node(tau=4.0, decay_input=False), DataWidth.WIDTH_8BIT, 9, True),
        ],
        ids=[
            "default_shared",
            "quality_tie_prefers_shared",
            "split_gain_too_small",
            "exact_sum_u1_range_exceeded",
            "ann_predecessor_fallback",
        ],
    )
    def test_select_shared_core_for_representative_cases(
        self, act, pred_out_width, window_size, allow_split_lif
    ):
        deploy_scheme = select_avgpool_lif_deployment(
            act=act,
            pred_out_width=pred_out_width,
            window_size=window_size,
            allow_split_lif=allow_split_lif,
        )
        assert deploy_scheme == AvgPoolDeployScheme.SHARED_CORE

    def test_score_avgpool_lif_candidates_reports_shared_and_split(self):
        act = make_lif_node(tau=4.0, decay_input=False)
        candidates = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
        )

        assert len(candidates) == 3
        assert all(isinstance(c, AvgPoolLIFCandidateScore) for c in candidates)
        assert {(c.scheme, c.uses_calibration) for c in candidates} == {
            (AvgPoolDeployScheme.SHARED_CORE, False),
            (AvgPoolDeployScheme.SHARED_CORE, True),
            (AvgPoolDeployScheme.SPLIT_CORE_LIF_EXACT_SUM, False),
        }

    def test_candidate_total_score_uses_rate_error(self):
        act = make_lif_node(tau=4.0, decay_input=False)
        candidates = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
        )

        for candidate in candidates:
            assert candidate.total_score == pytest.approx(
                candidate.rate_error + candidate.resource_cost
            )

    def test_random_probe_bank_is_reproducible_with_default_seeds(self):
        act = make_lif_node(tau=3.0, decay_input=False)

        candidates_1 = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
        )
        candidates_2 = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
        )

        assert candidates_1 == candidates_2

    def test_probe_seed_bank_can_change_aggregated_scores(self):
        act = make_lif_node(tau=3.0, decay_input=False)

        candidates_a = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
            probe_seeds=(42, 43, 44),
        )
        candidates_b = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
            probe_seeds=(0, 1, 2),
        )

        assert candidates_a != candidates_b

    def test_candidate_rate_error_is_mean_over_probe_bank(self):
        act = make_lif_node(tau=3.0, decay_input=False)
        probe_seeds = (42, 43)
        candidates = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=False,
            probe_seeds=probe_seeds,
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        decay_input = act.leak_multi_input == LeakMultiInputMode.ENABLE
        baseline_thres = deploy_scheme_mod._shared_core_baseline_threshold(
            act, 9, decay_input
        )
        per_probe_errors = []
        for seed in probe_seeds:
            probe = deploy_scheme_mod._make_eval_sum_input(
                DataWidth.WIDTH_1BIT, 9, n_probe_steps=32, seed=seed
            )
            ref_spikes = deploy_scheme_mod._simulate_ideal_lif(
                probe / 9,
                act.tau,
                act.thres_pos,
                act.reset_v,
                act.reset_mode,
                decay_input,
                act.init_v,
            )
            shared_spikes = deploy_scheme_mod._simulate_shared_core_candidate(
                act, probe, baseline_thres
            )
            ref_rate = ref_spikes.float().mean().item()
            shared_rate = shared_spikes.float().mean().item()
            per_probe_errors.append(abs(shared_rate - ref_rate))

        assert candidate.rate_error == pytest.approx(
            sum(per_probe_errors) / len(per_probe_errors)
        )

    def test_avg_divisor_changes_candidate_scoring(self):
        act = make_lif_node(tau=4.0, decay_input=True)

        candidates_default = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=4,
            allow_split_lif=False,
        )
        candidates_divisor_one = score_avgpool_lif_candidates(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=4,
            allow_split_lif=False,
            avg_divisor=1,
        )

        assert candidates_default != candidates_divisor_one

    def test_select_avgpool_lif_candidate_returns_full_winner(self):
        act = make_lif_node(tau=5.0, decay_input=False)

        best = select_avgpool_lif_candidate(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
        )

        assert isinstance(best, AvgPoolLIFCandidateScore)
        assert best.scheme == AvgPoolDeployScheme.SHARED_CORE
        assert best.uses_calibration is True

    def test_select_avgpool_lif_candidate_prefers_uncalibrated_on_tie(self):
        act = make_lif_node(tau=4.0, decay_input=True)

        best = select_avgpool_lif_candidate(
            act=act,
            pred_out_width=DataWidth.WIDTH_1BIT,
            window_size=9,
            allow_split_lif=True,
            try_calibration=True,
        )

        assert best.scheme == AvgPoolDeployScheme.SHARED_CORE
        assert best.uses_calibration is False


class TestCompensateAvgPool1dRepresentative:
    @pytest.mark.parametrize("window_size", [1, 2, 3, 16])
    def test_sumpool_lut_scaling_representative_windows(self, window_size):
        lut = LutData(
            thresholds=torch.tensor([0, 1, 10, 100], dtype=torch.int32),
            values=torch.tensor([0, 1, 2, 3], dtype=torch.int8),
        )
        result = compensate_avgpool_lut_for_sumpool(lut, window_size)
        expected = torch.tensor([0, window_size, 10 * window_size, 100 * window_size])
        assert torch.equal(result.thresholds, expected)


class TestCalibrateAvgPoolThreshold:
    """Tests for calibrate_avgpool_threshold()."""

    @pytest.mark.parametrize(
        ("baseline_thres", "n_steps"),
        [(4, 100), (10, 100)],
        ids=["smoke_fields", "search_bounds"],
    )
    def test_result_fields_and_search_bounds(self, baseline_thres, n_steps):
        act = make_lif_node(tau=9.0)
        result = calibrate_avgpool_threshold(
            act, 4, baseline_thres, n_steps=n_steps, seed=42
        )

        assert isinstance(result, CalibrationResult)
        assert result.baseline_thres == baseline_thres
        assert result.best_error >= 0
        assert result.n_candidates > 0
        assert result.best_thres <= baseline_thres
        assert result.best_thres >= 0
        assert 0 < result.alpha <= 1.0

    def test_power_of_2_tau_no_change(self):
        """For tau=2,4,8 (power of 2), chip dynamics match ideal: best == baseline."""
        for tau in [2.0, 4.0, 8.0]:
            act = make_lif_node(tau=tau)
            window_size = 4
            baseline_thres = 4  # 1 * 4

            result = calibrate_avgpool_threshold(
                act, window_size, baseline_thres, n_steps=200, seed=42
            )

            # For power-of-2 tau, chip dynamics match ideal, so baseline should be optimal
            assert result.best_thres == baseline_thres, f"tau={tau}"
            assert result.alpha == 1.0, f"tau={tau}"

    def test_non_power_of_2_reduces_threshold(self):
        """For non-power-of-2 tau (e.g., 9), best threshold may be < baseline."""
        act = make_lif_node(tau=9.0)
        window_size = 4
        baseline_thres = 4

        result = calibrate_avgpool_threshold(
            act, window_size, baseline_thres, n_steps=500, seed=42
        )

        # For non-power-of-2 tau, calibration may find a better threshold <= baseline
        assert result.best_thres <= baseline_thres

    def test_custom_calibration_input(self):
        """Accepts custom calibration input tensor."""
        act = make_lif_node(tau=9.0)
        window_size = 4
        baseline_thres = 4
        custom_input = torch.randint(0, 5, (200,), dtype=torch.float32)

        result = calibrate_avgpool_threshold(
            act,
            window_size,
            baseline_thres,
            calibration_input=custom_input,
            seed=42,
        )

        assert isinstance(result, CalibrationResult)
        assert result.n_candidates > 0

    @pytest.mark.parametrize(
        ("search_ratio", "expected_candidates"),
        [(0.0, 1), (0.5, 6)],
        ids=["zero_ratio", "half_ratio"],
    )
    def test_search_ratio_controls_candidate_count(
        self, search_ratio, expected_candidates
    ):
        act = make_lif_node(tau=9.0)
        result = calibrate_avgpool_threshold(
            act, 4, 10, n_steps=100, seed=42, search_ratio=search_ratio
        )

        assert result.n_candidates == expected_candidates

    def test_decay_input_false(self):
        """Power-of-2 tau keeps the baseline for decay_input=False."""
        act = make_lif_node(tau=4.0, decay_input=False, v_threshold=4.0)
        baseline_thres = 4
        calibration_input = torch.ones(128, dtype=torch.float32)

        result = calibrate_avgpool_threshold(
            act,
            window_size=4,
            baseline_thres=baseline_thres,
            calibration_input=calibration_input,
        )

        assert result.best_thres == baseline_thres
        assert result.alpha == 1.0

    def test_baseline_zero_supported(self):
        """A zero baseline still yields a valid single-candidate search."""
        act = make_lif_node(tau=9.0, decay_input=False)

        result = calibrate_avgpool_threshold(
            act, window_size=4, baseline_thres=0, n_steps=100
        )

        assert result.best_thres == 0
        assert result.alpha == 1.0
        assert result.n_candidates == 1
