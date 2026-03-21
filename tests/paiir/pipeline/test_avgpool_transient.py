"""Transient error tests for SNN+LIF AvgPool shared-core deployment.

Tests the transient quantization error when LIF neurons use tau values that
are not powers of 2. Chip uses S = ceil(log2(tau)), causing decay 1/2^S
instead of 1/tau.

Error formula: transient_error(t=1) = |tau/2^S - 1|

Key insight: AvgPool's threshold compensation ensures steady-state accuracy,
so transient error is equivalent to standard LIF tau quantization error.
"""

import math

import pytest
import torch
from paicorelib import RM
from torch import Tensor


def compute_chip_trajectory(tau: float, current: float, n_steps: int) -> Tensor:
    """Chip voltage trajectory: v(t+1) = (1-1/2^S)*v(t) + current/2^S, S = ceil(log2(tau))."""
    S = math.ceil(math.log2(tau))
    v = torch.zeros(n_steps)
    for t in range(1, n_steps):
        v[t] = (1 - 1 / (1 << S)) * v[t - 1] + current / (1 << S)
    return v


def compute_model_trajectory(
    tau: float, current: float, n_steps: int, decay_input: bool = True
) -> Tensor:
    """Model voltage trajectory: v(t+1) = (1-1/tau)*v(t) + current/tau."""
    v = torch.zeros(n_steps)
    for t in range(1, n_steps):
        v[t] = (1 - 1 / tau) * v[t - 1] + (current / tau if decay_input else current)
    return v


def compute_theoretical_transient_error(tau: float) -> float:
    """Theoretical error at t=1: |tau/2^S - 1|"""
    S = math.ceil(math.log2(tau))
    return abs(tau / (1 << S) - 1)


def simulate_ideal_lif_with_reset(
    tau: float,
    inputs: Tensor,
    thres: float,
    reset_v: float = 0.0,
    reset_mode: RM = RM.MODE_NORMAL,
    decay_input: bool = True,
    init_v: float | None = None,
) -> Tensor:
    """Simulate ideal LIF neuron with spike reset (reference model).

    Float arithmetic, no multiplicative leak (decay is inherent in the update equation).
    Matches SpikingJelly LIFNode dynamics.

    Args:
        tau: LIF time constant.
        inputs: Input current sequence (n_steps,).
        thres: Firing threshold.
        reset_v: Reset voltage after spike (default 0.0).
        reset_mode: Reset mode (NORMAL or LINEAR, default NORMAL).
        decay_input: Whether input is divided by tau (default True).
        init_v: Initial membrane potential (default reset_v).

    Returns:
        Binary spike train (n_steps,).
    """
    n_steps = len(inputs)
    spikes = torch.zeros(n_steps)

    if init_v is None:
        init_v = reset_v
    v = float(init_v)

    for t in range(n_steps):
        current = float(inputs[t])

        # Charge phase with ideal decay (match SpikingJelly LIFNode)
        # Hard reset (NORMAL): decay towards reset_v
        # Soft reset (LINEAR): decay towards 0
        if reset_mode == RM.MODE_NORMAL:
            # Hard reset: v = v + (current - (v - reset_v)) / tau
            if decay_input:
                v = v + (current - (v - reset_v)) / tau
            else:
                v = v - (v - reset_v) / tau + current
        else:
            # Soft reset: v = v + (current - v) / tau (decay towards 0)
            if decay_input:
                v = v + (current - v) / tau
            else:
                v = v * (1 - 1 / tau) + current

        if v >= thres:
            spikes[t] = 1
            if reset_mode == RM.MODE_LINEAR:
                v -= thres
            elif reset_mode == RM.MODE_NORMAL:
                v = float(reset_v)

    return spikes


def simulate_chip_lif_with_reset(
    tau: float,
    inputs: Tensor,
    thres: float,
    reset_v: float = 0.0,
    reset_mode: RM = RM.MODE_NORMAL,
    init_v: float | None = None,
) -> Tensor:
    """Simulate chip LIF neuron with spike reset (hardware model).

    Integer arithmetic with bit-shift operations.
    Multiplicative leak is applied after threshold comparison.

    Args:
        tau: LIF time constant (used to compute shift S = ceil(log2(tau))).
        inputs: Input current sequence (n_steps,).
        thres: Firing threshold.
        reset_v: Reset voltage after spike (default 0.0).
        reset_mode: Reset mode (NORMAL or LINEAR, default NORMAL).
        init_v: Initial membrane potential (default reset_v).

    Returns:
        Binary spike train (n_steps,).
    """
    n_steps = len(inputs)
    S = math.ceil(math.log2(tau))

    spikes = torch.zeros(n_steps)

    if init_v is None:
        init_v = reset_v
    v = int(init_v)
    thres_int = int(thres)
    reset_v_int = int(reset_v)

    for t in range(n_steps):
        current = int(inputs[t])

        # Charge phase: v += current >> S (integer right shift)
        v += current >> S

        # Threshold comparison and reset
        if v >= thres_int:
            spikes[t] = 1
            if reset_mode == RM.MODE_LINEAR:
                v -= thres_int
            elif reset_mode == RM.MODE_NORMAL:
                v = reset_v_int

        # Post-comparison multiplicative leak: v -= (v - reset_v) >> S
        v -= (v - reset_v_int) >> S

    return spikes


class TestTauQuantizationError:
    """Test tau quantization error boundaries."""

    @pytest.mark.parametrize(
        "tau, expected_S, expected_error",
        [
            (2.0, 1, 0.0),
            (3.0, 2, 0.25),
            (4.0, 2, 0.0),
            (5.0, 3, 0.375),
            (6.0, 3, 0.25),
            (7.0, 3, 0.125),
            (8.0, 3, 0.0),
            (9.0, 4, 0.4375),
        ],
    )
    def test_transient_error_formula(self, tau, expected_S, expected_error):
        """t=1 error = |tau/2^S - 1|"""
        assert math.ceil(math.log2(tau)) == expected_S
        assert compute_theoretical_transient_error(tau) == expected_error

    @pytest.mark.parametrize("tau", [2, 4, 8, 16, 32])
    def test_power_of_2_no_error(self, tau):
        """Power-of-2 tau has zero quantization error."""
        assert compute_theoretical_transient_error(tau) == 0

    @pytest.mark.parametrize("tau", [3, 5, 6, 7, 9])
    def test_non_power_of_2_has_error(self, tau):
        """Non-power-of-2 tau has quantization error in (0, 0.5)."""
        error = compute_theoretical_transient_error(tau)
        assert 0 < error < 0.5


class TestThresholdCompensation:
    """Test threshold compensation formula correctness."""

    @pytest.mark.parametrize(
        "thres, reset_v, window_size, decay_input, tau, expected",
        [
            (1, 0, 4, True, 2, 4),
            (1, 0, 9, True, 2, 9),
            (3, 1, 4, True, 2, 9),
            (1, 0, 4, False, 2, 2),
            (3, 1, 9, False, 3, 7),
        ],
    )
    def test_compensation_formula(
        self, thres, reset_v, window_size, decay_input, tau, expected
    ):
        """Verify thres' = r + (thres-r)*k^2 (or /tau if decay_input=False)."""
        if decay_input:
            thres_comp = reset_v + (thres - reset_v) * window_size
        else:
            thres_comp = reset_v + (thres - reset_v) * window_size / tau
        assert thres_comp == expected


class TestCompensationEffectiveness:
    """Verify compensation makes chip output match original model.

    Scenario:
    - Original: LIF with threshold thres, input current
    - Chip (shared-core with AvgPool): receives summed input current*k^2,
      threshold thres' = thres * k^2

    The compensation should make the chip's firing behavior match the original.
    """

    @pytest.mark.parametrize("tau", [3, 5, 6, 7])
    def test_steady_state_normalized_voltage_match(self, tau):
        """At steady state, normalized voltages match: v_chip/thres' == v_model/thres."""
        current = 1.0
        window_size = 9
        thres = 10.0
        thres_comp = thres * window_size
        n_steps = 1000

        # Chip receives summed input, model receives original
        v_chip = compute_chip_trajectory(tau, current * window_size, n_steps)
        v_model = compute_model_trajectory(tau, current, n_steps)

        v_chip_ss = v_chip[-100:].mean()
        v_model_ss = v_model[-100:].mean()

        chip_norm = v_chip_ss / thres_comp
        model_norm = v_model_ss / thres

        assert torch.allclose(
            chip_norm, model_norm, atol=0.01
        ), f"Normalized voltages don't match: chip={chip_norm:.4f}, model={model_norm:.4f}"

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_power_of_2_identical_normalized_trajectory(self, tau):
        """Power-of-2 tau: identical trajectories in normalized domain at all times."""
        current = 1.0
        window_size = 9
        thres = 10.0
        thres_comp = thres * window_size
        n_steps = 100

        v_chip = compute_chip_trajectory(tau, current * window_size, n_steps)
        v_model = compute_model_trajectory(tau, current, n_steps)

        v_chip_norm = v_chip / thres_comp
        v_model_norm = v_model / thres

        assert torch.allclose(v_chip_norm, v_model_norm, atol=1e-10)

    @pytest.mark.parametrize("tau", [3, 5, 6, 7])
    def test_first_spike_time_bounded(self, tau):
        """First spike time difference is bounded by 2 steps."""
        current = 5.0
        window_size = 9
        thres = 10.0
        thres_comp = thres * window_size
        n_steps = 100

        v_chip = compute_chip_trajectory(tau, current * window_size, n_steps)
        v_model = compute_model_trajectory(tau, current, n_steps)

        chip_spike = (v_chip >= thres_comp).nonzero()
        model_spike = (v_model >= thres).nonzero()

        if len(chip_spike) > 0 and len(model_spike) > 0:
            diff = abs(chip_spike[0].item() - model_spike[0].item())
            assert diff <= 2, f"Spike time diff too large: {diff}"


class TestSteadyStateBehavior:
    """Test steady-state voltage convergence (without compensation scaling)."""

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_power_of_2_identical_trajectory(self, tau):
        """Power-of-2 tau: identical trajectories at all times."""
        current = 1.0
        v_chip = compute_chip_trajectory(tau, current, 100)
        v_model = compute_model_trajectory(tau, current, 100)
        assert torch.equal(v_chip, v_model)

    @pytest.mark.parametrize("tau", [3, 5, 7])
    def test_transient_path_differs(self, tau):
        """Non-power-of-2 tau: different paths to steady state."""
        current = 1.0
        v_chip = compute_chip_trajectory(tau, current, 10)
        v_model = compute_model_trajectory(tau, current, 10)

        S = math.ceil(math.log2(tau))
        assert torch.allclose(v_model[1], torch.tensor(current / tau))
        assert torch.allclose(v_chip[1], torch.tensor(current / (1 << S)))
        assert not torch.allclose(v_chip[1], v_model[1])

    @pytest.mark.parametrize("tau", [2, 3, 4, 5, 6, 7, 8])
    def test_steady_state_voltage_identical(self, tau):
        """Both chip and model converge to v_ss = current (independent of decay factor)."""
        current = 1.0
        n_steps = 1000

        v_chip = compute_chip_trajectory(tau, current, n_steps)
        v_model = compute_model_trajectory(tau, current, n_steps)

        assert torch.allclose(v_chip[-100:].mean(), torch.tensor(current), atol=0.01)
        assert torch.allclose(v_model[-100:].mean(), torch.tensor(current), atol=0.01)


class TestFiringRateCompensation:
    """Test that compensation makes chip firing rate match model.

    Test scenario for shared-core AvgPool deployment:
    - Input: integer values (simulating summed spikes or quantized features)
    - Model: input I, threshold thres (reference)
    - Chip: input I*window_size, threshold thres*window_size (compensated)

    Due to tau quantization (S = ceil(log2(tau))), chip and model have different
    decay dynamics. Compensation fixes the threshold scaling issue, but for
    non-power-of-2 tau, transient error from tau quantization still causes
    firing rate differences.
    """

    @pytest.mark.parametrize("tau", [3, 5, 6, 7])
    def test_firing_rate_error_bounded(self, tau):
        """After compensation, chip firing rate error should be bounded (<50%).

        Compares compensated chip output against reference model.
        """
        torch.manual_seed(42)

        n_steps = 2000
        window_size = 9
        thres = 8

        inputs = torch.randint(0, 16, (n_steps,))

        # Reference model: original input, original threshold
        model_spikes = simulate_ideal_lif_with_reset(tau, inputs, thres)
        model_rate = model_spikes.mean()

        # Chip with compensation: summed input, compensated threshold
        chip_spikes = simulate_chip_lif_with_reset(
            tau, inputs * window_size, thres * window_size
        )
        chip_rate = chip_spikes.mean()

        # Relative error should be bounded
        relative_error = abs(chip_rate - model_rate) / (model_rate + 1e-6)
        assert relative_error < 0.50, (
            f"Firing rate mismatch for tau={tau}: "
            f"model={model_rate:.4f}, chip={chip_rate:.4f}, error={relative_error:.1%}"
        )

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_power_of_2_firing_rate_bounded(self, tau):
        """Power-of-2 tau: firing rates should be close but not exact due to multiplicative leak.

        The chip applies multiplicative leak after threshold comparison, while
        the reference model has decay inherent in the charge equation. This
        causes differences even for power-of-2 τ.
        """
        torch.manual_seed(42)

        n_steps = 1000
        window_size = 9
        thres = 8

        inputs = torch.randint(0, 16, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(tau, inputs, thres)
        model_rate = model_spikes.mean()

        chip_spikes = simulate_chip_lif_with_reset(
            tau, inputs * window_size, thres * window_size
        )
        chip_rate = chip_spikes.mean()

        # For power-of-2 tau, rates should be close but not exact
        relative_error = abs(chip_rate - model_rate) / (model_rate + 1e-6)
        assert relative_error < 0.10, (
            f"Power-of-2 tau firing rate mismatch: "
            f"model={model_rate:.4f}, chip={chip_rate:.4f}"
        )

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_power_of_2_spike_timing_bounded(self, tau):
        """Power-of-2 tau: spike timing should be close but not exact due to multiplicative leak."""
        torch.manual_seed(42)

        n_steps = 1000
        window_size = 9
        thres = 8

        inputs = torch.randint(0, 16, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(tau, inputs, thres)
        chip_spikes = simulate_chip_lif_with_reset(
            tau, inputs * window_size, thres * window_size
        )

        # Compute spike timing match rate (within 2 timesteps tolerance)
        model_times = (model_spikes > 0).nonzero(as_tuple=True)[0]
        chip_times = (chip_spikes > 0).nonzero(as_tuple=True)[0]

        if len(model_times) == 0:
            match_rate = 1.0 if len(chip_times) == 0 else 0.0
        else:
            matched = sum(
                1
                for m in model_times
                if len(chip_times) > 0 and torch.abs(chip_times - m).min() <= 2
            )
            match_rate = matched / len(model_times)

        # For power-of-2 τ, match rate should be high but not 100%
        assert (
            match_rate > 0.80
        ), f"Spike match rate too low for tau={tau}: {match_rate:.1%}"

    @pytest.mark.parametrize("tau", [3, 5, 6, 7])
    def test_non_power_of_2_spike_timing_bounded(self, tau):
        """Non-power-of-2 tau: spike timing match rate is limited by tau quantization.

        Uses lower threshold (thres=4) to ensure sufficient spike activity.
        Match rates are typically 80%+ for thres=4.
        """
        torch.manual_seed(42)

        n_steps = 2000
        window_size = 9
        thres = 4  # Lower threshold for sufficient spike activity

        inputs = torch.randint(0, 16, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(tau, inputs, thres)
        chip_spikes = simulate_chip_lif_with_reset(
            tau, inputs * window_size, thres * window_size
        )

        # Compute spike timing match rate (within 2 timesteps tolerance)
        model_times = (model_spikes > 0).nonzero(as_tuple=True)[0]
        chip_times = (chip_spikes > 0).nonzero(as_tuple=True)[0]

        if len(model_times) == 0:
            match_rate = 1.0 if len(chip_times) == 0 else 0.0
        else:
            matched = sum(
                1
                for m in model_times
                if len(chip_times) > 0 and torch.abs(chip_times - m).min() <= 2
            )
            match_rate = matched / len(model_times)

        # Typical values with thres=4: tau=3 -> ~98%, tau=5/6/7 -> ~82-86%
        assert (
            match_rate > 0.75
        ), f"Spike match rate too low for tau={tau}: {match_rate:.1%}"


class TestResetVCompensation:
    """Test reset_v ≠ 0 threshold compensation.

    When reset_v ≠ 0, the compensation formula becomes:
        thres' = reset_v + (thres - reset_v) * window_size

    IMPORTANT: For reset_v ≠ 0, exact spike matching is NOT possible because
    the chip's reset voltage is fixed at reset_v (cannot scale to reset_v*k).
    This is a fundamental hardware limitation of shared-core AvgPool deployment.

    Tests verify bounded error rather than exact match for reset_v ≠ 0.
    Uses small reset_v relative to threshold to keep error bounded.
    """

    @pytest.mark.parametrize("tau, reset_v", [(3, 1), (5, 1), (6, 1), (7, 1)])
    def test_reset_v_firing_rate_bounded(self, tau, reset_v):
        """reset_v ≠ 0: firing rate error should be bounded after compensation.

        Due to hardware limitation (reset_v cannot scale), error is larger than
        reset_v = 0 case. Uses small reset_v (1) to keep error bounded.
        """
        torch.manual_seed(42)

        n_steps = 2000
        window_size = 9
        thres = 20  # Higher threshold to keep reset_v/thres ratio small
        # Compensation formula: thres' = reset_v + (thres - reset_v) * window_size
        thres_comp = reset_v + (thres - reset_v) * window_size

        inputs = torch.randint(0, 16, (n_steps,))

        # Reference model: original input, original threshold, with reset_v
        model_spikes = simulate_ideal_lif_with_reset(
            tau, inputs, thres, reset_v=reset_v, decay_input=True
        )
        model_rate = model_spikes.mean()

        # Chip with compensation: summed input, compensated threshold
        # NOTE: reset_v stays fixed (cannot scale) - this is the hardware limitation
        chip_spikes = simulate_chip_lif_with_reset(
            tau,
            inputs * window_size,
            thres_comp,
            reset_v=reset_v,  # Fixed reset_v, NOT reset_v * window_size
        )
        chip_rate = chip_spikes.mean()

        # Relative error should be bounded
        relative_error = abs(chip_rate - model_rate) / (model_rate + 1e-6)
        assert relative_error < 0.60, (
            f"Firing rate mismatch for tau={tau}, reset_v={reset_v}: "
            f"model={model_rate:.4f}, chip={chip_rate:.4f}, error={relative_error:.1%}"
        )

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_reset_v_spike_timing_bounded(self, tau):
        """Power-of-2 tau with reset_v ≠ 0: spike timing match rate is limited.

        Even for power-of-2 tau, exact match is impossible due to fixed reset_v.
        Match rate is typically 70-90% due to this hardware limitation.
        """
        torch.manual_seed(42)

        n_steps = 1000
        window_size = 9
        reset_v = 1
        thres = 20
        thres_comp = reset_v + (thres - reset_v) * window_size

        inputs = torch.randint(0, 16, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(
            tau, inputs, thres, reset_v=reset_v, decay_input=True
        )
        chip_spikes = simulate_chip_lif_with_reset(
            tau,
            inputs * window_size,
            thres_comp,
            reset_v=reset_v,
        )

        # Compute spike timing match rate (within 2 timesteps tolerance)
        model_times = (model_spikes > 0).nonzero(as_tuple=True)[0]
        chip_times = (chip_spikes > 0).nonzero(as_tuple=True)[0]

        if len(model_times) == 0:
            match_rate = 1.0 if len(chip_times) == 0 else 0.0
        else:
            matched = sum(
                1
                for m in model_times
                if len(chip_times) > 0 and torch.abs(chip_times - m).min() <= 2
            )
            match_rate = matched / len(model_times)

        # Lower threshold for reset_v ≠ 0 due to hardware limitation
        assert (
            match_rate > 0.65
        ), f"Spike match rate too low for tau={tau}, reset_v={reset_v}: {match_rate:.1%}"


class TestDecayInputFalseCompensation:
    """Test decay_input=False threshold compensation.

    When decay_input=False, the compensation formula becomes:
        thres' = reset_v + (thres - reset_v) * window_size / tau

    This is because the input is NOT divided by tau in the model.
    """

    @pytest.mark.parametrize("tau", [3, 5, 6, 7])
    def test_decay_input_false_firing_rate_bounded(self, tau):
        """decay_input=False: firing rate error should be bounded after compensation."""
        torch.manual_seed(42)

        n_steps = 2000
        window_size = 9
        thres = 4
        reset_v = 0
        # Compensation formula for decay_input=False
        thres_comp = reset_v + (thres - reset_v) * window_size / tau

        inputs = torch.randint(0, 4, (n_steps,))

        # Reference model: decay_input=False
        model_spikes = simulate_ideal_lif_with_reset(
            tau, inputs, thres, reset_v=reset_v, decay_input=False
        )
        model_rate = model_spikes.mean()

        # Chip with compensation: summed input, compensated threshold
        chip_spikes = simulate_chip_lif_with_reset(
            tau,
            inputs * window_size,
            thres_comp,
            reset_v=reset_v * window_size / tau,
        )
        chip_rate = chip_spikes.mean()

        # Relative error should be bounded
        relative_error = abs(chip_rate - model_rate) / (model_rate + 1e-6)
        assert relative_error < 0.50, (
            f"Firing rate mismatch for tau={tau} (decay_input=False): "
            f"model={model_rate:.4f}, chip={chip_rate:.4f}, error={relative_error:.1%}"
        )

    @pytest.mark.parametrize("tau", [2, 4, 8])
    def test_decay_input_false_power_of_2_bounded(self, tau):
        """Power-of-2 tau with decay_input=False: spike timing should be close but not exact."""
        torch.manual_seed(42)

        n_steps = 1000
        window_size = 9
        thres = 4
        reset_v = 0
        thres_comp = reset_v + (thres - reset_v) * window_size / tau

        inputs = torch.randint(0, 4, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(
            tau, inputs, thres, reset_v=reset_v, decay_input=False
        )
        chip_spikes = simulate_chip_lif_with_reset(
            tau,
            inputs * window_size,
            thres_comp,
            reset_v=reset_v * window_size / tau,
        )

        # Check spike timing match rate instead of exact match
        model_times = (model_spikes > 0).nonzero(as_tuple=True)[0]
        chip_times = (chip_spikes > 0).nonzero(as_tuple=True)[0]

        if len(model_times) == 0:
            match_rate = 1.0 if len(chip_times) == 0 else 0.0
        else:
            matched = sum(
                1
                for m in model_times
                if len(chip_times) > 0 and torch.abs(chip_times - m).min() <= 2
            )
            match_rate = matched / len(model_times)

        assert (
            match_rate > 0.90
        ), f"Spike match rate too low for tau={tau} (decay_input=False): {match_rate:.1%}"

    @pytest.mark.parametrize("tau, reset_v", [(3, 1), (5, 1), (6, 2)])
    def test_decay_input_false_with_reset_v_bounded(self, tau, reset_v):
        """decay_input=False with reset_v ≠ 0: firing rate error bounded.

        NOTE: reset_v stays fixed on chip (cannot scale) - this is a hardware limitation.
        The threshold compensation formula is: θ' = r + (θ-r)·k²/τ
        """
        torch.manual_seed(42)

        n_steps = 2000
        window_size = 9
        thres = 6
        # Threshold compensation: thres' = reset_v + (thres - reset_v) * window_size / tau
        thres_comp = reset_v + (thres - reset_v) * window_size / tau
        # reset_v stays FIXED - cannot be scaled on chip
        reset_v_chip = reset_v

        inputs = torch.randint(0, 4, (n_steps,))

        model_spikes = simulate_ideal_lif_with_reset(
            tau, inputs, thres, reset_v=reset_v, decay_input=False
        )
        model_rate = model_spikes.mean()

        chip_spikes = simulate_chip_lif_with_reset(
            tau,
            inputs * window_size,
            thres_comp,
            reset_v=reset_v_chip,
        )
        chip_rate = chip_spikes.mean()

        relative_error = abs(chip_rate - model_rate) / (model_rate + 1e-6)
        # Larger tolerance due to fixed reset_v limitation
        assert relative_error < 0.70, (
            f"Firing rate mismatch for tau={tau}, reset_v={reset_v} (decay_input=False): "
            f"model={model_rate:.4f}, chip={chip_rate:.4f}, error={relative_error:.1%}"
        )
