"""Calibration utilities for AvgPool-related LIF deployment."""

import math
from dataclasses import dataclass

import torch
from paicorelib import RM, LeakMultiInputMode
from torch import Tensor

from ...ir.core_neuron import CoreNeuronV25

__all__ = ["CalibrationResult", "calibrate_avgpool_threshold"]


@dataclass
class CalibrationResult:
    """Result of AvgPool+LIF threshold calibration."""

    best_thres: int
    baseline_thres: int
    alpha: float
    best_error: float
    n_candidates: int


def _simulate_ideal_lif(
    inputs: Tensor,
    tau: float,
    thres: float,
    reset_v: float,
    reset_mode: RM,
    decay_input: bool,
    init_v: float,
) -> Tensor:
    """Reference branch: ideal float AvgPool -> LIF dynamics."""
    n_steps = len(inputs)
    spikes = torch.zeros(n_steps, dtype=torch.int8)

    v = init_v
    for t in range(n_steps):
        current = float(inputs[t].item())

        if reset_mode == RM.MODE_NORMAL:
            if decay_input:
                v = v + (current - (v - reset_v)) / tau
            else:
                v = v - (v - reset_v) / tau + current
        else:
            if decay_input:
                v = v + (current - v) / tau
            else:
                v = v * (1 - 1 / tau) + current

        if v >= thres:
            spikes[t] = 1
            if reset_mode == RM.MODE_LINEAR:
                v -= thres
            elif reset_mode == RM.MODE_NORMAL:
                v = reset_v

    return spikes


def _simulate_quantized_lif(
    inputs: Tensor,
    tau: float,
    thres: float,
    reset_v: float,
    reset_mode: RM,
    init_v: float,
    decay_input: bool,
) -> Tensor:
    """Quantized chip branch for a single LIF candidate.

    ``inputs`` are already expressed in the candidate's working domain:

    - shared-core AvgPool+LIF: sum domain, and the chip always enables
      ``leak_multi_input`` to emulate AvgPool division by shift
    - split-core exact-sum LIF: sum domain emitted losslessly by Core 1, while
      Core 2 preserves the original ``decay_input`` setting
    """
    n_steps = len(inputs)
    spikes = torch.zeros(n_steps, dtype=torch.int8)
    shift = math.ceil(math.log2(tau))

    v = int(init_v)
    thres_int = int(thres)
    reset_v_int = int(reset_v)
    for t in range(n_steps):
        current = int(inputs[t].item())
        if decay_input:
            v += current >> shift
        else:
            v += current

        if v >= thres_int:
            spikes[t] = 1
            if reset_mode == RM.MODE_LINEAR:
                v -= thres_int
            elif reset_mode == RM.MODE_NORMAL:
                v = reset_v_int

        v -= (v - reset_v_int) >> shift

    return spikes


def calibrate_avgpool_threshold(
    act: CoreNeuronV25,
    window_size: int,
    baseline_thres: int,
    avg_divisor: int | None = None,
    calibration_input: Tensor | None = None,
    n_steps: int = 32,
    input_range: tuple[int, int] | None = None,
    search_ratio: float = 0.3,
    seed: int = 42,
    decay_input: bool | None = None,
) -> CalibrationResult:
    """Refine shared-core AvgPool+LIF threshold via offline integer search.

    The search keeps topology fixed and only replaces the analytically derived
    integer threshold with a nearby candidate that better matches the reference
    firing rate over a finite horizon.
    """
    if baseline_thres < 0:
        raise ValueError(f"baseline_thres must be non-negative, got {baseline_thres}")
    if avg_divisor is None:
        avg_divisor = window_size

    if calibration_input is None:
        if input_range is None:
            input_range = (0, window_size)

        generator = torch.Generator().manual_seed(seed)
        calibration_input = torch.randint(
            input_range[0], input_range[1] + 1, (n_steps,), generator=generator
        )
    else:
        n_steps = len(calibration_input)

    tau = act.tau
    if decay_input is None:
        decay_input = act.leak_multi_input == LeakMultiInputMode.ENABLE
    reset_v = act.reset_v
    reset_mode = act.reset_mode
    init_v = act.init_v

    # Reference branch: the same sampled sequence interpreted in the AvgPool
    # domain, i.e. divide the sampled sum-domain current back by window_size.
    ref_spikes = _simulate_ideal_lif(
        calibration_input / avg_divisor,
        tau,
        act.thres_pos,
        reset_v,
        reset_mode,
        decay_input,
        init_v,
    )
    ref_rate = ref_spikes.sum().item() / n_steps

    assert search_ratio < 1
    search_min = max(0, int(round(baseline_thres * (1 - search_ratio))))
    candidates = list(range(search_min, baseline_thres + 1))

    best_thres = baseline_thres
    best_error = float("inf")

    for thres in candidates:
        # Chip branch: keep the sampled sequence in the sum domain and only vary
        # the threshold register that the shared-core implementation will use.
        chip_spikes = _simulate_quantized_lif(
            calibration_input, tau, thres, reset_v, reset_mode, init_v, True
        )
        chip_rate = chip_spikes.sum().item() / n_steps
        error = abs(chip_rate - ref_rate) / (ref_rate + 1e-6)

        if error < best_error or (error == best_error and thres > best_thres):
            best_error = error
            best_thres = thres

    alpha = 1 if baseline_thres == 0 else best_thres / baseline_thres
    return CalibrationResult(
        best_thres, baseline_thres, alpha, best_error, len(candidates)
    )
