"""Deployment scheme helpers for AvgPool."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch
from paicorelib import DataWidth, LeakMultiInputMode

from ...ir.calc_params import NeuronParams
from ...ir.core_neuron import CoreNeuronV25
from .calibration import (
    _simulate_ideal_lif,
    _simulate_quantized_lif,
    calibrate_avgpool_threshold,
)
from .compensation import compensate_avgpool_neuron, compensate_sumpool_neuron

__all__ = [
    "AvgPoolDeployScheme",
    "AvgPoolLIFCandidateScore",
    "score_avgpool_lif_candidates",
    "select_avgpool_lif_candidate",
    "select_avgpool_lif_deployment",
]


class AvgPoolDeployScheme(Enum):
    """Supported deployment schemes for AvgPool patterns."""

    SHARED_CORE = auto()
    SPLIT_CORE_LIF_EXACT_SUM = auto()


@dataclass(frozen=True)
class AvgPoolLIFCandidateScore:
    """Score summary for one AvgPool+LIF deployment candidate."""

    scheme: AvgPoolDeployScheme
    uses_calibration: bool
    spike_mismatch: float
    rate_error: float
    resource_cost: float
    total_score: float


_LIF_PROBE_STEPS = 32
_DEFAULT_PROBE_SEEDS = (42, 43, 44)
# Resource scoring is intentionally a lightweight proxy rather than a full
# hardware cost model. Shared-core keeps AvgPool and LIF on one core, so it is
# used as the zero-cost baseline. Split-core exact-sum consumes an extra core
# and routing/state overhead, so it pays a fixed penalty that only serves to
# break near-ties in favour of the cheaper topology.
_SPLIT_CORE_RESOURCE_COST = 0.05


def _split_lif_exact_sum_feasible(
    pred_out_width: DataWidth | None, window_size: int
) -> bool:
    """Return True when Core 1 can emit the exact sum-domain code losslessly."""
    if pred_out_width == DataWidth.WIDTH_1BIT:
        return window_size <= 255
    if pred_out_width == DataWidth.WIDTH_2BIT:
        return window_size <= 127
    return False


def _make_eval_sum_input(
    pred_out_width: DataWidth | None, window_size: int, n_probe_steps: int, seed: int
) -> torch.Tensor:
    """Build one random sum-domain probe sequence for LIF candidate scoring."""
    if pred_out_width == DataWidth.WIDTH_2BIT:
        low, high = -window_size, window_size
    else:
        low, high = 0, window_size

    if n_probe_steps <= 1 or low == high:
        return torch.full((n_probe_steps,), low, dtype=torch.int32)

    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        low, high + 1, (n_probe_steps,), generator=generator, dtype=torch.int32
    )


def _make_eval_sum_input_bank(
    pred_out_width: DataWidth | None,
    window_size: int,
    n_probe_steps: int,
    probe_seeds: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    """Build a fixed-seed random probe bank for LIF candidate scoring."""
    seeds = tuple(_DEFAULT_PROBE_SEEDS if probe_seeds is None else probe_seeds)
    if len(seeds) == 0:
        raise ValueError("probe_seeds must contain at least one seed")

    return [
        _make_eval_sum_input(pred_out_width, window_size, n_probe_steps, seed)
        for seed in seeds
    ]


def _shared_core_baseline_threshold(
    act: CoreNeuronV25,
    window_size: int,
    decay_input: bool,
) -> int:
    """Return the analytically compensated shared-core threshold."""
    shared_params = compensate_avgpool_neuron(
        NeuronParams(
            thres_pos=act.thres_pos,
            thres_neg=act.thres_neg,
            reset_v=act.reset_v,
            init_v=act.init_v,
        ),
        window_size,
        decay_input,
        act.tau,
    )
    return int(shared_params.thres_pos)


def _simulate_shared_core_candidate(
    act: CoreNeuronV25,
    probe_sum_input: torch.Tensor,
    thres_pos: int,
) -> torch.Tensor:
    """Simulate one shared-core AvgPool+LIF candidate in the chip domain."""

    # Shared-core export always routes AvgPool through the leak-on-input path.
    return _simulate_quantized_lif(
        probe_sum_input,
        act.tau,
        thres_pos,
        act.reset_v,
        act.reset_mode,
        act.init_v,
        True,
    )


def _simulate_split_core_candidate(
    act: CoreNeuronV25,
    window_size: int,
    probe_sum_input: torch.Tensor,
    decay_input: bool,
) -> torch.Tensor:
    """Simulate the split-core exact-sum LIF candidate in the chip domain."""
    split_params = compensate_sumpool_neuron(
        NeuronParams(
            thres_pos=act.thres_pos,
            thres_neg=act.thres_neg,
            reset_v=act.reset_v,
            init_v=act.init_v,
        ),
        window_size,
    )
    return _simulate_quantized_lif(
        probe_sum_input,
        act.tau,
        split_params.thres_pos,
        split_params.reset_v,
        act.reset_mode,
        split_params.init_v,
        decay_input,
    )


def _score_candidate_over_probes(
    scheme: AvgPoolDeployScheme,
    uses_calibration: bool,
    act: CoreNeuronV25,
    window_size: int,
    probe_sum_inputs: Sequence[torch.Tensor],
    decay_input: bool,
    resource_cost: float,
    simulate_candidate: Callable[[torch.Tensor], torch.Tensor],
) -> AvgPoolLIFCandidateScore:
    """Score one candidate by averaging its metrics over a probe bank."""
    if len(probe_sum_inputs) == 0:
        raise ValueError("probe_sum_inputs must contain at least one probe")

    spike_mismatch = 0.0
    rate_error = 0.0

    for probe_sum_input in probe_sum_inputs:
        ref_spikes = _simulate_ideal_lif(
            probe_sum_input / window_size,
            act.tau,
            act.thres_pos,
            act.reset_v,
            act.reset_mode,
            decay_input,
            act.init_v,
        )
        candidate_spikes = simulate_candidate(probe_sum_input)
        spike_mismatch += (
            (candidate_spikes != ref_spikes).float().mean().item()
            if len(ref_spikes) > 0
            else 0.0
        )
        ref_rate = ref_spikes.float().mean().item() if len(ref_spikes) > 0 else 0.0
        cand_rate = (
            candidate_spikes.float().mean().item() if len(candidate_spikes) > 0 else 0.0
        )
        rate_error += abs(cand_rate - ref_rate)

    spike_mismatch /= len(probe_sum_inputs)
    rate_error /= len(probe_sum_inputs)
    total_score = rate_error + resource_cost

    return AvgPoolLIFCandidateScore(
        scheme, uses_calibration, spike_mismatch, rate_error, resource_cost, total_score
    )


def score_avgpool_lif_candidates(
    act: CoreNeuronV25,
    pred_out_width: DataWidth | None,
    window_size: int,
    allow_split_lif: bool = False,
    try_calibration: bool = False,
    *,
    n_probe_steps: int = _LIF_PROBE_STEPS,
    probe_seeds: tuple[int, ...] | None = None,
) -> list[AvgPoolLIFCandidateScore]:
    """Score feasible AvgPool+LIF deployment candidates.

    The current selector intentionally uses a lightweight compile-time probe
    instead of a full dataset-driven evaluation:

    - reference: ideal float ``AvgPool -> LIF`` on a fixed-seed random probe bank
    - shared-core: quantized chip dynamics after analytic compensation
    - shared-core + calibration: same topology, but with the offline threshold
      search applied independently on each probe in the same bank
    - split-core: exact-sum Core 1 plus quantized LIF Core 2

    This gives the compiler a concrete proxy for compensation quality while
    still preferring cheaper shared-core execution when the quality is tied.

    ``n_probe_steps`` controls the length of each probe in that compile-time
    bank rather than the deployed runtime horizon.

    Candidate selection is therefore driven by a compile-time proxy:

    - quality: compare mean firing-rate errors against the ideal reference
    - cost: a fixed resource penalty for split-core execution
    """
    decay_input = act.leak_multi_input == LeakMultiInputMode.ENABLE
    probe_sum_inputs = _make_eval_sum_input_bank(
        pred_out_width, window_size, n_probe_steps, probe_seeds
    )
    baseline_thres = _shared_core_baseline_threshold(act, window_size, decay_input)

    def simulate_shared_baseline(probe_sum_input: torch.Tensor) -> torch.Tensor:
        return _simulate_shared_core_candidate(act, probe_sum_input, baseline_thres)

    candidates = [
        _score_candidate_over_probes(
            AvgPoolDeployScheme.SHARED_CORE,
            False,
            act,
            window_size,
            probe_sum_inputs,
            decay_input,
            0.0,
            simulate_shared_baseline,
        )
    ]

    if try_calibration:

        def simulate_shared_calibrated(probe_sum_input: torch.Tensor) -> torch.Tensor:
            best_thres = calibrate_avgpool_threshold(
                act,
                window_size,
                baseline_thres,
                probe_sum_input,
                decay_input=decay_input,
            ).best_thres
            return _simulate_shared_core_candidate(act, probe_sum_input, best_thres)

        candidates.append(
            _score_candidate_over_probes(
                AvgPoolDeployScheme.SHARED_CORE,
                True,
                act,
                window_size,
                probe_sum_inputs,
                decay_input,
                0.0,
                simulate_shared_calibrated,
            )
        )

    if allow_split_lif and _split_lif_exact_sum_feasible(pred_out_width, window_size):

        def simulate_split_core(probe_sum_input: torch.Tensor) -> torch.Tensor:
            return _simulate_split_core_candidate(
                act, window_size, probe_sum_input, decay_input
            )

        candidates.append(
            _score_candidate_over_probes(
                AvgPoolDeployScheme.SPLIT_CORE_LIF_EXACT_SUM,
                False,
                act,
                window_size,
                probe_sum_inputs,
                decay_input,
                _SPLIT_CORE_RESOURCE_COST,
                simulate_split_core,
            )
        )

    return candidates


def select_avgpool_lif_candidate(
    act: CoreNeuronV25,
    pred_out_width: DataWidth | None,
    window_size: int,
    allow_split_lif: bool = False,
    try_calibration: bool = False,
    *,
    n_probe_steps: int = _LIF_PROBE_STEPS,
    probe_seeds: tuple[int, ...] | None = None,
) -> AvgPoolLIFCandidateScore:
    """Choose the best scored AvgPool+LIF candidate."""
    candidates = score_avgpool_lif_candidates(
        act,
        pred_out_width,
        window_size,
        allow_split_lif,
        try_calibration,
        n_probe_steps=n_probe_steps,
        probe_seeds=probe_seeds,
    )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.total_score,
            candidate.resource_cost,
            candidate.uses_calibration,
        ),
    )


def select_avgpool_lif_deployment(
    act: CoreNeuronV25,
    pred_out_width: DataWidth | None,
    window_size: int,
    allow_split_lif: bool = False,
    try_calibration: bool = False,
    *,
    n_probe_steps: int = _LIF_PROBE_STEPS,
    probe_seeds: tuple[int, ...] | None = None,
) -> AvgPoolDeployScheme:
    """Choose the best AvgPool+LIF deployment scheme from scored candidates.

    Ordering rules:

    1. lower ``total_score`` wins
    2. for equal scores, lower ``resource_cost`` wins
    3. if still tied, prefer the non-calibrated candidate

    In other words, the compiler first looks for the closest firing-rate match
    to the ideal floating-point reference, then prefers the cheaper and simpler
    topology when that firing-rate quality is effectively tied.
    """
    best = select_avgpool_lif_candidate(
        act,
        pred_out_width,
        window_size,
        allow_split_lif,
        try_calibration,
        n_probe_steps=n_probe_steps,
        probe_seeds=probe_seeds,
    )
    return best.scheme
