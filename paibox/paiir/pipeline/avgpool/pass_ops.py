"""AvgPool-related calibration pass operations."""

import torch
from paicorelib import LeakMultiInputMode

from ...ir.graph import PAIIRGraph
from ...ir.op_node import SequentialOp
from .calibration import CalibrationResult, calibrate_avgpool_threshold
from .utils import get_avgpool_divisor, get_pool_window_size, is_avgpool

__all__ = ["calibrate_avgpool_thresholds"]


def calibrate_avgpool_thresholds(
    graph: PAIIRGraph,
    n_steps: int = 32,
    input_range: tuple[int, int] | None = None,
    seed: int = 42,
    search_ratio: float = 0.3,
) -> dict[str, CalibrationResult]:
    """Refine shared-core AvgPool+LIF thresholds via offline integer search."""
    results: dict[str, CalibrationResult] = {}

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, SequentialOp):
            continue
        if not is_avgpool(node.comp):
            continue
        if not node.act.has_lif_dynamics:
            continue
        if isinstance(node.act.thres_pos, torch.Tensor):
            raise ValueError(
                "AvgPool threshold calibration does not support per-channel thres_pos"
            )

        sum_window_size = get_pool_window_size(node.comp)
        avg_divisor = get_avgpool_divisor(node.comp)
        avgpool_deploy_metadata = node.avgpool_deploy_metadata
        if (
            avgpool_deploy_metadata is not None
            and not avgpool_deploy_metadata.uses_calibration
        ):
            continue
        source_decay_input = (
            avgpool_deploy_metadata.source_decay_input
            if avgpool_deploy_metadata is not None
            else node.act.leak_multi_input == LeakMultiInputMode.ENABLE
        )
        # Baseline starts from the analytic shared-core rule; the calibration
        # sweep only searches a neighbourhood around this integer threshold.
        factor = avg_divisor if source_decay_input else avg_divisor / node.act.tau
        baseline = round(
            node.act.reset_v + (node.act.thres_pos - node.act.reset_v) * factor
        )
        node_input_range = (
            input_range if input_range is not None else (0, sum_window_size)
        )

        result = calibrate_avgpool_threshold(
            node.act,
            sum_window_size,
            baseline,
            avg_divisor=avg_divisor,
            n_steps=n_steps,
            input_range=node_input_range,
            search_ratio=search_ratio,
            seed=seed,
            decay_input=source_decay_input,
        )
        node.act.thres_pos = result.best_thres
        results[node.name] = result

    return results
