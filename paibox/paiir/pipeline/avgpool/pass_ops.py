"""AvgPool-related calibration pass operations."""

from paicorelib import LeakMultiInputMode

from ...ir.graph import PAIIRGraph
from ...ir.op_node import SequentialOp, _get_pool_window_size, _is_avgpool
from .calibration import CalibrationResult, calibrate_avgpool_threshold

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
        if not _is_avgpool(node.comp):
            continue
        if not node.act.has_lif_dynamics:
            continue

        window_size = _get_pool_window_size(node.comp)
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
        factor = window_size if source_decay_input else window_size / node.act.tau
        baseline = round(
            node.act.reset_v + (node.act.thres_pos - node.act.reset_v) * factor
        )
        node_input_range = input_range if input_range is not None else (0, window_size)

        result = calibrate_avgpool_threshold(
            node.act,
            window_size,
            baseline,
            n_steps=n_steps,
            input_range=node_input_range,
            search_ratio=search_ratio,
            seed=seed,
            decay_input=source_decay_input,
        )
        node.act.thres_pos = result.best_thres
        results[node.name] = result

    return results
