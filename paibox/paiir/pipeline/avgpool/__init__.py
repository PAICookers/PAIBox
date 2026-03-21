"""AvgPool-specific deployment logic for PAIIR."""

from .calibration import CalibrationResult, calibrate_avgpool_threshold
from .compensation import (
    apply_avgpool_leak_params,
    apply_avgpool_lut_compensation,
    apply_avgpool_snn_compensation,
    apply_sumpool_snn_compensation,
    compensate_avgpool_lut,
    compensate_avgpool_lut_for_sumpool,
    compensate_avgpool_neuron,
    compensate_sumpool_neuron,
)
from .deploy_scheme import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
    score_avgpool_lif_candidates,
    select_avgpool_lif_candidate,
    select_avgpool_lif_deployment,
)
from .metadata import AvgPoolDeployMetadata
from .pass_ops import calibrate_avgpool_thresholds

__all__ = [
    "AvgPoolDeployScheme",
    "AvgPoolDeployMetadata",
    "AvgPoolLIFCandidateScore",
    "CalibrationResult",
    "apply_avgpool_leak_params",
    "apply_avgpool_lut_compensation",
    "apply_avgpool_snn_compensation",
    "apply_sumpool_snn_compensation",
    "calibrate_avgpool_threshold",
    "calibrate_avgpool_thresholds",
    "score_avgpool_lif_candidates",
    "select_avgpool_lif_candidate",
    "select_avgpool_lif_deployment",
    "compensate_avgpool_lut",
    "compensate_avgpool_lut_for_sumpool",
    "compensate_avgpool_neuron",
    "compensate_sumpool_neuron",
]
