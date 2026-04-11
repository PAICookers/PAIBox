"""AvgPool-specific deployment logic for PAIIR."""

from .calibration import CalibrationResult, calibrate_avgpool_threshold
from .deploy_scheme import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
    score_avgpool_lif_candidates,
    select_avgpool_lif_candidate,
    select_avgpool_lif_deployment,
)
from .pass_ops import calibrate_avgpool_thresholds
from .standalone_rewrite import rewrite_standalone_avgpools

__all__ = [
    "AvgPoolDeployScheme",
    "AvgPoolLIFCandidateScore",
    "CalibrationResult",
    "calibrate_avgpool_threshold",
    "calibrate_avgpool_thresholds",
    "score_avgpool_lif_candidates",
    "select_avgpool_lif_candidate",
    "select_avgpool_lif_deployment",
    "rewrite_standalone_avgpools",
]
