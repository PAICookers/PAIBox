"""Structured node metadata for AvgPool deployment."""

from dataclasses import dataclass

__all__ = ["AvgPoolDeployMetadata"]


@dataclass(slots=True)
class AvgPoolDeployMetadata:
    """AvgPool-specific deployment facts captured during fusion."""

    source_decay_input: bool
    uses_calibration: bool
