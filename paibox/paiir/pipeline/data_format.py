"""Data format inference for offline core parameters.

Provides functions to infer ``DataSign`` and ``DataWidth`` for the three
data-format groups in :class:`OfflineCoreParams`:

- **Output format** -- determined by the activation module configuration.
- **Weight format** -- determined by the quantized weight tensor values.
- **Input format** -- propagated from predecessor output formats.
"""

from collections.abc import Sequence

import torch
from paicorelib import DataSign, DataWidth, ThresholdNegMode

from ..ir.core_neuron import CoreNeuronV25
from ..ir.value_code import (
    SIGNED_VALUE_CODE_RANGES,
    UNSIGNED_VALUE_CODE_RANGES,
    fits_value_code_range,
)

__all__ = [
    "DataFormat",
    "fits_value_code_range",
    "infer_output_code_range",
    "infer_output_format",
    "infer_weight_format",
    "merge_data_formats",
]


DataFormat = tuple[DataSign, DataWidth]


def _infer_narrowest_range_format(
    value_min: int, value_max: int, sign: DataSign, label: str
) -> tuple[DataSign, DataWidth]:
    ranges = (
        SIGNED_VALUE_CODE_RANGES
        if sign == DataSign.SIGNED
        else UNSIGNED_VALUE_CODE_RANGES
    )

    for width, (lo, hi) in ranges.items():
        if lo <= value_min and value_max <= hi:
            return sign, width

    raise ValueError(
        f"{label} range [{value_min}, {value_max}] exceeds {sign.name.lower()} "
        "8-bit capacity"
    )


def infer_output_format(act: CoreNeuronV25) -> tuple[DataSign, DataWidth]:
    """Infer output data format from a :class:`CoreNeuronV25` activation.

    Rules:

    - SNN mode (``lut is None``) with ``thres_neg_mode == FIRE``:
      outputs {-1, 0, +1} -> ``SIGNED, WIDTH_2BIT``.
    - SNN mode with ``thres_neg_mode == FLOOR``:
      outputs {0, +1} -> ``UNSIGNED, WIDTH_1BIT``.
    - ANN mode (``lut is not None``) preserves the LUT-declared sign and uses
      the narrowest width that covers the stored LUT activation codes.
    - Float LUTs conservatively fall back to 8-bit because the deploy path uses
      integer LUT activation tables.
    """
    if act.is_snn:
        # SNN mode
        if act.thres_neg_mode == ThresholdNegMode.FIRE:
            return DataSign.SIGNED, DataWidth.WIDTH_2BIT  # including negative spike
        return DataSign.UNSIGNED, DataWidth.WIDTH_1BIT

    # ANN mode
    sign = DataSign.SIGNED if act.output_sign == 1 else DataSign.UNSIGNED
    lut = act.lut
    if lut is None or lut.is_float:
        return sign, DataWidth.WIDTH_8BIT

    value_min = int(lut.lut_values.min().item())
    value_max = int(lut.lut_values.max().item())
    return _infer_narrowest_range_format(value_min, value_max, sign, "LUT activation")


def infer_output_code_range(act: CoreNeuronV25) -> tuple[int, int] | None:
    """Infer the exact integer VALUE-code range emitted by an activation."""
    if act.is_snn:
        if act.thres_neg_mode == ThresholdNegMode.FIRE:
            return -1, 1
        return 0, 1

    lut = act.lut
    if lut is None:
        return None

    values = lut.lut_values.detach().to(torch.float32)
    if not torch.allclose(values, values.round()):
        return None

    value_min = int(values.min().item())
    value_max = int(values.max().item())
    if not fits_value_code_range(value_min, value_max):
        return None
    return value_min, value_max


def infer_weight_format(weight_min: int, weight_max: int) -> tuple[DataSign, DataWidth]:
    """Infer weight data format from quantized weight value range.

    Finds the narrowest ``(DataSign, DataWidth)`` that covers the range
    ``[weight_min, weight_max]``.

    Args:
        weight_min: Minimum weight value (integer).
        weight_max: Maximum weight value (integer).

    Returns:
        A ``(DataSign, DataWidth)`` tuple using the smallest width that
        fits the value range.

    Raises:
        ValueError: If the range exceeds 8-bit capacity.
    """
    sign = DataSign.SIGNED if weight_min < 0 else DataSign.UNSIGNED
    return _infer_narrowest_range_format(weight_min, weight_max, sign, "weight")


def merge_data_formats(
    formats: Sequence[tuple[DataSign, DataWidth]],
) -> tuple[DataSign, DataWidth]:
    """Merge multiple data formats by taking the widest representation.

    Used when a node has multiple predecessors whose output formats may
    differ.  The result is the smallest format that can represent all
    inputs without loss:

    - Sign: ``SIGNED`` if any input is signed.
    - Width: the maximum width among all inputs.

    Args:
        formats: Non-empty sequence of ``(DataSign, DataWidth)`` tuples.

    Returns:
        A merged ``(DataSign, DataWidth)`` tuple.
    """
    if not formats:
        raise ValueError("formats must not be empty")

    sign = max(f[0] for f in formats)  # SIGNED(1) > UNSIGNED(0)
    width = max(f[1] for f in formats)  # WIDTH_8BIT(3) > ... > WIDTH_1BIT(0)
    return sign, width
