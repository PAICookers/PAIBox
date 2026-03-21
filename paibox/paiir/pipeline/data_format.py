"""Data format inference for offline core parameters.

Provides functions to infer ``DataSign`` and ``DataWidth`` for the three
data-format groups in :class:`OfflineCoreParams`:

- **Output format** -- determined by the activation module configuration.
- **Weight format** -- determined by the quantized weight tensor values.
- **Input format** -- propagated from predecessor output formats.
"""

from collections.abc import Sequence

from paicorelib import DataSign, DataWidth, ThresholdNegMode

from ..ir.core_neuron import CoreNeuronV25

__all__ = [
    "DataFormat",
    "infer_output_format",
    "infer_weight_format",
    "merge_data_formats",
]


DataFormat = tuple[DataSign, DataWidth]

# Bit ranges for each DataWidth (signed)
_SIGNED_RANGES: dict[DataWidth, tuple[int, int]] = {
    DataWidth.WIDTH_1BIT: (-1, 0),
    DataWidth.WIDTH_2BIT: (-2, 1),
    DataWidth.WIDTH_4BIT: (-8, 7),
    DataWidth.WIDTH_8BIT: (-128, 127),
}

# Bit ranges for each DataWidth (unsigned)
_UNSIGNED_RANGES: dict[DataWidth, tuple[int, int]] = {
    DataWidth.WIDTH_1BIT: (0, 1),
    DataWidth.WIDTH_2BIT: (0, 3),
    DataWidth.WIDTH_4BIT: (0, 15),
    DataWidth.WIDTH_8BIT: (0, 255),
}


def infer_output_format(act: CoreNeuronV25) -> tuple[DataSign, DataWidth]:
    """Infer output data format from a :class:`CoreNeuronV25` activation.

    Rules:

    - SNN mode (``lut is None``) with ``thres_neg_mode == FIRE``:
      outputs {-1, 0, +1} -> ``SIGNED, WIDTH_2BIT``.
    - SNN mode with ``thres_neg_mode == FLOOR``:
      outputs {0, +1} -> ``UNSIGNED, WIDTH_1BIT``.
    - ANN mode (``lut is not None``) with ``output_sign == 1``:
      outputs [-128, 127] -> ``SIGNED, WIDTH_8BIT``.
    - ANN mode with ``output_sign == 0``:
      outputs [0, 255] -> ``UNSIGNED, WIDTH_8BIT``.
    """
    if act.is_snn:
        # SNN mode
        if act.thres_neg_mode == ThresholdNegMode.FIRE:
            return DataSign.SIGNED, DataWidth.WIDTH_2BIT  # including negative spike
        return DataSign.UNSIGNED, DataWidth.WIDTH_1BIT
    else:
        # ANN mode
        if act.output_sign == 1:
            return DataSign.SIGNED, DataWidth.WIDTH_8BIT
        else:
            return DataSign.UNSIGNED, DataWidth.WIDTH_8BIT


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
    signed = weight_min < 0
    sign = DataSign.SIGNED if signed else DataSign.UNSIGNED
    ranges = _SIGNED_RANGES if signed else _UNSIGNED_RANGES

    for w, (lo, hi) in ranges.items():
        if lo <= weight_min and weight_max <= hi:
            return sign, w

    raise ValueError(
        f"weight range [{weight_min}, {weight_max}] exceeds 8-bit capacity"
    )


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
