"""Shared VALUE-code range helpers used by IR and compile passes."""

from collections.abc import Sequence

from paicorelib import DataSign, DataWidth

__all__ = [
    "SIGNED_VALUE_CODE_RANGES",
    "UNSIGNED_VALUE_CODE_RANGES",
    "ValueCodeRange",
    "code_range_for_data_format",
    "code_range_for_format",
    "fits_value_code_range",
    "merge_code_ranges",
]

ValueCodeRange = tuple[int, int]

SIGNED_VALUE_CODE_RANGES: dict[DataWidth, ValueCodeRange] = {
    DataWidth.WIDTH_1BIT: (-1, 0),
    DataWidth.WIDTH_2BIT: (-2, 1),
    DataWidth.WIDTH_4BIT: (-8, 7),
    DataWidth.WIDTH_8BIT: (-128, 127),
}

UNSIGNED_VALUE_CODE_RANGES: dict[DataWidth, ValueCodeRange] = {
    DataWidth.WIDTH_1BIT: (0, 1),
    DataWidth.WIDTH_2BIT: (0, 3),
    DataWidth.WIDTH_4BIT: (0, 15),
    DataWidth.WIDTH_8BIT: (0, 255),
}


def code_range_for_format(sign: DataSign, width: DataWidth) -> ValueCodeRange:
    ranges = (
        SIGNED_VALUE_CODE_RANGES
        if sign == DataSign.SIGNED
        else UNSIGNED_VALUE_CODE_RANGES
    )
    return ranges[width]


def code_range_for_data_format(
    fmt: tuple[DataSign, DataWidth],
) -> ValueCodeRange:
    sign, width = fmt
    return code_range_for_format(sign, width)


def fits_value_code_range(value_min: int, value_max: int) -> bool:
    """Return whether one integer code range fits the 8-bit VALUE path."""
    if value_min < 0:
        lo, hi = SIGNED_VALUE_CODE_RANGES[DataWidth.WIDTH_8BIT]
        return lo <= value_min and value_max <= hi

    lo, hi = UNSIGNED_VALUE_CODE_RANGES[DataWidth.WIDTH_8BIT]
    return lo <= value_min and value_max <= hi


def merge_code_ranges(
    code_ranges: Sequence[ValueCodeRange],
) -> ValueCodeRange | None:
    if not code_ranges:
        return None
    return min(lo for lo, _ in code_ranges), max(hi for _, hi in code_ranges)
