from collections.abc import Sequence
from enum import Enum, unique

from paicorelib import DataSign, DataWidth

__all__ = [
    "DataSemanticType",
    "get_max_data_semantic_type",
    "merge_data_semantic_types",
    "infer_input_specs_by_semantic_type",
]


@unique
class DataSemanticType(Enum):
    UNKNOWN = 0
    SPIKE = 1
    ACTIVATION = 2
    POTENTIAL = 3


def get_max_data_semantic_type(types: Sequence[DataSemanticType]) -> DataSemanticType:
    return max(types, key=lambda x: x.value)


def merge_data_semantic_types(types: Sequence[DataSemanticType]) -> DataSemanticType:
    """Merge multiple semantic types by compatibility (take max).

    Order: potential > activation > spike > unknown
    """
    ts = list(types)
    if len(ts) == 0:
        return DataSemanticType.UNKNOWN

    return get_max_data_semantic_type(ts)


def infer_input_specs_by_semantic_type(
    type: DataSemanticType, signed: bool = False
) -> tuple[DataSign, DataWidth]:
    # TODO how to define the sign?
    if type == DataSemanticType.SPIKE:
        width = DataWidth.WIDTH_1BIT
    else:  # activation or unknown or potential(input sign/width will ignored)
        width = DataWidth.WIDTH_8BIT

    sign = DataSign.SIGNED if signed else DataSign.UNSIGNED
    return sign, width
