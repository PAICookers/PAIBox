from enum import Enum, IntEnum, unique

__all__ = [
    "WeightPrecisionType",
    "LCNExtensionType",
    "InputWidthFormatType",
    "SpikeWidthFormatType",
    "MaxPoolingEnableType",
    "SNNModeEnableType",
    "CoreType",
    "CoreMode",
]

"""
    Type defines of parameters of registers & parameters of neuron RAM.
    See Section 2.4.1 in V2.1 Manual for details.
"""


@unique
class WeightPrecisionType(IntEnum):
    """Weight precision of crossbar. 2-bit.

    - `WEIGHT_WIDTH_XBIT` for X-bit. Default value is `WEIGHT_WIDTH_8BIT`.
    """

    WEIGHT_WIDTH_1BIT = 0
    WEIGHT_WIDTH_2BIT = 1
    WEIGHT_WIDTH_4BIT = 2
    WEIGHT_WIDTH_8BIT = 3  # Default value.
    WEIGHT_WIDTH_MAX = 4


@unique
class LCNExtensionType(IntEnum):
    """Scale of Fan-in extension. 4-bit.

    - X-time LCN extension. Default value is `LCN_1X`.

    NOTE:
    - For ANN mode, `LCN_1X` = 144x.
    - For BANN/SNN mode, `LCN_1X` = 1152x.
    """

    LCN_1X = 0  # Default value.
    LCN_2X = 1
    LCN_4X = 2
    LCN_8X = 3
    LCN_16X = 4
    LCN_32X = 5
    LCN_64X = 6
    LCN_MAX = 7


@unique
class InputWidthFormatType(Enum):
    """Format of input spike. 1-bit.

    - `WIDTH_1BIT`: 1-bit spike. Default value.
    - `WIDTH_8BIT`: 8-bit activation.
    """

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class SpikeWidthFormatType(Enum):
    """Format of output spike. 1-bit.

    - `WIDTH_1BIT`: 1-bit spike. Default value.
    - `WIDTH_8BIT`: 8-bit activation.
    """

    WIDTH_1BIT = 0  # Default value.
    WIDTH_8BIT = 1


@unique
class MaxPoolingEnableType(Enum):
    """Enable max pooling or not in 8-bit input format. 1-bit.

    - `MAX_POOLING_DISABLE`: pooling max disable. Default value.
    - `MAX_POOLING_ENABLE`: pooling max enable.
    """

    DISABLE = 0  # Default value.
    ENABLE = 1


@unique
class SNNModeEnableType(Enum):
    """Enable SNN mode or not. 1-bit.

    - `SNN_MODE_DISABLE`: SNN mode disable.
    - `SNN_MODE_ENABLE`: SNN mode enable. Default value.
    """

    DISABLE = 0
    ENABLE = 1  # Default value.


@unique
class CoreType(Enum):
    TYPE_OFFLINE = "OFFLINE"
    TYPE_ONLINE = "ONLINE"


@unique
class CoreMode(Enum):
    """Working mode of cores.

    Decided by `input_width`, `spike_width` and `SNN_EN` of core parameters registers.

    NOTE: See table below for details.

    Mode                        input_width    spike_width    SNN_EN
    BANN                            0               0           0
    SNN                             0               0           1
    BANN/SNN to ANN                 0               1           0
    BANN/SNN to SNN with values     0               1           1
    ANN to BANN/SNN                 1               0       Don't care
    ANN                             1               1       Don't care
    """

    MODE_UNKNOWN = 0
    MODE_ANN = 1
    MODE_SNN = 2
    MODE_BANN = 3
    MODE_BANN_OR_SNN_TO_ANN = 4
    MODE_BANN_OR_SNN_TO_SNN = 5
    MODE_ANN_TO_BANN_OR_SNN = 6


def get_core_mode(
    iwidth_format: InputWidthFormatType,
    swidth_format: SpikeWidthFormatType,
    snn_en: SNNModeEnableType,
) -> CoreMode:
    """Get the working mode of the core.

    Decided by `input_width`, `spike_width` and `SNN_EN` of core parameters registers.

    NOTE: See table below for details.

    Mode            input_width    spike_width    SNN_EN
    BANN                0               0           0
    SNN                 0               0           1
    BANN/SNN to ANN     0               1           0
    BANN/SNN to SNN     0               1           1
    ANN to BANN/SNN     1               0       Don't care
    ANN                 1               1       Don't care
    """
    if iwidth_format is InputWidthFormatType.WIDTH_1BIT:
        if swidth_format is SpikeWidthFormatType.WIDTH_1BIT:
            if snn_en is SNNModeEnableType.DISABLE:
                # 0 / 0 / 0
                return CoreMode.MODE_BANN
            else:
                # 0 / 0 / 1
                return CoreMode.MODE_SNN
        else:
            if snn_en is SNNModeEnableType.DISABLE:
                # 0 / 1 / 0
                return CoreMode.MODE_BANN_OR_SNN_TO_ANN
            else:
                # 0 / 1 / 1
                return CoreMode.MODE_BANN_OR_SNN_TO_SNN
    elif swidth_format is SpikeWidthFormatType.WIDTH_1BIT:
        # 1 / 0 / *
        return CoreMode.MODE_ANN_TO_BANN_OR_SNN
    else:
        # 1 / 1 / *
        return CoreMode.MODE_ANN
