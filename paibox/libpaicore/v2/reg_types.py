from enum import Enum, IntEnum, unique
from typing import Dict, Tuple

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


@unique
class LCNExtensionType(IntEnum):
    """Scale of Fan-in extension. 4-bit.

    - X-time LCN extension. Default value is `LCN_1X`.

    NOTE:
    - For `MODE_ANN`, `LCN_1X` = 144x.
    - For `MODE_SNN` or `MODE_BANN`, `LCN_1X` = 1152x.
    """

    LCN_1X = 0  # Default value.
    LCN_2X = 1
    LCN_4X = 2
    LCN_8X = 3
    LCN_16X = 4
    LCN_32X = 5
    LCN_64X = 6


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
    """Type of core. Reserved."""

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

    MODE_SNN = 0
    MODE_BANN = 1  # SNN mode like.
    MODE_ANN = 2
    MODE_BANN_OR_SNN_TO_ANN = 3
    MODE_BANN_OR_SNN_TO_SNN = 4
    MODE_ANN_TO_BANN_OR_SNN = 5


_ModeParamRef = Tuple[InputWidthFormatType, SpikeWidthFormatType, SNNModeEnableType]


CoreModeDict: Dict[CoreMode, _ModeParamRef] = {
    CoreMode.MODE_BANN: (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.DISABLE,
    ),
    CoreMode.MODE_SNN: (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.ENABLE,
    ),
    CoreMode.MODE_BANN_OR_SNN_TO_ANN: (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.DISABLE,
    ),
    CoreMode.MODE_BANN_OR_SNN_TO_SNN: (
        InputWidthFormatType.WIDTH_1BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.ENABLE,
    ),
    CoreMode.MODE_ANN_TO_BANN_OR_SNN: (
        InputWidthFormatType.WIDTH_8BIT,
        SpikeWidthFormatType.WIDTH_1BIT,
        SNNModeEnableType.DISABLE,
    ),
    CoreMode.MODE_ANN: (
        InputWidthFormatType.WIDTH_8BIT,
        SpikeWidthFormatType.WIDTH_8BIT,
        SNNModeEnableType.DISABLE,
    ),
}
