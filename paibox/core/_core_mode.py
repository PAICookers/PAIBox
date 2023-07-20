from enum import Enum, unique

__all__ = ["CoreMode"]


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
