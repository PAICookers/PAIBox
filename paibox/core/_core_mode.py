from enum import Enum, unique
from .reg_types import InputWidthFormatType, SpikeWidthFormatType, SNNModeEnableType

from .reg_types import InputWidthFormatType, SNNModeEnableType, SpikeWidthFormatType

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
<<<<<<< HEAD
        return CoreMode.MODE_ANN
=======
        return CoreMode.MODE_ANN
>>>>>>> 86a2555 (修改参数定义文件)
