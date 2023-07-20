from enum import Enum, IntEnum, unique

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

    - `MAX_POOLING_DISABLE`: pooling max disable.
    - `MAX_POOLING_ENABLE`: pooling max enable. Default value.
    """

    DISABLE = 0
    ENABLE = 1  # Default value.


@unique
class SNNModeEnableType(Enum):
    """Enable SNN mode or not. 1-bit.

    - `SNN_MODE_DISABLE`: SNN mode disable.
    - `SNN_MODE_ENABLE`: SNN mode enable. Default value.
    """

    DISABLE = 0
    ENABLE = 1  # Default value.


"""
    Type defines of RAM parameters.
    See Section 2.4.2 in V2.1 Manual for details.
"""


@unique
class ResetMode(Enum):
    """Reset modes of neurons. 2-bit.

    - `MODE_NORMAL`: normal mode. Default value.
    - `MODE_LINEAR`: linear mode.
    - `MODE_NONRESET`: non-reset mode.

    NOTE: Same as `reset_mode` in V2.1.
    """

    MODE_NORMAL = 0  # Default value.
    MODE_LINEAR = 1
    MODE_NONRESET = 2


@unique
class LeakingComparisonMode(Enum):
    """Leak after comparison or before. 1-bit.

    - `LEAK_BEFORE_COMP`: leak before comparison.
    - `LEAK_AFTER_COMP`: leak after comparison. Default value.

    NOTE: Same as `leak_post` in V2.1.
    """

    LEAK_BEFORE_COMP = 0
    LEAK_AFTER_COMP = 1  # Default value.


@unique
class NegativeThresholdMode(Enum):
    """Modes of negative threshold. 1-bit.

    - `MODE_RESET`: reset mode. Default value.
    - `MODE_SATURATION`: saturation(floor) mode.

    NOTE: Same as `threshold_neg_mode` in V2.1.
    """

    MODE_RESET = 0  # Default value.
    MODE_SATURATION = 1


@unique
class LeakingDirectionMode(Enum):
    """Direction of leaking, forward or reversal.

    - `FORWARD`: forward leaking. Default value.
    - `REVERSAL`: reversal leaking.

    NOTE: Same as `leak_reversal_flag` in V2.1.
    """

    MODE_FORWARD = 0  # Default value.
    MODE_REVERSAL = 1


@unique
class LeakingIntegrationMode(Enum):
    """Mode of leaking integration, deterministic or stochastic.

    - `MODE_DETERMINISTIC`: deterministic leaking. Default value.
    - `MODE_STOCHASTIC`: stochastic leaking.

    NOTE: Same as `leak_det_stoch` in V2.1.
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1


@unique
class SynapticIntegrationMode(Enum):
    """Modes of synaptic integration, deterministic or stochastic.

    - `MODE_DETERMINISTIC`: deterministic weights. Default value.
    - `MODE_STOCHASTIC`: stochastic weights.

    NOTE: Same as `weight_det_stoch` in V2.1.
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1
