from enum import Enum, Flag, unique

"""
    Type defines of Parameter Registers.
    See Section 2.4.1 in V2.1 Manual for details.
"""


@unique
class WeightPrecisionType(Enum):
    """Weight precision of crossbar. 2-bit.

    `WEIGHT_WIDTH_XBIT` for X-bit.
    """

    WEIGHT_WIDTH_1BIT = 0
    WEIGHT_WIDTH_2BIT = 1
    WEIGHT_WIDTH_4BIT = 2
    WEIGHT_WIDTH_8BIT = 3  # Default value.


@unique
class LCNExtensionType(Enum):
    """Scale of Fan-in extension. 4-bit.

    - For ANN mode, LCN_1X = 144x.
    - For BANN/SNN mode, LCN_1X = 1152x.
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

    - `INPUT_WIDTH_1BIT`: 1-bit spikes, 0.
    - `INPUT_WIDTH_8BIT`: 8-bit activation.
    """

    INPUT_WIDTH_1BIT = 0  # Default value.
    INPUT_WIDTH_8BIT = 1


@unique
class SpikeWidthFormatType(Enum):
    """Format of output spike. 1-bit.

    - `SPIKE_WIDTH_1BIT`: 1-bit spikes, 0.
    - `SPIKE_WIDTH_8BIT`: 8-bit activation.
    """

    SPIKE_WIDTH_1BIT = 0  # Default value.
    SPIKE_WIDTH_8BIT = 1


@unique
class PoolMaxEnableType(Enum):
    """Enable max pooling or not in 8-bit input format. 1-bit.

    - `POOL_MAX_DISABLE`: pooling max disable, 0.
    - `POOL_MAX_ENABLE`: pooling max enable.
    """

    POOL_MAX_DISABLE = 0
    POOL_MAX_ENABLE = 1  # Default value.


@unique
class SNNModeEnableType(Enum):
    """Enable SNN mode or not. 1-bit.

    - `SNN_MODE_DISABLE`: SNN mode disable, 0.
    - `SNN_MODE_ENABLE`: SNN mode enable.

    """

    SNN_MODE_DISABLE = 0
    SNN_MODE_ENABLE = 1  # Default value.


"""
    Type defines of RAM parameters.
    See Section 2.4.2 in V2.1 Manual for details.
"""


class CoreAddrX_EX_Type(Flag):
    """
    Broadcast address X of the destination code.
    询问钟这个是否是核间路由

    通过flag的或的方式，获得路由的规则，从而实现*通配符的等效替代

    Usage:

    NOTE: *addr_core_x_ex* in Section 2.4.2.
    """

    pass


@unique
class ResetModeType(Enum):
    """Reset modes of cores. 2-bit.

    - MODE_NORMAL: normal mode.
    - MODE_LINEAR: linear mode.
    - MODE_NONRESET: non-reset mode.

    NOTE: *reset_mode*
    """

    MODE_NORMAL = 0  # Default value.
    MODE_LINEAR = 1
    MODE_NONRESET = 2


@unique
class LeakingComparisonType(Enum):
    """Leak after comparison or before. 1-bit.

    - LEAK_BEFORE_COMP: leak before comparison.
    - LEAK_AFTER_COMP: leak after comparison.

    NOTE: *leak_post*
    """

    LEAK_BEFORE_COMP = 0
    LEAK_AFTER_COMP = 1  # Default value.


@unique
class NegativeThresModeType(Enum):
    """Modes of negative threshold. 1-bit.

    - MODE_RESET: reset mode.
    - MODE_SATURATION: saturation mode.

    NOTE: *threshold_neg_mode*
    """

    MODE_RESET = 0  # Default value.
    MODE_SATURATION = 1


@unique
class LeakingDirectionType(Enum):
    """Direction of leaking, forward or reversal.

    - MODE_FORWARD: forward leaking.
    - MODE_REVERSAL: reversal leaking.

    NOTE: *leak_reversal_flag*
    """

    MODE_FORWARD = 0  # Default value.
    MODE_REVERSAL = 1


@unique
class LeakingModeType(Enum):
    """Modes of leaking, deterministic or stochastic.

    - MODE_DETERMINISTIC: deterministic leaking.
    - MODE_STOCHASTIC: stochastic leaking.

    NOTE: *leak_det_stoch*
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1


@unique
class WeightModeType(Enum):
    """Modes of weights, deterministic or stochastic.

    - MODE_DETERMINISTIC: deterministic weights
    - MODE_STOCHASTIC: stochastic weights

    NOTE: *weight_det_stoch*
    """

    MODE_DETERMINISTIC = 0  # Default value.
    MODE_STOCHASTIC = 1


if __name__ == "__main__":
    a = LCNExtensionType.LCN_32X

    print(a == 5)
