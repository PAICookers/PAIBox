from enum import Enum, unique

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

    - `MODE_FORWARD`: forward leaking. Default value.
    - `MODE_REVERSAL`: reversal leaking.

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


@unique
class ThresholdMode(Enum):
    MODE_UNSET = 0
    MODE_POSITIVE = 1
    MODE_NEGATIVE = 2