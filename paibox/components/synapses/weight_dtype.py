import numpy as np
from paicorelib import WeightWidth as WW

from paibox.types import WeightType

MAX_INT1 = 1
MIN_INT1 = 0
MAX_INT2 = 1
MIN_INT2 = -2
MAX_INT4 = 7
MIN_INT4 = -8
MAX_INT8 = np.iinfo(np.int8).max
MIN_INT8 = np.iinfo(np.int8).min


weight_width2range = {
    WW.WEIGHT_WIDTH_1BIT: (MIN_INT1, MAX_INT1),
    WW.WEIGHT_WIDTH_2BIT: (MIN_INT2, MAX_INT2),
    WW.WEIGHT_WIDTH_4BIT: (MIN_INT4, MAX_INT4),
    WW.WEIGHT_WIDTH_8BIT: (MIN_INT8, MAX_INT8),
}


def get_weight_width(weight: WeightType, enable_wp_opt: bool = True) -> WW:
    """Get the actual width of the weight."""
    _max, _min = np.max(weight), np.min(weight)

    if enable_wp_opt:
        if _max <= MAX_INT1 and _min >= MIN_INT1:
            return WW.WEIGHT_WIDTH_1BIT
        elif _max <= MAX_INT2 and _min >= MIN_INT2:
            return WW.WEIGHT_WIDTH_2BIT
        elif _max <= MAX_INT4 and _min >= MIN_INT4:
            return WW.WEIGHT_WIDTH_4BIT
        else:
            return WW.WEIGHT_WIDTH_8BIT
    else:
        return WW.WEIGHT_WIDTH_8BIT
