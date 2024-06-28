import warnings
from typing import Union

import numpy as np
from paicorelib.framelib.utils import _mask
from paicorelib.ram_model import LEAK_V_BIT_MAX, LEAK_V_MAX, LEAK_V_MIN
from paicorelib.ram_model import NEG_THRES_MAX as NEG_THRES_UNSIGNED_MAX
from paicorelib.ram_model import VJT_MAX, VJT_MIN, VJT_PRE_BIT_MAX

from paibox.exceptions import FunctionalError, PAIBoxWarning
from paibox.types import LeakVType, VoltageType

NEG_THRES_MIN = -NEG_THRES_UNSIGNED_MAX


SIGNED_PARAM_OVERFLOW_TEXT = "{0} overflow, beyond the range of {1}-bit signed integer."
VJT_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format(
    "membrane potential", VJT_PRE_BIT_MAX
)
LEAK_V_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format("leak voltage", LEAK_V_BIT_MAX)
VJT_RANGE_LIMIT = VJT_MAX - VJT_MIN


def _is_vjt_overflow(vjt: VoltageType, strict: bool = False) -> bool:
    # NOTE: In most cases, membrane potential overflow won't occur,
    # otherwise the result is incorrect.
    if np.any(vjt > VJT_MAX) or np.any(vjt < VJT_MIN):
        if strict:
            raise FunctionalError(VJT_OVERFLOW_TEXT)
        else:
            warnings.warn(VJT_OVERFLOW_TEXT, PAIBoxWarning)

        return False

    return True


def vjt_overflow(vjt: VoltageType, strict: bool = False) -> VoltageType:
    """Handle the overflow of the membrane potential.

    NOTE: If the incoming membrane potential (30-bit signed) overflows, the chip will   \
        automatically handle it. This behavior needs to be implemented in simulation.
    """
    _is_vjt_overflow(vjt, strict)

    return np.where(
        vjt > VJT_MAX,
        vjt - VJT_RANGE_LIMIT,
        np.where(
            vjt < VJT_MIN,
            vjt + VJT_RANGE_LIMIT,
            vjt,
        ),
    ).astype(np.int32)


def _is_leak_v_overflow(leak_v: Union[int, LeakVType], strict: bool = True) -> None:
    if isinstance(leak_v, int):
        if leak_v > LEAK_V_MAX or leak_v < LEAK_V_MIN:
            if strict:
                raise FunctionalError(LEAK_V_OVERFLOW_TEXT)
            else:
                warnings.warn(LEAK_V_OVERFLOW_TEXT, PAIBoxWarning)
    elif np.any(leak_v > LEAK_V_MAX) or np.any(leak_v < LEAK_V_MIN):
        if strict:
            raise FunctionalError(LEAK_V_OVERFLOW_TEXT)
        else:
            warnings.warn(LEAK_V_OVERFLOW_TEXT, PAIBoxWarning)
