import warnings

import numpy as np

from paibox.exceptions import FunctionalError, PAIBoxWarning
from paibox.types import VoltageType

try:
    from paicorelib.ram_model import (
        NEGATIVE_THRESHOLD_VALUE_BIT_MAX as _NEG_THRES_BIT_MAX,
    )
    from paicorelib.ram_model import VJT_PRE_BIT_MAX as _VJT_PRE_BIT_MAX

except ImportError:
    _VJT_PRE_BIT_MAX = 30
    _NEG_THRES_BIT_MAX = 29


VJT_MAX_LIMIT = 1 << (_VJT_PRE_BIT_MAX - 1) - 1
VJT_MIN_LIMIT = -(VJT_MAX_LIMIT + 1)
VJT_LIMIT = 1 << _VJT_PRE_BIT_MAX
VJT_OVERFLOW_TEXT = f"Membrane potential overflow, beyond the range of {_VJT_PRE_BIT_MAX}-bit signed integer."
NEG_THRES_MIN = 1 - (1 << _NEG_THRES_BIT_MAX)


def _is_vjt_overflow(vjt: VoltageType, strict: bool = False) -> bool:
    # NOTE: In most cases, membrane potential overflow won't occur,
    # otherwise the result is incorrect.
    if np.any(vjt > VJT_MAX_LIMIT) or np.any(vjt < VJT_MIN_LIMIT):
        if strict:
            raise FunctionalError(VJT_OVERFLOW_TEXT)
        else:
            warnings.warn(VJT_OVERFLOW_TEXT, PAIBoxWarning)

        return False

    return True


def vjt_overflow(vjt: VoltageType, strict: bool = False) -> VoltageType:
    """Handle the overflow of the membrane potential.

    NOTE: If the incoming membrane potential (30-bit signed) overflows, the chip\
        will automatically handle it. This behavior needs to be implemented in  \
        simulation.
    """
    _is_vjt_overflow(vjt, strict)

    return np.where(
        vjt > VJT_MAX_LIMIT,
        vjt - VJT_LIMIT,
        np.where(
            vjt < VJT_MIN_LIMIT,
            vjt + VJT_LIMIT,
            vjt,
        ),
    ).astype(np.int32)
