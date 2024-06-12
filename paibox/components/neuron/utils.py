import warnings

import numpy as np

from paibox.exceptions import FunctionalError, PAIBoxWarning
from paibox.types import VoltageType
from paicorelib import __version__ as __plib_version__

if __plib_version__ > "1.1.4":
    from paicorelib.ram_model import PRN_THRES_BIT_MAX, NEGATIVE_THRES_BIT_MAX
else:
    PRN_THRES_BIT_MAX = 29
    NEGATIVE_THRES_BIT_MAX = 29

try:
    from paicorelib.framelib.frame_defs import _mask
    from paicorelib.ram_model import (
        LEAK_V_BIT_MAX,
        VJT_PRE_BIT_MAX,
    )
except ImportError:

    def _mask(mask_bit: int) -> int:
        return (1 << mask_bit) - 1

    LEAK_V_BIT_MAX = 30
    VJT_PRE_BIT_MAX = 30


VJT_MAX_LIMIT = _mask(VJT_PRE_BIT_MAX - 1)
VJT_MIN_LIMIT = -(VJT_MAX_LIMIT + 1)
VJT_LIMIT = VJT_MAX_LIMIT - VJT_MIN_LIMIT
VJT_OVERFLOW_TEXT = f"Membrane potential overflow, beyond the range of {VJT_PRE_BIT_MAX}-bit signed integer."
NEG_THRES_MIN = -_mask(NEGATIVE_THRES_BIT_MAX)
LEAK_V_MAX = _mask(LEAK_V_BIT_MAX - 1)


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
