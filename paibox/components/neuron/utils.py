import numpy as np

from paibox.types import VoltageType

VJT_MAX_LIMIT: int = 2**29 - 1
VJT_MIN_LIMIT: int = -(2**29)
VJT_LIMIT: int = 2**30


def _is_vjt_overflow(vjt: VoltageType) -> bool:
    return bool(np.any(vjt > VJT_MAX_LIMIT) or np.any(vjt < VJT_MIN_LIMIT))


def _vjt_overflow(vjt: VoltageType) -> VoltageType:
    """Handle the overflow of the membrane potential.

    NOTE: If the incoming membrane potential (30-bit signed) overflows, the chip\
        will automatically handle it. This behavior needs to be implemented in  \
        simulation.
    """
    return np.where(
        vjt > VJT_MAX_LIMIT,
        vjt - VJT_LIMIT,
        np.where(
            vjt < VJT_MIN_LIMIT,
            vjt + VJT_LIMIT,
            vjt,
        ),
    ).astype(np.int32)
