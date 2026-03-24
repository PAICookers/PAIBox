"""IR-level signal-domain annotations."""

from enum import Enum, auto

__all__ = ["SignalDomain"]


class SignalDomain(Enum):
    """Semantic output domain of a PAIIR node."""

    POTENTIAL = auto()
    VALUE = auto()
