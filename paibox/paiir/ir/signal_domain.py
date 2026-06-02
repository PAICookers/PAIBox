"""IR-level signal-domain annotations."""

from dataclasses import dataclass
from enum import Enum, auto

__all__ = ["SignalDomain", "SignalSemantics"]


class SignalDomain(Enum):
    """Semantic output domain of a PAIIR node."""

    POTENTIAL = auto()
    VALUE = auto()


@dataclass(slots=True)
class SignalSemantics:
    """Node-level signal semantics derived during compilation.

    Attributes:
        output_domain: Coarse VALUE/POTENTIAL domain annotation.
        known_code_range: Exact VALUE code range when derivable; otherwise
            ``None``.
    """

    output_domain: SignalDomain | None = None
    known_code_range: tuple[int, int] | None = None
