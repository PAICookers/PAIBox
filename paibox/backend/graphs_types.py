import sys
from enum import Enum, auto
from typing import NamedTuple

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paibox.base import NeuDyn

NodeName: TypeAlias = str
EdgeName: TypeAlias = str


class NodePosition(Enum):
    """Charactor of a node in the directed graph."""

    MEMBER = auto()
    """As a member layer."""
    INPUT = auto()
    """As an input node."""
    OUTPUT = auto()
    """As an output node."""


class NodeDegree(NamedTuple):
    """In/Out-degree of a node in the directed graph."""

    in_degree: int = 0
    out_degree: int = 0


class NodeAttr(NamedTuple):
    obj: NeuDyn
    position: NodePosition
    degree: NodeDegree


class EdgeAttr(NamedTuple):
    edge: EdgeName
    distance: int
