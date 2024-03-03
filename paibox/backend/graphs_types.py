import sys
from enum import Enum, auto, unique
from typing import NamedTuple, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paibox.base import NeuDyn
from paibox.projection import InputProj
from paibox.synapses import SynSys

NodeName: TypeAlias = str
EdgeName: TypeAlias = str
NodeType: TypeAlias = Union[InputProj, NeuDyn]
EdgeType: TypeAlias = SynSys
SourceNodeType: TypeAlias = NodeType
DestNodeType: TypeAlias = NeuDyn


@unique
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
    node: NodeType
    position: NodePosition
    degree: NodeDegree


class EdgeAttr(NamedTuple):
    edge: EdgeType
    distance: int
