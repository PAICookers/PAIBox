import sys

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto, unique
from numpy.typing import NDArray
from typing import NamedTuple, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paibox.base import NeuDyn
from paibox.components import FullConnectedSyn, InputProj

__all__ = [
    "NodeName",
    "EdgeName",
    "NodeType",
    "EdgeType",
    "SourceNodeType",
    "DestNodeType",
    "NodePosition",
    "NodeDegree",
    "NodeAttr",
    "EdgeAttr",
    "PartitionedEdges",
]

NodeName: TypeAlias = str
EdgeName: TypeAlias = str
NodeType: TypeAlias = Union[InputProj, NeuDyn]
EdgeType: TypeAlias = FullConnectedSyn
SourceNodeType: TypeAlias = NodeType
DestNodeType: TypeAlias = NeuDyn

WeightRamType: TypeAlias = NDArray[np.uint64]  # uint64 weights mapped in weight RAM
_COORD_UNSET = 0


@unique
class NodePosition(Enum):
    """Charactor of a node in the directed graph."""

    MEMBER = auto()
    INPUT = auto()
    OUTPUT = auto()


_DEGREE_UNSET = -1


@dataclass
class NodeDegree:
    """In/Out-degree of a node in the directed graph."""

    in_degree: int = _DEGREE_UNSET
    out_degree: int = _DEGREE_UNSET

    def __copy__(self) -> "NodeDegree":
        return self.__deepcopy__()

    def __deepcopy__(self) -> "NodeDegree":
        return NodeDegree(self.in_degree, self.out_degree)

    def copy(self) -> "NodeDegree":
        return self.__deepcopy__()


class NodeAttr(NamedTuple):
    node: NodeType
    position: NodePosition
    degree: NodeDegree


class EdgeAttr(NamedTuple):
    edge: EdgeType
    distance: int


class PartitionedEdges(NamedTuple):
    edges: set[EdgeType]
    rg_id: int
