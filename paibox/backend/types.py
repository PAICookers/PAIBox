import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import CoreMode, HwConfig

from paibox.base import PAIBoxObject
from paibox.components import FullConnectedSyn, InputProj, Neuron

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
    "NeuSlice",
    "NeuSegment",
    "NeuSegOfCorePlm",
    "NeuSegOfCoreBlock",
    "AxonCoord",
    "AxonSegment",
    "CoreAbstract",
]

NodeName: TypeAlias = str
EdgeName: TypeAlias = str
NodeType: TypeAlias = Union[InputProj, Neuron]
EdgeType: TypeAlias = FullConnectedSyn
SourceNodeType: TypeAlias = NodeType
DestNodeType: TypeAlias = Neuron

WRAM_UNPACKED_DTYPE = np.uint8
WRAM_PACKED_DTYPE = np.uint64
# Type of unpacked weight in WRAM
WRAMUnpackedType: TypeAlias = NDArray[WRAM_UNPACKED_DTYPE]
# Type of packed weight in WRAM
WRAMPackedType: TypeAlias = NDArray[WRAM_PACKED_DTYPE]
N_BIT_PACKED_WEIGHT = WRAM_PACKED_DTYPE(1).nbytes * 8  # #N bits of packed weight

_COORD_UNSET = 0
_DEGREE_UNSET = -1


@unique
class NodePosition(Enum):
    """Charactor of a node in the directed graph."""

    MEMBER = auto()
    INPUT = auto()
    OUTPUT = auto()


@dataclass
class NodeDegree:
    """In/Out-degree of a node in the directed graph."""

    in_degree: int = _DEGREE_UNSET
    out_degree: int = _DEGREE_UNSET

    def __copy__(self) -> "NodeDegree":
        return self.__deepcopy__()

    def __deepcopy__(self, memo=None) -> "NodeDegree":
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
    rt_mode: CoreMode = CoreMode.MODE_SNN  # XXX Temp solution


NeuSlice: TypeAlias = slice


@dataclass(frozen=True)
class NeuSegment:
    target: DestNodeType
    index: NeuSlice  # slice like slice(x, y, 1)
    offset: int
    repeat: int = 1

    @property
    def n_neuron(self) -> int:
        return self.index.stop - self.index.start

    @property
    def n_addr(self) -> int:
        return self.repeat * self.n_neuron

    @property
    def addr_ram(self) -> list[int]:
        """Convert index of neuron into RAM address."""
        return list(range(self.offset, self.addr_max, 1))

    @property
    def addr_max(self) -> int:
        if (
            _addr_max := self.offset + self.repeat * self.n_neuron
        ) > HwConfig.ADDR_RAM_MAX + 1:
            raise ValueError(
                f"neuron RAM address out of range {HwConfig.ADDR_RAM_MAX + 1} ({_addr_max})."
            )

        return _addr_max

    @property
    def addr_slice(self) -> slice:
        """Display the RAM address in slice format."""
        return slice(self.offset, self.addr_max, self.repeat)

    def __str__(self) -> str:
        return f"NeuSeg {self.target.name} at offset {self.offset}"


NeuSegOfCorePlm: TypeAlias = list[NeuSegment]
NeuSegOfCoreBlock: TypeAlias = list[NeuSegOfCorePlm]


class AxonCoord(NamedTuple):
    tick_relative: int
    addr_axon: int


class AxonSegment(NamedTuple):
    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset + addr_width)."""
    addr_offset: int
    """The offset of the assigned address."""


class CoreAbstract(PAIBoxObject, ABC):
    """Abstract core class."""

    rt_mode: CoreMode

    @property
    @abstractmethod
    def n_core_required(self) -> int:
        """#N of cores required to accommodate neurons inside self."""
        ...

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs): ...


if hasattr(CoreMode, "is_iw8"):

    def is_iw8(mode: CoreMode) -> bool:
        return mode.is_iw8  # type: ignore

else:

    def is_iw8(mode: CoreMode) -> bool:
        return mode is CoreMode.MODE_ANN_TO_BANN_OR_SNN or mode is CoreMode.MODE_ANN
