import sys
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import Coord, CoreMode, HwConfig
from paicorelib import ReplicationId as RId
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

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
    "NeuSliceType",
    "NeuSegment",
    "NeuSegOfCorePlm",
    "NeuSegOfCoreBlock",
    "AxonCoord",
    "AxonSegment",
]

NodeName: TypeAlias = str
EdgeName: TypeAlias = str
NodeType: TypeAlias = Union[InputProj, Neuron]
EdgeType: TypeAlias = FullConnectedSyn
SourceNodeType: TypeAlias = NodeType
DestNodeType: TypeAlias = Neuron

WRAM_UNPACKED_DTYPE = np.uint8
WRAM_PACKED_DTYPE = np.uint64  # Type of one frame of data package
# Type of unpacked weight in WRAM
WRAMUnpackedType: TypeAlias = NDArray[WRAM_UNPACKED_DTYPE]
# Type of packed weight in WRAM
WRAMPackedType: TypeAlias = NDArray[WRAM_PACKED_DTYPE]
N_BIT_PACKED_WEIGHT = np.iinfo(WRAM_PACKED_DTYPE).bits

# TODO `Coord` will be called as read-only object in the future.
_COORD_UNSET = Coord(0, 0)
_RID_UNSET = RId(0, 0)
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


@dataclass
class NodeAttr:
    node: NodeType
    position: NodePosition
    degree: NodeDegree


@dataclass
class EdgeAttr:  # TODO FIXME distance?
    edge: EdgeType
    distance: int


NeuSliceType: TypeAlias = slice


@dataclass(frozen=True)
class NeuSegAddr:
    """The neuron segment in neuron address space."""

    n_neuron: int
    addr_offset: int
    """Same as `NeuSegment.offset`."""
    interval: int
    """Same as `NeuSegment.repeat`."""
    idx_offset: int
    """The offset of the starting address of this neuron corresponding to the neuron node   \
        in which it is located."""


@dataclass(frozen=True)
class NeuSegment:
    """`NeuSegment` describes the arrangement of neurons in neuron address space.

    Mapping between logical neuron indexes, neuron addresses & SRAM addresses:
                                        |<----------- index --------->|
        Logical idx:           ...     [1]                           [2]         ...
                            |<----- repeat ----->|        |<------ repeat ------>|
        Neuron address:    [0]      [1]   ...   [7]      [8]         ...        [15]
        SRAM address:    [0*4+:4] [1*4+:4] ... [7*4+:4] [8*4+:4]     ...     [15*4+:4]

    NOTE: Not necessary to descibe the neuron mapping in the SRAM address space. The SRAM address space \
        is only used to store the frame data.
    """

    target: DestNodeType
    index: NeuSliceType  # slice like slice(x, y, 1)
    offset: int
    """The offset at which the segment starts in the neuron address space."""
    repeat: int = 1
    """The number of times the neuron in the segment is repeated."""

    def __getitem__(self, s: slice):
        """Get a sub-segment of the current segment. The `start` & `stop` in `s` are the offsets    \
            relative to `index.start`.

        NOTE:
                    index.start                                index.stop
                        |                                           |
            index       |<----------- s.stop ---------->|           |
                   [0]  |<-- s.start -->|<-- sub-seg -->|           |    [end-1]
                    |   |<--------------- NeuSegment -------------->|       |
            target  ---------------------------------------------------------
        """
        if s.start is None:
            _start_offset = 0
        else:
            assert s.start >= 0
            if s.start > self.n_neuron:
                raise IndexError(f"Index out of range: {s.start} > {self.n_neuron}")

            _start_offset = s.start

        if s.stop is None:
            _stop_offset = self.n_neuron
        else:
            assert s.stop >= 0
            if s.stop > self.n_neuron:
                raise IndexError(f"Index out of range: {s.stop} > {self.n_neuron}")

            _stop_offset = s.stop

        new_slice = slice(
            self.index.start + _start_offset,
            self.index.start + _stop_offset,
            self.index.step,
        )

        return type(self)(
            self.target, new_slice, self.offset + _start_offset, self.repeat
        )

    @property
    def n_neuron(self) -> int:
        """The number of logical neurons in the segment."""
        return self.index.stop - self.index.start

    @property
    def n_occupied_in_addr(self) -> int:
        """#N of neuron addresses the segment occupies in the neuron address space."""
        return self.repeat * self.n_neuron

    @property
    def attrs(self) -> dict[str, Any]:
        return self.target._slice_attrs(self.index)

    @property
    def occupied_addr(self) -> list[int]:
        """Return the occupied neuron address range in the neuron address space."""
        return list(range(self.offset, self.offset + self.n_occupied_in_addr, 1))

    @property
    def _occupied_addr_repr(self) -> slice:
        """Represent the occupied neuron address range in the neuron address space with 'repeat'."""
        return slice(self.offset, self.offset + self.n_occupied_in_addr, self.repeat)

    @property
    def neu_seg_addr(self) -> NeuSegAddr:
        return NeuSegAddr(self.n_neuron, self.offset, self.repeat, self.index.start)


NeuSegOfCorePlm: TypeAlias = list[NeuSegment]
NeuSegOfCoreBlock: TypeAlias = list[NeuSegOfCorePlm]


@dataclass(frozen=True)
class AxonCoord:
    tick_relative: int
    addr_axon: int

    @classmethod
    def build(cls, tick_relative: int, addr_axon: int) -> "AxonCoord":
        return cls(tick_relative % HwConfig.N_TIMESLOT_MAX, addr_axon)


@dataclass(frozen=True)
class AxonSegment:
    """The axons will be arranged as a segment on the axon side of the core, and the segment    \
        starts at `addr_offset` & has a width of `addr_width`.
    """

    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset + addr_width)."""
    addr_offset: int
    """The offset of the assigned address."""
    start_offset: int
    """"The start offset of the source slice."""


if hasattr(CoreMode, "is_iw8"):

    def is_iw8(mode: CoreMode) -> bool:
        return mode.is_iw8  # type: ignore

else:

    def is_iw8(mode: CoreMode) -> bool:
        return mode is CoreMode.MODE_ANN_TO_BANN_OR_SNN or mode is CoreMode.MODE_ANN


def _coord2index(coord: Coord) -> str:
    index = 0

    for i in range(MAX_ROUTING_PATH_LENGTH):
        shift = 4 - i
        value_x, value_y = (coord.x >> shift) & 0b1, (coord.y >> shift) & 0b1
        if HwConfig.COORD_Y_PRIORITY:
            index = (index << 2) | (value_x << 1) | value_y
        else:
            index = (index << 2) | (value_y << 1) | value_x

    return f"{bin(index)[2:].zfill(10)}({index})"


if hasattr(Coord, "to_bin_str"):

    def _coord_to_bin_str(coord: Coord) -> str:
        return coord.to_bin_str()  # type: ignore

else:

    def _to_bin(n: int, keep_bits: int) -> str:
        """Convert an integer to a binary string with a fixed number of bits, removing the prefix '0b'."""
        assert 0 <= n < (1 << keep_bits)
        return bin(n)[2:].zfill(keep_bits)

    def _coord_to_bin_str(coord: Coord) -> str:
        return f"({_to_bin(coord.x, HwConfig.N_BIT_COORD_ADDR)},{_to_bin(coord.y, HwConfig.N_BIT_COORD_ADDR)})"


def _1st_core_coord_repr(coord_lst: list[Coord]) -> str:
    """Represent the first core coordinate in a list of coordinates as a binary string & an index."""
    if coord_lst:
        return _coord_to_bin_str(coord_lst[0]) + ", " + _coord2index(coord_lst[0])
    else:
        return ""
