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
class NeuSegment:
    target: DestNodeType
    index: NeuSliceType
    """Indicate that this segment is cut from the target, i.e. `target[index]`."""
    offset_nram: int
    """The address offset of the segment in the NRAM."""
    repeat: int = 1

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
            self.target, new_slice, self.offset_nram + _start_offset, self.repeat
        )

    @property
    def n_neuron(self) -> int:
        """#N of unique neurons in this segment."""
        return self.index.stop - self.index.start

    @property
    def n_occupied_addr(self) -> int:
        """#N of neuron addresses the segment occupies in the RAM."""
        return self.repeat * self.n_neuron

    @property
    def attrs(self) -> dict[str, Any]:
        return self.target._slice_attrs(self.index)

    @property
    def addr_ram(self) -> list[int]:
        """Convert index of neuron into RAM address."""
        return list(range(self.offset_nram, self.offset_nram + self.n_occupied_addr, 1))

    @property
    def _addr_ram_repr(self) -> slice:
        """Represent the slice of neuron RAM address."""
        return slice(
            self.offset_nram, self.offset_nram + self.n_occupied_addr, self.repeat
        )


NeuSegOfCorePlm: TypeAlias = list[NeuSegment]
NeuSegOfCoreBlock: TypeAlias = list[NeuSegOfCorePlm]


@dataclass(frozen=True)
class AxonCoord:
    tick_relative: int
    addr_axon: int

    @classmethod
    def build(cls, tick_relative: int, addr_axon: int):
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
        return f"({_to_bin(coord.x, HwConfig.N_BIT_CORE_X)},{_to_bin(coord.y, HwConfig.N_BIT_CORE_Y)})"


def _1st_core_coord_repr(coord_lst: list[Coord]) -> str:
    """Represent the first core coordinate in a list of coordinates as a binary string & an index."""
    if coord_lst:
        return _coord_to_bin_str(coord_lst[0]) + ", " + _coord2index(coord_lst[0])
    else:
        return ""
