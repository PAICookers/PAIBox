import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Any, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import Coord, CoreMode
from paicorelib import ReplicationId as RId

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
    "SuccGroup",
    "MergedSuccGroup",
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
class SuccGroup:
    """A node and all its successor nodes & edges are grouped into a `SuccGroup`."""

    input: NodeType
    nodes: list[NodeType]
    edges: list[EdgeType]  # len(edges) == len(nodes)

    def __eq__(self, other: "SuccGroup") -> bool:
        return self.input == other.input

    def __hash__(self) -> int:
        return hash(self.input)


class MergedSuccGroup:
    """SuccGroups with intersecting nodes will be merged into a `MergedSuccGroup`."""

    def __init__(self, *init_sgrp: SuccGroup) -> None:
        self.nodes: set[NodeType] = set()
        self.groups: list[SuccGroup] = list()
        self.input_nodes: list[NodeType] = list()

        if init_sgrp:
            for sgrp in init_sgrp:
                self.add_group(sgrp)

    def add_group(self, group: SuccGroup) -> None:
        self.groups.append(group)
        self.nodes.update(group.nodes)
        self.input_nodes.append(group.input)

    @property
    def outputs(self) -> dict[NodeType, list[EdgeType]]:
        onodes = defaultdict(list)
        for group in self.groups:
            for node, edge in zip(group.nodes, group.edges):
                assert edge.dest.name == node.name
                onodes[node].append(edge)

        return onodes

    @property
    def num_in(self) -> int:
        return sum(input_node.num_out for input_node in self.input_nodes)

    @classmethod
    def merge(cls, merged_sgrps: list["MergedSuccGroup"]) -> "MergedSuccGroup":
        merged = cls()
        for merged_sgrp in merged_sgrps:
            merged.nodes.update(merged_sgrp.nodes)
            merged.groups.extend(merged_sgrp.groups)
            merged.input_nodes.extend(merged_sgrp.input_nodes)
        return merged

    def __hash__(self) -> int:
        return hash(tuple(self.nodes))

    def dump(self) -> None:
        print("MergedSuccGroup:")
        for group in self.groups:
            print(f"\tGroup: of {group.input.name}")
            for node, edge in zip(group.nodes, group.edges):
                print(
                    f"\t\tnode: {node.name}, edge: {edge.name}: {edge.source.name} -> {edge.dest.name}"
                )
        print("\tNodes:")
        for node in self.nodes:
            print(f"\t\tnode: {node.name}")


@dataclass(frozen=True)
class NeuSegment:
    target: DestNodeType
    index: NeuSlice  # slice like slice(x, y, 1)
    offset: int
    repeat: int = 1

    def __getitem__(self, s: slice) -> "NeuSegment":
        # if isinstance(idx, int):
        #     if idx < 0:
        #         _idx = self.n_neuron + idx
        #         if _idx < 0:
        #             raise ValueError(f"index out of range: {idx} < 0")
        #     else:
        #         _idx = idx
        #         if _idx > self.n_neuron - 1:
        #             raise ValueError(f"index out of range: {idx} > {self.n_neuron-1}")

        #     start = self.index.start + _idx
        #     end = start + 1

        #     return NeuSegment(
        #         self.target,
        #         NeuSlice(start, end, self.index.step),
        #         self.offset + idx,
        #         self.repeat,
        #     )
        _idx_start = s.start if s.start is not None else 0
        if s.stop is None:
            _idx_stop = self.n_neuron
        elif s.stop < 0:
            _idx_stop = self.n_neuron + s.stop
        else:
            _idx_stop = s.stop

        if (_n_idx := _idx_stop - _idx_start) > self.n_neuron:
            raise IndexError(f"index out of range: {_n_idx} > {self.n_neuron}")

        start = self.index.start + _idx_start
        end = self.index.start + _idx_stop

        return NeuSegment(
            self.target,
            NeuSlice(start, end, self.index.step),
            self.offset + _idx_start,
            self.repeat,
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
        return list(range(self.offset, self.offset + self.n_occupied_addr, 1))

    @property
    def _addr_ram_repr(self) -> slice:
        """Represent the slice of neuron RAM address."""
        return slice(self.offset, self.offset + self.n_occupied_addr, self.repeat)


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
