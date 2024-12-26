import sys
from collections.abc import Sequence
from typing import Generic, Optional, Protocol, TypeVar, Union, cast

from paicorelib import MaxPoolingEnable, WeightWidth

from paibox.components import InputProj, Neuron, NeuronSubView
from paibox.types import WeightType

from .types import EdgeType, NodeType

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


__all__ = [
    "PartitionedSlice",
    "NodeSlice",
    "EdgeSlice",
    "InputSlice",
    "NeuronSlice",
    "EdgeSlice",
    "NodeSliceType",
    "EdgeSliceType",
    "SourceSliceType",
    "DestSliceType",
]

_NT = TypeVar("_NT", bound=NodeType)
PrttnSliceType = slice  # Partitioned slice type


def sl_overlap(slice1: PrttnSliceType, slice2: PrttnSliceType) -> bool:
    """Check wether 2 partitioned slice are overlapped."""
    return slice1.start < slice2.stop and slice2.start < slice1.stop


def sl_cover(part: PrttnSliceType, whole: PrttnSliceType) -> bool:
    """Check wherher a partitioned slice is covered by another."""
    return whole.start <= part.start and whole.stop >= part.stop


class _HasAttrNumOut(Protocol):

    @property
    def num_out(self) -> int: ...


def _idx2slice(
    target: _HasAttrNumOut, index: Optional[Union[int, slice]]
) -> PrttnSliceType:
    if index is None:
        return slice(0, target.num_out)
    if isinstance(index, int):
        return slice(index, index)

    return index


class PartitionedSlice:
    pass


class NodeSlice(PartitionedSlice, Generic[_NT]):
    __slots__ = ["target", "index"]

    def __init__(self, target: _NT, index: Optional[Union[int, slice]] = None) -> None:
        self.target = target
        self.index = _idx2slice(target, index)

    def overlap(self, other: Union["NodeSlice", Sequence["NodeSlice"]]) -> bool:
        """Check whether self & the other one are overlapped."""

        def _overlap(o: NodeSlice) -> bool:
            return self.target == o.target and sl_overlap(self.index, o.index)

        if isinstance(other, NodeSlice):
            return _overlap(other)
        else:
            return any(_overlap(sl) for sl in other)

    def covered_by(self, other: Union["NodeSlice", Sequence["NodeSlice"]]) -> bool:
        """Check wherher self is covered by the other one."""

        def _cover(o: NodeSlice) -> bool:
            return self.target == o.target and sl_cover(self.index, o.index)

        if isinstance(other, NodeSlice):
            return _cover(other)
        else:
            return any(_cover(sl) for sl in other)

    @property
    def num_out(self) -> int:
        return self.index.stop - self.index.start

    @property
    def num_in(self) -> int:
        return self.num_in

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} {self.target.name}"
            + f"[{self.index.start}:{self.index.stop}]"
        )

    def __eq__(self, other: "NodeSlice") -> bool:
        return self.target == other.target and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.target, self.index.start, self.index.stop))


class InputSlice(NodeSlice[InputProj]):
    pass


class NeuronSlice(NodeSlice[Neuron]):
    @property
    def unrolling_factor(self) -> int:
        return self.target.unrolling_factor

    @property
    def tick_wait_start(self) -> int:
        return self.target.tick_wait_start

    @property
    def tick_wait_end(self) -> int:
        return self.target.tick_wait_end

    @property
    def pool_max(self) -> MaxPoolingEnable:
        return self.target.pool_max

    @property
    def view(self) -> NeuronSubView:
        return NeuronSubView(self.target, self.index)


class EdgeSlice(PartitionedSlice):
    __slots__ = ["target", "in_index", "out_index"]

    def __init__(
        self,
        target: EdgeType,
        in_index: Optional[Union[int, slice]] = None,
        out_index: Optional[Union[int, slice]] = None,
    ) -> None:
        self.target = target
        self.in_index = _idx2slice(target, in_index)
        self.out_index = _idx2slice(target, out_index)

    @property
    def source(self) -> Union[InputSlice, NeuronSlice]:
        if isinstance(self.target.source, InputProj):
            return InputSlice(self.target.source, self.in_index)
        else:
            return NeuronSlice(cast(Neuron, self.target.source), self.in_index)

    @property
    def dest(self) -> NeuronSlice:
        return NeuronSlice(cast(Neuron, self.target.dest), self.out_index)

    @property
    def weight_width(self) -> WeightWidth:
        return self.target.weight_width

    @property
    def connectivity(self) -> WeightType:
        return self.target.connectivity[self.in_index, self.out_index]

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} {self.target.name}"
            + f"[{self.in_index.start}:{self.in_index.stop}]"
            + f"[{self.out_index.start}:{self.out_index.stop}]"
        )

    def __eq__(self, other: "EdgeSlice") -> bool:
        return (
            self.target == other.target
            and self.in_index == other.in_index
            and self.out_index == other.out_index
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.target,
                self.in_index.start,
                self.in_index.stop,
                self.out_index.start,
                self.out_index.stop,
            )
        )


NodeSliceType: TypeAlias = Union[InputSlice, NeuronSlice]
EdgeSliceType: TypeAlias = EdgeSlice
SourceSliceType: TypeAlias = NodeSliceType
DestSliceType: TypeAlias = NeuronSlice


def node_sl_lst_overlap(
    node_or_sl: Union[NodeType, NodeSliceType, Sequence[NodeSliceType]],
    node_sl_lst: Sequence[NodeSliceType],
) -> bool:
    """Check whether a single node, node slice or list of node slices overlaps with another list of node slices `node_sl_lst`."""
    if isinstance(node_or_sl, Sequence):
        for node in node_or_sl:
            if node.overlap(node_sl_lst):
                return True

        return False
    else:
        if isinstance(node_or_sl, NodeType):
            node_sl = NodeSlice(node_or_sl)
        else:
            node_sl = node_or_sl

        return node_sl.overlap(node_sl_lst)


def node_sl_lst_covered_by(
    node_or_sl: Union[NodeType, NodeSliceType, Sequence[NodeSliceType]],
    whole_sl_lst: Sequence[NodeSliceType],
) -> bool:
    """Check whether another list of node slices `whole_sl_lst` covers a single node, node slice, or list of node slices."""
    if isinstance(node_or_sl, Sequence):
        for node in node_or_sl:
            if node.covered_by(whole_sl_lst):
                return True

        return False
    else:
        if isinstance(node_or_sl, NodeType):
            node_sl = NodeSlice(node_or_sl)
        else:
            node_sl = node_or_sl

        return node_sl.covered_by(whole_sl_lst)
