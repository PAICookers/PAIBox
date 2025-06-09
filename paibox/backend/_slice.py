import sys
from collections.abc import Sequence
from typing import Generic, Optional, Protocol, TypeVar, Union, cast, runtime_checkable

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


class _HasAttrNumOut(Protocol):
    __slots__ = ()

    @property
    def num_out(self) -> int: ...


@runtime_checkable
class _HasNodeSliceIntf(Protocol):
    __slots__ = ()

    @property
    def target(self) -> NodeType: ...

    @property
    def index(self) -> slice: ...


def sl_overlap(slice1: PrttnSliceType, slice2: PrttnSliceType) -> bool:
    """Check wether 2 partitioned slice are overlapped."""
    return slice1.start < slice2.stop and slice2.start < slice1.stop


def sl_cover(part: PrttnSliceType, whole: PrttnSliceType) -> bool:
    """Check wherher a partitioned slice is covered by another."""
    return whole.start <= part.start and whole.stop >= part.stop


def _cover(self: _HasNodeSliceIntf, o: _HasNodeSliceIntf) -> bool:
    return self.target is o.target and sl_cover(self.index, o.index)


def _overlap(self: _HasNodeSliceIntf, o: _HasNodeSliceIntf) -> bool:
    return self.target is o.target and sl_overlap(self.index, o.index)


def _idx2slice(target: _HasAttrNumOut, index: Optional[slice]) -> PrttnSliceType:
    _nmax = target.num_out

    if index is None:
        return slice(0, _nmax, 1)

    return slice(*index.indices(_nmax))


class PartitionedSlice:
    pass


class NodeSlice(PartitionedSlice, Generic[_NT]):
    """NodeSlice represents a slice of a input projection or neuron."""

    target: _NT
    __slots__ = ("target", "index")

    def __init__(self, target: _NT, index: Optional[slice] = None) -> None:
        self.target = target
        self.index = _idx2slice(target, index)

    def covered_by(
        self, other: Union[_HasNodeSliceIntf, Sequence[_HasNodeSliceIntf]]
    ) -> bool:
        """Check wherher `self` is covered by the other one or a list of node slices."""
        if isinstance(other, _HasNodeSliceIntf):
            return _cover(self, other)

        return any(_cover(self, sl) for sl in other)

    def overlap(
        self, other: Union[_HasNodeSliceIntf, Sequence[_HasNodeSliceIntf]]
    ) -> bool:
        """Check whether `self` & the other one or a list of node slices are overlapped."""
        if isinstance(other, _HasNodeSliceIntf):
            return _overlap(self, other)

        return any(_overlap(self, sl) for sl in other)

    @property
    def num_in(self) -> int:
        return self.index.stop - self.index.start

    @property
    def num_out(self) -> int:
        return self.index.stop - self.index.start

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
    """Input slice represents a slice of a `InputProj`."""

    pass


class NeuronSlice(NodeSlice[Neuron]):
    """Neuron slice represents a slice of a `Neuron`."""

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
    def target_chip_idx(self) -> int:
        return self.target.target_chip_idx

    @property
    def view(self) -> NeuronSubView:
        return NeuronSubView(self.target, self.index)


class EdgeSlice(PartitionedSlice):
    """EdgeSlice records the slices corresponding to the two end nodes of the target synapse."""

    target: EdgeType
    __slots__ = ("target", "in_index", "out_index")

    def __init__(
        self,
        target: EdgeType,
        in_index: Optional[slice] = None,
        out_index: Optional[slice] = None,
    ) -> None:
        self.target = target
        self.in_index = _idx2slice(target.source, in_index)
        self.out_index = _idx2slice(target.dest, out_index)

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
            + f"({self.target.source.name} -> {self.target.dest.name})"
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
        return hash((self.target, self.in_index, self.out_index))


NodeSliceType: TypeAlias = Union[InputSlice, NeuronSlice]
EdgeSliceType: TypeAlias = EdgeSlice
SourceSliceType: TypeAlias = NodeSliceType
DestSliceType: TypeAlias = NeuronSlice


def covered_by(node_slice_like: _HasNodeSliceIntf, node_slice: NodeSliceType) -> bool:
    """Check wherher `node_slice_like` is covered by `node_slice`.

    NOTE: A generic implementation of `NodeSlice.covered_by`.
    """
    return _cover(node_slice_like, node_slice)


def overlap(node_slice_like: _HasNodeSliceIntf, node_slice: NodeSliceType) -> bool:
    """Check whether `node_slice_like` & `node_slice` are overlapped.

    NOTE: A generic implementation of `NodeSlice.overlap`.
    """
    return _overlap(node_slice_like, node_slice)


def node_sl_lst_overlap(
    node_or_sl: Union[NodeType, NodeSliceType, Sequence[NodeSliceType]],
    node_sl_lst: Sequence[NodeSliceType],
) -> bool:
    """Check whether a single node, node slice or list of node slices overlaps with another list of node slices `node_sl_lst`."""
    if isinstance(node_or_sl, Sequence):
        return any(node.overlap(node_sl_lst) for node in node_or_sl)

    if isinstance(node_or_sl, (InputProj, Neuron)):
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
        return any(node.covered_by(whole_sl_lst) for node in node_or_sl)

    if isinstance(node_or_sl, (InputProj, Neuron)):
        node_sl = NodeSlice(node_or_sl)
    else:
        node_sl = node_or_sl

    return node_sl.covered_by(whole_sl_lst)
