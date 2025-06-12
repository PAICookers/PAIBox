import sys
from collections.abc import Sequence
from typing import Generic, Optional, List, Protocol, TypeVar, Union, cast, runtime_checkable

from paicorelib import MaxPoolingEnable, WeightWidth

from paibox.components import InputProj, Neuron, NeuronSubView
from paibox.types import WeightType
from paibox.components import MatMul2d, Conv2d, FullConn

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
PrttnSliceType = List[slice]  # Partitioned slice type


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
    def index(self) -> List[slice]: ...


def sl_overlap(slice1: PrttnSliceType, slice2: PrttnSliceType) -> bool:
    """Check wether 2 partitioned slice are overlapped."""
    return slice1[3].start < slice2[3].stop and slice2[3].start < slice1[3].stop
    # return slice1[0].start < slice2[0].stop and slice2[0].start < slice1[0].stop and \
    #        slice1[1].start < slice2[1].stop and slice2[1].start < slice1[1].stop and \
    #        slice1[2].start < slice2[2].stop and slice2[2].start < slice1[2].stop


def sl_cover(part: PrttnSliceType, whole: PrttnSliceType) -> bool:
    """Check wherher a partitioned slice is covered by another."""
    return whole[3].start <= part.start and whole[3].stop >= part.stop
    # return whole[0].start <= part[0].start and whole[0].stop >= part[0].stop and \
    #        whole[1].start <= part[1].start and whole[1].stop >= part[1].stop and \
    #        whole[2].start <= part[2].start and whole[2].stop >= part[2].stop


def _cover(self: _HasNodeSliceIntf, o: _HasNodeSliceIntf) -> bool:
    return self.target is o.target and sl_cover(self.index, o.index)


def _overlap(self: _HasNodeSliceIntf, o: _HasNodeSliceIntf) -> bool:
    return self.target is o.target and sl_overlap(self.index, o.index)


def _idx2slice(target: _HasAttrNumOut, index: Optional[List[slice]]) -> PrttnSliceType:
    _nmax = target.num_out

    if index is None:
        # Default to full range for all three dimensions
        shape = target.shape if hasattr(target, 'shape') else (target.num_out, 1, 1)
        return [
            slice(0, shape[0], 1),
            slice(0, shape[1], 1),
            slice(0, shape[2], 1),
            slice(0, _nmax, 1)
        ]
    return index

    # _nmax = target.num_out
    # # print("_nmax", _nmax)

    # if index is None:
    #     return slice(0, _nmax, 1)

    # return slice(*index.indices(_nmax))


class PartitionedSlice:
    pass


class NodeSlice(PartitionedSlice, Generic[_NT]):
    """NodeSlice represents a slice of a input projection or neuron."""

    target: _NT
    __slots__ = ("target", "index")

    def __init__(self, target: _NT, index: Optional[List[slice]] = None) -> None:
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
        # c_len = self.index[0].stop - self.index[0].start
        # h_len = self.index[1].stop - self.index[1].start
        # w_len = self.index[2].stop - self.index[2].start
        # return c_len * h_len * w_len
        return self.index[3].stop - self.index[3].start

    @property
    def num_out(self) -> int:
        # c_len = self.index[0].stop - self.index[0].start
        # h_len = self.index[1].stop - self.index[1].start
        # w_len = self.index[2].stop - self.index[2].start
        # return c_len * h_len * w_len
        return self.index[3].stop - self.index[3].start

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} {self.target.name}"
            # + f"[{self.index[0].start}:{self.index[0].stop}]"
            # + f"[{self.index[1].start}:{self.index[1].stop}]"
            # + f"[{self.index[2].start}:{self.index[2].stop}]"
            + f"[{self.index[3].start}:{self.index[3].stop}]"   
        )

    def __eq__(self, other: "NodeSlice") -> bool:
        # return self.target == other.target and self.index[0] == other.index[0] and \
        #        self.index[1] == other.index[1] and self.index[2] == other.index[2]
        return self.target == other.target and self.index[3] == other.index[3]

    def __hash__(self) -> int:
        # print(f"[INFO] hash {self.index}")
        return hash((self.target, self.index[3].start, self.index[3].stop))
        # return hash((self.target, 
        #             self.index[0].start, self.index[0].stop,
        #             self.index[1].start, self.index[1].stop,
        #             self.index[2].start, self.index[2].stop))


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
        in_index: Optional[List[slice]] = None,
        out_index: Optional[List[slice]] = None,
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
        if isinstance(self.target, Conv2d):
            shape_in = self.target.shape_in
            shape_out = self.target.shape_out
            conn = self.target.connectivity.reshape(shape_in[0], shape_in[1], shape_in[2],
                                                    shape_out[0], shape_out[1], shape_out[2])
            conn_slice = conn[self.in_index[0], self.in_index[1], self.in_index[2], 
                                self.out_index[0], self.out_index[1], self.out_index[2]]
            shape = conn_slice.shape
            conn_slice = conn_slice.reshape(
                shape[0] * shape[1] * shape[2],
                shape[3] * shape[4] * shape[5]
            )
            return conn_slice
        if isinstance(self.target, FullConn):
            return self.target.connectivity[self.in_index[3], self.out_index[3]]

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} {self.target.name}"
            + f"({self.target.source.name} -> {self.target.dest.name})"
            + f"({self.in_index[3].start}:{self.in_index[3].stop})"
            + f"({self.out_index[3].start}:{self.out_index[3].stop})"
            # + f"[{self.in_index[0].start}:{self.in_index[0].stop}]"
            # + f"[{self.in_index[1].start}:{self.in_index[1].stop}]"
            # + f"[{self.in_index[2].start}:{self.in_index[2].stop}]"
            # + f"[{self.out_index[0].start}:{self.out_index[0].stop}]"
            # + f"[{self.out_index[1].start}:{self.out_index[1].stop}]"
            # + f"[{self.out_index[2].start}:{self.out_index[2].stop}]"
        )

    def __eq__(self, other: "EdgeSlice") -> bool:
        return (
            self.target == other.target
            and self.in_index == other.in_index
            and self.out_index == other.out_index
        )

    def __hash__(self) -> int:
        return hash((self.target, self.in_index[3], self.out_index[3]))


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
