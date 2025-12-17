from collections import defaultdict
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, cast

import numpy as np
from paicorelib import MaxPoolingEnable, WeightWidth

from paibox.components import InputProj, Neuron, OfflineNeuron, OnlineNeuron
from paibox.types import WeightType

from .types import CustomIndex, EdgeType, NodeType

__all__ = [
    "SubEdge",
    "SubNode",
    "SubInput",
    "SubNeuron",
    "SubEdgeType",
    "SubNodeType",
    "SubSourceType",
    "SubDestType",
]

_NT = TypeVar("_NT", bound=NodeType)


class _HasAttrNumOut(Protocol):
    __slots__ = ()

    @property
    def num_out(self) -> int: ...


def get_index(
    target: _HasAttrNumOut,
    index: list[int] | None = None,
    copy_id: list[int] | None = None,
    custom_index: list[CustomIndex] | None = None,
) -> tuple[list[CustomIndex], bool]:
    _nmax = target.num_out
    if custom_index is not None:
        return custom_index, any(c.copy_id for c in custom_index)
    elif index is None and copy_id is None:
        return [CustomIndex(i, 0) for i in range(_nmax)], False
    elif index is not None and copy_id is None:
        return [CustomIndex(i, 0) for i in index], False
    elif index is not None and copy_id is not None:
        assert len(index) == len(copy_id), "index and copy_id must have the same length"
        contain_copy = any(c for c in copy_id)
        return [CustomIndex(i, c) for i, c in zip(index, copy_id)], contain_copy
    else:
        raise ValueError("index is None but copy_id is not None")


def list_to_str(lst: list, show_all: bool = False) -> str:
    """
    返回列表元素的字符串表示。
    - 元素数量 <= 5，返回完整列表字符串；
    - 元素数量 > 5，返回前3个和最后2个，中间用 ... 省略。
    """
    n = len(lst)
    if n <= 5 or show_all:
        return f"[{', '.join(str(x) for x in lst)}]"
    else:
        return f"[{', '.join(str(x) for x in lst[:3]) + ' ... ' + ' '.join(str(x) for x in lst[-2:])}]"


class SubNodeBase:
    pass


class SubNode(SubNodeBase, Generic[_NT]):
    target: _NT
    __slots__ = (
        "target",
        "index",
        "contain_copy",
        "raw_index_list",
        "custom_index_set",
    )

    def __init__(
        self,
        target: _NT,
        raw_index: list[int] | None = None,
        copy_id: list[int] | None = None,
        custom_index: list[CustomIndex] | None = None,
    ) -> None:
        self.target = target

        # the index order matters, so we use list instead of set
        self.index, self.contain_copy = get_index(
            target, raw_index, copy_id, custom_index
        )

        # prepare a list of raw indexes for easy access in connectivity slicing
        self.raw_index_list = [i.index for i in self.index]

        # sometimes we need to quickly check intersection, so we prepare a set
        self.custom_index_set = set(self.index)

    @property
    def num_in(self) -> int:
        return len(self.index)

    @property
    def num_out(self) -> int:
        return len(self.index)

    def handle_copy(self, copy_map: dict[int, list[int]]) -> None:
        """copy_list: dict[index, list[copy_id]]"""
        for raw_index, copy_id_list in copy_map.items():
            if CustomIndex(raw_index, 0) in self.custom_index_set:
                for copy_id in copy_id_list:
                    if CustomIndex(raw_index, copy_id) not in self.custom_index_set:
                        self.index.append(CustomIndex(raw_index, copy_id))
                        self.custom_index_set.add(CustomIndex(raw_index, copy_id))
                        self.raw_index_list.append(raw_index)

    def __str__(self) -> str:
        info = f"{len(self.index)}"
        if self.contain_copy:
            info += ", copy"
        return f"{type(self).__name__} {self.target.name}({info})" + list_to_str(
            self.index
        )

    def __eq__(self, other: "SubNode") -> bool:
        return self.target == other.target and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.target, frozenset(self.index)))


class SubInput(SubNode[InputProj]):
    pass


class SubNeuron(SubNode[Neuron]):
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
        if isinstance(self.target, OfflineNeuron):
            return self.target.pool_max
        else:
            raise ValueError("OnlineNeuron don't have pool_max")

    @property
    def lateral_inhi_target(self) -> set[OnlineNeuron]:
        if isinstance(self.target, OnlineNeuron):
            return self.target.lateral_inhi_target
        else:
            raise ValueError("OfflineNeuron don't have later_inhi_target")

    @property
    def lateral_inhi_source(self) -> set[OnlineNeuron]:
        if isinstance(self.target, OnlineNeuron):
            return self.target.lateral_inhi_source
        else:
            raise ValueError("OfflineNeuron don't have later_inhi_source")

    @property
    def target_chip_idx(self) -> int:
        return self.target.target_chip_idx


class SubEdge:
    """EdgeSlice records the slices corresponding to the two end nodes of the target synapse."""

    target: EdgeType
    __slots__ = ("target", "dest", "source")

    def __init__(
        self,
        target: EdgeType,
        in_raw_index: list[int] | None = None,
        in_copy_id: list[int] | None = None,
        in_custom_index: list[CustomIndex] | None = None,
        out_raw_index: list[int] | None = None,
        out_copy_id: list[int] | None = None,
        out_custom_index: list[CustomIndex] | None = None,
    ) -> None:
        self.target = target
        self.source = self.get_source(in_raw_index, in_copy_id, in_custom_index)
        self.dest = self.get_dest(out_raw_index, out_copy_id, out_custom_index)

    def get_source(
        self,
        in_raw_index: list[int] | None = None,
        in_copy_id: list[int] | None = None,
        in_custom_index: list[CustomIndex] | None = None,
    ) -> SubInput | SubNeuron:
        if isinstance(self.target.source, InputProj):
            return SubInput(
                self.target.source,
                raw_index=in_raw_index,
                copy_id=in_copy_id,
                custom_index=in_custom_index,
            )
        else:
            return SubNeuron(
                cast(Neuron, self.target.source),
                raw_index=in_raw_index,
                copy_id=in_copy_id,
                custom_index=in_custom_index,
            )

    def get_dest(
        self,
        out_raw_index: list[int] | None = None,
        out_copy_id: list[int] | None = None,
        out_custom_index: list[CustomIndex] | None = None,
    ) -> SubNeuron:
        return SubNeuron(
            cast(Neuron, self.target.target),
            raw_index=out_raw_index,
            copy_id=out_copy_id,
            custom_index=out_custom_index,
        )

    def handle_copy(self, copy_map: dict[int, list[int]]) -> None:
        self.dest.handle_copy(copy_map)

    @property
    def weight_width(self) -> WeightWidth:
        return self.target.weight_width

    @property
    def in_index_list(self) -> list[int]:
        return self.source.raw_index_list

    @property
    def out_index_list(self) -> list[int]:
        return self.dest.raw_index_list

    @property
    def connectivity(self) -> WeightType:
        return self.target.connectivity[np.ix_(self.in_index_list, self.out_index_list)]

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} {self.target.name}"
            + f"({self.target.source.name} -> {self.target.target.name})"
            + list_to_str(self.source.index)
            + list_to_str(self.dest.index)
        )

    def __eq__(self, other: "SubEdge") -> bool:
        return (
            self.target == other.target
            and self.source == other.source
            and self.dest == other.dest
        )

    def __hash__(self) -> int:
        return hash((self.target, self.source, self.dest))


SubEdgeType = SubEdge
SubNodeType = SubInput | SubNeuron
SubSourceType = SubNodeType
SubDestType = SubNeuron


def sub_node_overlap(
    sub_node_a: NodeType | SubNodeType | Sequence[SubNodeType],
    sub_node_b: NodeType | SubNodeType | Sequence[SubNodeType],
) -> bool:
    """Check whether a single node, sub node or list of sub nodes overlaps with another one."""

    def to_sequence(
        node_or_subnode: NodeType | SubNodeType | Sequence[SubNodeType],
    ) -> Sequence[SubNodeType]:
        if isinstance(node_or_subnode, Sequence):
            return node_or_subnode
        elif isinstance(node_or_subnode, NodeType):
            if isinstance(node_or_subnode, InputProj):
                return [SubInput(node_or_subnode)]
            elif isinstance(node_or_subnode, Neuron):
                return [SubNeuron(node_or_subnode)]
            else:
                raise TypeError(f"Unsupported node type {type(node_or_subnode)}")
        else:
            return [node_or_subnode]

    def to_group(
        sub_node_seq: Sequence[SubNodeType],
    ) -> dict[int, list[set[CustomIndex]]]:
        group = defaultdict(list)
        for sub_node in sub_node_seq:
            group[id(sub_node.target)].append(sub_node.custom_index_set)
        return group

    seq_a = to_sequence(sub_node_a)
    seq_b = to_sequence(sub_node_b)

    group_a = to_group(seq_a)
    group_b = to_group(seq_b)

    for tgt in group_a.keys() & group_b.keys():
        for set_a in group_a[tgt]:
            for set_b in group_b[tgt]:
                if set_a & set_b:
                    return True

    return False
