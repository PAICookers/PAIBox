from collections import UserList, defaultdict
from collections.abc import Generator, Iterable
from typing import Optional, cast

from ..utils import check_elem_same
from .types import EdgeType, NodeType

__all__ = ["SuccGroup", "MergedSuccGroup"]


# class SuccGroup(UserList[EdgeType]):
#     """The successor edges of a node are grouped into a `SuccGroup`."""

#     def __init__(self, edges: Iterable[EdgeType]) -> None:
#         _edges = list(edges)
#         if not check_elem_same(e.source for e in _edges):
#             raise ValueError("All edges must have the same source.")

#         super().__init__(_edges)

#     def iter_nodes_and_edges(self) -> Generator[tuple[NodeType, EdgeType], None, None]:
#         return iter((cast(NodeType, e.target), e) for e in self)

#     def remove_node(self, node: NodeType):
#         """Create a new `SuccGroup` without the edges belonging to the given node. If the node is   \
#             not in the group, return self.
#         """
#         if node in self.nodes:
#             return SuccGroup(
#                 e for (n, e) in self.iter_nodes_and_edges() if n is not node
#             )

#         return self

#     @property
#     def input(self) -> NodeType:
#         return cast(NodeType, self[0].source)

#     @property
#     def edges(self) -> list[EdgeType]:
#         return self.data

#     @property
#     def nodes(self) -> list[NodeType]:
#         return [cast(NodeType, e.target) for e in self]

#     def __eq__(self, other: "SuccGroup") -> bool:
#         """Compare the included edges, but don’t care about the order."""
#         return set(self) == set(other)

#     def __hash__(self) -> int:
#         return hash(tuple(self))

#     def __str__(self) -> str:
#         ind1 = "\t"
#         _repr = f"{self.__class__.__name__}:\n"

#         for node, edge in self.iter_nodes_and_edges():
#             _repr += ind1 + f"Edge {edge.name}: {self.input.name} -> {node.name}\n"

#         return _repr


class SuccGroup:
    def __init__(
        self,
        edges: list[EdgeType],
        nodes: list[NodeType] = [],
        group_type: str = "data",
    ) -> None:
        if group_type == "data" and len(nodes) == 0:
            self.raw_edges = set(edges)
            self.raw_nodes = set([cast(NodeType, e.target) for e in self.raw_edges])
        else:
            self.raw_nodes = set(nodes)
            self.raw_edges = set(edges)
        self.group_type = group_type
        self.edges_dict: dict[NodeType, list[EdgeType]] = defaultdict(list)
        for node in self.raw_nodes:
            self.edges_dict[node] = []
        for e in edges:
            target = cast(NodeType, e.target)
            self.edges_dict[target].append(e)

    @property
    def input(self) -> Optional[NodeType]:
        if len(self.raw_edges) == 0:
            return None
        return cast(NodeType, next(iter(self.raw_edges)).source)

    @property
    def nodes(self) -> list[NodeType]:
        return list(self.raw_nodes)

    @property
    def edges(self) -> list[EdgeType]:
        return list(self.raw_edges)

    def remove_node(self, node: NodeType) -> "SuccGroup":
        if self.group_type == "inhi":
            raise ValueError("Cannot remove node from inhi group.")
        new_nodes = self.raw_nodes - {node}
        new_edges = self.raw_edges - set(self.edges_dict.get(node, []))
        return SuccGroup(list(new_edges), list(new_nodes), group_type=self.group_type)

    def reserve_node(self, reserve_nodes: set[NodeType]) -> Optional["SuccGroup"]:
        new_nodes = self.raw_nodes.intersection(reserve_nodes)
        if len(new_nodes) == 0:
            return None
        new_edges = set()
        for n in new_nodes:
            new_edges.update(self.edges_dict.get(n, []))
        return SuccGroup(list(new_edges), list(new_nodes), group_type=self.group_type)

    def __eq__(self, other: "SuccGroup") -> bool:
        return self.raw_edges == other.raw_edges and self.raw_nodes == other.raw_nodes

    def __hash__(self) -> int:
        return hash((frozenset(self.raw_edges), frozenset(self.raw_nodes)))

    def __str__(self, ind1="\t") -> str:
        _repr = f"{ind1}{self.__class__.__name__}:\n"

        ind1 += "\t"
        for node, edges in self.edges_dict.items():
            if self.group_type == "inhi":
                _repr += ind1 + f"Inhi node: {node.name}\n"
            else:
                for edge in edges:
                    _repr += (
                        ind1
                        + f"Edge {edge.name}: {edge.source.name} -> {edge.target.name}\n"
                    )

        return _repr


class MergedSuccGroup(UserList[SuccGroup]):
    """SuccGroups with intersecting nodes will be merged into a `MergedSuccGroup`."""

    def add_group(self, group: SuccGroup) -> None:
        self.append(group)

    def remove_node(self, node: NodeType) -> list[SuccGroup]:
        # Do not modify the original `SuccGroup` list in for loop.
        to_remove = []
        to_append = []

        for sgrp in self:
            if node in sgrp.nodes:
                to_remove.append(sgrp)
                if len(sgrp.nodes) > 1:
                    # Replace the original group with the new one if it's not empty.
                    new_sgrp = sgrp.remove_node(node)
                    to_append.append(new_sgrp)

        for r in to_remove:
            self.remove(r)

        for a in to_append:
            self.add_group(a)

        return to_remove

    def reserve_node(self, reserve_nodes: set[NodeType]) -> "MergedSuccGroup":
        new_sgrps = []
        for sgrp in self:
            new_sgrp = sgrp.reserve_node(reserve_nodes)
            if new_sgrp is not None:
                new_sgrps.append(new_sgrp)

        return MergedSuccGroup(new_sgrps)

    @property
    def inputs(self) -> list[NodeType]:
        result = []
        for g in self:
            if g.input is not None:
                result.append(g.input)

        return result

    @property
    def nodes(self) -> set[NodeType]:
        _nodes = set()
        for sgrp in self:
            _nodes.update(sgrp.nodes)

        return _nodes

    @property
    def outputs(self) -> dict[NodeType, list[EdgeType]]:
        onodes = defaultdict(list)
        for sgrp in self:
            for node, edges in sgrp.edges_dict.items():
                # A node may belong to multiple edges.
                onodes[node].extend(edges)

        return onodes

    @property
    def num_in(self) -> int:
        return sum(i.num_out for i in self.inputs)

    @classmethod
    def merge(cls, merged_sgrps: list["MergedSuccGroup"]):
        """Merge multiple `MergedSuccGroup` into a new one."""
        merged = cls()
        for m in merged_sgrps:
            merged.extend(m)

        return merged

    # def __eq__(self, other: "MergedSuccGroup") -> bool:
    #     """Compare the included `SuccGroup`, but don’t care about the order."""
    #     return set(self) == set(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __str__(self) -> str:
        ind1 = "\t"
        ind2 = "\t\t"
        _repr = f"{self.__class__.__name__}:\n"
        _repr += ind1 + "Nodes: " + ", ".join(n.name for n in self.nodes) + "\n"

        for sgrp in self:
            _repr += sgrp.__str__(ind1=ind2) + "\n"

        return _repr
