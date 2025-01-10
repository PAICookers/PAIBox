from collections import UserList, defaultdict
from collections.abc import Generator, Iterable
from typing import cast

from ..utils import check_elem_same
from .types import EdgeType, NodeType

__all__ = ["SuccGroup", "MergedSuccGroup"]


class SuccGroup(UserList[EdgeType]):
    """The successor edges of a node are grouped into a `SuccGroup`."""

    def __init__(self, edges: Iterable[EdgeType]) -> None:
        _edges = list(edges)
        if not check_elem_same(e.source for e in _edges):
            raise ValueError("All edges must have the same source.")

        super().__init__(_edges)

    def iter_nodes_and_edges(self) -> Generator[tuple[NodeType, EdgeType], None, None]:
        return iter((cast(NodeType, e.dest), e) for e in self)

    def remove_node(self, node: NodeType):
        """Create a new `SuccGroup` without the edges belonging to the given node. If the node is   \
            not in the group, return self.
        """
        if node in self.nodes:
            return SuccGroup(
                e for (n, e) in self.iter_nodes_and_edges() if n is not node
            )

        return self

    @property
    def input(self) -> NodeType:
        return cast(NodeType, self[0].source)

    @property
    def edges(self) -> list[EdgeType]:
        return self.data

    @property
    def nodes(self) -> list[NodeType]:
        return [cast(NodeType, e.dest) for e in self]

    def __eq__(self, other: "SuccGroup") -> bool:
        """Compare the included edges, but don’t care about the order."""
        return set(self) == set(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __str__(self) -> str:
        ind1 = "\t"
        _repr = f"{self.__class__.__name__}:\n"

        for node, edge in self.iter_nodes_and_edges():
            _repr += ind1 + f"Edge {edge.name}: {self.input.name} -> {node.name}\n"

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

    @property
    def inputs(self) -> list[NodeType]:
        return [g.input for g in self]

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
            for node, edge in sgrp.iter_nodes_and_edges():
                # A node may belong to multiple edges.
                onodes[node].append(edge)

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
        _repr += ind1 + f"Nodes: " + ", ".join(n.name for n in self.nodes) + "\n"

        for sgrp in self:
            _repr += ind1 + f"Group of {sgrp.input.name}:\n"
            for node, edge in sgrp.iter_nodes_and_edges():
                _repr += ind2 + f"Edge {edge.name}: {sgrp.input.name} -> {node.name}\n"

        return _repr
