from abc import abstractmethod
from collections import UserList, defaultdict
from collections.abc import Iterable
from typing import cast

from .types import EdgeType, NodeType

__all__ = ["BaseGroup", "InhiGroup", "DataGroup", "MergedGroup"]


class BaseGroup:
    def __init__(self, nodes: list[NodeType]) -> None:
        self.raw_nodes = set(nodes)

    @property
    def nodes(self) -> list[NodeType]:
        return list(self.raw_nodes)

    @property
    @abstractmethod
    def input(self) -> NodeType | None:
        pass

    @property
    @abstractmethod
    def edges(self) -> list[EdgeType]:
        pass

    @abstractmethod
    def reserve_node(self, reserve_nodes: Iterable[NodeType]) -> "BaseGroup":
        pass

    @abstractmethod
    def __eq__(self, other: "BaseGroup") -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __str__(self, ind1="\t") -> str:
        pass


class InhiGroup(BaseGroup):
    def __init__(self, nodes: list[NodeType]) -> None:
        super().__init__(nodes)

    @property
    def input(self):
        return None

    @property
    def edges(self) -> list[EdgeType]:
        return []

    def reserve_node(self, reserve_nodes: list[NodeType]) -> "InhiGroup":
        new_nodes = self.raw_nodes.intersection(reserve_nodes)
        return InhiGroup(list(new_nodes))

    def __eq__(self, other: "BaseGroup") -> bool:
        return isinstance(other, InhiGroup) and self.raw_nodes == other.raw_nodes

    def __hash__(self) -> int:
        return hash((frozenset(self.raw_nodes)))

    def __str__(self, ind1="\t") -> str:
        _repr = f"{ind1}{self.__class__.__name__}:\n"

        ind1 += "\t"
        for node in self.raw_nodes:
            _repr += ind1 + f"Inhi node: {node.name}\n"
        return _repr


class DataGroup(BaseGroup):
    def __init__(self, edges: list[EdgeType]) -> None:
        self.raw_edges = set(edges)
        super().__init__([cast(NodeType, e.target) for e in self.raw_edges])
        self.edges_dict: dict[NodeType, list[EdgeType]] = defaultdict(list)
        for e in edges:
            target = cast(NodeType, e.target)
            self.edges_dict[target].append(e)

    @property
    def input(self) -> NodeType | None:
        return cast(NodeType, next(iter(self.raw_edges)).source)

    @property
    def edges(self) -> list[EdgeType]:
        return list(self.raw_edges)

    def reserve_node(self, reserve_nodes: Iterable[NodeType]) -> "DataGroup":
        new_nodes = self.raw_nodes.intersection(reserve_nodes)
        new_edges = set()
        for n in new_nodes:
            new_edges.update(self.edges_dict.get(n, []))
        return DataGroup(list(new_edges))

    def __eq__(self, other: "BaseGroup") -> bool:
        if isinstance(other, DataGroup):
            return (
                self.raw_edges == other.raw_edges and self.raw_nodes == other.raw_nodes
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((frozenset(self.raw_edges), frozenset(self.raw_nodes)))

    def __str__(self, ind1="\t") -> str:
        _repr = f"{ind1}{self.__class__.__name__}:\n"

        ind1 += "\t"
        for _, edges in self.edges_dict.items():
            for edge in edges:
                _repr += (
                    ind1
                    + f"Edge {edge.name}: {edge.source.name} -> {edge.target.name}\n"
                )

        return _repr


class MergedGroup(UserList[BaseGroup]):
    """BaseGroups with intersecting nodes will be merged into a `MergedGroup`."""

    def add_group(self, group: BaseGroup) -> None:
        self.append(group)

    def reserve_node(self, reserve_nodes: Iterable[NodeType]) -> "MergedGroup":
        new_grps = []
        for grp in self:
            new_grp = grp.reserve_node(reserve_nodes)
            if len(new_grp.nodes) > 0:
                new_grps.append(new_grp)

        return MergedGroup(new_grps)

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
        for grp in self:
            _nodes.update(grp.nodes)

        return _nodes

    @property
    def outputs(self) -> dict[NodeType, list[EdgeType]]:
        onodes = defaultdict(list)
        for grp in self:
            if not isinstance(grp, DataGroup):
                continue
            for node, edges in grp.edges_dict.items():
                # A node may belong to multiple edges.
                onodes[node].extend(edges)

        return onodes

    @property
    def num_in(self) -> int:
        return sum(i.num_out for i in self.inputs)

    @classmethod
    def merge(cls, merged_grps: list["MergedGroup"]):
        """Merge multiple `MergedGroup` into a new one."""
        merged = cls()
        for m in merged_grps:
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
