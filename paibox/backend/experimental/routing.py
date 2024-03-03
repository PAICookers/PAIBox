from typing import Any, List, Optional, Sequence, final

from paicorelib.v2.routing_defs import RoutingDirection as Direction
from paicorelib.v2.routing_defs import RoutingDirectionIdx as DirectionIdx
from paicorelib.v2.routing_defs import RoutingNodeLevel as Level
from paicorelib.v2.routing_defs import RoutingNodeStatus as NodeStatus
from paicorelib.v2.routing_defs import get_node_consumption

from ...exceptions import NotSupportedError
from ..placement import CorePlacement

"""
    This is an alternative to the routing tree that \
    does the same thing as the development version \
    but is more complex.

    Some functions are still not implemented and \
    will not be developed until the solution is \
    reconsidered later.
"""


class RoutingNode:
    def __init__(
        self,
        level: Level,
        data: Optional[Any] = None,
        *,
        tag: Optional[str] = None,
    ) -> None:
        """Instance a tree node with `level`. \
            For a node with level Lx > 0, after created, \
                the length of children is `node_capacity`.

            For a node with level L0, it is a leaf node.

        Args:
            - level: the node level.
            - data: the data hanging on the node. Optional.
            - tag: a tag for user to identify. Optional.
        """
        self._level = level
        self._children: List["RoutingNode"] = []
        self.item = data
        self.tag = tag

        self._status = NodeStatus.ALL_EMPTY

    def add_item(self, data: Any) -> None:
        """Add data to its item. Only used for L0-level node."""
        self.item = data
        self._status = NodeStatus.OCCUPIED

    def add_child(self, child: "RoutingNode") -> bool:
        if self.level == Level.L0:
            # L0-level node cannot add child.
            raise AttributeError(f"L0-level node cannot add child")

        if self.level - child.level != 1:
            raise AttributeError(
                f"The node with level {child.level} can not be a child"
            )

        if self.is_full():
            return False

        self._children.append(child)

        return True

    def get_avail_child(self, method: str = "nearest") -> Optional["RoutingNode"]:
        if self.is_children_all_status(NodeStatus.OCCUPIED):
            return None

        for child in self.children:
            if child.status != NodeStatus.OCCUPIED:
                return child

    def find_node_by_path(self, path: Sequence[Direction]) -> "RoutingNode":
        """Find node by the path of `Direction`.

        Description:
            Find by starting at this level based on the path provided. \
            Take `path[0]` each time and then do a recursive search.

        NOTE: The length of path <= the level of this node.
        """
        if len(path) == 0:
            return self

        if len(path) > self.level:
            raise ValueError(
                f"The length of path {len(path)} > level of node {self.level}"
            )

        idx = path[0].to_index()
        if idx > len(self.children) - 1:
            raise IndexError(f"Index out of range: {idx} > {len(self.children) - 1}")

        sub_node = self.children[idx]

        if len(path) > 1:
            return sub_node.find_node_by_path(path[1:])
        else:
            return sub_node

    def find_node_by_tag(self, tag: str) -> Optional["RoutingNode"]:
        """Searches for nodes by tag using DFS.

        Args:
            - tag: the tag string.

        Returns:
            - the node if found. Otherwise return `None`.
        """

        def dfs_preorder(root: RoutingNode, tag: str) -> Optional[RoutingNode]:
            if root.tag == tag:
                return root
            elif root.level == Level.L0:
                return None
            else:
                for child in root.children:
                    node = dfs_preorder(child, tag)
                    if node:
                        return node

        return dfs_preorder(self, tag)

    def get_node_path(self, node: "RoutingNode") -> List[Direction]:
        """Return a direction path from L4 to the level of `node`.

        Args:
            - node: the node with level <= `self.level`.

        Return:
            - A list of `Direction` from L4 to L0.
        """
        if node.level > self.level:
            raise ValueError(f"The node with level {node.level} is not in self")

        if node.level == self.level:
            if node != self:
                raise ValueError(f"The node with level {node.level} is not in self")

            return []

        path = []

        def dfs_preorder(root: RoutingNode) -> bool:
            i = 0
            for child in root.children:
                path.append(DirectionIdx[i])
                if child is node:
                    return True
                else:
                    if dfs_preorder(child):
                        return True
                    else:
                        path.pop(-1)

                i += 1

            return False

        if dfs_preorder(self):
            return path
        else:
            raise ValueError(f"The node with level {node.level} is not in self")

    def get_lx_nodes(self, lx: Level, method: str = "nearest") -> List["RoutingNode"]:
        if lx > self.level:
            raise ValueError(f"The node with level {lx} is not in self")

        if lx == self.level:
            return [self]

        nodes = []

        def dfs_preorder(root: RoutingNode, lx: Level, method: str = "nearest") -> None:
            if root.level == lx + 1:
                nodes.extend(root.children)
                return

            for child in root.children:
                dfs_preorder(child, lx, method)

        dfs_preorder(self, lx, method)

        return nodes

    def _find_lx_node_with_n_child_avail(
        self, lx: Level, n_child_avail: int, method: str = "nearest"
    ) -> Optional["RoutingNode"]:
        """Find the child of level `lx` with at least \
            `n_child_avail` children available.
        """
        if lx > self.level:
            raise ValueError(f"The node with level {lx} is not in self")

        if lx == self.level:
            if self.n_child_not_occpuied() >= n_child_avail:
                return self
            else:
                return None

        if lx < self.level:
            for child in self.children:
                node = child._find_lx_node_with_n_child_avail(lx, n_child_avail, method)
                if node is not None:
                    return node

            return None

    def _find_lx_node_all_empty(
        self, lx: Level, method: str = "nearest"
    ) -> Optional["RoutingNode"]:
        if lx > self.level:
            raise ValueError(f"The node with level {lx} is not in self")

        if lx == self.level:
            if self.status == NodeStatus.ALL_EMPTY:
                return self
            else:
                return None

        if lx < self.level:
            for child in self.children:
                node = child._find_lx_node_all_empty(lx, method)
                if node is not None:
                    return node

            return None

    def find_lx_node_for_routing(
        self,
        lx: Level,
        n_child_avail: int = 1,
        method: str = "nearest",
    ) -> List["RoutingNode"]:
        """Find lx-level node for placing.

        Args:
            - lx: the level of node to be found(lx > L0).
            - n_child_avail: find the node with at least `N` free child left.
            - method: nearest or by the path. The paremeter is reserved.
        """

        def _get_child_nodes(
            routing_node: RoutingNode,
        ) -> List["RoutingNode"]:
            if n_child_avail == 4:
                return routing_node.children
            elif n_child_avail == 2:
                not_empty = routing_node.n_child_not_empty()
                if not_empty > 0:
                    return routing_node.children[2:]
                else:
                    return routing_node.children[:2]
            elif n_child_avail == 1:
                avail_child = routing_node.get_avail_child(method)
                if avail_child:
                    return [avail_child]
            else:
                # TODO Hard to describe
                raise NotSupportedError

            return []

        if lx > self.level:
            raise ValueError(f"The node with level {lx} is not in self")

        if lx == self.level:
            node = self._find_lx_node_with_n_child_avail(lx, n_child_avail, method)
            if node is not None:
                return _get_child_nodes(node)
        else:
            for child in self.children:
                # Find the Lx-level node with `n_child_avail` Lx-1-level children.
                lx_node = child._find_lx_node_with_n_child_avail(
                    lx, n_child_avail, method
                )
                if lx_node is not None:
                    return _get_child_nodes(lx_node)

        return []

    def add_item_to_L0_node(self, data: Any, method: str = "nearest") -> bool:
        """Add item to the nearest available L0-level node."""
        if self.level == Level.L0:
            self.add_item(data)
            return True

        # Find the nearest available L1-level node.
        L1_node = self._find_lx_node_with_n_child_avail(Level.L1, 1)

        if L1_node is None:
            # No available L1-level node found.
            return False

        # Find the nearest available L0-level node.
        L0_node = L1_node.get_avail_child(method)
        if L0_node is None:
            return False

        L0_node.add_item(data)
        return True

    def n_child_occupied(self) -> int:
        """Get #N of occpuied children."""
        return sum(child.status == NodeStatus.OCCUPIED for child in self.children)

    def n_child_not_occpuied(self) -> int:
        return self.node_capacity - self.n_child_occupied()

    def n_child_empty(self) -> int:
        """Get #N of empty children."""
        return sum(child.status == NodeStatus.ALL_EMPTY for child in self.children)

    def n_child_not_empty(self) -> int:
        return self.node_capacity - self.n_child_empty()

    def is_full(self) -> bool:
        return len(self.children) == self.node_capacity

    def is_empty(self) -> bool:
        return len(self.children) == 0

    def is_children_all_status(self, status: NodeStatus) -> bool:
        return all(child.status == status for child in self.children)

    def is_sub_node_all_status(self, status: NodeStatus) -> bool:
        if self.level == Level.L1:
            return self.is_children_all_status(status)

        for child in self.children:
            if not child.is_sub_node_all_status(status):
                return False

        return True

    def __getitem__(self, index: int) -> "RoutingNode":
        return self.children[index]

    def __contains__(self, item: "RoutingNode") -> bool:
        return item in self.children

    @property
    def level(self) -> Level:
        return self._level

    @property
    def node_capacity(self) -> int:
        return 4 if self.level > Level.L0 else 0

    @property
    def children(self) -> List["RoutingNode"]:
        return self._children

    @property
    def status(self) -> NodeStatus:
        return self._status

    @status.setter
    def status(self, new_status: NodeStatus) -> None:
        self._status = new_status

    def node_status_update(self, method: str = "nearest") -> None:
        """Update the status of the node and its children \
            of all levels(from `self.level` to L1).
        """

        def dfs_preorder(root: RoutingNode, method: str) -> None:
            if root.level > Level.L1:
                for child in root.children:
                    dfs_preorder(child, method)

            root._status_update()

        if self.level > Level.L0:
            dfs_preorder(self, method)

    def _status_update(self) -> None:
        """Update the status of the node."""
        if self.is_sub_node_all_status(NodeStatus.OCCUPIED):
            self._status = NodeStatus.OCCUPIED
        elif self.is_sub_node_all_status(NodeStatus.ALL_EMPTY):
            self._status = NodeStatus.ALL_EMPTY
        else:
            self._status = NodeStatus.AVAILABLE


@final
class RoutingRoot(RoutingNode):
    def __init__(self, empty_root: bool = False, **kwargs) -> None:
        """Initialize a routing quadtree root. \
            The level of the root is L5.

        Args:
            empty_root: whether to create a empty root. Default is false.
        """
        super().__init__(Level.L5, **kwargs)

        if not empty_root:
            for i in range(self.node_capacity):
                L4_child = create_lx_full_tree(Level.L4, f"L4_{i}")
                self.add_child(L4_child)

    def insert_gsyn_on_core(self, *cb_on_core: CorePlacement) -> None:
        """Insert the grouped synapse on core into the tree.

        Steps:
            - 1. Get the routing node consumption.
            - 2. Based on the routing level, find the available node of the routing level.
        """
        n_core_total = len(cb_on_core)

        cost = get_node_consumption(n_core_total)
        level, next_n = cost.get_routing_level()

        # Find L2-level node with at least 2 L1 children available.
        routing_node = self.find_lx_node_for_routing(level, next_n)
        if routing_node is None:
            raise ValueError

        for gsyn_on_core in cb_on_core:
            leaf = RoutingNode(
                Level.L0, gsyn_on_core, tag=f"leaf of {gsyn_on_core.name}"
            )


def create_lx_full_tree(lx: Level, root_tag: Optional[str] = None) -> RoutingNode:
    """Create a full Lx-level routing tree.

    If creating a L4 routing tree, it will return:
    L4 with #N children
        -> L3 with #N children
            -> L2 with #N children
                -> L1 with #N children
                    -> L0 with no child

    where #N is `node_capacity`.
    """
    root = RoutingNode(lx, tag=root_tag)

    if lx > Level.L0:
        for i in range(root.node_capacity):
            child = create_lx_full_tree(Level(lx - 1), f"L{lx-1}_{i}")
            root.add_child(child)

    return root


def get_parent(tree: RoutingNode, node: RoutingNode) -> Optional[RoutingNode]:
    """Get the parent node of the given node. \
        If not found, return None.
    """

    def dfs_preorder(tree, node) -> Optional[RoutingNode]:
        if tree is node:
            return None

        for child in tree.children:
            if child is node:
                return tree
            else:
                return dfs_preorder(child, node)

    return dfs_preorder(tree, node)
