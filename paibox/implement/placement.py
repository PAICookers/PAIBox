from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, final

from paibox.libpaicore.v2.route import RoutingDirection as Direction
from paibox.libpaicore.v2.route import RoutingDirectionIdx as DirectionIdx
from paibox.libpaicore.v2.route import RoutingNodeLevel as Level
from paibox.libpaicore.v2.route import RoutingNodeStatus as Status
from paibox.libpaicore.v2.route import get_node_consumption

from .grouping import GroupedSynOnCore


class RoutingNode:
    def __init__(
        self,
        level: Level,
        data: Optional[Any] = None,
        *,
        status: Optional[Status] = Status.AVAILABLE,
        tag: Optional[str] = None,
    ) -> None:
        """Instance a tree node with `level`. \
            For a node with level Lx > 0, after created, \
                the length of children is `node_capacity`.

            For a node with level L0, it is a leaf node.

        Args:
            - level: the node level.
            - data: the data hanging on the node. Optional.
            - tag: a tag for user to identify.
        """
        self._level = level
        self._children: Dict[Direction, RoutingNode] = defaultdict()
        self.item = data
        self.tag = tag

        # Only set the attribute for L0-level node.
        if self.level == Level.L0:
            setattr(self, "status", status)

    def create_child(self, force: bool = False, **kwargs) -> Optional["RoutingNode"]:
        """Create a child. If full, return None."""
        child = RoutingNode(Level(self.level - 1), **kwargs)

        if not self.add_child(child, force=force):
            return None

        return child

    def add_child(
        self, child: "RoutingNode", method: str = "nearest", force: bool = False
    ) -> bool:
        if self.level == Level.L0:
            # L0-level node cannot add child.
            # TODO
            raise ValueError

        if self.is_full():
            return False

        # Traverse from X0Y0 to X1Y1.
        for d in DirectionIdx:
            if d not in self.children:
                return self.add_child_to(child, d, force)

        return False

    def add_child_to(
        self, child: "RoutingNode", direction: Direction, force: bool = False
    ) -> bool:
        if self.level - child.level != 1:
            raise ValueError

        if not force and direction in self.children:
            return False

        self._children[direction] = child

        return True

    def find_node_by_path(self, path: Sequence[Direction]) -> Optional["RoutingNode"]:
        """Find the node by given a path of `Direction`.

        Description:
            Find by starting at this level based on the path provided. \
            Take `path[0]` each time and then do a recursive search.

        NOTE: The length of path <= the level of this node.
        """
        if len(path) == 0:
            return self

        if len(path) > self.level:
            # TODO
            raise ValueError

        if path[0] not in self.children:
            return None

        sub_node = self[path[0]]

        if len(path) > 1:
            return sub_node.find_node_by_path(path[1:])
        else:
            return sub_node

    def is_full(self) -> bool:
        return self.n_child == self.node_capacity

    def is_empty(self) -> bool:
        return self.n_child == 0

    def n_child_avail(self) -> int:
        return self.node_capacity - self.n_child

    def _find_lx_node_with_n_child_avail(
        self, lx: Level, n_child_avail: int, method: str = "nearest"
    ) -> Optional["RoutingNode"]:
        """Find the child of level `lx` with at least \
            `n_child_avail` children available.
        """
        if lx > self.level:
            raise ValueError

        if lx == self.level:
            if self.n_child_avail() >= n_child_avail:
                return self
            else:
                return None

        if not self.is_empty():
            for d in DirectionIdx:
                if d in self.children:
                    node = self[d]._find_lx_node_with_n_child_avail(
                        lx, n_child_avail, method
                    )
                    if node is not None:
                        return node

        child = self.create_child()
        if not child:
            return None

        return child._find_lx_node_with_n_child_avail(lx, n_child_avail, method)

    def add_subtree(
        self,
        subtree: "RoutingNode",
        method: str = "nearest",
    ) -> bool:
        """Add the subtree's children to itself. \
            If successful, return the added parent node."""
        if subtree.level > self.level:
            raise ValueError

        if subtree.level == self.level:
            # TODO Check if `sub_n_child` is legal.
            # Only be 1, 2, or 4?
            sub_n_child = len(subtree.children)
            if self.n_child_avail() < sub_n_child:
                return False

            if sub_n_child == 1:
                self.add_child(subtree.children[Direction.X0Y0])

            if sub_n_child == 2:
                if self.n_child == 0:
                    self.add_child_to(subtree.children[Direction.X0Y0], Direction.X0Y0)
                    self.add_child_to(subtree.children[Direction.X0Y1], Direction.X0Y1)
                else:
                    self.add_child_to(subtree.children[Direction.X0Y0], Direction.X1Y0)
                    self.add_child_to(subtree.children[Direction.X0Y1], Direction.X1Y1)

            else:
                self._children = subtree.children

            return True

        if not self.is_empty():
            for d in DirectionIdx:
                if d in self.children:
                    flag = self[d].add_subtree(subtree, method)
                    if flag:
                        return flag

        child = self.create_child()
        if not child:
            return False

        return child.add_subtree(subtree, method)

    @classmethod
    def create_lx_full_tree(
        cls, lx: Level, root_tag: Optional[str] = None
    ) -> "RoutingNode":
        root = RoutingNode(lx, tag=root_tag)

        if lx > Level.L1:
            for i in range(root.node_capacity):
                child = cls.create_lx_full_tree(Level(lx - 1), f"L{lx-1}_{i}")
                if not root.add_child(child):
                    raise ValueError

        return root

    @classmethod
    def create_routing_tree(cls, lx: Level, n_branch: int) -> "RoutingNode":
        """Create a routing tree with `n_branch` children.

        NOTE: When lx == L1, do not create the L0-level children. \
            WHen lx > L1, create the lx-1 level children.
        """
        if lx == Level.L0 or n_branch < 0:
            raise ValueError

        root = RoutingNode(lx)

        # Create `n_branch` children when lx > L1.
        if lx > Level.L1:
            for _ in range(n_branch):
                child = cls.create_lx_full_tree(Level(lx - 1))
                if not root.add_child(child):
                    raise ValueError

        return root

    def add_L0_for_placing(self, data: Any, **kwargs) -> bool:
        """Add L0 node for placing in the routing tree.

        Args:
            - data: the data attached to the L0-level node.
            - kwargs: other arguments of the L0-level node, status, tag...
        """
        node = RoutingNode(Level.L0, data, **kwargs)

        L1_node = self._find_lx_node_with_n_child_avail(Level.L1, 1)
        if not L1_node:
            return False

        return L1_node.add_child(node)

    def find_nodes_at_level(
        self, lx: Level, n_child_avail_low: int = 0
    ) -> List["RoutingNode"]:
        """Find all nodes at a `lx` level with at least \
            `n_child_avail_low` child nodes.
        """
        if lx > self.level:
            raise ValueError

        nodes = []

        def dfs_preorder(root: RoutingNode) -> None:
            if root.level == lx:
                if root.n_child_avail() >= n_child_avail_low:
                    nodes.append(root)

                return

            for d in DirectionIdx:
                if d in root.children:
                    dfs_preorder(root[d])

        dfs_preorder(self)
        return nodes

    def find_empty_nodes_at_level(self, lx: Level) -> List["RoutingNode"]:
        if lx == Level.L0:
            return []

        return self.find_nodes_at_level(lx, self.node_capacity)

    def __getitem__(self, key: Direction) -> "RoutingNode":
        return self.children[key]

    @property
    def level(self) -> Level:
        return self._level

    # @status.setter
    # def status(self, new_status: Status):
    #     self._status = new_status

    @property
    def node_capacity(self) -> int:
        return 4 if self.level > Level.L0 else 0

    @property
    def children(self):
        return self._children

    @property
    def n_child(self) -> int:
        return len(self._children)


@final
class RoutingRoot(RoutingNode):
    def __init__(self, **kwargs) -> None:
        """Initialize a routing quadtree root(L5-level)."""
        super().__init__(Level.L5, **kwargs)

    def insert_gsyn_on_core(self, *gsyns_on_core: GroupedSynOnCore) -> bool:
        """Insert a list of `gsyn_on_core` in the routing tree.

        TODO add error descriptions.
        """
        parent_name = gsyns_on_core[0].obj.name
        n_core = len(gsyns_on_core)

        cost = get_node_consumption(n_core)
        level, next_n = cost.get_routing_level()

        routing_root = RoutingNode.create_routing_tree(level, next_n)

        for i in range(cost.n_L0):
            if i < n_core:
                if not routing_root.add_L0_for_placing(
                    data=gsyns_on_core[i],
                    status=Status.USED,
                    tag=f"Used by {gsyns_on_core[i].name}",
                ):
                    raise RuntimeError(f"Cannot place {gsyns_on_core[i].name} on core")
            else:
                # Other L0 nodes are unused but occupied.
                if not routing_root.add_L0_for_placing(
                    data=None, status=Status.OCCUPIED, tag=f"Occupied by {parent_name}"
                ):
                    raise RuntimeError(f"Cannot place!")

        return self.add_subtree(routing_root)

    def breadth_of_lx_nodes(self, lx: Level) -> int:
        """Get the number of nodes in the routing tree at the given level."""
        nodes = self.find_nodes_at_level(lx, 0)

        return len(nodes)


def get_parent(tree: RoutingNode, node: RoutingNode) -> Optional[RoutingNode]:
    """Get the parent node of the given node. \
        If not found, return None.
    """
    assert tree != node

    def dfs_preorder(tree: RoutingNode, node: RoutingNode) -> Optional[RoutingNode]:
        for d in DirectionIdx:
            if d in tree.children:
                if tree[d] is node:
                    return tree
                else:
                    parent = dfs_preorder(tree[d], node)
                    if parent:
                        return parent

        return None

    return dfs_preorder(tree, node)
