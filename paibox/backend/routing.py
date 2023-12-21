from typing import Any, Dict, List, Optional, Sequence, final

from paicorelib import HwConfig
from paicorelib import RoutingDirection as Direction
from paicorelib import RoutingDirectionIdx as DirectionIdx
from paicorelib import RoutingNodeCoord as NodeCoord
from paicorelib import RoutingNodeCost as NodeCost
from paicorelib import RoutingNodeLevel as Level
from paicorelib import RoutingNodeStatus as Status
from paicorelib import get_node_consumption

from paibox._types import FrozenOrderedSet
from paibox.exceptions import NotSupportedError

from .placement import CoreBlock, CorePlacement


class RoutingNode:
    def __init__(
        self,
        level: Level,
        data: Optional[CorePlacement] = None,
        *,
        direction: Direction = Direction.ANY,
        status: Optional[Status] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Instance a tree node with `level`. \
            For a node with level Lx > 0, after created, \
                the length of children is `node_capacity`.

            For a node with level L0, it is a leaf node.

        Args:
            - level: the node level.
            - data: the data hanging on the node. Optional.
            - direction: the direction of the node itself. Default is `Direction.ANY`.
            - tag: a tag for user to identify. Optional.

        Attributes:
            - level: the node level.
            - children: the children of the node.
            - direction: the direction of the node iteself.
            - item: the data hanging on the node.
            - tag: a tag for user to identify.
            - status: the status of the node. It's only for L0-level leaves.

        NOTE: Do not add methods `__len__` & `__contains__`.
        """
        self._level = level
        self._children: Dict[Direction, RoutingNode] = dict()
        self._direction = direction
        self.item = data
        self.tag = tag

        # Only set the attribute for L0-level node.
        if self.level == Level.L0:
            setattr(self, "status", status)

    def clear(self) -> None:
        """Clear the tree."""

        def dfs(root: RoutingNode) -> None:
            root.children.clear()
            if root.level == Level.L1:
                return

            for child in root.children.values():
                dfs(child)

            return None

        if self.level > Level.L0:
            dfs(self)

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
            raise AttributeError(f"L0-level node cannot add child")

        if self.is_full():
            return False

        # Traverse from X0Y0 to X1Y1.
        for d in DirectionIdx:
            if d not in self.children:
                return self.add_child_to(child, d, force)

        return False

    def add_child_to(
        self, child: "RoutingNode", d: Direction, force: bool = False
    ) -> bool:
        """Add a child node to a certain `direction`."""
        if self.level - child.level != 1:
            raise ValueError

        if not force and d in self.children:
            return False

        child.direction = d
        self[d] = child

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
            raise ValueError(
                f"The length of the {path} should be less than or equal to level, but yours is greater than"
            )

        if path[0] not in self.children:
            return None

        sub_node = self[path[0]]

        if len(path) > 1:
            return sub_node.find_node_by_path(path[1:])
        else:
            return sub_node

    def get_node_path(self, node: "RoutingNode") -> Optional[List[Direction]]:
        """Return a direction path from L4 to the level of `node`.

        Args:
            - node: the node with level <= `self.level`.

        Return:
            - A list of `Direction` from L4 to L0.
        """
        if node.level > self.level:
            raise ValueError

        if node.level == self.level:
            if node != self:
                return None

            return []

        path = []

        def dfs(root: RoutingNode) -> bool:
            for d, child in root.children.items():
                path.append(d)

                if child is node:
                    return True
                elif dfs(child):
                    return True
                else:
                    path.pop(-1)

            return False

        if dfs(self):
            return path

    def is_full(self) -> bool:
        return len(self.children) == self.node_capacity

    def is_empty(self) -> bool:
        return len(self.children) == 0

    def n_child_avail(self) -> int:
        return self.node_capacity - len(self.children)

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
            sub_n_child = len(subtree.children)
            if self.n_child_avail() < sub_n_child:
                return False

            if sub_n_child == 1:
                self.add_child(subtree.children[Direction.X0Y0])

            elif sub_n_child == 2:
                if len(self.children) == 0:
                    self.add_child_to(subtree.children[Direction.X0Y0], Direction.X0Y0)
                    self.add_child_to(subtree.children[Direction.X0Y1], Direction.X0Y1)
                else:
                    self.add_child_to(subtree.children[Direction.X0Y0], Direction.X1Y0)
                    self.add_child_to(subtree.children[Direction.X0Y1], Direction.X1Y1)

            elif sub_n_child == 4:
                self._children = subtree.children
            else:
                # Only support 1, 2, & 4.
                raise NotSupportedError(
                    f"#N of {sub_n_child} children not supported yet."
                )

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
        cls,
        lx: Level,
        d: Direction = Direction.X0Y0,
        root_tag: Optional[str] = None,
    ) -> "RoutingNode":
        root = RoutingNode(lx, direction=d, tag=root_tag)

        if lx > Level.L1:
            for i in range(root.node_capacity):
                child = cls.create_lx_full_tree(
                    Level(lx - 1), DirectionIdx[i], f"L{lx-1}_{i}"
                )
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

        root = RoutingNode(lx, direction=Direction.X0Y0)

        # Create `n_branch` children when lx > L1.
        if lx > Level.L1:
            for i in range(n_branch):
                child = cls.create_lx_full_tree(Level(lx - 1), DirectionIdx[i])
                if not root.add_child(child):
                    raise ValueError

        return root

    def add_L0_for_placing(self, data: Any = None, **kwargs) -> "RoutingNode":
        """Add L0 node for placing in the routing tree.

        Args:
            - data: the data attached to the L0-level node.
            - kwargs: other arguments of the L0-level node, status, tag...
        """
        node = RoutingNode(Level.L0, data, **kwargs)

        L1_node = self._find_lx_node_with_n_child_avail(Level.L1, 1)
        if not L1_node:
            raise RuntimeError

        if not L1_node.add_child(node):
            raise RuntimeError

        return node

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

    def find_leaf_at_level(self, lx: Level) -> List["RoutingNode"]:
        """Find nodes with no child at the `lx` level."""
        if lx == Level.L0:
            return []

        return self.find_nodes_at_level(lx, self.node_capacity)

    def breadth_of_lx_nodes(self, lx: Level) -> int:
        """Get the number of nodes in the routing tree at the given level."""
        nodes = self.find_nodes_at_level(lx, 0)

        return len(nodes)

    def __getitem__(self, key: Direction) -> "RoutingNode":
        return self.children[key]

    def __setitem__(self, key: Direction, value: "RoutingNode") -> None:
        self._children[key] = value

    @property
    def level(self) -> Level:
        return self._level

    @property
    def node_capacity(self) -> int:
        return HwConfig.N_SUB_ROUTING_NODE if self.level > Level.L0 else 0

    @property
    def children(self):
        return self._children

    @property
    def direction(self) -> Direction:
        return self._direction

    @direction.setter
    def direction(self, d: Direction) -> None:
        self._direction = d


class RoutingGroup(FrozenOrderedSet):
    """Core blocks located within a routing group are routable."""

    def __init__(self, *cb: CoreBlock):
        super().__init__(cb)

    def n_core_at(self, idx: int) -> int:
        if idx >= len(self) or idx < 0:
            raise IndexError

        return self[idx].n_core_required

    @property
    def n_core_required(self) -> int:
        return sum(cb.n_core_required for cb in self)

    @property
    def routing_cost(self) -> NodeCost:
        return get_node_consumption(self.n_core_required)

    @property
    def routing_level(self) -> Level:
        return self.routing_cost.get_routing_level()


@final
class RoutingRoot(RoutingNode):
    def __init__(self, **kwargs) -> None:
        """Initialize a routing quadtree root(L5-level)."""
        super().__init__(Level.L5, **kwargs)

    def get_leaf_coord(self, node: "RoutingNode") -> NodeCoord:
        """Return the routing coordinate of the node(must be a L0 leaf)."""
        path = self.get_node_path(node)
        if path:
            return NodeCoord(*path)

        raise RuntimeError(f"Get leaf node {node.tag} coordinate failed.")

    @classmethod
    def insert_coreblock(cls, root: "RoutingRoot", cb: CoreBlock) -> bool:
        """Insert a `CoreBlock` in the routing tree. Assign the core placements \
            of its routing coordinates & make sure they are routable.
            
        TODO Will be deprecated.
        """
        leaves = []
        coords = []
        n_core = cb.n_core_required
        cost = get_node_consumption(n_core)
        level = cost.get_routing_level()
        # Create a sub-tree node
        routing_node = RoutingNode.create_routing_tree(level, cost[level - 1])

        for i in range(cost.n_L0):
            if i < n_core:
                # Valid L0 nodes
                node = routing_node.add_L0_for_placing(
                    data=f"{cb.name}_{i}", status=Status.USED, tag=f"{cb.name}_{i}"
                )

                leaves.append(node)
            else:
                # Other L0 nodes are unused but occupied.
                node = routing_node.add_L0_for_placing(
                    status=Status.OCCUPIED, tag=f"{cb.name}_{i}"
                )

        # Add the sub-tree to the root.
        flag = root.add_subtree(routing_node)
        if not flag:
            return False

        for node in leaves:
            coord = root.get_leaf_coord(node)
            coords.append(coord.coordinate)

        cb.core_coords = coords

        return True

    def insert_routing_group(self, routing_group: RoutingGroup) -> bool:
        """Insert a `RoutingGroup` in the routing tree. Assign each core blocks \
            with routing coordinates & make sure they are routable.
        """
        leaves = []
        coords = []
        cost = routing_group.routing_cost
        level = routing_group.routing_level
        # Create a sub-tree node
        routing_node = RoutingNode.create_routing_tree(level, cost[level - 1])

        for i in range(cost.n_L0):
            if i < routing_group.n_core_required:
                # Valid L0 nodes
                node = routing_node.add_L0_for_placing(
                    data=f"{id(routing_group)}_{i}",
                    status=Status.USED,
                    tag=f"{id(routing_group)}_{i}",
                )

                leaves.append(node)
            else:
                # Other L0 nodes are unused but occupied.
                node = routing_node.add_L0_for_placing(
                    status=Status.OCCUPIED, tag=f"{id(routing_group)}_{i}"
                )

        # Add the sub-tree to the root.
        flag = self.add_subtree(routing_node)
        if not flag:
            return False

        for node in leaves:
            coord = self.get_leaf_coord(node)
            coords.append(coord.coordinate)

        # Assign the coordinates to each core block inside the routing group.
        cur_i = 0
        for cb in routing_group:
            n = cb.n_core_required
            cb.core_coords = coords[cur_i : cur_i + n]
            cur_i += n

        return True

    @property
    def n_L0_nodes(self) -> int:
        return self.breadth_of_lx_nodes(Level.L0)


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
