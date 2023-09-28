from typing import final, List, Optional

from .grouping import GroupedSyn, GroupedSynOnCore

from paibox.base import PAIBoxObject
from paibox.libpaicore.v2.coordinate import CoordOffset
from paibox.libpaicore.v2._types import RouterLevel

GUESS_PLACE_LEVEL = RouterLevel.L2
ROUTER_TREE_NODE_TAG = "Node_%d_%d"


class RouterTreeNode(PAIBoxObject):
    _direction_table: List[CoordOffset] = [
        CoordOffset(0, 0),
        CoordOffset(0, 1),
        CoordOffset(1, 0),
        CoordOffset(1, 1),
    ]

    _node_capacity = len(_direction_table)

    def __init__(
        self,
        level: RouterLevel,
        data: Optional[GroupedSynOnCore] = None,
    ) -> None:
        super().__init__()

        self._level = level
        self._children: List["RouterTreeNode"] = []
        self.item = data

    def add_child(self, child: "RouterTreeNode") -> bool:
        if self.level == RouterLevel.L0 or self.level - child.level != 1:
            raise ValueError

        if len(self.children) == self._node_capacity:
            return False

        self._children.append(child)
        return True

    def find_child(self, _child: "RouterTreeNode") -> bool:
        if _child.level == RouterLevel.L0:
            raise ValueError

        for child in self._children:
            if child == _child:
                return True

        return False

    # def get_node_by_tag(self, tag: str):
    #     def dfs_preorder(root: RouterTreeNode, tag: str):
    #         if root.level == RouterLevel.L0 and root.tag != tag:
    #             return

    #         if root.tag == tag:
    #             return root

    #         for child in root.children:
    #             dfs_preorder(child, tag)

    #     dfs_preorder(self, tag)

    def get_lx_nodes(self, lx: RouterLevel) -> List["RouterTreeNode"]:
        if lx > self.level:
            raise ValueError

        if lx == self.level:
            return [self]

        nodes = []

        def dfs_preorder(root: RouterTreeNode, lx: RouterLevel) -> None:
            if root.level == lx + 1:
                nodes.extend(root.children)
                return

            for child in root.children:
                dfs_preorder(child, lx)

        dfs_preorder(self, lx)

        return nodes

    def is_full(self) -> bool:
        return len(self.children) == self.capacity

    def is_empty(self) -> bool:
        return len(self.children) == 0

    # def is_full(self) -> bool:
    #     """All children are full, then full. \
    #         One child is not full, then not full.
    #     """
    #     # L0 never has children.
    #     if self.level == RouterLevel.L0:
    #         if len(self.children) == self.capacity:
    #             return True

    #     for child in self.children:
    #         if not child.is_full():
    #             return False

    #     return False

    # def is_empty(self) -> bool:
    #     """All children are empty, then empty. \
    #         One child is not empty, then not empty.
    #     """
    #     # L0 never has children.
    #     if self.level == RouterLevel.L0:
    #         return False

    #     for child in self.children:
    #         if not child.is_empty():
    #             return False

    #     return True

    @classmethod
    def create_node(cls, gsyn: GroupedSyn):
        level = cls._get_gsyn_level(gsyn.n_core)

        syns_on_core = gsyn.build_syn_on_core()

        # At least, create a L1 router node. (>=L1)
        root = RouterTreeNode(level, data=None)
        for syn in syns_on_core:
            child = RouterTreeNode(level=RouterLevel.L0, data=syn)
            root.add_child(child)

        return root

    @staticmethod
    def _get_gsyn_level(n_core: int) -> RouterLevel:
        if n_core <= 4**RouterLevel.L1:
            return RouterLevel.L1
        elif n_core <= 4**RouterLevel.L2:
            return RouterLevel.L2
        elif n_core <= 4**RouterLevel.L3:
            return RouterLevel.L3
        else:
            raise NotImplementedError

    @property
    def level(self) -> RouterLevel:
        return self._level

    @property
    def children(self) -> List["RouterTreeNode"]:
        return self._children

    @property
    def capacity(self) -> int:
        return self._node_capacity


@final
class RouterTreeRoot(RouterTreeNode):
    def __init__(self) -> None:
        super().__init__(RouterLevel.L5)

        for i in range(self.capacity):
            L4_child = create_lx_full_tree(RouterLevel.L4)
            self.add_child(L4_child)

    def nearest_avail_L1(self):
        """Whether L1 level nodes are all full"""
        l1_nodes = self.get_lx_nodes(RouterLevel.L1)

        for node in l1_nodes:
            if not node.is_full():
                return node

        return None

    def insert_gsyn_on_core(self, gsyn_on_core: GroupedSynOnCore) -> bool:
        node = RouterTreeNode(RouterLevel.L0, gsyn_on_core)
        l1_node = self.nearest_avail_L1()

        if l1_node:
            return l1_node.add_child(node)

        return False

    def get_L0_node_road(self):
        road: List[CoordOffset] = []

        def dfs_preorder(root: RouterTreeNode, lx: RouterLevel) -> None:
            if root.level == RouterLevel.L0:
                if root.item:
                    root.item.road = road
                road.pop(-1)
                return

            for i in range(root.capacity):
                road.append(root._direction_table[i])
                dfs_preorder(root.children[i], RouterLevel(lx - 1))

            road.pop(-1)

        dfs_preorder(self, RouterLevel.L5)


def create_lx_full_tree(level: RouterLevel) -> RouterTreeNode:
    """Create a full Lx-level router tree.

    If creating a L4 router tree, it will return:
    L4 with 4 children
        -> L3 with 4 children
            -> L2 with 4 children
                -> L1 with 4 children
                    -> L0 with no child
    """
    root = RouterTreeNode(level)

    if level > RouterLevel.L0:
        for i in range(root.capacity):
            child = create_lx_full_tree(RouterLevel(level - 1))
            root.add_child(child)

    return root


router_tree_root = RouterTreeRoot()
