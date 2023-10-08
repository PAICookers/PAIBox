from typing import ClassVar, final, Literal, List, Optional, Sequence

from .grouping import GroupedSyn, GroupedSynOnCore

from paibox.libpaicore.v2._types import RouterLevel, RouterDirection
from paibox.libpaicore.v2.router import RouterCoordinate, idx2router_direction


class RouterTreeNode:
    node_capacity: ClassVar[int] = 4

    def __init__(
        self,
        level: RouterLevel,
        data: Optional[GroupedSynOnCore] = None,
        *,
        tag: Optional[str] = None,
    ) -> None:
        self._level = level
        self._children: List["RouterTreeNode"] = []
        self.item = data
        self.tag = tag

    def add_child(self, child: "RouterTreeNode") -> bool:
        if self.level == RouterLevel.L0:
            # L0-level node cannot add child.
            # TODO
            raise ValueError

        if self.level - child.level != 1:
            raise ValueError

        if len(self.children) == self.node_capacity:
            return False

        # TODO append? method X/Y-priority?
        self._children.append(child)

        return True

    def find_child(self, _child: "RouterTreeNode") -> bool:
        for child in self.children:
            if child == _child:
                return True

        return False

    def find_node_by_path(
        self, path: Sequence[RouterDirection], method: Literal["X", "Y"] = "Y"
    ) -> "RouterTreeNode":
        """Find node by the path of `RouterDirection`.
        
        Description: Find by start at this level based on the path provided. \
            Take `path[0]` each time and then do a recursive search.
        
        NOTE: The length of path <= the level of this node.
        """
        if len(path) == 0:
            return self

        if len(path) > self.level:
            # TODO
            raise ValueError

        idx = path[0].to_index(method)
        if idx > len(self.children):
            raise IndexError

        sub_node = self.children[idx]

        if len(path) > 1:
            return sub_node.find_node_by_path(path[1:], method)
        else:
            return sub_node

    def find_node_by_tag(self, tag: str) -> Optional["RouterTreeNode"]:
        """Searches for nodes by tag using DFS.

        Args:
            - tag: the tag string.

        Returns:
            - the node if found. Otherwise return `None`.
        """

        def dfs_preorder(root: RouterTreeNode, tag: str) -> Optional[RouterTreeNode]:
            if root.tag == tag:
                return root
            elif root.level == RouterLevel.L0:
                return None
            else:
                for child in root.children:
                    node = dfs_preorder(child, tag)
                    if node:
                        return node

        return dfs_preorder(self, tag)

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
        return len(self.children) == self.node_capacity

    def is_empty(self) -> bool:
        return len(self.children) == 0

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


@final
class RouterTreeRoot(RouterTreeNode):
    def __init__(self, empty_root: bool = False, **kwargs) -> None:
        """Initialize a router tree root.

        Args:
            empty_root: whether to create a empty root. Default is false.
        """
        super().__init__(RouterLevel.L5, **kwargs)

        if not empty_root:
            for i in range(self.node_capacity):
                L4_child = create_lx_full_tree(RouterLevel.L4)
                self.add_child(L4_child)

    def nearest_avail_lx_node(self, lx: RouterLevel) -> Optional[RouterTreeNode]:
        """Get the nearest available Lx node.
        Search from the X0Y0 direction of every node.
        """
        nodes = self.get_lx_nodes(lx)

        for node in nodes:
            # Return an available Lx node.
            if not node.is_full():
                return node

        return None

    # def insert_gsyn(self, gsyn: GroupedSyn) -> bool:
    #     gsyn_on_cores = gsyn.build_syn_on_core()

    #     for gsyn_on_core in gsyn_on_cores:
    #         if not self.insert_gsyn_on_core(gsyn_on_core):
    #             raise ValueError

    #     return True

    def insert_gsyn_on_core(self, gsyn_on_core: GroupedSynOnCore) -> bool:
        l1_node = self.nearest_avail_lx_node(RouterLevel.L1)

        if not l1_node:
            return False

        node = RouterTreeNode(RouterLevel.L0, gsyn_on_core)
        return l1_node.add_child(node)

    def get_L0_node_path(
        self, node: RouterTreeNode, method: Literal["X", "Y"] = "Y"
    ) -> RouterCoordinate:
        """Return a direction path from L4 to L0.

        Args:
            - node: the L0 node.
            - method: use X/Y-priority method.

        Return:
            - A list of `RouterDirection` from L4 to L0.
        """
        assert node.level == RouterLevel.L0

        path = []

        def dfs_preorder(root: RouterTreeNode) -> bool:
            i = 0
            if root.level == RouterLevel.L1:
                for child in root.children:
                    if child is node:
                        path.append(idx2router_direction(i, method))
                        return True

                    i += 1
            else:
                for child in root.children:
                    path.append(idx2router_direction(i, method))
                    if dfs_preorder(child):
                        return True

                    i += 1

            return False

        if dfs_preorder(self):
            return RouterCoordinate.build_from_path(path)

        raise ValueError


def create_lx_nbranch(lx: RouterLevel, nbranch: int) -> RouterTreeNode:
    root = RouterTreeNode(lx)

    if lx > RouterLevel.L1:
        for i in range(nbranch):
            child = create_lx_nbranch(RouterLevel(lx - 1), nbranch)
            root.add_child(child)

    return root


def create_lx_full_tree(lx: RouterLevel) -> RouterTreeNode:
    """Create a full Lx-level router tree.

    If creating a L4 router tree, it will return:
    L4 with 4 children
        -> L3 with 4 children
            -> L2 with 4 children
                -> L1 with no L0 child
    """
    return create_lx_nbranch(lx, nbranch=RouterTreeNode.node_capacity)
