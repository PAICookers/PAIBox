from typing import Any, Dict, Iterator, List, Optional, Sequence, final

from paicorelib import Coord, HwConfig
from paicorelib.routing_defs import ROUTING_DIRECTIONS_IDX, RoutingCoord, RoutingCost
from paicorelib.routing_defs import RoutingDirection as Direction
from paicorelib.routing_defs import RoutingLevel as Level
from paicorelib.routing_defs import RoutingStatus as Status
from paicorelib.routing_defs import get_routing_consumption

from paibox.exceptions import NotSupportedError

from .placement import CoreBlock, CorePlacement

__all__ = ["RoutingGroup", "RoutingRoot"]


class RoutingCluster:
    def __init__(
        self,
        level: Level,
        data: Optional[CorePlacement] = None,
        *,
        direction: Direction = Direction.ANY,
        status: Optional[Status] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Instance a tree cluster with `level`.
        - For a Lx(>0)-level cluster, after created, the length of children is `node_capacity`.
        - For a L0-level cluster, it's a leaf.

        Args:
            - level: the cluster level.
            - data: the data hanging on the cluster. Optional.
            - direction: the direction of the cluster itself. Default is `Direction.ANY`.
            - tag: a tag for user to identify. Optional.

        Attributes:
            - level: the cluster level.
            - children: the children of the cluster.
            - direction: the direction of the cluster iteself.
            - item: the data hanging on the cluster.
            - tag: a tag for user to identify.
            - status: the status of the cluster. It's only for L0-level leaves.

        NOTE: Do not add methods `__len__` & `__contains__`.
        """
        self._level = level
        self._children: Dict[Direction, RoutingCluster] = dict()
        self._direction = direction
        self.item = data
        self.tag = tag

        # Only set the attribute for L0-level cluster.
        if self.level == Level.L0:
            setattr(self, "status", status)

    def clear(self) -> None:
        """Clear the tree."""

        def dfs(root: RoutingCluster) -> None:
            root.children.clear()
            if root.level == Level.L1:
                return

            for child in root.children.values():
                dfs(child)

            return None

        if self.level > Level.L0:
            dfs(self)

    def create_child(self, force: bool = False, **kwargs) -> Optional["RoutingCluster"]:
        """Create a child. If full, return None."""
        child = RoutingCluster(Level(self.level - 1), **kwargs)

        if not self.add_child(child, force=force):
            return None

        return child

    def add_child(
        self, child: "RoutingCluster", method: str = "nearest", force: bool = False
    ) -> bool:
        if self.level == Level.L0:
            # L0-level cluster cannot add child.
            raise AttributeError(f"L0-level cluster cannot add child")

        if self.is_full():
            return False

        # Traverse from X0Y0 to X1Y1.
        for d in ROUTING_DIRECTIONS_IDX:
            if d not in self.children:
                return self.add_child_to(child, d, force)

        return False

    def add_child_to(
        self, child: "RoutingCluster", d: Direction, force: bool = False
    ) -> bool:
        """Add a child cluster to a certain `direction`."""
        if self.level - child.level != 1:
            raise ValueError

        if not force and d in self.children:
            return False

        child.direction = d
        self[d] = child

        return True

    def find_cluster_by_path(
        self, path: Sequence[Direction]
    ) -> Optional["RoutingCluster"]:
        """Find the cluster by given a path of `Direction`.

        Description:
            Find by starting at this level based on the path provided. \
            Take `path[0]` each time and then do a recursive search.

        NOTE: The length of path <= the level of this cluster.
        """
        if len(path) == 0:
            return self

        if len(path) > self.level:
            raise ValueError(
                f"The length of the {path} should be less than or equal to level, but yours is greater than"
            )

        if path[0] not in self.children:
            return None

        sub_cluster = self[path[0]]

        if len(path) > 1:
            return sub_cluster.find_cluster_by_path(path[1:])
        else:
            return sub_cluster

    def get_routing_path(self, cluster: "RoutingCluster") -> Optional[List[Direction]]:
        """Return a direction path from L4 to the level of `cluster`.

        Args:
            - cluster: the cluster with level <= `self.level`.

        Return:
            - A list of `Direction` from L4 to L0.
        """
        if cluster.level > self.level:
            raise ValueError

        if cluster.level == self.level:
            if cluster != self:
                return None

            return []

        path = []

        def dfs(root: RoutingCluster) -> bool:
            for d, child in root.children.items():
                path.append(d)

                if child is cluster:
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

    def _find_lx_cluster_with_n_child_avail(
        self, lx: Level, n_child_avail: int, method: str = "nearest"
    ) -> Optional["RoutingCluster"]:
        """Find the child of level `lx` with at least `n_child_avail` children available."""
        if lx > self.level:
            raise ValueError

        if lx == self.level:
            if self.n_child_avail() >= n_child_avail:
                return self
            else:
                return None

        if not self.is_empty():
            for d in ROUTING_DIRECTIONS_IDX:
                if d in self.children:
                    cluster = self[d]._find_lx_cluster_with_n_child_avail(
                        lx, n_child_avail, method
                    )
                    if cluster is not None:
                        return cluster

        child = self.create_child()
        if not child:
            return None

        return child._find_lx_cluster_with_n_child_avail(lx, n_child_avail, method)

    def add_subtree(
        self,
        subtree: "RoutingCluster",
        method: str = "nearest",
    ) -> bool:
        """Add the subtree's children to itself. \
            If successful, return the added parent cluster."""
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
                    if HwConfig.COORD_Y_PRIORITY:
                        self.add_child_to(
                            subtree.children[Direction.X0Y0], Direction.X0Y0
                        )
                        self.add_child_to(
                            subtree.children[Direction.X0Y1], Direction.X0Y1
                        )
                    else:
                        self.add_child_to(
                            subtree.children[Direction.X0Y0], Direction.X0Y0
                        )
                        self.add_child_to(
                            subtree.children[Direction.X0Y1], Direction.X1Y0
                        )
                else:
                    if HwConfig.COORD_Y_PRIORITY:
                        self.add_child_to(
                            subtree.children[Direction.X0Y0], Direction.X1Y0
                        )
                        self.add_child_to(
                            subtree.children[Direction.X0Y1], Direction.X1Y1
                        )
                    else:
                        self.add_child_to(
                            subtree.children[Direction.X0Y0], Direction.X0Y1
                        )
                        self.add_child_to(
                            subtree.children[Direction.X0Y1], Direction.X1Y1
                        )

            elif sub_n_child == 4:
                self._children = subtree.children
            else:
                # Only support 1, 2, & 4.
                raise NotSupportedError(
                    f"#N of {sub_n_child} children not supported yet."
                )

            return True

        if not self.is_empty():
            for d in ROUTING_DIRECTIONS_IDX:
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
    ) -> "RoutingCluster":
        root = RoutingCluster(lx, direction=d, tag=root_tag)

        if lx > Level.L1:
            for i in range(root.node_capacity):
                child = cls.create_lx_full_tree(
                    Level(lx - 1), ROUTING_DIRECTIONS_IDX[i], f"L{lx-1}_{i}"
                )
                if not root.add_child(child):
                    raise ValueError

        return root

    @classmethod
    def create_routing_tree(cls, lx: Level, n_branch: int) -> "RoutingCluster":
        """Create a routing tree with `n_branch` children.

        NOTE: When lx == L1, do not create the L0-level children. \
            WHen lx > L1, create the lx-1 level children.
        """
        if lx == Level.L0 or n_branch < 0:
            raise ValueError

        root = RoutingCluster(lx, direction=Direction.X0Y0)

        # Create `n_branch` children when lx > L1.
        if lx > Level.L1:
            for i in range(n_branch):
                child = cls.create_lx_full_tree(
                    Level(lx - 1), ROUTING_DIRECTIONS_IDX[i]
                )
                if not root.add_child(child):
                    raise ValueError

        return root

    def add_L0_for_placing(self, data: Any = None, **kwargs) -> "RoutingCluster":
        """Add L0 cluster for placing in the routing tree.

        Args:
            - data: the data attached to the L0-level cluster.
            - kwargs: other arguments of the L0-level cluster, status, tag, etc.
        """
        cluster = RoutingCluster(Level.L0, data, **kwargs)

        L1_cluster = self._find_lx_cluster_with_n_child_avail(Level.L1, 1)
        if not L1_cluster:
            raise RuntimeError("Available L1 cluster not found!")

        if not L1_cluster.add_child(cluster):
            raise RuntimeError(f"Add child into L1 cluster failed!")

        return cluster

    def find_clusters_at_level(
        self, lx: Level, n_child_avail_low: int = 0
    ) -> List["RoutingCluster"]:
        """Find all clusters at a `lx` level with at least `n_child_avail_low` child clusters."""
        if lx > self.level:
            raise ValueError

        clusters = []

        def dfs_preorder(root: RoutingCluster) -> None:
            if root.level == lx:
                if root.n_child_avail() >= n_child_avail_low:
                    clusters.append(root)

                return

            for d in ROUTING_DIRECTIONS_IDX:
                if d in root.children:
                    dfs_preorder(root[d])

        dfs_preorder(self)
        return clusters

    def find_leaf_at_level(self, lx: Level) -> List["RoutingCluster"]:
        """Find clusters with no child at the `lx` level."""
        if lx == Level.L0:
            return []

        return self.find_clusters_at_level(lx, self.node_capacity)

    def breadth_of_lx_clusters(self, lx: Level) -> int:
        """Get the number of clusters in the routing tree at the given level."""
        clusters = self.find_clusters_at_level(lx, 0)

        return len(clusters)

    def __getitem__(self, key: Direction) -> "RoutingCluster":
        return self.children[key]

    def __setitem__(self, key: Direction, value: "RoutingCluster") -> None:
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


class RoutingGroup(List[CoreBlock]):
    """Core blocks located within a routing group are routable.

    NOTE: Axon groups within a routing group are the same.
    """

    def __init__(self, *cb: CoreBlock) -> None:
        self.cb = list(cb)

        self.assigned_coords: List[Coord] = []
        """Assigned core coordinates for the routing group."""
        self.wasted_coords: List[Coord] = []
        """Wasted core coordinates for the routing group."""

    def assign(self, assigned: List[Coord], wasted: List[Coord]) -> None:
        self.assigned_coords = assigned
        self.wasted_coords = wasted

        # Assign the coordinates to each core block inside the routing group.
        cur_i = 0
        for cb in self:
            n = cb.n_core_required
            cb.core_coords = assigned[cur_i : cur_i + n]
            cur_i += n

    def __getitem__(self, idx: int) -> CoreBlock:
        if idx >= len(self.cb) or idx < 0:
            raise IndexError(f"Index out of range [0, {len(self.cb)}), {idx}.")

        return self.cb[idx]

    def __len__(self) -> int:
        return len(self.cb)

    def __iter__(self) -> Iterator[CoreBlock]:
        return self.cb.__iter__()

    def __contains__(self, key: CoreBlock) -> bool:
        return key in self.cb

    @property
    def n_core_required(self) -> int:
        return sum(cb.n_core_required for cb in self)

    @property
    def routing_cost(self) -> RoutingCost:
        return get_routing_consumption(self.n_core_required)

    @property
    def routing_level(self) -> Level:
        return self.routing_cost.get_routing_level()


@final
class RoutingRoot(RoutingCluster):
    def __init__(self, **kwargs) -> None:
        """Initialize a routing quadtree root(L5-level)."""
        super().__init__(Level.L5, **kwargs)

    def get_leaf_coord(self, cluster: "RoutingCluster") -> RoutingCoord:
        """Return the routing coordinate of the cluster(must be a L0 leaf)."""
        path = self.get_routing_path(cluster)
        if path:
            return RoutingCoord(*path)

        raise RuntimeError(f"Get leaf cluster {cluster.tag} coordinate failed.")

    def insert_routing_group(self, routing_group: RoutingGroup) -> bool:
        """Insert a `RoutingGroup` in the routing tree. Assign each core blocks with \
            routing coordinates & make sure they are routable.

        NOTE: Use depth-first search to insert each core block into the routing tree \
            to ensure that no routing deadlock occurs between core blocks.
        """
        cost = routing_group.routing_cost
        level = routing_group.routing_level
        # Create a routing cluster
        routing_cluster = RoutingCluster.create_routing_tree(level, cost[level - 1])

        # `n_L0` physical cores will be occupied.
        #   - For the first `n_core_required` cores, they are used for placement.
        #   - For the rest, they are unused.
        # Make sure the routing cluster is successfully inserted to the root
        # then assign coordinates & status.
        leaves = []
        wasted = []
        for i in range(cost.n_L0):
            if i < routing_group.n_core_required:
                cluster = routing_cluster.add_L0_for_placing(
                    data=f"{id(routing_group)}_{i}",
                    status=Status.USED,
                    tag=f"{id(routing_group)}_{i}",
                )
                leaves.append(cluster)

            else:
                cluster = routing_cluster.add_L0_for_placing(
                    status=Status.OCCUPIED, tag=f"{id(routing_group)}_{i}"
                )
                wasted.append(cluster)

        # Add the sub-tree to the root.
        flag = self.add_subtree(routing_cluster)
        if not flag:
            return False

        valid_coords = []
        wasted_coords = []
        for cluster in leaves:
            coord = self.get_leaf_coord(cluster)
            valid_coords.append(coord.coordinate)

        for cluster in wasted:
            coord = self.get_leaf_coord(cluster)
            wasted_coords.append(coord.coordinate)

        routing_group.assign(valid_coords, wasted_coords)

        return True

    @property
    def n_L0_clusters(self) -> int:
        return self.breadth_of_lx_clusters(Level.L0)


def get_parent(
    tree: RoutingCluster, cluster: RoutingCluster
) -> Optional[RoutingCluster]:
    """Get the parent cluster of the given cluster. If not found, return None."""
    assert tree != cluster

    def dfs_preorder(
        tree: RoutingCluster, cluster: RoutingCluster
    ) -> Optional[RoutingCluster]:
        for d in ROUTING_DIRECTIONS_IDX:
            if d in tree.children:
                if tree[d] is cluster:
                    return tree
                else:
                    parent = dfs_preorder(tree[d], cluster)
                    if parent:
                        return parent

        return None

    return dfs_preorder(tree, cluster)
