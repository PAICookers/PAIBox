from typing import Any, Dict, Iterator, List, Optional, Sequence, final

from paicorelib import ChipCoord, Coord, HwConfig
from paicorelib.routing_defs import ROUTING_DIRECTIONS_IDX as DIREC_IDX
from paicorelib.routing_defs import RoutingCoord, RoutingCost
from paicorelib.routing_defs import RoutingDirection as Direction
from paicorelib.routing_defs import RoutingLevel as Level
from paicorelib.routing_defs import RoutingStatus as Status
from paicorelib.routing_defs import get_routing_consumption

from paibox.exceptions import ResourceError, RoutingError

from .conf_template import CorePlmConfInChip
from .placement import CoreBlock, CorePlacement, EmptyCorePlacement

__all__ = ["RoutingGroup", "RoutingRoot"]


class RoutingCluster:
    def __init__(
        self,
        level: Level,
        direction: Direction = Direction.ANY,
        *,
        data: Optional[CorePlacement] = None,
        status: Optional[Status] = None,
        tag: Optional[str] = None,
        include_online: bool = False,
    ) -> None:
        """Instance a tree cluster with `level` and `direction`.
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
            - d: the direction of the cluster, relative to its parent.
            - item: the data hanging on the cluster.
            - tag: a tag for user to identify.
            - status: the status of the cluster. Only for L0-level leaves.
        """
        self.level = level
        self.children: Dict[Direction, RoutingCluster] = dict()
        self.d = direction
        self.item = data
        self.tag = tag
        self.include_online = include_online

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

        if self.level > Level.L0:
            dfs(self)

    def create_child(self, **kwargs) -> Optional["RoutingCluster"]:
        """Create a child. If full, return None."""
        child = RoutingCluster(Level(self.level - 1), **kwargs)

        if not self.add_child(child):
            return None

        return child

    def add_child(
        self, child: "RoutingCluster", check_hit_online: bool = False
    ) -> bool:
        if self.level == Level.L0:
            # L0-level cluster cannot add child.
            raise AttributeError(f"L0-level cluster cannot add child.")

        if self.is_full():
            return False

        for d in DIREC_IDX:
            if d not in self:
                return self.add_child_to(child, d, check_hit_online)

        return False

    def add_child_to(
        self, child: "RoutingCluster", d: Direction, check_hit_online: bool = False
    ) -> bool:
        """Add a child cluster to a certain `direction`."""
        if self.level - child.level != 1:
            raise ValueError(f"Cannot skip more than 1 level.")

        if d in self:
            return False

        if d == Direction.X1Y1:
            if self.include_online and check_hit_online:
                return False
            else:
                child.include_online = True

        # child.direction = d. Already done in `self[d]`(__setitem__).
        self[d] = child

        return True

    def remove_child(
        self,
        d: Direction,
        revert_direc: Direction = Direction.ANY,
        strict: bool = False,
    ) -> Optional["RoutingCluster"]:
        child = self.children.pop(d, None)

        if child is None:
            if strict:
                raise RoutingError(f"Removed child of {d} from {self} failed.")
            else:
                return None

        # Revert the properties that were modified in the previous insertion.
        child.include_online = False
        child.d = revert_direc

        return child

    def find_cluster_by_path(
        self, path: Sequence[Direction]
    ) -> Optional["RoutingCluster"]:
        """Find the cluster by given a routing path.

        Description:
            Find by starting at this level based on the routing path provided.
            Take `path[0]` each time and do a recursive search.
        """
        if len(path) == 0:
            return self

        if len(path) > self.level:
            raise ValueError(
                f"the length of path must be no more than {self.level}, but got {len(path)}."
            )

        if path[0] not in self:
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
            raise ValueError(
                f"Cannot get routing path because the level cluster is higher."
            )

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
        """Find the child of level `lx` with at least `n_child_avail` child available."""
        if lx > self.level:
            raise ValueError

        if lx == self.level:
            if self.n_child_avail() >= n_child_avail:
                return self
            else:
                return None

        if not self.is_empty():
            for d in DIREC_IDX:
                if d in self:
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
        check_hit_online: bool,
    ) -> bool:
        """Add the subtree's children to itself. If successful, return the added parent cluster."""
        if subtree.level > self.level:
            raise ValueError(
                f"subtree's level {subtree.level} must be no more than the current level {self.level}."
            )

        if subtree.level == self.level:
            sub_n_child = len(subtree.children)
            if self.n_child_avail() < sub_n_child:
                return False

            if sub_n_child == 1:
                self.add_child(subtree[Direction.X0Y0], check_hit_online)

            elif sub_n_child == 2:
                n_cur_child = len(self.children)
                hit_online = False

                for i in range(sub_n_child):
                    success = self.add_child_to(
                        subtree[DIREC_IDX[i]],
                        DIREC_IDX[n_cur_child + i],
                        check_hit_online,
                    )
                    hit_online |= not success

                if hit_online:
                    # If any of the subtrees fail to insert, the inserted subtrees are removed.
                    for i in range(sub_n_child):
                        removed = self.remove_child(
                            DIREC_IDX[n_cur_child + i], DIREC_IDX[i], strict=False
                        )

                    return False

            elif sub_n_child == 4:
                if self.include_online and check_hit_online:
                    return False

                self.children = subtree.children
                # Because the tree is inserted using depth-first order, when a node is
                # encountered with no child, it must be on the far right.
                self[Direction.X1Y1].include_online = True

            else:
                raise ValueError(f"the number of {sub_n_child} child is invalid.")

            return True

        if not self.is_empty():
            for d in DIREC_IDX:
                if d in self:
                    flag = self[d].add_subtree(subtree, check_hit_online)
                    if flag:
                        return True

        child = self.create_child()
        if not child:
            return False

        return child.add_subtree(subtree, check_hit_online)

    @classmethod
    def create_lx_full_tree(
        cls,
        lx: Level,
        d: Direction = Direction.X0Y0,
        root_tag: Optional[str] = None,
    ) -> "RoutingCluster":
        root = RoutingCluster(lx, d, tag=root_tag)

        if lx > Level.L1:
            for i in range(root.node_capacity):
                child = cls.create_lx_full_tree(
                    Level(lx - 1), DIREC_IDX[i], f"L{lx-1}_{i}"
                )
                if not root.add_child(child):
                    raise ValueError

        return root

    @classmethod
    def create_routing_tree(cls, lx: Level, n_branch: int) -> "RoutingCluster":
        """Create a routing tree with `n_branch` child.

        NOTE: When lx == L1, do not create the L0-level child. \
            WHen lx > L1, create the lx-1 level child.
        """
        if n_branch < 0 or n_branch > HwConfig.N_SUB_ROUTING_NODE:
            raise ValueError(f"#N of branches out of range, got {n_branch}.")

        if lx == Level.L0:
            raise ValueError(f"do not create L0-level node directly.")

        root = RoutingCluster(lx, Direction.X0Y0)

        # Create `n_branch` children when lx > L1.
        if lx > Level.L1:
            for i in range(n_branch):
                child = cls.create_lx_full_tree(Level(lx - 1), DIREC_IDX[i])
                if not root.add_child(child):
                    raise ValueError(f"add child {child} failed.")

        return root

    def add_L0_for_placing(self, data: Any = None, **kwargs) -> "RoutingCluster":
        """Add L0 cluster for placing in the routing tree.

        Args:
            - data: the data attached to the L0-level cluster.
            - kwargs: other arguments of the L0-level cluster, status, tag, etc.
        """
        cluster = RoutingCluster(Level.L0, data=data, **kwargs)

        L1_cluster = self._find_lx_cluster_with_n_child_avail(Level.L1, 1)
        if not L1_cluster:
            raise RoutingError("available L1 cluster not found.")

        if not L1_cluster.add_child(cluster):
            raise RoutingError(f"add child to L1 cluster failed.")

        return cluster

    def find_lx_clusters(
        self, lx: Level, n_child_avail_low: int = 0
    ) -> List["RoutingCluster"]:
        """Find all clusters at a `lx` level with at least `n_child_avail_low` child clusters."""
        if lx > self.level:
            return []

        clusters = []

        def dfs_preorder(root: RoutingCluster) -> None:
            if root.level == lx:
                if root.n_child_avail() >= n_child_avail_low:
                    clusters.append(root)

                return None

            for d in DIREC_IDX:
                if d in root:
                    dfs_preorder(root[d])

        dfs_preorder(self)
        return clusters

    def find_leaf_at_level(self, lx: Level) -> List["RoutingCluster"]:
        """Find clusters with no child at the `lx` level."""
        if lx == Level.L0:
            return []

        return self.find_lx_clusters(lx, self.node_capacity)

    def breadth_of_lx(self, lx: Level) -> int:
        """Get the number of clusters in the routing tree at the given level."""
        clusters = self.find_lx_clusters(lx, 0)

        return len(clusters)

    def __getitem__(self, d: Direction) -> "RoutingCluster":
        return self.children[d]

    def __setitem__(self, d: Direction, child: "RoutingCluster") -> None:
        self.children[d] = child
        child.d = d  # Set the direction of the child.

    def __str__(self) -> str:
        _name = id(self) if self.tag is None else self.tag
        return f"tree {_name} at {self.d.name} at level {self.level}"

    def __iter__(self) -> Iterator[Direction]:
        return self.children.__iter__()

    def __contains__(self, d: Direction) -> bool:
        return d in self.children

    @property
    def node_capacity(self) -> int:
        return HwConfig.N_SUB_ROUTING_NODE if self.level > Level.L0 else 0


class RoutingGroup(List[CoreBlock]):
    """Core blocks located within a routing group are routable.

    NOTE: Axon groups within a routing group are the same.
    """

    def __init__(self, *cb: CoreBlock) -> None:
        self.core_blocks = list(cb)
        self.assigned_coords: List[Coord] = []
        """Assigned core coordinates in the routing group"""
        self.wasted_coords: List[Coord] = []
        """Wasted core coordinates in routing group"""
        self.wasted_core_plm: Dict[Coord, EmptyCorePlacement] = {}
        """Wasted core placements"""

    def assign(
        self, assigned: List[Coord], wasted: List[Coord], chip_coord: Coord
    ) -> None:
        self.assigned_coords = assigned
        self.wasted_coords = wasted

        # Assign the coordinates to each core block inside the routing group.
        cur_i = 0
        for cb in self:
            n = cb.n_core_required
            cb.core_coords = assigned[cur_i : cur_i + n]
            cb.chip_coord = chip_coord
            cur_i += n

    def core_block_alloc(self) -> None:
        for cb in self:
            cb.core_plm_alloc()

        # Allocate blank core placements for the wasted coordinates.
        for coord in self.wasted_coords:
            self.wasted_core_plm[coord] = EmptyCorePlacement.build(coord)

    def get_wasted_cplm_config(self) -> CorePlmConfInChip:
        return {
            coord: core_plm.export_core_plm_config()
            for coord, core_plm in self.wasted_core_plm.items()
        }

    def get_n_core_occupied(self) -> int:
        """Get the #N of cores occupied by the routing group."""
        return len(self.assigned_coords) + len(self.wasted_coords)

    def __getitem__(self, idx: int) -> CoreBlock:
        if idx >= len(self.core_blocks) or idx < 0:
            raise IndexError(
                f"index out of range [0, {len(self.core_blocks)}), ({idx})."
            )

        return self.core_blocks[idx]

    def __len__(self) -> int:
        return len(self.core_blocks)

    def __iter__(self) -> Iterator[CoreBlock]:
        return self.core_blocks.__iter__()

    def __contains__(self, key: CoreBlock) -> bool:
        return key in self.core_blocks

    @property
    def n_core_required(self) -> int:
        return sum(cb.n_core_required for cb in self)

    @property
    def routing_cost(self) -> RoutingCost:
        return get_routing_consumption(self.n_core_required)

    @property
    def routing_level(self) -> Level:
        return self.routing_cost.get_routing_level()

    @property
    def chip_coord(self) -> ChipCoord:
        if not all(cb.chip_coord == self[0].chip_coord for cb in self):
            raise RoutingError(
                "Chip coordinates in the routing group is not consistent."
            )

        return self[0].chip_coord


@final
class RoutingRoot:
    def __init__(self, chip_list: List[ChipCoord], **kwargs) -> None:
        """Initialize a routing quadtree root."""
        self.chip_list = chip_list
        # Every L5 routing cluster is unique in each chip root.
        self.chip_roots = [
            RoutingCluster(Level.L5, include_online=True) for _ in range(len(chip_list))
        ]

    def get_leaf_coord(
        self, root: RoutingCluster, leaf: RoutingCluster
    ) -> RoutingCoord:
        """Return the routing coordinate of the L0 leaf."""
        path = root.get_routing_path(leaf)
        if path:
            return RoutingCoord(*path)

        raise RoutingError(f"get leaf {leaf.tag} coordinate failed.")

    def insert_routing_group(self, routing_group: RoutingGroup) -> bool:
        """Insert a `RoutingGroup` in the routing tree. Assign each core blocks with \
            routing coordinates & make sure they are routable.

        NOTE: Use depth-first search to insert each core block into the routing tree \
            to ensure that no routing deadlock occurs between core blocks.
        """
        cost = routing_group.routing_cost
        level = routing_group.routing_level
        if cost.n_L0 > HwConfig.N_CORE_OFFLINE:
            raise ResourceError(
                f"the number of cores required exceeds the hardware limit, {cost.n_L0} > {HwConfig.N_CORE_OFFLINE}."
            )

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
                l0 = routing_cluster.add_L0_for_placing(
                    data=f"{id(routing_group)}_{i}",
                    status=Status.USED,
                    tag=f"{id(routing_group)}_{i}",
                )
                leaves.append(l0)

            else:
                l0 = routing_cluster.add_L0_for_placing(
                    status=Status.OCCUPIED, tag=f"{id(routing_group)}_{i}"
                )
                wasted.append(l0)

        # If #N of wasted cores > 16, it won't hit online L2 cluster.
        check_hit_online = len(wasted) <= HwConfig.N_CORE_ONLINE

        # Add the sub-tree to the root.
        flag = False
        # TODO For now, use sequential attempt.
        for chip_coord, chip_root in zip(self.chip_list, self.chip_roots):
            flag = chip_root.add_subtree(routing_cluster, check_hit_online)
            if flag:
                break

        if not flag:
            raise RoutingError(
                f"insert routing group {routing_group} into the routing tree failed, "
                f"cannot insert to any chip."
            )

        # TODO Consider obtaining the root coord of the `routing_cluster` applied for when inserting,
        # and calculate all leaf coords according to the size of the routing group. Instead of
        # recording all the leaves and then looking up their coordinates in the tree.
        valid_coords = []
        wasted_coords = []
        for cluster in leaves:
            coord = self.get_leaf_coord(chip_root, cluster)
            valid_coords.append(coord.to_coord())

        for cluster in wasted:
            coord = self.get_leaf_coord(chip_root, cluster)
            wasted_coords.append(coord.to_coord())

        routing_group.assign(valid_coords, wasted_coords, chip_coord)

        return True

    def clear(self) -> None:
        for root in self:
            root.clear()

    def breadth_of_lx_at(self, lx: Level, chip_idx: int) -> int:
        return self[chip_idx].breadth_of_lx(lx)

    def breadth_of_lx(self, lx: Level) -> int:
        return sum(chip_root.breadth_of_lx(lx) for chip_root in self)

    def __getitem__(self, index: int) -> RoutingCluster:
        return self.chip_roots[index]

    def __iter__(self) -> Iterator[RoutingCluster]:
        return self.chip_roots.__iter__()


def get_parent(
    tree: RoutingCluster, cluster: RoutingCluster
) -> Optional[RoutingCluster]:
    """Get the parent cluster of the given cluster. If not found, return None."""
    assert tree != cluster

    def dfs_preorder(
        tree: RoutingCluster, cluster: RoutingCluster
    ) -> Optional[RoutingCluster]:
        for d in DIREC_IDX:
            if d in tree:
                if tree[d] is cluster:
                    return tree
                else:
                    parent = dfs_preorder(tree[d], cluster)
                    if parent:
                        return parent

        return None

    return dfs_preorder(tree, cluster)
