import itertools
import math
from collections.abc import Generator, Iterator, Sequence
from typing import Any, Optional, Union, final

from paicorelib import ROUTING_DIRECTIONS_IDX as DIREC_IDX
from paicorelib import ChipCoord, Coord, HwConfig, RoutingCoord, RoutingCost
from paicorelib import RoutingDirection as Direction
from paicorelib import RoutingLevel as Level
from paicorelib import RoutingStatus as Status
from paicorelib import get_routing_consumption
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

from paibox.exceptions import ResourceError, RoutingError

from .conf_types import CorePlmConfInChip
from paibox.exceptions import ResourceError, RoutingError, GraphBuildError
from .placement import CoreBlock, CorePlacement, EmptyCorePlacement
from .types import *
__all__ = ["RoutingGroup", "RoutingRoot"]

def Coord2RoutingCoord(coord: Coord) -> RoutingCoord:
    directions: list[Direction] = []
    x = coord.x
    y = coord.y

    for i in range(MAX_ROUTING_PATH_LENGTH):
        # 每个循环，提取最高位（移动了 4-i 位）到最低位，恢复 value_x 和 value_y
        shift = 4 - i
        value_x = (x >> shift) & 0b1  # 取出当前位的值
        value_y = (y >> shift) & 0b1
        directions.append(Direction((value_x, value_y)))
    return RoutingCoord(*directions)    

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
        parent: Optional["RoutingCluster"] = None,
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
        self.children: dict[Direction, RoutingCluster] = dict()
        self.d = direction
        self.item = data
        self.tag = tag
        self.include_online = include_online
        self.parent = parent

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
            raise AttributeError("L0-level cluster cannot add child.")

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
            raise ValueError("cannot skip more than 1 level.")

        if d in self:
            return False

        if d == Direction.X1Y1 or self.level in (Level.L1, Level.L2):
            if self.include_online and check_hit_online:
                return False
            else:
                child.include_online = self.include_online

        # child.direction = d. Already done in `self[d]`(__setitem__).
        self[d] = child
        child.parent = self

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
                raise RoutingError(f"removed child of {d} from {self} failed.")
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

    def get_routing_path(self, cluster: "RoutingCluster") -> Optional[list[Direction]]:
        """Return a direction path from L4 to the level of `cluster`.

        Args:
            - cluster: the cluster with level <= `self.level`.

        Return:
            - A list of `Direction` from L4 to L0.
        """
        if cluster.level > self.level:
            raise ValueError(
                "cannot get routing path because the level cluster is higher."
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
                return self.add_child(subtree[Direction.X0Y0], check_hit_online)

            elif sub_n_child == 2:
                # len(self.children) == 0, place in [0,1]
                # len(self.children) == 1, place in [2,3]
                if len(self.children) == 0:
                    _place_idx = (0, 1)
                else:  # 2 & 3
                    _place_idx = (2, 3)

                hit_online = False

                for i in range(sub_n_child):
                    success = self.add_child_to(
                        subtree[DIREC_IDX[i]],
                        DIREC_IDX[_place_idx[i]],
                        check_hit_online,
                    )
                    hit_online |= not success

                if hit_online:
                    # If any of the subtrees fail to insert, the inserted subtrees are removed.
                    for i in range(sub_n_child):
                        removed = self.remove_child(
                            DIREC_IDX[_place_idx[i]], DIREC_IDX[i], strict=False
                        )
                        subtree[DIREC_IDX[i]].parent = subtree

                    return False

            elif sub_n_child == 4:
                if self.include_online and check_hit_online:
                    return False

                self.children = subtree.children
                # Because the tree is inserted using depth-first order, when a node is
                # encountered with no child, it must be on the far right.
                self[Direction.X1Y1].include_online = True
                for child in self.children.values():
                    child.parent = self

            else:
                raise ValueError(f"the number of {sub_n_child} child is invalid.")

            return True

        # subtree.level < self.level
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
            raise ValueError("do not create L0-level node directly.")

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
            raise RoutingError("add child to L1 cluster failed.")

        return cluster

    def find_lx_clusters(
        self, lx: Level, n_child_avail_low: int = 0
    ) -> list["RoutingCluster"]:
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

    def find_leaf_at_level(self, lx: Level) -> list["RoutingCluster"]:
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

    @property
    def routing_coord(self) -> RoutingCoord:
        cur_cluster = self
        path = [self.d]

        while cur_cluster.parent is not None:
            path.append(cur_cluster.parent.d)
            cur_cluster = cur_cluster.parent

        path = path[:-1]

        for _ in range(cur_cluster.level, Level.L5):
            path.append(Direction.X0Y0)

        for _ in range(self.level):
            path.insert(0, Direction.ANY)

        return RoutingCoord(*reversed(path))

# each sub routing group should be able to route by single coord
class SubRoutingGroup:
    index = 0
    def __init__(self, unorder_elements: list["CoreBlock|SubRoutingGroup"], ordered_elements: list["SubRoutingGroup"]) -> None:
        self.unorder_elements:list["CoreBlock|SubRoutingGroup"] = unorder_elements
        self.ordered_elements:list["SubRoutingGroup"] = ordered_elements
        self.routing_elements:list["CoreBlock|SubRoutingGroup"] = unorder_elements + ordered_elements
        self.offset:list[int] = list()
        self.n_core_required:int = 0
        self.tail_wasted:int = 0
        self.name = f"SubRoutingGroup[{SubRoutingGroup.index}]"
        axons:set[SourceNodeType] = set()
        for element in self.routing_elements:
            axons.update(element.axons)
        self.axons:list[SourceNodeType] = list(axons)
        SubRoutingGroup.index += 1
    
    def set_config(self):
        for element in self.routing_elements:
            if isinstance(element, SubRoutingGroup):
                element.set_config()

        # unorder elements sorted from big to small, avoiding assigning waste.
        unorder_elements = sorted(self.unorder_elements, key=lambda x: x.n_core_required, reverse=True)
        ordered_elements = self.ordered_elements
        for element in unorder_elements:
            n_core_required = element.n_core_required
            self.offset.append(self.n_core_required)
            self.n_core_required += n_core_required
        
        # ordered elements should be assgined first
        for element in ordered_elements:
            n_core_required = element.n_core_required
            n_core_assigned = _nearest_multiple_above(self.n_core_required, n_core_required)
            self.offset.append(n_core_assigned)
            self.n_core_required = n_core_assigned + n_core_required
        
        #routing elements should satisfy topological order
        self.routing_elements:list["CoreBlock|SubRoutingGroup"] = unorder_elements + ordered_elements
        
        
        sub_tail_wasted = 0 if isinstance(self.routing_elements[-1], CoreBlock) else self.routing_elements[-1].tail_wasted
        assigned_n_core_required = 1 << (self.n_core_required - 1).bit_length()
        self.tail_wasted += assigned_n_core_required - self.n_core_required + sub_tail_wasted
        self.n_core_required = assigned_n_core_required
        
        
    # return Coord that wasted in subrouting group
    def assign(self, allocated: list[Coord], chip_coord: Coord) -> tuple[list[Coord], list[Coord]]:
        cur_i = 0
        assigned_coords:list[Coord] = []
        wasted_coords:list[Coord] = []
        for element, offset in zip(self.routing_elements, self.offset):
            if offset > cur_i:
                wasted_coords = wasted_coords + allocated[cur_i : offset]
            cur_i = offset
            
            n = element.n_core_required
            print(f"element: {element.name}, {n} cores, start at {Coord2RoutingCoord(allocated[cur_i])}")
            assigned, wasted = element.assign(allocated[cur_i : cur_i + n], chip_coord)
            assigned_coords = assigned_coords + assigned
            wasted_coords = wasted_coords + wasted
            cur_i += n
        return assigned_coords, wasted_coords + allocated[cur_i:]
    
    # use list to keep the order of axons
    def group_axons(self, multicast_axons: list[SourceNodeType]) -> None:
        private_multicast_axons = multicast_axons.copy()
        axons_count:list[int] = [0] * len(self.axons)
        for element in self.routing_elements:
            for axon in element.axons:
                idx = self.axons.index(axon)
                axons_count[idx] += 1
        for i, axon in enumerate(self.axons):
            if axons_count[i] > 1 and axon not in private_multicast_axons:
                private_multicast_axons.append(axon)
        
        for element in self.routing_elements:
            element.group_axons(private_multicast_axons)
        
    @property
    def core_blocks(self) -> list[CoreBlock]:
        cbs = []
        for element in self.routing_elements:
            if isinstance(element, CoreBlock):
                cbs.append(element)
            else:
                cbs += element.core_blocks
        return cbs
    
    @classmethod
    def build(cls, route_group: RouteGroup) -> "SubRoutingGroup":
        
        if len(route_group.nodes) == 0:
            return None
        sub_group = RouteGroup()
        remaining_group = RouteGroup()
        for group in route_group.groups:
            if group.input in route_group.nodes:
                sub_group.add_group(group)
            else:
                remaining_group.add_group(group)
                
        remaining_group.nodes = remaining_group.nodes - sub_group.nodes
        unorder_elements:list[CoreBlock] = CoreBlock.build_core_blocks(remaining_group)
        ordered_elements:list[SubRoutingGroup] = []
        sub_routing_group: SubRoutingGroup = SubRoutingGroup.build(sub_group)
        if sub_routing_group is not None:
            ordered_elements = [sub_routing_group]
        return cls(unorder_elements, ordered_elements)
    
    def dump(self, i:int = 0):
        tabs = "\t" * i
        print(f"{tabs}SubRoutingGroup: {self.name} with {self.n_core_required} cores:")
        for element in self.routing_elements:
            if isinstance(element, SubRoutingGroup):
                element.dump(i+1)
            else:
                print(f"{tabs}\t{element.name} with {element.n_core_required} cores:")
                for edge in element._parents:
                    print(f"{tabs}\t\t{edge.name}: {edge.source.name} -> {edge.target.name}")
                

class RoutingGroup:
    """Core blocks located within a routing group are routable.

    NOTE: Axon groups within a routing group are the same.
    """

    def __init__(self, route_group:RouteGroup) -> None:
        self.sub_routing_group: SubRoutingGroup = SubRoutingGroup.build(route_group)
        self.core_blocks = self.sub_routing_group.core_blocks
        self.assigned_coords: list[Coord] = []
        """Assigned core coordinates in the routing group"""
        self.wasted_coords: list[Coord] = []
        """Wasted core coordinates in routing group"""
        self.wasted_core_plm: dict[Coord, EmptyCorePlacement] = {}
        """Wasted core placements"""
        self.sub_n_core_wasted = 0

    def assign(
        self, allocated: list[Coord], chip_coord: Coord
    ) -> None:
        print(f"route_group: {self.sub_routing_group.name} assigned from {Coord2RoutingCoord(allocated[0])}")
        assigned, wasted = self.sub_routing_group.assign(allocated, chip_coord)
        self.assigned_coords = assigned
        self.wasted_coords = wasted

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

    @property
    def n_core_required(self) -> int:
        """The actual number of cores required by the routing group."""
        return sum(cb.n_core_required for cb in self)

    @property
    def n_core_cost(self) -> int:
        return self.sub_routing_group.n_core_required
    
    @property
    def tail_wasted(self) -> int:
        return self.sub_routing_group.tail_wasted

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
                "chip coordinates in the routing group are not consistent."
            )

        return self[0].chip_coord

    def __contains__(self, cb: CoreBlock) -> bool:
        return cb in self.core_blocks

    def __getitem__(self, idx: int) -> CoreBlock:
        return self.core_blocks[idx]

    def __iter__(self) -> Iterator[CoreBlock]:
        return self.core_blocks.__iter__()

    def group_axons(self) -> None:
        for cb in self.core_blocks:
            if not cb._lcn_locked:
                raise GraphBuildError("get axon segments after 'lcn_ex' is locked.")
        self.sub_routing_group.group_axons([])
        
        

@final
class RoutingRoot:
    def __init__(self, chip_list: list[ChipCoord], **kwargs) -> None:
        """Initialize a routing quadtree root."""
        self.chip_list: list[ChipCoord] = chip_list
        # Every L5 routing cluster is unique in each chip root.
        self.chip_roots = [
            RoutingCluster(Level.L5, include_online=True) for _ in range(len(chip_list))
        ]
        self.used_L2_clusters: list[list[RoutingCoord]] = [
            list() for _ in range(len(chip_list))
        ]
        """Used L2 clusters in each chip root. The clocks of unused L2 clusters can be turned off   \
            through the serial port to reduce power consumption.
        """
        self.n_core_total: int = 0
        self.n_core_per_chip: list[int] = [0] * len(chip_list)

    def get_leaf_coord(
        self, root: RoutingCluster, leaf: RoutingCluster
    ) -> RoutingCoord:
        """Return the routing coordinate of the L0 leaf."""
        path = root.get_routing_path(leaf)
        if path:
            return RoutingCoord(*path)

        raise RoutingError(f"get leaf {leaf.tag} coordinate failed.")

    def get_insert_location(
        self, n_core_incoming: int, n_core_wasted: int
    ) -> tuple[int, int, list[Direction]]:
        """Look for the insertion location of the incoming routing group."""
        n_core_aligned = _nearest_multiple_above(self.n_core_total, n_core_incoming)

        n_core_predicted = n_core_aligned + n_core_incoming
        n_core_inchip = _num_inchip(n_core_predicted)

        # If online cores are hit, start from the next chip
        if n_core_inchip - n_core_wasted > HwConfig.N_CORE_OFFLINE:
            n_core_aligned = _nearest_multiple_above(
                n_core_aligned, HwConfig.N_CORE_MAX_INCHIP
            )

        core_loc = n_core_aligned

        if (chip_idx_loc := core_loc // HwConfig.N_CORE_MAX_INCHIP) >= len(
            self.chip_list
        ):
            raise ResourceError(
                f"the number of required chips exceeds the limit {len(self.chip_list)} ({chip_idx_loc+1})."
            )

        self.n_core_total = n_core_aligned + n_core_incoming
        self.n_core_per_chip[chip_idx_loc] = _num_inchip(self.n_core_total)

        routing_idx = core_loc % HwConfig.N_CORE_MAX_INCHIP
        routing_path = []

        # From L0 to L4
        for _ in range(MAX_ROUTING_PATH_LENGTH):
            routing_idx, re = divmod(routing_idx, HwConfig.N_SUB_ROUTING_NODE)
            routing_path.append(DIREC_IDX[re])

        return core_loc, chip_idx_loc, routing_path

    def place_routing_group(self, routing_group: RoutingGroup) -> None:
        """Place a routing group in the chip list. Assign each core blocks with routing coordinates &   \
            make sure they are routable.
        """
        print(f"Routing Group:")
        for cb in routing_group:
            print(f"\t{cb.name}")
        
        n_core_cost = routing_group.n_core_cost
        tail_wasted = routing_group.tail_wasted
        n_core_req = n_core_cost - tail_wasted
        print(f"\tcost: {n_core_cost}, tail_wasted: {tail_wasted}")

        if  n_core_req > HwConfig.N_CORE_OFFLINE:
            raise ResourceError(
                "the number of cores required by the routing group exceeds the hardware limit, "
                f"{n_core_req} > {HwConfig.N_CORE_OFFLINE}."
            )

        core_insert_loc, chip_idx_loc, rpath_start = self.get_insert_location(
            n_core_cost, tail_wasted
        )
        allocated_coords:list[Coord] = []

        for i, rpath in _routing_path_generator(n_core_cost, rpath_start):
            leaf_coord = RoutingCoord(*reversed(rpath))
            # Record the used L2 clusters
            if (core_insert_loc + i) % (HwConfig.N_SUB_ROUTING_NODE**Level.L2) == 0:
                L2_coord = RoutingCoord(*reversed(rpath[Level.L2 :]))
                self.used_L2_clusters[chip_idx_loc].append(L2_coord)
            allocated_coords.append(leaf_coord.to_coord())

        routing_group.assign(allocated_coords, self.chip_list[chip_idx_loc])
        print()

    def insert_routing_group(self, routing_group: RoutingGroup) -> bool:
        """Insert a `RoutingGroup` in the routing tree. Assign each core blocks with \
            routing coordinates & make sure they are routable.

        NOTE: Use depth-first search to insert each core block into the routing tree \
            to ensure that no routing deadlock occurs between core blocks.
        """
        cost = routing_group.routing_cost
        level = routing_group.routing_level
        if routing_group.n_core_required > HwConfig.N_CORE_OFFLINE:
            raise ResourceError(
                f"the number of cores required by the routing group exceeds the hardware limit, "
                f"{routing_group.n_core_required} > {HwConfig.N_CORE_OFFLINE}."
            )

        routing_cluster = RoutingCluster.create_routing_tree(level, cost[level - 1])

        # `n_L0` physical cores will be occupied.
        #   - For the first `n_core_required` cores, they are used for placement.
        #   - For the rest, they are unused.
        # Make sure the routing cluster is successfully inserted to the root
        # then assign coordinates & status.
        leaves = []
        wasted = []

        if cost.n_L0 > HwConfig.N_CORE_OFFLINE:
            _max_n_l0 = HwConfig.N_CORE_OFFLINE
        else:
            _max_n_l0 = cost.n_L0

        for i in range(routing_group.n_core_required):
            l0 = routing_cluster.add_L0_for_placing(
                data=f"rg_{id(routing_group)}_{i}",
                status=Status.USED,
                tag=f"rg_{id(routing_group)}_{i}",
            )
            leaves.append(l0)

        for i in range(routing_group.n_core_required, _max_n_l0):
            l0 = routing_cluster.add_L0_for_placing(
                status=Status.OCCUPIED, tag=f"rg_{id(routing_group)}_{i}"
            )
            wasted.append(l0)

        # If #N of wasted cores > 16, it won't hit online L2 cluster.
        # XXX 'check_hit_online' conditions could be more precise, but
        # there is no clear benefit to doing so at the moment.
        check_hit_online = (
            _max_n_l0 - routing_group.n_core_required
        ) <= HwConfig.N_CORE_ONLINE

        # Add the sub-tree to the root.
        flag = False
        # TODO For now, use sequential attempt.
        for chip_coord, chip_root in zip(self.chip_list, self.chip_roots):
            flag = chip_root.add_subtree(routing_cluster, check_hit_online)
            if flag:
                break

        if not flag:
            raise RoutingError(
                f"insert routing group 0x{id(routing_group):x} into the routing tree failed, "
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

    def breadth_of_lx(self, lx: Union[Level, int], chip_idx: int = -1) -> int:
        """Get the breadth of the given level at chip root #idx.

        Args:
            - lx: the level to find.
            - chip_idx: the chip root index. If it is -1, return the sum of the breadth on all roots.
        """
        if chip_idx == -1:
            return sum(chip_root.breadth_of_lx(Level(lx)) for chip_root in self)

        return self[chip_idx].breadth_of_lx(Level(lx))

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


def _nearest_multiple_above(a: int, x: int) -> int:
    """Return the nearest number greater than or equal to `a`, and is an integer multiple of `x`."""
    # (above + multiple - 1) // multiple
    return math.ceil(a / x) * x


def _num_inchip(n: int) -> int:
    return (n - 1) % HwConfig.N_CORE_MAX_INCHIP + 1


def _routing_path_generator(
    n_times: int, rpath: list[Direction]
) -> Generator[tuple[int, list[Direction]], Any, None]:
    for i in range(n_times):
        yield i, rpath

        for lx in range(len(rpath)):
            if rpath[lx] == DIREC_IDX[-1]:
                rpath[lx] = DIREC_IDX[0]
            else:
                rpath[lx] = DIREC_IDX[(DIREC_IDX.index(rpath[lx]) + 1) % len(DIREC_IDX)]
                break


# def _routing_path_generator(
#     n_times: int, rpath: RoutingPath, yield_first: bool = True
# ) -> Generator[tuple[int, RoutingPath], Any, None]:
#     _len = len(rpath)
#     lx_iter = reversed(range(_len)) if rpath.reversed else range(_len)

#     for i in range(n_times):
#         if yield_first:
#             yield i, rpath

#         for lx in lx_iter:
#             if rpath[lx] == DIREC_IDX[-1]:
#                 rpath[lx] = DIREC_IDX[0]
#             else:
#                 rpath[lx] = DIREC_IDX[(DIREC_IDX.index(rpath[lx]) + 1) % len(DIREC_IDX)]
#                 break

#         if not yield_first:
#             yield i, rpath


def _all_lx_clusters(lx: Union[Level, int]) -> list[RoutingCoord]:
    return [
        RoutingCoord(*path)
        for path in itertools.product(DIREC_IDX, repeat=MAX_ROUTING_PATH_LENGTH - lx)
    ]


def get_unused_lx(
    used_lx: list[RoutingCoord], lx: Union[Level, int] = Level.L2
) -> list[RoutingCoord]:
    all_lx = _all_lx_clusters(lx)

    for l in set(used_lx):  # make used_lx unduplicated
        all_lx.remove(l)  # keep the rest clusters in order

    return all_lx
