from dataclasses import dataclass, field
from enum import Enum, IntEnum, unique
from typing import List, Sequence, Set, Tuple

from ._types import ReplicationFlag as RFlag
from .coordinate import Coord, ReplicationId as RId
from .hw_defs import HwConfig


@unique
class RoutingOP(Enum):
    UP = 0
    DOWN_UNICAST = 1
    DOWN_MULTICAST = 2


@unique
class RoutingNodeLevel(IntEnum):
    L0 = 0
    """Leaves of tree to store the data. A L0-layer is a core."""
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5
    """The root."""


@unique
class RoutingDirection(Enum):
    """Indicate the 4 children of a node.

    NOTE: There is an X/Y coordinate priority method \
        to specify the order of the 4 children.
    """

    X0Y0 = (0, 0)
    X0Y1 = (0, 1)
    X1Y0 = (1, 0)
    X1Y1 = (1, 1)
    ANY = (-1, -1)
    """Don't care when a level direction is `ANY`."""

    def to_index(self) -> int:
        """Convert the direction to index in children list."""
        if self is RoutingDirection.ANY:
            # TODO
            raise ValueError

        x, y = self.value

        return (x << 1) + y


@unique
class RoutingNodeStatus(IntEnum):
    """Indicate the status of L0-level nodes."""

    AVAILABLE = 0
    """Available for item to attach."""
    
    USED = 1
    """An item is attached to this node."""

    OCCUPIED = 2
    """Wasted. It will be an optimization goal."""


@dataclass
class RoutingNodeCost:
    n_L0: int
    n_L1: int
    n_L2: int
    n_L3: int
    n_L4: int

    def get_routing_level(self) -> Tuple[RoutingNodeLevel, int]:
        """Return the routing level.

        If the #N of Lx-level > 1, then we need A node with level Lx+1.
            And we need the #N of routing sub-level nodes.
        """
        if self.n_L4 > 1:
            return RoutingNodeLevel.L5, self.n_L4
        elif self.n_L3 > 1:
            return RoutingNodeLevel.L4, self.n_L3
        elif self.n_L2 > 1:
            return RoutingNodeLevel.L3, self.n_L2
        elif self.n_L1 > 1:
            return RoutingNodeLevel.L2, self.n_L1
        else:
            return RoutingNodeLevel.L1, self.n_L0


def get_node_consumption(n_core: int) -> RoutingNodeCost:
    """Get the nodes consumption at different levels given the `n_core`."""

    def min_n_L0_nodes(n_core: int) -> int:
        """Find the nearest #N(=2^X) to accommodate \
            `n_core` L0-level nodes.
        
        If n_core = 5, return 8.
        If n_core = 20, return 32.
        """
        n_L0_nodes = 1
        while n_core > n_L0_nodes:
            n_L0_nodes *= 2

        return n_L0_nodes

    n_sub_node = HwConfig.N_SUB_ROUTER_NODE

    n_L0 = min_n_L0_nodes(n_core)

    n_L1 = 1 if n_L0 < n_sub_node else (n_L0 // n_sub_node)
    n_L2 = 1 if n_L1 < n_sub_node else (n_L1 // n_sub_node)
    n_L3 = 1 if n_L2 < n_sub_node else (n_L2 // n_sub_node)
    n_L4 = 1 if n_L3 < n_sub_node else (n_L3 // n_sub_node)

    return RoutingNodeCost(n_L0, n_L1, n_L2, n_L3, n_L4)


def lx_need_copy(rflag: RFlag, lx: int) -> bool:
    """Return the bit Lx wether needs do replication.

    Arguments:
        - rflag: the feplication flag of X or Y.
        - lx: the bit of Lx.
    """
    return rflag & RFlag(1 << lx) == RFlag(1 << lx)


def get_replication_id(dest_coords: Sequence[Coord]) -> RId:
    """
    Arguments:
        - dest_coords: the list of coordinates which are the destinations of a frame.

    Return:
        The replication ID.
    """
    base_coord = dest_coords[0]
    rid = RId(0, 0)

    for coord in dest_coords[1:]:
        rid |= base_coord ^ coord

    return rid


def get_multicast_cores(base_coord: Coord, rid: RId) -> Set[Coord]:
    cores: Set[Coord] = set()

    # countx = 0
    # county = 0

    corex = set()
    corex.add(base_coord.x)
    corey = set()
    corey.add(base_coord.y)

    for lx in range(5):
        if lx_need_copy(rid.rflags[0], lx):
            temp = set()
            for x in corex:
                # countx += 1
                temp.add(x ^ (1 << lx))

            corex = corex.union(temp)

        if lx_need_copy(rid.rflags[1], lx):
            temp = set()
            for y in corey:
                # county += 1
                temp.add(y ^ (1 << lx))

            corey = corey.union(temp)

    for x in corex:
        for y in corey:
            cores.add(Coord(x, y))

    return cores


# def get_router_road(cur_coord: Coord, dest_coord: Coord, rid: RId) -> RouterRoad:
#     """
#     TODO
#     """
#     road = []

#     max_level = max(dest_coord.router_level, rid.router_level)

#     cur_level = cur_coord.router_level
#     while cur_level != max_level:
#         if cur_level < max_level:
#             # Go up
#             road.append(RoutingOP.UP)
#             cur_level += 1
#         elif cur_level > max_level:
#             road.append(RoutingOP.DOWN_MULTICAST)
#         else:
#             pass

#     return road


def coord2level(rid: Coord) -> RoutingNodeLevel:
    x_high = y_high = RoutingNodeLevel.L1

    for level in RoutingNodeLevel:
        if (rid.x >> level.value) == 0:
            x_high = level
            break

    for level in RoutingNodeLevel:
        if (rid.y >> level.value) == 0:
            y_high = level
            break

    return max(x_high, y_high, key=lambda x: x.value)


RoutingDirectionIdx = (
    RoutingDirection.X0Y0,
    RoutingDirection.X0Y1,
    RoutingDirection.X1Y0,
    RoutingDirection.X1Y1,
)


@dataclass
class RoutingNodeCoord:
    """Use router directions to represent the coordinate of a node."""

    L4: RoutingDirection = field(default=RoutingDirection.ANY)
    L3: RoutingDirection = field(default=RoutingDirection.ANY)
    L2: RoutingDirection = field(default=RoutingDirection.ANY)
    L1: RoutingDirection = field(default=RoutingDirection.ANY)
    L0: RoutingDirection = field(default=RoutingDirection.ANY)

    level_table = [
        (L4, RoutingNodeLevel.L5),
        (L3, RoutingNodeLevel.L4),
        (L2, RoutingNodeLevel.L3),
        (L1, RoutingNodeLevel.L2),
        (L0, RoutingNodeLevel.L1),
    ]

    @classmethod
    def build_from_path(cls, path: List[RoutingDirection]):
        if len(path) > 5:
            # TODO
            raise ValueError

        return cls(*path)

    @property
    def level(self) -> RoutingNodeLevel:
        for level in self.level_table:
            if level[0].value == RoutingDirection.ANY:
                return level[1]

        return RoutingNodeLevel.L0

    @property
    def coordinate(self) -> Coord:
        x = (
            (self.L4.value[0] << 4)
            + (self.L3.value[0] << 3)
            + (self.L2.value[0] << 2)
            + (self.L1.value[0] << 1)
            + self.L0.value[0]
        )

        y = (
            (self.L4.value[1] << 4)
            + (self.L3.value[1] << 3)
            + (self.L2.value[1] << 2)
            + (self.L1.value[1] << 1)
            + self.L0.value[1]
        )

        return Coord(x, y)
