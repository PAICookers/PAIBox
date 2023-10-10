from dataclasses import dataclass, field
from enum import Enum, IntEnum, unique
from typing import List, Sequence, Set

from ._types import ReplicationFlag as RFlag
from .coordinate import Coord, ReplicationId as RId


__all__ = ["RouterCoordinate"]


@unique
class RouterOp(Enum):
    UP = 0
    DOWN_UNICAST = 1
    DOWN_MULTICAST = 2


@unique
class RouterLevel(IntEnum):
    L0 = 0  # Leaves for storing the data. A L0-layer is a core.
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


@unique
class RouterDirection(Enum):
    """Indicate the 4 children of a node.

    NOTE: There is an X/Y coordinate priority method \
        to specify the order of the 4 children.
    """

    X0Y0 = (0, 0)
    X0Y1 = (0, 1)
    X1Y0 = (1, 0)
    X1Y1 = (1, 1)
    ANY = (-1, -1)  # Don't care when a level direction is `ANY`.

    def to_index(self) -> int:
        """Convert the direction to index in children list."""
        if self is RouterDirection.ANY:
            # TODO
            raise ValueError

        x, y = self.value

        return (x << 1) + y


@unique
class RouterNodeStatus(Enum):
    """Indicate the status of a Lx-level(Lx > L0) node."""

    AVAILABLE = 0
    """There is at least 1 child node which is `AVAILABLE`."""
    OCCUPIED = 1
    """Full."""

    AVAILABLE_BUT_WASTED = 2
    """Wasted because it's a destination for broadcasting."""


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
#             road.append(RouterOp.UP)
#             cur_level += 1
#         elif cur_level > max_level:
#             road.append(RouterOp.DOWN_MULTICAST)
#         else:
#             pass

#     return road


def coord2level(rid: Coord) -> RouterLevel:
    x_high = y_high = RouterLevel.L1

    for level in RouterLevel:
        if (rid.x >> level.value) == 0:
            x_high = level
            break

    for level in RouterLevel:
        if (rid.y >> level.value) == 0:
            y_high = level
            break

    return max(x_high, y_high, key=lambda x: x.value)


RouterDirectionIdx = (
    RouterDirection.X0Y0,
    RouterDirection.X0Y1,
    RouterDirection.X1Y0,
    RouterDirection.X1Y1,
)


@dataclass
class RouterCoordinate:
    """Use router directions to represent the coordinate of a node."""

    L4: RouterDirection = field(default=RouterDirection.ANY)
    L3: RouterDirection = field(default=RouterDirection.ANY)
    L2: RouterDirection = field(default=RouterDirection.ANY)
    L1: RouterDirection = field(default=RouterDirection.ANY)
    L0: RouterDirection = field(default=RouterDirection.ANY)

    level_table = [
        (L4, RouterLevel.L5),
        (L3, RouterLevel.L4),
        (L2, RouterLevel.L3),
        (L1, RouterLevel.L2),
        (L0, RouterLevel.L1),
    ]

    @classmethod
    def build_from_path(cls, path: List[RouterDirection]):
        if len(path) > 5:
            # TODO
            raise ValueError

        return cls(*path)

    @property
    def level(self) -> RouterLevel:
        for level in self.level_table:
            if level[0].value == RouterDirection.ANY:
                return level[1]

        return RouterLevel.L0

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
