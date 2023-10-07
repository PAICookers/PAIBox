from dataclasses import dataclass
from typing import List, Literal, Sequence, Set

from ._types import ReplicationFlag as RFlag, RouterDirection, RouterLevel, RouterOp
from .coordinate import Coord, ReplicationId as RId

RouterRoad = List[RouterOp]
RouterStatus = List[RouterLevel]


__all__ = ["RouterCoordinate"]


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


def get_router_road(cur_coord: Coord, dest_coord: Coord, rid: RId) -> RouterRoad:
    """
    TODO
    """
    road = []

    max_level = max(dest_coord.router_level, rid.router_level)

    cur_level = cur_coord.router_level
    while cur_level != max_level:
        if cur_level < max_level:
            # Go up
            road.append(RouterOp.UP)
            cur_level += 1
        elif cur_level > max_level:
            road.append(RouterOp.DOWN_MULTICAST)
        else:
            pass

    return road


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


def idx2router_direction(idx: int, method: Literal["X", "Y"] = "Y") -> RouterDirection:
    """Get the router direction given an index of the node.
    
    Args:
        - idx: the index of the node for its parent.
        - method: use X/Y-priority method.
    """
    if idx > 3:
        raise ValueError

    if method == "Y":
        directions = (
            RouterDirection.X0Y0,
            RouterDirection.X0Y1,
            RouterDirection.X1Y0,
            RouterDirection.X1Y1,
        )
    else:
        directions = (
            RouterDirection.X0Y0,
            RouterDirection.X1Y0,
            RouterDirection.X0Y1,
            RouterDirection.X1Y1,
        )

    return directions[idx]


@dataclass
class RouterCoordinate:
    L4: RouterDirection
    L3: RouterDirection
    L2: RouterDirection
    L1: RouterDirection
    L0: RouterDirection
    
    @classmethod
    def build_from_road(cls, road: List[RouterDirection]):
        if len(road) != 5:
            # TODO
            raise ValueError
        
        return cls(*road)

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
