from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset

from .route_solver import G_X_MAX, G_X_MIN, G_Y_MAX, G_Y_MIN

RouteSide = tuple[Literal["x", "y", "xy", "local"], int]


def route_len(offset: CoordZXYOffset) -> int:
    return abs(offset.z) + abs(offset.x) + abs(offset.y)


def terminal_route_side(offset: CoordZXYOffset) -> RouteSide:
    """Return the terminal route side after Z/XY, then X, then Y routing."""
    if offset.y != 0:
        return ("y", 1 if offset.y > 0 else -1)
    if offset.x != 0:
        return ("x", 1 if offset.x > 0 else -1)
    if offset.z != 0:
        return ("xy", 1 if offset.z > 0 else -1)
    return ("local", 0)


def route_coord_path(
    start_coord: CoordXY, offset: CoordZXYOffset
) -> tuple[CoordXY, ...]:
    """Return the coordinate path produced by backendv2 Z, X, then Y routing."""
    x, y = start_coord.x, start_coord.y
    points = [start_coord]

    def walk(step_count: int, dx: int, dy: int) -> None:
        nonlocal x, y
        for _ in range(step_count):
            x += dx
            y += dy
            points.append(CoordXY(x, y))

    if offset.z != 0:
        walk(abs(offset.z), 1 if offset.z > 0 else -1, 1 if offset.z > 0 else -1)
    if offset.x != 0:
        walk(abs(offset.x), 1 if offset.x > 0 else -1, 0)
    if offset.y != 0:
        walk(abs(offset.y), 0, 1 if offset.y > 0 else -1)

    return tuple(points)


def route_stays_in_grid(
    start_coord: CoordXY, offset: CoordZXYOffset, target_coord: CoordXY
) -> bool:
    path = route_coord_path(start_coord, offset)
    return path[-1] == target_coord and all(
        G_X_MIN <= coord.x <= G_X_MAX and G_Y_MIN <= coord.y <= G_Y_MAX
        for coord in path
    )


def candidate_offsets(
    start_coord: CoordXY, target_coord: CoordXY
) -> tuple[CoordZXYOffset, ...]:
    """Enumerate legal offsets to one fixed target without changing the target."""
    dcoord = target_coord - start_coord
    offsets: list[CoordZXYOffset] = []
    for z in range(-31, 32):
        x = dcoord.x - z
        y = dcoord.y - z
        if not (-31 <= x <= 31 and -31 <= y <= 31):
            continue
        offset = CoordZXYOffset(z, x, y)
        if route_stays_in_grid(start_coord, offset, target_coord):
            offsets.append(offset)

    return tuple(
        sorted(offsets, key=lambda offset: (route_len(offset), offset.to_tuple()))
    )
