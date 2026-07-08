"""Fixed backendv2 board route scopes.

The route solver and completion planners consume a :class:`RouteScope` instead
of hard-coded single-chip constants. Board descriptions stay internal because
the supported hardware layouts are fixed product targets, not user-provided
placement inputs.
"""

from dataclasses import dataclass
from typing import Literal

from paicorelib import CoordXY

__all__ = ["CpuEndpoint", "RouteScope", "TargetBoard", "get_route_scope"]

TargetBoard = Literal["single", "array_2x2"]

CHIP_ROUTE_SIZE = 9
CHIP_ARRAY_STRIDE = (9, 9)

_SINGLE_CPU = CoordXY(0, 0)
_SINGLE_OFFLINE = frozenset(
    CoordXY(x, y) for x in range(CHIP_ROUTE_SIZE) for y in range(2, CHIP_ROUTE_SIZE)
)
_SINGLE_GLOBAL = frozenset(
    CoordXY(x, y) for x in range(CHIP_ROUTE_SIZE) for y in range(CHIP_ROUTE_SIZE)
)
_SINGLE_ONLINE = frozenset(
    CoordXY(x, y)
    for x in range(CHIP_ROUTE_SIZE)
    for y in range(2)
    if CoordXY(x, y) != _SINGLE_CPU
)


@dataclass(frozen=True, slots=True)
class CpuEndpoint:
    """A CPU tile that can be selected as a board endpoint."""

    chip_id: int
    chip_x: int
    chip_y: int
    coord: CoordXY


@dataclass(frozen=True, slots=True)
class RouteScope:
    """Resolved routing resources for one fixed backendv2 board target."""

    name: TargetBoard
    cpu_endpoints: tuple[CpuEndpoint, ...]
    default_cpu: CpuEndpoint
    offline_core_coords: frozenset[CoordXY]
    online_core_coords: frozenset[CoordXY]
    global_route_coords: frozenset[CoordXY]
    copy_limits: tuple[int, int, int]

    @property
    def cpu_coords(self) -> frozenset[CoordXY]:
        return frozenset(cpu.coord for cpu in self.cpu_endpoints)

    @property
    def configurable_coords(self) -> frozenset[CoordXY]:
        return self.offline_core_coords | self.online_core_coords

    @property
    def offline_bounds(self) -> tuple[int, int, int, int]:
        return _bounds(self.offline_core_coords)

    @property
    def global_bounds(self) -> tuple[int, int, int, int]:
        return _bounds(self.global_route_coords | self.cpu_coords)

    def is_global_route_coord(self, coord: CoordXY) -> bool:
        return coord in self.global_route_coords

    def is_online_core_coord(self, coord: CoordXY) -> bool:
        return coord in self.online_core_coords

    def is_configurable_thread_core_coord(self, coord: CoordXY) -> bool:
        return coord in self.configurable_coords

    def chip_id(self, chip_x: int, chip_y: int) -> int:
        """Return the row-major chip id for a chip coordinate.

        Args:
            chip_x: Chip column in the selected board target.
            chip_y: Chip row in the selected board target.

        Returns:
            Stable row-major chip identifier.

        Raises:
            ValueError: If the chip coordinate is not present in this scope.
        """
        for cpu in self.cpu_endpoints:
            if cpu.chip_x == chip_x and cpu.chip_y == chip_y:
                return cpu.chip_id
        raise ValueError(
            f"Chip coordinate ({chip_x}, {chip_y}) is not in target_board={self.name!r}."
        )

    def chip_xy(self, chip_id: int) -> tuple[int, int]:
        """Return the chip coordinate for a stable chip id.

        Args:
            chip_id: Stable row-major chip identifier.

        Returns:
            ``(chip_x, chip_y)`` coordinate in the selected board target.

        Raises:
            ValueError: If the chip id is not present in this scope.
        """
        for cpu in self.cpu_endpoints:
            if cpu.chip_id == chip_id:
                return cpu.chip_x, cpu.chip_y
        raise ValueError(f"Chip id {chip_id} is not in target_board={self.name!r}.")

    def route_path_valid(self, path: tuple[CoordXY, ...], target: CoordXY) -> bool:
        """Return whether a Z/X/Y route path stays inside this scope.

        CPU tiles are endpoints, not pass-through coordinates. The final target
        may be a CPU coordinate; every other coordinate must be in
        ``global_route_coords``.

        Args:
            path: Absolute coordinate path produced by route-offset expansion.
            target: Required final coordinate of the path.

        Returns:
            Whether the path ends at `target` and stays within the selected
            route scope.
        """
        if not path or path[-1] != target:
            return False
        for idx, coord in enumerate(path):
            if coord in self.global_route_coords:
                continue
            if idx == len(path) - 1 and coord == target and coord in self.cpu_coords:
                continue
            return False
        return True


def get_route_scope(target_board: TargetBoard = "single") -> RouteScope:
    """Return the internal route scope for a supported fixed board target.

    Args:
        target_board: Fixed board topology to resolve. Defaults to ``"single"``.

    Returns:
        Immutable `RouteScope` for the selected board target.
    """
    if target_board not in _ROUTE_SCOPES:
        supported = ", ".join(sorted(_ROUTE_SCOPES))
        raise ValueError(
            f"Unsupported target_board {target_board!r}; expected one of: {supported}."
        )
    return _ROUTE_SCOPES[target_board]


def _bounds(coords: frozenset[CoordXY]) -> tuple[int, int, int, int]:
    if not coords:
        raise ValueError("Route scope coordinate set cannot be empty.")
    xs = [coord.x for coord in coords]
    ys = [coord.y for coord in coords]
    return min(xs), max(xs), min(ys), max(ys)


def _shift(coords: frozenset[CoordXY], dx: int, dy: int) -> frozenset[CoordXY]:
    return frozenset(CoordXY(coord.x + dx, coord.y + dy) for coord in coords)


def _single_scope() -> RouteScope:
    cpu = CpuEndpoint(0, 0, 0, _SINGLE_CPU)
    return RouteScope(
        "single",
        (cpu,),
        cpu,
        _SINGLE_OFFLINE,
        _SINGLE_ONLINE,
        _SINGLE_GLOBAL - frozenset((cpu.coord,)),
        (6, 8, 6),
    )


def _array_2x2_scope() -> RouteScope:
    offline: set[CoordXY] = set()
    online: set[CoordXY] = set()
    global_route: set[CoordXY] = set()
    cpus: list[CpuEndpoint] = []

    stride_x, stride_y = CHIP_ARRAY_STRIDE
    for chip_y in range(2):
        for chip_x in range(2):
            chip_id = chip_y * 2 + chip_x
            dx = chip_x * stride_x
            dy = chip_y * stride_y
            cpu = CpuEndpoint(chip_id, chip_x, chip_y, CoordXY(dx, dy))
            cpus.append(cpu)
            offline.update(_shift(_SINGLE_OFFLINE, dx, dy))
            online.update(_shift(_SINGLE_ONLINE, dx, dy))
            global_route.update(_shift(_SINGLE_GLOBAL, dx, dy))

    cpu_coords = {cpu.coord for cpu in cpus}
    return RouteScope(
        "array_2x2",
        tuple(cpus),
        cpus[0],
        frozenset(offline),
        frozenset(online),
        frozenset(global_route - cpu_coords),
        (6, 17, 6),
    )


_ROUTE_SCOPES: dict[TargetBoard, RouteScope] = {
    "single": _single_scope(),
    "array_2x2": _array_2x2_scope(),
}
