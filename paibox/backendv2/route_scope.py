"""Fixed backendv2 board route scopes.

The route solver and completion planners consume a :class:`RouteScope` instead
of hard-coded single-chip constants. Board descriptions stay internal because
the supported hardware layouts are fixed product targets, not user-provided
placement inputs.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from paicorelib import (
    AERPacket,
    AERPacketZXYCopy,
    CoordXY,
    CoordXYLike,
    CoordZXYOffset,
    aer_packet_walk,
    route_coord_path,
    to_coordxy,
)

__all__ = [
    "CpuEndpoint",
    "RouteAuditFailure",
    "RouteAuditResult",
    "RouteScope",
    "TargetBoard",
    "expand_packet_targets",
    "get_route_scope",
]

TargetBoard = Literal["single", "array_2x2"]
RoutePath = tuple[CoordXY, ...]
LocalCoords = tuple[CoordXY, ...]
PacketAuditKey = tuple[TargetBoard, int, int, int, int, int, int, int, int]
TargetExpansionKey = tuple[int, int, int, int, int]
FailureCode = Literal[
    "cpu_transit",
    "path_out_of_scope",
    "path_target_mismatch",
    "expected_target_outside_offline",
    "online_local_delivery",
    "local_target_set_mismatch",
]
FailureStage = Literal["path", "expected_local", "actual_local", "target_set"]

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
class RouteAuditFailure:
    """First failure found while auditing one complete AER packet."""

    code: FailureCode
    stage: FailureStage
    coord: CoordXY | None
    message: str


@dataclass(frozen=True, slots=True)
class RouteAuditResult:
    """Result of simulating one complete DATA AER packet.

    Attributes:
        path: Coordinates visited by the non-multicast Z/X/Y route.
        actual_local: Coordinates that receive the packet locally according to
            the hardware multicast expansion.
        expected_local: Coordinates implied by the destination base and
            copy fields.
        failure: First structured reason when the packet is not legal.
    """

    path: RoutePath
    actual_local: LocalCoords
    expected_local: LocalCoords
    failure: RouteAuditFailure | None = None

    @property
    def valid(self) -> bool:
        """Return whether the complete packet matches its offline targets."""
        return self.failure is None


@lru_cache(maxsize=4096)
def _expand_packet_targets_cached(key: TargetExpansionKey) -> LocalCoords:
    """Expand copy-only states without constructing a complete AER packet."""
    base_x, base_y, copy_z, copy_x, copy_y = key
    targets: list[CoordXY] = []
    seen: set[tuple[int, int]] = set()

    def add(x: int, y: int) -> None:
        point = (x, y)
        if point not in seen:
            seen.add(point)
            targets.append(CoordXY(x, y))

    # This is the PAIlib copy state machine with packet offsets removed.  The
    # queue is required because every multicast branch continues with its own
    # remaining copy counts; flattening the dimensions would change the
    # hardware's reachable target set for mixed-sign copies.
    queue = [(base_x, base_y, copy_z, copy_x, copy_y)]
    queue_index = 0
    while queue_index < len(queue):
        x, y, z_copy, x_copy, y_copy = queue[queue_index]
        queue_index += 1
        while True:
            if z_copy > 0:
                z_copy -= 1
                add(x, y)
                queue.append((x + 1, y + 1, z_copy, x_copy, y_copy))
            elif z_copy < 0:
                z_copy += 1
                add(x, y)
                queue.append((x - 1, y - 1, z_copy, x_copy, y_copy))
            elif x_copy > 0:
                x_copy -= 1
                add(x, y)
                queue.append((x + 1, y, z_copy, x_copy, y_copy))
            elif x_copy < 0:
                x_copy += 1
                add(x, y)
                queue.append((x - 1, y, z_copy, x_copy, y_copy))
            elif y_copy > 0:
                y_copy -= 1
                add(x, y)
                queue.append((x, y + 1, z_copy, x_copy, y_copy))
            elif y_copy < 0:
                y_copy += 1
                add(x, y)
                queue.append((x, y - 1, z_copy, x_copy, y_copy))
            else:
                add(x, y)
                break

    return tuple(targets)


def expand_packet_targets(
    base: CoordXYLike, copy_config: AERPacketZXYCopy
) -> LocalCoords:
    """Return deterministic local target coordinates for a copy configuration.

    This target-only helper deliberately does not call ``aer_packet_walk``.
    Use :meth:`RouteScope.audit_aer_packet` when the complete hardware route,
    including transit and actual local footholds, must be validated.
    """
    point = to_coordxy(base)
    return _expand_packet_targets_cached(
        (point.x, point.y, copy_config.z, copy_config.x, copy_config.y)
    )


@lru_cache(maxsize=8192)
def _audit_packet_cached(key: PacketAuditKey) -> RouteAuditResult:
    """Evaluate a normalized packet once and reuse its immutable result."""
    (
        scope_name,
        source_x,
        source_y,
        offset_z,
        offset_x,
        offset_y,
        copy_z,
        copy_x,
        copy_y,
    ) = key
    scope = get_route_scope(scope_name)
    return scope._audit_aer_packet(
        CoordXY(source_x, source_y),
        CoordZXYOffset(offset_z, offset_x, offset_y),
        AERPacketZXYCopy(copy_z, copy_x, copy_y),
    )


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

    def audit_aer_packet(
        self, source: CoordXYLike, offset: CoordZXYOffset, copy_config: AERPacketZXYCopy
    ) -> RouteAuditResult:
        """Validate route geometry and exact local delivery of one DATA packet.

        The PAICORE router consumes each core offset dimension before the next
        one and handles the matching copy dimension immediately after it. The
        PAIlib primitives are the source of truth for this ordered simulation;
        this method adds board policy: CPU coordinates are endpoints only, and
        normal DATA may be delivered only to offline cores.

        Args:
            source: Absolute source core or CPU coordinate.
            offset: Complete packet core offset in Z/X/Y order.
            copy_config: Complete packet copy count in Z/X/Y order.

        Returns:
            An immutable audit containing the physical path, actual local
            footholds, expected target footholds, and optional failure details.
        """
        source = to_coordxy(source)
        key: PacketAuditKey = (
            self.name,
            source.x,
            source.y,
            offset.z,
            offset.x,
            offset.y,
            copy_config.z,
            copy_config.x,
            copy_config.y,
        )
        return _audit_packet_cached(key)

    def _audit_aer_packet(
        self, source: CoordXY, offset: CoordZXYOffset, copy_config: AERPacketZXYCopy
    ) -> RouteAuditResult:
        """Run the uncached hardware simulation for a normalized packet."""
        path = route_coord_path(source, offset)
        target = path[-1]
        expected = expand_packet_targets(target, copy_config)
        actual_local = tuple(aer_packet_walk(AERPacket(source, offset, copy_config)))

        failure = self._packet_path_failure(path, target)
        if failure is None:
            invalid_expected = [
                coord
                for coord in expected
                if coord not in self.offline_core_coords and coord != target
            ]
            if invalid_expected:
                failure = RouteAuditFailure(
                    "expected_target_outside_offline",
                    "expected_local",
                    invalid_expected[0],
                    "expected local targets leave offline cores: "
                    f"{invalid_expected[0]}",
                )

        if failure is None:
            unexpected_online = [
                coord for coord in actual_local if coord in self.online_core_coords
            ]
            if unexpected_online:
                failure = RouteAuditFailure(
                    "online_local_delivery",
                    "actual_local",
                    unexpected_online[0],
                    "DATA packet performs TO_LOCAL on online core "
                    f"{unexpected_online[0]}",
                )

        if failure is None and set(actual_local) != set(expected):
            missing = sorted(
                set(expected) - set(actual_local), key=lambda c: (c.x, c.y)
            )
            unexpected = sorted(
                set(actual_local) - set(expected), key=lambda c: (c.x, c.y)
            )
            coord = missing[0] if missing else (unexpected[0] if unexpected else None)
            failure = RouteAuditFailure(
                "local_target_set_mismatch",
                "target_set",
                coord,
                "local delivery set does not match expected targets "
                f"(missing={missing[:1]}, unexpected={unexpected[:1]})",
            )

        return RouteAuditResult(path, actual_local, expected, failure)

    def _packet_path_failure(
        self, path: RoutePath, target: CoordXY
    ) -> RouteAuditFailure | None:
        """Return the first board-policy violation in a packet's core path."""
        for idx, coord in enumerate(path):
            if coord in self.global_route_coords:
                continue
            if idx == 0 and coord in self.cpu_coords:
                continue
            if idx == len(path) - 1 and coord in self.cpu_coords:
                continue
            if coord in self.cpu_coords:
                return RouteAuditFailure(
                    "cpu_transit",
                    "path",
                    coord,
                    f"CPU endpoint {coord} cannot be a route transit",
                )
            return RouteAuditFailure(
                "path_out_of_scope",
                "path",
                coord,
                f"route coordinate {coord} is outside board scope",
            )
        if path[-1] != target:
            return RouteAuditFailure(
                "path_target_mismatch",
                "path",
                path[-1],
                f"route ended at {path[-1]}, expected {target}",
            )
        return None


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
