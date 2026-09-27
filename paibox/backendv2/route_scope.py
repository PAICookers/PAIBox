"""Fixed backendv2 board route scopes.

The route solver and completion planners consume a :class:`RouteScope` instead
of hard-coded single-chip constants. Board descriptions stay internal because
the supported hardware layouts are fixed product targets, not user-provided
placement inputs.
"""

from collections.abc import Iterator
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

from .board import (
    BoardCoreAddr,
    BoardName,
    BoardProfile,
    ChipCoord,
    TargetBoard,
    get_board_profile,
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

RoutePath = tuple[CoordXY, ...]
LocalCoords = tuple[CoordXY, ...]
PacketAuditKey = tuple[str, int, int, int, int, int, int, int, int]
TargetExpansionKey = tuple[int, int, int, int, int]
FailureCode = Literal[
    "path_out_of_scope",
    "path_target_mismatch",
    "missing_data_edge",
    "cross_chip_xy",
    "expected_target_outside_offline",
    "online_local_delivery",
    "local_target_set_mismatch",
]
FailureStage = Literal["path", "expected_local", "actual_local", "target_set"]


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

    cpu_endpoints: tuple[CpuEndpoint, ...]
    default_cpu: CpuEndpoint
    offline_core_coords: frozenset[CoordXY]
    online_core_coords: frozenset[CoordXY]
    global_route_coords: frozenset[CoordXY]
    board: BoardProfile

    @property
    def name(self) -> str:
        return self.board.name.value

    @property
    def copy_limits(self) -> tuple[int, int, int]:
        """Maximum absolute ``(Z, X, Y)`` AER copy counts for this board."""
        return self.board.copy_limits

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

        CPU coordinates remain ordinary NoC transit coordinates. A CPU
        coordinate is a terminal only when it is the final target.

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
            if coord not in self.global_route_coords:
                return False
            if idx and self._step_failure(path[idx - 1], coord) is not None:
                return False
        return target in self.global_route_coords and (
            target not in self.cpu_coords or target == self.default_cpu.coord
        )

    def audit_aer_packet(
        self, source: CoordXYLike, offset: CoordZXYOffset, copy_config: AERPacketZXYCopy
    ) -> RouteAuditResult:
        """Validate route geometry and exact local delivery of one DATA packet.

        The PAICORE router consumes each core offset dimension before the next
        one and handles the matching copy dimension immediately after it. The
        PAIlib primitives are the source of truth for this ordered simulation;
        this method adds board policy: CPU coordinates may be NoC transit
        positions, while normal DATA may be delivered only to offline cores.

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

    def data_edge_claims(
        self,
        source: CoordXYLike,
        offset: CoordZXYOffset,
        copy_config: AERPacketZXYCopy,
    ) -> frozenset[tuple[ChipCoord, ChipCoord]]:
        """Return directed physical DATA edges traversed by one packet."""
        source = to_coordxy(source)
        path = route_coord_path(source, offset)
        claims: set[tuple[ChipCoord, ChipCoord]] = set()

        def add_step(left: CoordXY, right: CoordXY) -> None:
            failure = self._step_failure(left, right)
            if failure is not None:
                raise ValueError(failure.message)
            left_chip, right_chip = (
                self.chip_for_coord(left),
                self.chip_for_coord(right),
            )
            if left_chip != right_chip:
                claims.add((left_chip, right_chip))

        for left, right in zip(path, path[1:]):
            add_step(left, right)
        for left, right in _iter_copy_steps(path[-1], copy_config):
            add_step(left, right)
        return frozenset(claims)

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
            failure = self._copy_path_failure(target, copy_config)
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

    def _copy_path_failure(
        self, base: CoordXY, copy_config: AERPacketZXYCopy
    ) -> RouteAuditFailure | None:
        """Audit every multicast branch, including cross-chip copy moves."""
        for left, right in _iter_copy_steps(base, copy_config):
            failure = self._step_failure(left, right)
            if failure is not None:
                return failure
        return None

    def _packet_path_failure(
        self, path: RoutePath, target: CoordXY
    ) -> RouteAuditFailure | None:
        """Return the first board-policy violation in a packet's core path."""
        for idx, coord in enumerate(path):
            if coord not in self.global_route_coords:
                return RouteAuditFailure(
                    "path_out_of_scope",
                    "path",
                    coord,
                    f"route coordinate {coord} is outside board scope",
                )
            if idx:
                failure = self._step_failure(path[idx - 1], coord)
                if failure is not None:
                    return failure
        if path[-1] != target:
            return RouteAuditFailure(
                "path_target_mismatch",
                "path",
                path[-1],
                f"route ended at {path[-1]}, expected {target}",
            )
        if target in self.cpu_coords and target != self.default_cpu.coord:
            return RouteAuditFailure(
                "path_out_of_scope",
                "path",
                target,
                "only the control CPU may be a final CPU destination",
            )
        return None

    def _step_failure(
        self, source: CoordXY, target: CoordXY
    ) -> RouteAuditFailure | None:
        dx, dy = target.x - source.x, target.y - source.y
        src_chip = self.chip_for_coord(source)
        dst_chip = self.chip_for_coord(target)
        if abs(dx) == 1 and abs(dy) == 1:
            if src_chip != dst_chip:
                return RouteAuditFailure(
                    "cross_chip_xy",
                    "path",
                    target,
                    "XY route cannot cross a chip boundary",
                )
            return None
        if abs(dx) + abs(dy) != 1:
            return RouteAuditFailure(
                "path_out_of_scope",
                "path",
                target,
                f"invalid adjacent route step {source}->{target}",
            )
        if src_chip != dst_chip and not self.board.has_edge(src_chip, dst_chip):
            return RouteAuditFailure(
                "missing_data_edge",
                "path",
                target,
                f"DATA route crosses undeclared edge {src_chip.xy}->{dst_chip.xy}",
            )
        return None

    def chip_for_coord(self, coord: CoordXY) -> ChipCoord:
        min_x, max_x, min_y, max_y = self.global_bounds
        if not min_x <= coord.x <= max_x or not min_y <= coord.y <= max_y:
            raise ValueError(f"coordinate {coord} is outside board bounds")
        chip = ChipCoord(
            coord.x // self.board.core_grid_width,
            coord.y // self.board.core_grid_height,
        )
        return self.board.chip_at(chip.x, chip.y)


def get_route_scope(target_board: TargetBoard = "single") -> RouteScope:
    """Return the internal route scope for a supported fixed board target.

    Args:
        target_board: Fixed board topology to resolve. Defaults to ``"single"``.

    Returns:
        Immutable `RouteScope` for the selected board target.
    """
    return _scope_from_profile(get_board_profile(target_board).name)


def _bounds(coords: frozenset[CoordXY]) -> tuple[int, int, int, int]:
    if not coords:
        raise ValueError("Route scope coordinate set cannot be empty.")
    xs = [coord.x for coord in coords]
    ys = [coord.y for coord in coords]
    return min(xs), max(xs), min(ys), max(ys)


def _shift(coords: frozenset[CoordXY], dx: int, dy: int) -> frozenset[CoordXY]:
    return frozenset(CoordXY(coord.x + dx, coord.y + dy) for coord in coords)


def _iter_copy_steps(
    base: CoordXY, copy_config: AERPacketZXYCopy
) -> Iterator[tuple[CoordXY, CoordXY]]:
    """Yield every multicast branch step in PAICORE copy order."""
    queue = [(base.x, base.y, copy_config.z, copy_config.x, copy_config.y)]
    index = 0
    while index < len(queue):
        x, y, z_copy, x_copy, y_copy = queue[index]
        index += 1
        if z_copy > 0:
            next_state = (x + 1, y + 1, z_copy - 1, x_copy, y_copy)
        elif z_copy < 0:
            next_state = (x - 1, y - 1, z_copy + 1, x_copy, y_copy)
        elif x_copy > 0:
            next_state = (x + 1, y, z_copy, x_copy - 1, y_copy)
        elif x_copy < 0:
            next_state = (x - 1, y, z_copy, x_copy + 1, y_copy)
        elif y_copy > 0:
            next_state = (x, y + 1, z_copy, x_copy, y_copy - 1)
        elif y_copy < 0:
            next_state = (x, y - 1, z_copy, x_copy, y_copy + 1)
        else:
            continue
        queue.append(next_state)
        yield CoordXY(x, y), CoordXY(next_state[0], next_state[1])


@lru_cache(maxsize=None)
def _scope_from_profile(name: BoardName) -> RouteScope:
    """Derive compiler coordinate pools from one authoritative board profile."""
    profile = get_board_profile(name)
    offline: set[CoordXY] = set()
    online: set[CoordXY] = set()
    global_route: set[CoordXY] = set()
    cpus = tuple(_cpu_endpoint(profile, endpoint) for endpoint in profile.cpu_ports)

    stride_x, stride_y = profile.core_grid_size
    local_offline, local_online, local_global = _local_core_sets(profile)
    for chip in profile.chips:
        # Board coordinates are local 9x9 coordinates translated by chip xy.
        dx = chip.x * stride_x
        dy = chip.y * stride_y
        offline.update(_shift(local_offline, dx, dy))
        online.update(_shift(local_online, dx, dy))
        global_route.update(_shift(local_global, dx, dy))
    # CPU tiles remain in global_route for NoC transit, but are not resources.
    offline.difference_update(cpu.coord for cpu in cpus)
    online.difference_update(cpu.coord for cpu in cpus)

    return RouteScope(
        cpus,
        _find_control_cpu(profile, cpus),
        frozenset(offline),
        frozenset(online),
        frozenset(global_route),
        profile,
    )


def _cpu_endpoint(profile: BoardProfile, address: BoardCoreAddr) -> CpuEndpoint:
    """Convert one profile CPU port into its global NoC route coordinate."""
    return CpuEndpoint(
        profile.chips.index(address.chip),
        address.chip.x,
        address.chip.y,
        CoordXY(
            address.chip.x * profile.core_grid_width + address.core.x,
            address.chip.y * profile.core_grid_height + address.core.y,
        ),
    )


def _local_core_sets(
    profile: BoardProfile,
) -> tuple[frozenset[CoordXY], frozenset[CoordXY], frozenset[CoordXY]]:
    """Build local offline, online, and route coordinates from board geometry."""
    width, height = profile.core_grid_size
    offline = frozenset(
        CoordXY(x, y)
        for x in range(width)
        for y in range(profile.online_row_count, height)
    )
    global_route = frozenset(
        CoordXY(x, y) for x in range(width) for y in range(height)
    )
    online = frozenset(
        CoordXY(x, y)
        for x in range(width)
        for y in range(profile.online_row_count)
    )
    return offline, online, global_route


def _find_control_cpu(
    profile: BoardProfile, cpus: tuple[CpuEndpoint, ...]
) -> CpuEndpoint:
    control = _cpu_endpoint(profile, profile.control_cpu)
    return next(cpu for cpu in cpus if cpu.coord == control.coord)
