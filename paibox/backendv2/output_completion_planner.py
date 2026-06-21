"""Output DATA and completion-frame route planner for backendv2."""

from collections.abc import Set
from dataclasses import dataclass
from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset

from .core_config import TEST_DEST_CORE
from .global_signal import EmptyRelayCoreKind, solve_global_signal_tree
from .route_solver import (
    CPU_COORD,
    G_X_MAX,
    G_X_MIN,
    G_Y_MAX,
    G_Y_MIN,
    ONLINE_CORE_COORDS,
    is_configurable_thread_core_coord,
    is_global_route_coord,
    is_online_core_coord,
)

__all__ = [
    "OutputCompletionPlan",
    "OutputCpuIngressPlan",
    "OutputProducer",
    "select_output_completion_plan",
]

RouteSide = tuple[Literal["x", "y", "xy", "local"], int]
CompletionCoreTier = Literal["used", "empty_offline", "empty_online"]


@dataclass(frozen=True, slots=True)
class OutputRouteDecision:
    """Selected Z/X/Y offset for one output producer's DATA route."""

    producer_coord: CoordXY
    target_coord: CoordXY
    offset: CoordZXYOffset


@dataclass(frozen=True, slots=True)
class OutputCpuIngressPlan:
    """DATA and complete route offsets that share one CPU ingress side."""

    output_routes: tuple[OutputRouteDecision, ...]
    control_offset: CoordZXYOffset
    ingress_side: RouteSide

    def output_route_offsets(self) -> dict[tuple[CoordXY, CoordXY], CoordZXYOffset]:
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }


def route_len(offset: CoordZXYOffset) -> int:
    return abs(offset.z) + abs(offset.x) + abs(offset.y)


def terminal_route_side(offset: CoordZXYOffset) -> RouteSide:
    """Returns the terminal route side after backendv2 Z, X, then Y routing."""
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
    """Returns the coordinate path produced by backendv2 Z, X, then Y routing."""
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
    """Enumerates legal offsets to one fixed target without retargeting CPU."""
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


class OutputCompletionError(RuntimeError):
    """Base error for output DATA / complete-frame planning failures."""


class OutputCompletionNoFeasiblePlanError(OutputCompletionError):
    """Raised when DATA and completion paths cannot share a valid suffix."""


@dataclass(frozen=True, slots=True)
class OutputProducer:
    """Output-producing core endpoint and its static output-entry weight."""

    coord: CoordXY
    target_coord: CoordXY = TEST_DEST_CORE
    weight: int = 1


@dataclass(frozen=True, slots=True)
class EmptyRelayCore:
    """Selected empty shell core added to the current thread configuration."""

    coord: CoordXY
    kind: EmptyRelayCoreKind


@dataclass(frozen=True, slots=True)
class OutputCompletionPlan:
    """Chosen DATA routes and matching complete/control route."""

    output_routes: tuple[OutputRouteDecision, ...]
    control_offset: CoordZXYOffset
    ingress_side: RouteSide
    global_signal_root: CoordXY
    completion_join_point: CoordXY
    required_route_cores: tuple[EmptyRelayCore, ...]
    relay_cores: tuple[EmptyRelayCore, ...]

    def output_route_offsets(self) -> dict[tuple[CoordXY, CoordXY], CoordZXYOffset]:
        """Returns per-output-core DATA offsets keyed by producer and target."""
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }

    def relay_core_kinds(self) -> dict[CoordXY, EmptyRelayCoreKind]:
        """Returns selected empty core kinds for completion and signal relays."""
        relay_core_kinds: dict[CoordXY, EmptyRelayCoreKind] = {}
        for relay in (*self.required_route_cores, *self.relay_cores):
            relay_core_kinds[relay.coord] = relay.kind
        return relay_core_kinds

    def to_cpu_ingress_plan(self) -> OutputCpuIngressPlan:
        """Converts the completion decision to CPU ingress route metadata."""
        return OutputCpuIngressPlan(
            self.output_routes, self.control_offset, self.ingress_side
        )


@dataclass(frozen=True, slots=True)
class _ThreadCorePools:
    configured: frozenset[CoordXY]
    empty_offline: frozenset[CoordXY]
    empty_online: frozenset[CoordXY]


@dataclass(frozen=True, slots=True)
class _PlannerRequest:
    producers: tuple[OutputProducer, ...]
    pools: _ThreadCorePools


@dataclass(frozen=True, slots=True)
class _DataRouteOption:
    producer: OutputProducer
    offset: CoordZXYOffset
    path: tuple[CoordXY, ...]
    path_len: int


@dataclass(frozen=True, slots=True)
class _CommonSuffix:
    join_point: CoordXY
    suffix: tuple[CoordXY, ...]


@dataclass(frozen=True, slots=True)
class _JoinCandidate:
    suffix: _CommonSuffix
    data_routes: tuple[_DataRouteOption, ...]
    required_route_cores: tuple[EmptyRelayCore, ...]
    required_route_core_coords: frozenset[CoordXY]
    data_total_len: int
    join_tier: CompletionCoreTier
    join_to_cpu_path_len: int


@dataclass(frozen=True, slots=True)
class _CompletionCandidate:
    join_candidate: _JoinCandidate
    source: CoordXY
    source_tier: CompletionCoreTier
    control_offset: CoordZXYOffset
    control_path: tuple[CoordXY, ...]
    required_route_cores: tuple[EmptyRelayCore, ...]
    relay_cores: tuple[EmptyRelayCore, ...]
    control_path_len: int
    control_required_route_core_count: int


def _coord_key(coord: CoordXY) -> tuple[int, int]:
    return coord.x, coord.y


def _coords_key(coords: tuple[CoordXY, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(_coord_key(coord) for coord in coords)


def _empty_core_key(core: EmptyRelayCore) -> tuple[int, int, EmptyRelayCoreKind]:
    return core.coord.x, core.coord.y, core.kind


def _sorted_empty_cores(cores: Set[EmptyRelayCore]) -> tuple[EmptyRelayCore, ...]:
    return tuple(sorted(cores, key=_empty_core_key))


def _path_suffix_from(
    path: tuple[CoordXY, ...], join_point: CoordXY
) -> tuple[CoordXY, ...] | None:
    for index, coord in enumerate(path[:-1]):
        if coord == join_point:
            return path[index:]
    return None


def _completion_core_tier(
    coord: CoordXY, pools: _ThreadCorePools
) -> CompletionCoreTier | None:
    if coord in pools.configured:
        return "used"
    if coord in pools.empty_offline:
        return "empty_offline"
    if coord in pools.empty_online:
        return "empty_online"
    return None


def _empty_completion_core(
    coord: CoordXY, pools: _ThreadCorePools
) -> EmptyRelayCore | None:
    if coord in pools.configured:
        return None
    if coord in pools.empty_offline:
        return EmptyRelayCore(coord, "offline")
    if coord in pools.empty_online:
        return EmptyRelayCore(coord, "online")
    return None


def _tier_rank(tier: CompletionCoreTier) -> int:
    if tier == "used":
        return 0
    if tier == "empty_offline":
        return 1
    return 2


def _common_suffixes_for_path(path: tuple[CoordXY, ...]) -> tuple[_CommonSuffix, ...]:
    suffixes: list[_CommonSuffix] = []
    seen: set[_CommonSuffix] = set()
    for index, coord in enumerate(path[:-1]):
        if not is_configurable_thread_core_coord(coord):
            continue
        suffix = _CommonSuffix(coord, path[index:])
        if suffix in seen:
            continue
        seen.add(suffix)
        suffixes.append(suffix)
    return tuple(suffixes)


def _plan_paths(
    plan: OutputCompletionPlan,
) -> tuple[tuple[CoordXY, ...], list[tuple[CoordXY, ...]]]:
    complete_path = route_coord_path(plan.global_signal_root, plan.control_offset)
    data_paths = [
        route_coord_path(route.producer_coord, route.offset)
        for route in plan.output_routes
    ]
    return complete_path, data_paths


class OutputCompletionPlanner:
    """Plans output DATA routes and their matching completion route.

    The planner keeps the safety algorithm staged: enumerate legal DATA routes,
    select the best common join suffix, then select a completion source for that
    join. Public callers should normally use ``select_output_completion_plan``.
    """

    def __init__(
        self,
        producers: list[OutputProducer],
        configured_thread_core_coords: Set[CoordXY],
        available_empty_offline_coords: Set[CoordXY],
        available_empty_online_coords: Set[CoordXY] | None = None,
        allow_empty_online: bool = True,
    ) -> None:
        """Initializes immutable planning inputs.

        Args:
            producers: Output producer cores to route to CPU.
            configured_thread_core_coords: Cores already configured in this
                thread.
            available_empty_offline_coords: Empty offline cores that can be
                added to this thread.
            available_empty_online_coords: Empty online cores that can be added
                to this thread. If omitted and online relays are enabled, all
                unused online-core coordinates are considered available.
            allow_empty_online: Whether empty online shell cores may be added.
        """
        self._raw_producers = tuple(producers)
        self._configured_thread_core_coords = frozenset(configured_thread_core_coords)
        self._available_empty_offline_coords = frozenset(available_empty_offline_coords)
        self._available_empty_online_coords = (
            None
            if available_empty_online_coords is None
            else frozenset(available_empty_online_coords)
        )
        self._allow_empty_online = allow_empty_online
        self._request = self._normalize_request()

    def plan(self) -> OutputCompletionPlan:
        """Builds and validates an output completion plan."""
        if not self._request.producers:
            return self._plan_without_output_producers()

        join_candidates = self._select_join_candidates()
        selected = self._select_completion_candidate(join_candidates)
        if selected is None:
            raise OutputCompletionNoFeasiblePlanError(
                "Cannot find output DATA and completion routes with a shared "
                "thread-local suffix. DATA relay cores are not supported by this "
                "planner. producers="
                f"{[(p.coord.x, p.coord.y, p.weight) for p in self._request.producers]}, "
                f"used={sorted((c.x, c.y) for c in self._request.pools.configured)}"
            )

        plan = self._build_plan(selected)
        validate_output_completion_plan(
            plan,
            self._request.producers,
            self._request.pools.configured,
            self._allow_empty_online,
        )
        return plan

    def _normalize_request(self) -> _PlannerRequest:
        configured_coords = self._configured_thread_core_coords
        if not self._allow_empty_online:
            empty_online_coords: frozenset[CoordXY] = frozenset()
        elif self._available_empty_online_coords is None:
            empty_online_coords = frozenset(ONLINE_CORE_COORDS - configured_coords)
        else:
            empty_online_coords = self._available_empty_online_coords

        producers = tuple(
            OutputProducer(
                producer.coord, producer.target_coord, max(producer.weight, 1)
            )
            for producer in self._raw_producers
        )
        if producers and any(
            producer.target_coord != producers[0].target_coord for producer in producers
        ):
            raise OutputCompletionNoFeasiblePlanError(
                "Output completion currently requires one shared CPU target."
            )

        return _PlannerRequest(
            producers,
            _ThreadCorePools(
                configured_coords,
                self._available_empty_offline_coords,
                empty_online_coords,
            ),
        )

    def _plan_without_output_producers(self) -> OutputCompletionPlan:
        root = min(self._request.pools.configured, key=_coord_key)
        control_offset = candidate_offsets(root, TEST_DEST_CORE)[0]
        return OutputCompletionPlan(
            (), control_offset, terminal_route_side(control_offset), root, root, (), ()
        )

    def _classify_relay_cores(
        self,
        coords: Set[CoordXY],
        configured_thread_core_coords: Set[CoordXY],
    ) -> tuple[EmptyRelayCore, ...] | None:
        relay_cores: set[EmptyRelayCore] = set()
        pools = self._request.pools
        for coord in coords:
            if coord in configured_thread_core_coords:
                continue
            if coord in pools.empty_offline:
                relay_cores.add(EmptyRelayCore(coord, "offline"))
                continue
            if coord in pools.empty_online:
                relay_cores.add(EmptyRelayCore(coord, "online"))
                continue
            return None
        return _sorted_empty_cores(relay_cores)

    def _enumerate_data_route_options(
        self, producer: OutputProducer
    ) -> dict[_CommonSuffix, _DataRouteOption]:
        offsets = candidate_offsets(producer.coord, producer.target_coord)
        if not offsets:
            raise OutputCompletionNoFeasiblePlanError(
                "No legal output DATA route offset from "
                f"({producer.coord.x},{producer.coord.y}) to "
                f"({producer.target_coord.x},{producer.target_coord.y})."
            )

        options: dict[_CommonSuffix, _DataRouteOption] = {}
        for offset in offsets:
            path = route_coord_path(producer.coord, offset)
            if path[-1] != producer.target_coord:
                continue
            if not all(is_global_route_coord(coord) for coord in path):
                continue

            option = _DataRouteOption(producer, offset, path, route_len(offset))
            self._record_best_option_by_suffix(options, option)
        return options

    def _record_best_option_by_suffix(
        self, options: dict[_CommonSuffix, _DataRouteOption], option: _DataRouteOption
    ) -> None:
        for suffix in _common_suffixes_for_path(option.path):
            current = options.get(suffix)
            if current is None or self._data_route_key(option) < self._data_route_key(
                current
            ):
                options[suffix] = option

    def _data_route_key(
        self, option: _DataRouteOption
    ) -> tuple[int, tuple[int, int, int]]:
        return option.path_len, option.offset.to_tuple()

    def _find_common_suffix_candidates(self) -> tuple[_JoinCandidate, ...]:
        producer_options = [
            self._enumerate_data_route_options(producer)
            for producer in self._request.producers
        ]
        if any(not options for options in producer_options):
            return ()

        common_suffixes = set(producer_options[0])
        for options in producer_options[1:]:
            common_suffixes &= set(options)

        candidates: list[_JoinCandidate] = []
        pools = self._request.pools
        for suffix in common_suffixes:
            join_tier = _completion_core_tier(suffix.join_point, pools)
            if join_tier is None:
                continue

            data_routes = tuple(options[suffix] for options in producer_options)
            selected_join_core = _empty_completion_core(suffix.join_point, pools)
            required_cores = () if selected_join_core is None else (selected_join_core,)
            required_coords = frozenset(core.coord for core in required_cores)
            candidates.append(
                _JoinCandidate(
                    suffix,
                    data_routes,
                    required_cores,
                    required_coords,
                    sum(route.path_len for route in data_routes),
                    join_tier,
                    len(suffix.suffix) - 1,
                )
            )
        return tuple(candidates)

    def _select_join_candidates(self) -> tuple[_JoinCandidate, ...]:
        candidates = self._find_common_suffix_candidates()
        if not candidates:
            return ()

        min_data_total_len = min(candidate.data_total_len for candidate in candidates)
        shortest_data_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.data_total_len == min_data_total_len
        )
        best_join = min(shortest_data_candidates, key=self._join_selection_key)
        return tuple(
            candidate
            for candidate in shortest_data_candidates
            if candidate.suffix == best_join.suffix
        )

    def _join_selection_key(
        self, candidate: _JoinCandidate
    ) -> tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]:
        return (
            _tier_rank(candidate.join_tier),
            candidate.join_to_cpu_path_len,
            len(candidate.required_route_core_coords),
            candidate.suffix.join_point.x,
            candidate.suffix.join_point.y,
            _coords_key(candidate.suffix.suffix),
        )

    def _build_completion_candidates(
        self, join_candidate: _JoinCandidate
    ) -> tuple[_CompletionCandidate, ...]:
        join_point = join_candidate.suffix.join_point
        suffix = join_candidate.suffix.suffix
        data_required_coords = join_candidate.required_route_core_coords
        pools = self._request.pools
        candidate_thread_coords = pools.configured | data_required_coords
        source_coords = (
            candidate_thread_coords | pools.empty_offline | pools.empty_online
        ) - {CPU_COORD}
        candidates: list[_CompletionCandidate] = []

        for source in sorted(source_coords, key=_coord_key):
            if not is_configurable_thread_core_coord(source):
                continue
            candidates.extend(
                self._completion_candidates_for_source(
                    join_candidate, source, join_point, suffix
                )
            )
        return tuple(candidates)

    def _completion_candidates_for_source(
        self,
        join_candidate: _JoinCandidate,
        source: CoordXY,
        join_point: CoordXY,
        suffix: tuple[CoordXY, ...],
    ) -> tuple[_CompletionCandidate, ...]:
        candidates: list[_CompletionCandidate] = []
        pools = self._request.pools
        source_tier = _completion_core_tier(source, pools)
        if source_tier is None:
            return ()

        for control_offset in candidate_offsets(source, TEST_DEST_CORE):
            control_path = route_coord_path(source, control_offset)
            if _path_suffix_from(control_path, join_point) != suffix:
                continue

            if not all(is_global_route_coord(coord) for coord in control_path):
                continue

            source_core = _empty_completion_core(source, pools)
            selected_source_cores = () if source_core is None else (source_core,)
            combined_required_cores = {
                *join_candidate.required_route_cores,
                *selected_source_cores,
            }
            combined_required_coords = frozenset(
                core.coord for core in combined_required_cores
            )
            combined_thread_coords = pools.configured | combined_required_coords
            global_signal_tree = solve_global_signal_tree(
                list(combined_thread_coords), source, verbose=False
            )
            relay_cores = self._classify_relay_cores(
                frozenset(global_signal_tree.added),
                combined_thread_coords,
            )
            if relay_cores is None:
                continue

            candidates.append(
                _CompletionCandidate(
                    join_candidate,
                    source,
                    source_tier,
                    control_offset,
                    control_path,
                    _sorted_empty_cores(combined_required_cores),
                    relay_cores,
                    route_len(control_offset),
                    len(combined_required_coords),
                )
            )
        return tuple(candidates)

    def _select_completion_candidate(
        self, join_candidates: tuple[_JoinCandidate, ...]
    ) -> _CompletionCandidate | None:
        candidates = [
            completion_candidate
            for join_candidate in join_candidates
            for completion_candidate in self._build_completion_candidates(
                join_candidate
            )
        ]
        if not candidates:
            return None

        best_tier_rank = min(
            _tier_rank(candidate.source_tier) for candidate in candidates
        )
        tier_candidates = [
            candidate
            for candidate in candidates
            if _tier_rank(candidate.source_tier) == best_tier_rank
        ]
        return min(tier_candidates, key=self._completion_selection_key)

    def _completion_selection_key(
        self, candidate: _CompletionCandidate
    ) -> tuple[int, int, int, int, int]:
        return (
            candidate.control_path_len,
            candidate.control_required_route_core_count,
            len(candidate.relay_cores),
            candidate.source.x,
            candidate.source.y,
        )

    def _build_plan(self, candidate: _CompletionCandidate) -> OutputCompletionPlan:
        output_routes = tuple(
            OutputRouteDecision(
                data_route.producer.coord,
                data_route.producer.target_coord,
                data_route.offset,
            )
            for data_route in candidate.join_candidate.data_routes
        )
        return OutputCompletionPlan(
            output_routes,
            candidate.control_offset,
            terminal_route_side(candidate.control_offset),
            candidate.source,
            candidate.join_candidate.suffix.join_point,
            candidate.required_route_cores,
            candidate.relay_cores,
        )


def validate_output_completion_plan(
    plan: OutputCompletionPlan,
    producers: tuple[OutputProducer, ...],
    configured_thread_core_coords: Set[CoordXY],
    allow_empty_online: bool = True,
) -> None:
    """Validates the safety constraints of an output completion plan."""
    configured_coords = frozenset(configured_thread_core_coords)
    if (
        plan.completion_join_point == TEST_DEST_CORE
        or not is_configurable_thread_core_coord(plan.completion_join_point)
    ):
        raise OutputCompletionError("Completion join point must be inside the thread.")

    required_coords = {relay.coord for relay in plan.required_route_cores}
    relay_coords = {relay.coord for relay in plan.relay_cores}
    if required_coords & configured_coords:
        raise OutputCompletionError(
            "Required route cores must be empty configurable thread cores."
        )
    if relay_coords & configured_coords:
        raise OutputCompletionError(
            "Relay cores must be empty configurable thread cores."
        )

    for relay in (*plan.required_route_cores, *plan.relay_cores):
        if not is_configurable_thread_core_coord(relay.coord):
            raise OutputCompletionError(
                "Relay core must be a configurable thread core."
            )
        expected_kind: EmptyRelayCoreKind = (
            "online" if is_online_core_coord(relay.coord) else "offline"
        )
        if relay.kind != expected_kind:
            raise OutputCompletionError("Relay core kind does not match its region.")
        if not allow_empty_online and relay.kind == "online":
            raise OutputCompletionError("Online relay cores are disabled.")

    final_thread_coords = configured_coords | required_coords | relay_coords
    if plan.global_signal_root not in final_thread_coords:
        raise OutputCompletionError(
            "Completion source must be configured in the thread."
        )
    if plan.completion_join_point not in final_thread_coords:
        raise OutputCompletionError(
            "Completion join point must be configured in the thread."
        )

    complete_path, data_paths = _plan_paths(plan)
    if complete_path[-1] != TEST_DEST_CORE:
        raise OutputCompletionError("Completion route must target the CPU core.")
    suffix = _path_suffix_from(complete_path, plan.completion_join_point)
    if suffix is None:
        raise OutputCompletionError("Completion route must pass the join point.")
    if terminal_route_side(plan.control_offset) != plan.ingress_side:
        raise OutputCompletionError("Completion ingress side must match the plan.")

    route_by_endpoint = {
        (route.producer_coord, route.target_coord): route
        for route in plan.output_routes
    }
    for producer, data_path in zip(producers, data_paths, strict=True):
        route = route_by_endpoint.get((producer.coord, producer.target_coord))
        if route is None:
            raise OutputCompletionError(
                "Missing output route for producer "
                f"({producer.coord.x},{producer.coord.y})."
            )
        if data_path[-1] != route.target_coord:
            raise OutputCompletionError("Output DATA route must target the CPU core.")
        if _path_suffix_from(data_path, plan.completion_join_point) != suffix:
            raise OutputCompletionError(
                "Completion route must match the output DATA common suffix."
            )
        if terminal_route_side(route.offset) != plan.ingress_side:
            raise OutputCompletionError(
                "Output DATA ingress side must match completion."
            )


def select_output_completion_plan(
    producers: list[OutputProducer],
    configured_thread_core_coords: Set[CoordXY],
    available_empty_offline_coords: Set[CoordXY],
    available_empty_online_coords: Set[CoordXY] | None = None,
    allow_empty_online: bool = False,
) -> OutputCompletionPlan:
    """Chooses DATA routes, a shared join, and a completion source."""
    return OutputCompletionPlanner(
        producers,
        configured_thread_core_coords,
        available_empty_offline_coords,
        available_empty_online_coords,
        allow_empty_online,
    ).plan()


def debug_output_completion_plan(
    plan: OutputCompletionPlan,
    producers: tuple[OutputProducer, ...],
    used_core_coords: Set[CoordXY],
) -> tuple[str, ...]:
    """Returns stable debug lines for the selected completion plan."""
    del producers, used_core_coords
    complete_path, data_paths = _plan_paths(plan)
    suffix = _path_suffix_from(complete_path, plan.completion_join_point)
    lines = [
        f"source=({plan.global_signal_root.x},{plan.global_signal_root.y})",
        f"join=({plan.completion_join_point.x},{plan.completion_join_point.y})",
        f"control_offset={plan.control_offset.to_tuple()} ingress={plan.ingress_side}",
        "selected_empty_completion_cores="
        + str(
            sorted((c.coord.x, c.coord.y, c.kind) for c in plan.required_route_cores)
        ),
        "relay_cores="
        + str(sorted((c.coord.x, c.coord.y, c.kind) for c in plan.relay_cores)),
        "shared_suffix=" + str([(coord.x, coord.y) for coord in (suffix or ())]),
        "complete_path=" + str([(coord.x, coord.y) for coord in complete_path]),
    ]
    for route, path in zip(plan.output_routes, data_paths, strict=True):
        lines.append(
            "data_path "
            f"producer=({route.producer_coord.x},{route.producer_coord.y}) "
            f"offset={route.offset.to_tuple()} "
            f"path={[(coord.x, coord.y) for coord in path]}"
        )
    return tuple(lines)


def render_output_completion_plan_ascii(
    plan: OutputCompletionPlan,
    producers: tuple[OutputProducer, ...],
    used_core_coords: Set[CoordXY],
) -> str:
    """Renders a compact ASCII view of DATA and complete route overlap."""
    complete_path, data_paths = _plan_paths(plan)
    suffix = set(_path_suffix_from(complete_path, plan.completion_join_point) or ())
    complete_only = set(complete_path) - suffix
    data_only: set[CoordXY] = set()
    for path in data_paths:
        data_only.update(set(path) - suffix)
    required_coords = {core.coord for core in plan.required_route_cores}
    producer_labels = {producer.coord: f"P{i}" for i, producer in enumerate(producers)}

    def cell(coord: CoordXY) -> str:
        if coord == TEST_DEST_CORE:
            return "C"
        if coord == plan.global_signal_root and coord == plan.completion_join_point:
            return "SM"
        if coord == plan.global_signal_root:
            return "S"
        if coord == plan.completion_join_point:
            return "M"
        if coord in producer_labels:
            return producer_labels[coord]
        if coord in suffix:
            return "*"
        if coord in complete_only:
            return "c"
        if coord in data_only:
            return "d"
        if coord in required_coords:
            return "E"
        if coord in used_core_coords:
            return "U"
        return "."

    rows = []
    for y in range(G_Y_MAX, G_Y_MIN - 1, -1):
        cells = " ".join(
            f"{cell(CoordXY(x, y)):>2}" for x in range(G_X_MIN, G_X_MAX + 1)
        )
        rows.append(f"y={y} {cells}")
    rows.append("    " + " ".join(f"{x:>2}" for x in range(G_X_MIN, G_X_MAX + 1)))
    rows.append(
        "legend: C=CPU S=source M=join *=shared d=DATA-only "
        "c=complete-only E=required-empty U=used"
    )
    rows.append(
        "producers: "
        + ", ".join(
            f"P{i}=({producer.coord.x},{producer.coord.y})"
            for i, producer in enumerate(producers)
        )
    )
    return "\n".join(rows)
