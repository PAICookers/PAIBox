"""Output DATA and completion-frame route planner for backendv2."""

from collections.abc import Iterable, Set
from dataclasses import dataclass
from itertools import chain
from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset

from .core_config import TEST_DEST_CORE
from .global_signal import EmptyRelayCoreKind as EmptyThreadCoreKind
from .global_signal import solve_global_signal_tree
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
_JoinSelectionKey = tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]


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
class EmptyThreadCore:
    """Selected empty shell core added to the current thread configuration."""

    coord: CoordXY
    kind: EmptyThreadCoreKind


@dataclass(frozen=True, slots=True)
class OutputCompletionPlan:
    """Chosen DATA routes and matching complete/control route."""

    output_routes: tuple[OutputRouteDecision, ...]
    control_offset: CoordZXYOffset
    ingress_side: RouteSide
    global_signal_root: CoordXY
    completion_join_point: CoordXY
    completion_thread_cores: tuple[EmptyThreadCore, ...]
    global_signal_relay_cores: tuple[EmptyThreadCore, ...]

    def output_route_offsets(self) -> dict[tuple[CoordXY, CoordXY], CoordZXYOffset]:
        """Returns per-output-core DATA offsets keyed by producer and target."""
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }

    def empty_thread_core_kinds(self) -> dict[CoordXY, EmptyThreadCoreKind]:
        """Returns all selected empty shell cores keyed by coordinate."""
        empty_core_kinds: dict[CoordXY, EmptyThreadCoreKind] = {}
        for core in chain(self.completion_thread_cores, self.global_signal_relay_cores):
            empty_core_kinds[core.coord] = core.kind
        return empty_core_kinds

    def to_cpu_ingress_plan(self) -> OutputCpuIngressPlan:
        """Converts the completion decision to CPU ingress route metadata."""
        return OutputCpuIngressPlan(
            self.output_routes, self.control_offset, self.ingress_side
        )


@dataclass(frozen=True, slots=True)
class _ThreadCorePools:
    """Current-thread core ownership pools used by completion planning."""

    # FIXME: A future multi-thread allocator must remove cores already owned by
    # other threads from these empty pools before the planner sees them.
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
    """One shared DATA suffix candidate anchored at a completion join point."""

    suffix: _CommonSuffix
    data_routes: tuple[_DataRouteOption, ...]
    completion_thread_cores: tuple[EmptyThreadCore, ...]
    """Empty cores on producer-to-M DATA prefixes that must join this thread."""
    completion_thread_core_coords: frozenset[CoordXY]
    data_total_len: int
    """Weighted total DATA route length across all output producers."""
    join_tier: CompletionCoreTier
    """Join ownership tier: used core, available empty offline, or empty online."""
    join_to_cpu_path_len: int
    """Length of the shared completion_join_point-to-CPU suffix."""


@dataclass(frozen=True, slots=True)
class _CompletionCandidate:
    """A complete candidate after selecting DATA routes, join, and source."""

    join_candidate: _JoinCandidate
    source: CoordXY
    source_tier: CompletionCoreTier
    """Source ownership tier; source is the selected completion join point."""
    control_offset: CoordZXYOffset
    control_path: tuple[CoordXY, ...]
    completion_thread_cores: tuple[EmptyThreadCore, ...]
    """Completion-thread cores include join/source empty cores and DATA-prefix \
        empty cores; global-signal relay cores are only tree-connectivity helpers."""
    global_signal_relay_cores: tuple[EmptyThreadCore, ...]
    control_path_len: int
    """Length of the source-to-CPU complete/control path."""
    completion_thread_core_count: int


def _coord_key(coord: CoordXY) -> tuple[int, int]:
    return coord.x, coord.y


def _coords_key(coords: tuple[CoordXY, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(_coord_key(coord) for coord in coords)


def _empty_core_key(core: EmptyThreadCore) -> tuple[int, int, EmptyThreadCoreKind]:
    return core.coord.x, core.coord.y, core.kind


def _sorted_empty_cores(
    cores: Iterable[EmptyThreadCore],
) -> tuple[EmptyThreadCore, ...]:
    return tuple(sorted(set(cores), key=_empty_core_key))


def _path_suffix_from(
    path: tuple[CoordXY, ...], join_point: CoordXY
) -> tuple[CoordXY, ...] | None:
    for index, coord in enumerate(path[:-1]):
        if coord == join_point:
            return path[index:]
    return None


def _path_prefix_to(
    path: tuple[CoordXY, ...], join_point: CoordXY
) -> tuple[CoordXY, ...] | None:
    for index, coord in enumerate(path[:-1]):
        if coord == join_point:
            return path[: index + 1]
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
) -> EmptyThreadCore | None:
    if coord in pools.configured:
        return None
    if coord in pools.empty_offline:
        return EmptyThreadCore(coord, "offline")
    if coord in pools.empty_online:
        return EmptyThreadCore(coord, "online")
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


def _empty_data_prefix_thread_cores(
    path: tuple[CoordXY, ...], join_point: CoordXY, pools: _ThreadCorePools
) -> tuple[EmptyThreadCore, ...] | None:
    """Returns empty thread cores needed before DATA reaches the join point."""
    prefix = _path_prefix_to(path, join_point)
    if prefix is None:
        return None

    thread_cores: set[EmptyThreadCore] = set()
    # DATA must reach the join point before the shared suffix can serialize it
    # with complete. Empty configurable cores on this prefix must join the
    # thread/global-signal tree; shared suffix transit remains route-only.
    # TODO: Model real DATA relay computation cores separately if such a
    # direction-changing relay is ever introduced.
    for coord in prefix:
        if coord == CPU_COORD or coord in pools.configured:
            continue
        if not is_configurable_thread_core_coord(coord):
            continue
        empty_core = _empty_completion_core(coord, pools)
        if empty_core is None:
            return None
        thread_cores.add(empty_core)

    return _sorted_empty_cores(thread_cores)


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

    Safety contract:
    - Each output producer emits DATA directly to the fixed CPU target through
      one legal Z/X/Y route offset; the planner does not create DATA relay
      computation cores.
    - All DATA routes must pass the non-CPU ``completion_join_point`` and
      share the exact same suffix from that point to the CPU.
    - The global-signal root is the completion join point. This conservative
      choice makes every DATA route intersect the root before the shared suffix.
    - Empty configurable cores on producer-to-join DATA prefixes become
      ``completion_thread_cores`` so their route completion is visible to the
      global signal tree.
    - Empty configurable cores on the shared join-to-CPU suffix are route-only
      transit and do not become thread members.
    - ``allow_empty_online`` only controls whether empty online cores may become
      join/source/completion-thread/global-signal-relay cores; online suffix
      transit remains legal.

    Public callers should normally use ``select_output_completion_plan``.
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

        join_candidates = self._find_common_suffix_candidates()
        selected = self._select_completion_candidate(join_candidates)
        if selected is None:
            raise OutputCompletionNoFeasiblePlanError(
                "Cannot find output DATA and completion routes with a shared "
                "thread-local suffix. DATA computation relay cores are not "
                "supported by this planner. producers="
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

    def _classify_global_signal_relay_cores(
        self,
        coords: Set[CoordXY],
        configured_thread_core_coords: Set[CoordXY],
    ) -> tuple[EmptyThreadCore, ...] | None:
        global_signal_relay_cores: set[EmptyThreadCore] = set()
        pools = self._request.pools
        for coord in coords:
            if coord in configured_thread_core_coords:
                continue
            if coord in pools.empty_offline:
                global_signal_relay_cores.add(EmptyThreadCore(coord, "offline"))
                continue
            if coord in pools.empty_online:
                global_signal_relay_cores.add(EmptyThreadCore(coord, "online"))
                continue
            return None
        return _sorted_empty_cores(global_signal_relay_cores)

    def _enumerate_data_route_options(
        self, producer: OutputProducer
    ) -> dict[_CommonSuffix, tuple[_DataRouteOption, ...]]:
        offsets = candidate_offsets(producer.coord, producer.target_coord)
        if not offsets:
            raise OutputCompletionNoFeasiblePlanError(
                "No legal output DATA route offset from "
                f"({producer.coord.x},{producer.coord.y}) to "
                f"({producer.target_coord.x},{producer.target_coord.y})."
            )

        option_groups: dict[_CommonSuffix, list[_DataRouteOption]] = {}
        for offset in offsets:
            path = route_coord_path(producer.coord, offset)
            if path[-1] != producer.target_coord:
                continue
            if not all(is_global_route_coord(coord) for coord in path):
                continue

            option = _DataRouteOption(producer, offset, path, route_len(offset))
            for suffix in _common_suffixes_for_path(option.path):
                option_groups.setdefault(suffix, []).append(option)

        return {
            suffix: tuple(sorted(options, key=self._data_route_key))
            for suffix, options in option_groups.items()
        }

    def _data_route_key(
        self, option: _DataRouteOption
    ) -> tuple[int, tuple[int, int, int]]:
        return option.path_len, option.offset.to_tuple()

    def _best_data_route_for_suffix(
        self, options: tuple[_DataRouteOption, ...], suffix: _CommonSuffix
    ) -> tuple[_DataRouteOption, tuple[EmptyThreadCore, ...]] | None:
        for option in options:
            thread_cores = _empty_data_prefix_thread_cores(
                option.path, suffix.join_point, self._request.pools
            )
            if thread_cores is not None:
                return option, thread_cores
        return None

    def _find_common_suffix_candidates(self) -> tuple[_JoinCandidate, ...]:
        producer_options = [
            self._enumerate_data_route_options(producer)
            for producer in self._request.producers
        ]
        if any(not options for options in producer_options):
            return ()

        common_suffixes = set(producer_options[0]).intersection(*producer_options[1:])

        candidates: list[_JoinCandidate] = []
        pools = self._request.pools
        for suffix in common_suffixes:
            join_tier = _completion_core_tier(suffix.join_point, pools)
            if join_tier is None:
                continue

            route_choices = tuple(
                self._best_data_route_for_suffix(options[suffix], suffix)
                for options in producer_options
            )
            if any(choice is None for choice in route_choices):
                continue

            data_routes = tuple(choice[0] for choice in route_choices if choice)
            completion_thread_cores = _sorted_empty_cores(
                chain.from_iterable(choice[1] for choice in route_choices if choice)
            )
            completion_thread_coords = frozenset(
                core.coord for core in completion_thread_cores
            )
            candidates.append(
                _JoinCandidate(
                    suffix,
                    data_routes,
                    completion_thread_cores,
                    completion_thread_coords,
                    sum(route.path_len for route in data_routes),
                    join_tier,
                    len(suffix.suffix) - 1,
                )
            )
        return tuple(candidates)

    def _join_selection_key(self, candidate: _JoinCandidate) -> _JoinSelectionKey:
        # Prefer used M, then shorter suffix and fewer selected empty cores.
        return (
            _tier_rank(candidate.join_tier),
            candidate.join_to_cpu_path_len,
            len(candidate.completion_thread_core_coords),
            candidate.suffix.join_point.x,
            candidate.suffix.join_point.y,
            _coords_key(candidate.suffix.suffix),
        )

    def _build_completion_candidates(
        self, join_candidate: _JoinCandidate
    ) -> tuple[_CompletionCandidate, ...]:
        join_point = join_candidate.suffix.join_point
        suffix = join_candidate.suffix.suffix
        # Keep completion source and DATA join identical. Allowing a separate
        # source can satisfy suffix equality through a downstream M, but then
        # some DATA routes do not visibly intersect the global-signal root.
        if not is_configurable_thread_core_coord(join_point):
            return ()
        return self._completion_candidates_for_source(
            join_candidate, join_point, join_point, suffix
        )

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
            combined_required_cores = _sorted_empty_cores(
                chain(join_candidate.completion_thread_cores, selected_source_cores)
            )
            combined_required_coords = frozenset(
                core.coord for core in combined_required_cores
            )
            combined_thread_coords = pools.configured | combined_required_coords
            global_signal_tree = solve_global_signal_tree(
                list(combined_thread_coords), source, verbose=False
            )
            global_signal_relay_cores = self._classify_global_signal_relay_cores(
                frozenset(global_signal_tree.added),
                combined_thread_coords,
            )
            if global_signal_relay_cores is None:
                continue

            candidates.append(
                _CompletionCandidate(
                    join_candidate,
                    source,
                    source_tier,
                    control_offset,
                    control_path,
                    combined_required_cores,
                    global_signal_relay_cores,
                    route_len(control_offset),
                    len(combined_required_coords),
                )
            )
        return tuple(candidates)

    def _select_completion_candidate(
        self, join_candidates: tuple[_JoinCandidate, ...]
    ) -> _CompletionCandidate | None:
        candidates = tuple(
            chain.from_iterable(
                self._build_completion_candidates(join_candidate)
                for join_candidate in join_candidates
            )
        )
        if not candidates:
            return None

        return min(candidates, key=self._completion_selection_key)

    def _completion_selection_key(self, candidate: _CompletionCandidate):
        join_candidate = candidate.join_candidate
        # Rank only complete candidates: DATA cost, M quality, source quality.
        return (
            join_candidate.data_total_len,
            *self._join_selection_key(join_candidate),
            _tier_rank(candidate.source_tier),
            candidate.control_path_len,
            candidate.completion_thread_core_count,
            len(candidate.global_signal_relay_cores),
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
            candidate.completion_thread_cores,
            candidate.global_signal_relay_cores,
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

    completion_thread_coords = {core.coord for core in plan.completion_thread_cores}
    global_signal_relay_coords = {core.coord for core in plan.global_signal_relay_cores}
    if completion_thread_coords & configured_coords:
        raise OutputCompletionError(
            "Completion thread cores must be empty configurable thread cores."
        )
    if global_signal_relay_coords & configured_coords:
        raise OutputCompletionError(
            "Global signal relay cores must be empty configurable thread cores."
        )

    for core in chain(plan.completion_thread_cores, plan.global_signal_relay_cores):
        if not is_configurable_thread_core_coord(core.coord):
            raise OutputCompletionError(
                "Empty thread core must be a configurable thread core."
            )
        expected_kind: EmptyThreadCoreKind = (
            "online" if is_online_core_coord(core.coord) else "offline"
        )
        if core.kind != expected_kind:
            raise OutputCompletionError(
                "Empty thread core kind does not match its region."
            )
        if not allow_empty_online and core.kind == "online":
            raise OutputCompletionError("Online empty thread cores are disabled.")

    final_thread_coords = (
        configured_coords | completion_thread_coords | global_signal_relay_coords
    )
    if plan.global_signal_root not in final_thread_coords:
        raise OutputCompletionError(
            "Completion source must be configured in the thread."
        )
    if plan.completion_join_point not in final_thread_coords:
        raise OutputCompletionError(
            "Completion join point must be configured in the thread."
        )
    if plan.global_signal_root != plan.completion_join_point:
        raise OutputCompletionError(
            "Completion source must be the output DATA join point."
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
        prefix = _path_prefix_to(data_path, plan.completion_join_point)
        if prefix is None:
            raise OutputCompletionError(
                "Output DATA route must pass the completion join point."
            )
        missing_prefix_cores = [
            coord
            for coord in prefix
            if is_configurable_thread_core_coord(coord)
            and coord not in final_thread_coords
        ]
        if missing_prefix_cores:
            raise OutputCompletionError(
                "Output DATA prefix completion thread cores must be configured "
                "in the thread."
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
        "completion_thread_cores="
        + str(
            sorted((c.coord.x, c.coord.y, c.kind) for c in plan.completion_thread_cores)
        ),
        "global_signal_relay_cores="
        + str(
            sorted(
                (c.coord.x, c.coord.y, c.kind) for c in plan.global_signal_relay_cores
            )
        ),
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
    completion_thread_coords = {core.coord for core in plan.completion_thread_cores}
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
        if coord in completion_thread_coords:
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
