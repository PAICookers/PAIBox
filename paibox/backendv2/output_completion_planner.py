"""Output DATA and completion-frame route planner for backendv2."""

from collections.abc import Iterable, Set
from dataclasses import dataclass
from functools import cache
from itertools import chain
from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset, route_coord_path, to_coordxys

from .global_signal import EmptyRelayCoreKind as EmptyThreadCoreKind
from .global_signal import solve_global_signal_tree
from .route_scope import RouteScope, TargetBoard, get_route_scope

__all__ = [
    "OutputCompletionPlan",
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


def terminal_route_side(offset: CoordZXYOffset) -> RouteSide:
    """Return the terminal route side after backendv2 Z, X, then Y routing."""
    if offset.y != 0:
        return ("y", 1 if offset.y > 0 else -1)
    if offset.x != 0:
        return ("x", 1 if offset.x > 0 else -1)
    if offset.z != 0:
        return ("xy", 1 if offset.z > 0 else -1)
    return ("local", 0)


@cache
def _candidate_offset_paths(
    start_coord: CoordXY, target_coord: CoordXY, scope_name: TargetBoard
) -> tuple[tuple[CoordZXYOffset, tuple[CoordXY, ...]], ...]:
    scope = get_route_scope(scope_name)
    dcoord = target_coord - start_coord
    routes: list[tuple[CoordZXYOffset, tuple[CoordXY, ...]]] = []
    for z in range(-31, 32):
        x = dcoord.x - z
        y = dcoord.y - z
        if not (-31 <= x <= 31 and -31 <= y <= 31):
            continue
        offset = CoordZXYOffset(z, x, y)
        path = route_coord_path(start_coord, offset)
        if scope.route_path_valid(path, target_coord):
            routes.append((offset, path))

    return tuple(
        sorted(routes, key=lambda item: (item[0].l1_norm(), item[0].to_tuple()))
    )


def candidate_offsets(
    start_coord: CoordXY, target_coord: CoordXY, scope: RouteScope | None = None
) -> tuple[CoordZXYOffset, ...]:
    """Enumerate legal offsets to one fixed target without retargeting CPU."""
    resolved_scope = scope or get_route_scope("single")
    return tuple(
        offset
        for offset, _ in _candidate_offset_paths(
            start_coord, target_coord, resolved_scope.name
        )
    )


class OutputCompletionError(RuntimeError):
    """Base error for output DATA / complete-frame planning failures."""


class OutputCompletionNoFeasiblePlanError(OutputCompletionError):
    """Raised when DATA and completion paths cannot share a valid suffix."""


@dataclass(frozen=True, slots=True)
class OutputProducer:
    """Output-producing core endpoint and its static output-entry weight."""

    coord: CoordXY
    target_coord: CoordXY | None = None
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
        """Return per-output-core DATA offsets keyed by producer and target."""
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }

    def empty_thread_core_kinds(self) -> dict[CoordXY, EmptyThreadCoreKind]:
        """Return all selected empty shell cores keyed by coordinate."""
        empty_core_kinds: dict[CoordXY, EmptyThreadCoreKind] = {}
        for core in chain(self.completion_thread_cores, self.global_signal_relay_cores):
            empty_core_kinds[core.coord] = core.kind
        return empty_core_kinds


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
    l1_norm: int


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
    data_total_l1_norm: int
    """Weighted total DATA route L1 norm across all output producers."""
    join_tier: CompletionCoreTier
    """Join ownership tier: used core, available empty offline, or empty online."""


@dataclass(frozen=True, slots=True)
class _CompletionCandidate:
    """A complete candidate after selecting DATA routes and join point."""

    join_candidate: _JoinCandidate
    control_offset: CoordZXYOffset
    completion_thread_cores: tuple[EmptyThreadCore, ...]
    """Completion-thread cores include join empty cores and DATA-prefix \
        empty cores; global-signal relay cores are only tree-connectivity helpers."""
    global_signal_relay_cores: tuple[EmptyThreadCore, ...]


def _sorted_empty_cores(
    cores: Iterable[EmptyThreadCore],
) -> tuple[EmptyThreadCore, ...]:
    return tuple(
        sorted(set(cores), key=lambda core: (*core.coord.to_tuple(), core.kind))
    )


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


def _common_suffixes_for_path(
    path: tuple[CoordXY, ...], scope: RouteScope
) -> tuple[_CommonSuffix, ...]:
    suffixes: list[_CommonSuffix] = []
    seen: set[_CommonSuffix] = set()
    configurable_coords = scope.configurable_coords
    for index, coord in enumerate(path[:-1]):
        if coord not in configurable_coords:
            continue
        suffix = _CommonSuffix(coord, path[index:])
        if suffix in seen:
            continue
        seen.add(suffix)
        suffixes.append(suffix)
    return tuple(suffixes)


def _empty_data_prefix_thread_cores(
    path: tuple[CoordXY, ...],
    join_point: CoordXY,
    pools: _ThreadCorePools,
    scope: RouteScope,
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
        if coord in scope.cpu_coords or coord in pools.configured:
            continue
        if coord not in scope.configurable_coords:
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
        scope: RouteScope | None = None,
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
            scope: Selected board route scope for CPU target and grid checks.
        """
        self._scope = scope or get_route_scope("single")
        self._cpu_coord = self._scope.default_cpu.coord
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
        """Build and validate an output completion plan.

        Returns:
            Selected DATA routes, completion route, and empty thread cores.

        Raises:
            OutputCompletionNoFeasiblePlanError: If DATA and completion routes
                cannot share a valid thread-local suffix.
            OutputCompletionError: If a selected candidate violates the planner
                safety contract.
        """
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
            self._scope,
        )
        return plan

    def _normalize_request(self) -> _PlannerRequest:
        configured_coords = self._configured_thread_core_coords
        if not self._allow_empty_online:
            empty_online_coords: frozenset[CoordXY] = frozenset()
        elif self._available_empty_online_coords is None:
            empty_online_coords = frozenset(
                self._scope.online_core_coords - configured_coords
            )
        else:
            empty_online_coords = self._available_empty_online_coords

        producers = tuple(
            OutputProducer(
                producer.coord,
                producer.target_coord or self._cpu_coord,
                max(producer.weight, 1),
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
        root = min(self._request.pools.configured, key=lambda coord: coord.to_tuple())
        control_offset = _candidate_offset_paths(
            root, self._cpu_coord, self._scope.name
        )[0][0]
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
        target_coord = producer.target_coord or self._cpu_coord
        offset_paths = _candidate_offset_paths(
            producer.coord, target_coord, self._scope.name
        )
        if not offset_paths:
            raise OutputCompletionNoFeasiblePlanError(
                "No legal output DATA route offset from "
                f"({producer.coord.x},{producer.coord.y}) to "
                f"({target_coord.x},{target_coord.y})."
            )

        option_groups: dict[_CommonSuffix, list[_DataRouteOption]] = {}
        for offset, path in offset_paths:
            option = _DataRouteOption(producer, offset, path, offset.l1_norm())
            for suffix in _common_suffixes_for_path(option.path, self._scope):
                option_groups.setdefault(suffix, []).append(option)

        return {
            suffix: tuple(sorted(options, key=self._data_route_key))
            for suffix, options in option_groups.items()
        }

    def _data_route_key(
        self, option: _DataRouteOption
    ) -> tuple[int, tuple[int, int, int]]:
        return option.l1_norm, option.offset.to_tuple()

    def _best_data_route_for_suffix(
        self, options: tuple[_DataRouteOption, ...], suffix: _CommonSuffix
    ) -> tuple[_DataRouteOption, tuple[EmptyThreadCore, ...]] | None:
        for option in options:
            thread_cores = _empty_data_prefix_thread_cores(
                option.path, suffix.join_point, self._request.pools, self._scope
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
            candidates.append(
                _JoinCandidate(
                    suffix,
                    data_routes,
                    completion_thread_cores,
                    sum(route.l1_norm for route in data_routes),
                    join_tier,
                )
            )
        return tuple(candidates)

    def _join_selection_key(self, candidate: _JoinCandidate):
        # Prefer used M, then shorter suffix and fewer selected empty cores.
        return (
            _tier_rank(candidate.join_tier),
            len(candidate.suffix.suffix) - 1,
            len(candidate.completion_thread_cores),
            candidate.suffix.join_point.x,
            candidate.suffix.join_point.y,
            to_coordxys(candidate.suffix.suffix),
        )

    def _build_completion_candidates(
        self, join_candidate: _JoinCandidate
    ) -> tuple[_CompletionCandidate, ...]:
        join_point = join_candidate.suffix.join_point
        suffix = join_candidate.suffix.suffix
        # Keep completion source and DATA join identical. Allowing a separate
        # source can satisfy suffix equality through a downstream M, but then
        # some DATA routes do not visibly intersect the global-signal root.
        candidates: list[_CompletionCandidate] = []
        pools = self._request.pools

        for control_offset, control_path in _candidate_offset_paths(
            join_point, self._cpu_coord, self._scope.name
        ):
            if _path_suffix_from(control_path, join_point) != suffix:
                continue

            join_core = _empty_completion_core(join_point, pools)
            selected_join_cores = () if join_core is None else (join_core,)
            combined_required_cores = _sorted_empty_cores(
                chain(join_candidate.completion_thread_cores, selected_join_cores)
            )
            combined_required_coords = frozenset(
                core.coord for core in combined_required_cores
            )
            combined_thread_coords = pools.configured | combined_required_coords
            global_signal_tree = solve_global_signal_tree(
                list(combined_thread_coords), join_point, verbose=False
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
                    control_offset,
                    combined_required_cores,
                    global_signal_relay_cores,
                )
            )
        return tuple(candidates)

    def _select_completion_candidate(
        self, join_candidates: tuple[_JoinCandidate, ...]
    ) -> _CompletionCandidate | None:
        return min(
            chain.from_iterable(
                self._build_completion_candidates(join_candidate)
                for join_candidate in join_candidates
            ),
            key=self._completion_selection_key,
            default=None,
        )

    def _completion_selection_key(self, candidate: _CompletionCandidate):
        join_candidate = candidate.join_candidate
        # Rank only complete candidates: DATA cost, M quality, control path.
        return (
            join_candidate.data_total_l1_norm,
            *self._join_selection_key(join_candidate),
            candidate.control_offset.l1_norm(),
            len(candidate.completion_thread_cores),
            len(candidate.global_signal_relay_cores),
        )

    def _build_plan(self, candidate: _CompletionCandidate) -> OutputCompletionPlan:
        output_routes = tuple(
            OutputRouteDecision(
                data_route.producer.coord,
                data_route.producer.target_coord or self._cpu_coord,
                data_route.offset,
            )
            for data_route in candidate.join_candidate.data_routes
        )
        return OutputCompletionPlan(
            output_routes,
            candidate.control_offset,
            terminal_route_side(candidate.control_offset),
            candidate.join_candidate.suffix.join_point,
            candidate.join_candidate.suffix.join_point,
            candidate.completion_thread_cores,
            candidate.global_signal_relay_cores,
        )


def validate_output_completion_plan(
    plan: OutputCompletionPlan,
    producers: tuple[OutputProducer, ...],
    configured_thread_core_coords: Set[CoordXY],
    allow_empty_online: bool = True,
    scope: RouteScope | None = None,
) -> None:
    """Validate the safety constraints of an output completion plan.

    Args:
        plan: Candidate output completion plan to validate.
        producers: Output producers that must be represented by DATA routes.
        configured_thread_core_coords: Coordinates already configured in the
            current thread.
        allow_empty_online: Whether selected empty online cores are legal.
            Defaults to ``True``.
        scope: Board route scope used for CPU target and route-grid checks.
            Defaults to the single-chip scope.

    Raises:
        OutputCompletionError: If the plan violates target, thread membership,
            route-scope, suffix-sharing, or ingress-side constraints.
    """
    resolved_scope = scope or get_route_scope("single")
    cpu_coord = resolved_scope.default_cpu.coord
    configured_coords = frozenset(configured_thread_core_coords)
    if (
        plan.completion_join_point == cpu_coord
        or not resolved_scope.is_configurable_thread_core_coord(
            plan.completion_join_point
        )
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
        if not resolved_scope.is_configurable_thread_core_coord(core.coord):
            raise OutputCompletionError(
                "Empty thread core must be a configurable thread core."
            )
        expected_kind: EmptyThreadCoreKind = (
            "online" if resolved_scope.is_online_core_coord(core.coord) else "offline"
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
    if complete_path[-1] != cpu_coord:
        raise OutputCompletionError("Completion route must target the CPU core.")
    if not resolved_scope.route_path_valid(complete_path, cpu_coord):
        raise OutputCompletionError("Completion route leaves the selected route scope.")
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
        producer_target = producer.target_coord or cpu_coord
        route = route_by_endpoint.get((producer.coord, producer_target))
        if route is None:
            raise OutputCompletionError(
                "Missing output route for producer "
                f"({producer.coord.x},{producer.coord.y})."
            )
        if data_path[-1] != route.target_coord:
            raise OutputCompletionError("Output DATA route must target the CPU core.")
        if not resolved_scope.route_path_valid(data_path, route.target_coord):
            raise OutputCompletionError("Output DATA route leaves the selected scope.")
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
            if resolved_scope.is_configurable_thread_core_coord(coord)
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
    scope: RouteScope | None = None,
) -> OutputCompletionPlan:
    """Choose DATA routes, a shared join, and a completion source.

    Args:
        producers: Output-producing cores that need DATA routes to the CPU.
        configured_thread_core_coords: Coordinates already configured in the
            current thread.
        available_empty_offline_coords: Empty offline cores that may be added to
            the current thread.
        available_empty_online_coords: Empty online cores that may be added to
            the current thread. When ``None`` and online relays are enabled, all
            unused online cores in `scope` are considered available.
        allow_empty_online: Whether empty online cores may be selected.
            Defaults to ``False``.
        scope: Board route scope used for CPU target and route-grid checks.
            Defaults to the single-chip scope.

    Returns:
        Validated output completion plan.

    Raises:
        OutputCompletionNoFeasiblePlanError: If no valid shared-suffix plan can
            be found.
        OutputCompletionError: If a selected candidate fails validation.
    """
    return OutputCompletionPlanner(
        producers,
        configured_thread_core_coords,
        available_empty_offline_coords,
        available_empty_online_coords,
        allow_empty_online,
        scope,
    ).plan()
