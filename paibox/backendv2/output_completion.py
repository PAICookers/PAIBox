from dataclasses import dataclass
from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset

from .core_config import TEST_DEST_CORE
from .global_signal import (
    EmptyRelayCoreKind,
    solve_global_signal_tree,
)
from .output_route_offsets import (
    RouteSide,
    candidate_offsets,
    route_coord_path,
    route_len,
    terminal_route_side,
)
from .output_routes import OutputCpuIngressPlan, OutputRouteDecision
from .route_solver import G_X_MAX, G_X_MIN, G_Y_MAX, G_Y_MIN

RootKind = Literal["used", "empty_offline", "empty_online"]


class OutputCompletionError(RuntimeError):
    """Base error for output DATA / complete-frame planning failures."""


class OutputCompletionNoFeasiblePlanError(OutputCompletionError):
    """Raised when DATA paths cannot share a valid complete-frame suffix."""


@dataclass(frozen=True)
class OutputProducer:
    coord: CoordXY
    target_coord: CoordXY = TEST_DEST_CORE
    weight: int = 1


@dataclass(frozen=True)
class EmptyRelayCore:
    coord: CoordXY
    kind: EmptyRelayCoreKind


@dataclass(frozen=True)
class OutputCompletionPlan:
    output_routes: tuple[OutputRouteDecision, ...]
    control_offset: CoordZXYOffset
    ingress_side: RouteSide
    global_signal_root: CoordXY
    root_kind: RootKind
    relay_cores: tuple[EmptyRelayCore, ...]
    data_penalty: int
    global_signal_tree_max_depth: int
    global_signal_tree_total_edges: int
    complete_path_len: int
    diagnostics: tuple[str, ...]

    @property
    def root_is_unused(self) -> bool:
        return self.root_kind != "used"

    @property
    def root_is_online_empty(self) -> bool:
        return self.root_kind == "empty_online"

    def output_route_offsets(self) -> dict[tuple[CoordXY, CoordXY], CoordZXYOffset]:
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }

    def relay_core_kinds(self) -> dict[CoordXY, EmptyRelayCoreKind]:
        relay_core_kinds: dict[CoordXY, EmptyRelayCoreKind] = {
            relay.coord: relay.kind for relay in self.relay_cores
        }
        if self.root_kind in {"empty_offline", "empty_online"}:
            relay_core_kinds[self.global_signal_root] = (
                "online" if self.root_kind == "empty_online" else "offline"
            )
        return relay_core_kinds

    def to_cpu_ingress_plan(self) -> OutputCpuIngressPlan:
        return OutputCpuIngressPlan(
            self.output_routes, self.control_offset, self.ingress_side
        )


@dataclass(frozen=True)
class _ProducerPathChoice:
    offset: CoordZXYOffset
    path: tuple[CoordXY, ...]
    path_len: int
    penalty: int


@dataclass(frozen=True)
class _RootSuffixCandidate:
    root: CoordXY
    suffix: tuple[CoordXY, ...]
    choices: tuple[_ProducerPathChoice, ...]
    control_offset: CoordZXYOffset
    root_kind: RootKind
    relay_cores: tuple[EmptyRelayCore, ...]
    data_penalty: int
    global_signal_tree_max_depth: int
    global_signal_tree_total_edges: int


def _is_in_global_route_grid(coord: CoordXY) -> bool:
    return G_X_MIN <= coord.x <= G_X_MAX and G_Y_MIN <= coord.y <= G_Y_MAX


def _control_offset_for_suffix(
    root: CoordXY, target_coord: CoordXY, suffix: tuple[CoordXY, ...]
) -> CoordZXYOffset | None:
    for offset in candidate_offsets(root, target_coord):
        if route_coord_path(root, offset) == suffix:
            return offset
    return None


def _root_kind(
    root: CoordXY,
    used_core_coords: set[CoordXY],
    available_empty_offline_coords: set[CoordXY],
    available_empty_online_coords: set[CoordXY],
    allow_empty_online: bool,
    empty_online_frame_supported: bool,
) -> RootKind | None:
    if root in used_core_coords:
        return "used"
    if root in available_empty_offline_coords:
        return "empty_offline"
    if (
        allow_empty_online
        and empty_online_frame_supported
        and root in available_empty_online_coords
    ):
        return "empty_online"
    return None


def _classify_relay_cores(
    added_points: list[CoordXY],
    used_core_coords: set[CoordXY],
    available_empty_offline_coords: set[CoordXY],
    available_empty_online_coords: set[CoordXY],
    allow_empty_online: bool,
    empty_online_frame_supported: bool,
) -> tuple[EmptyRelayCore, ...] | None:
    relay_cores: list[EmptyRelayCore] = []
    for coord in added_points:
        if coord in used_core_coords:
            continue
        if coord in available_empty_offline_coords:
            relay_cores.append(EmptyRelayCore(coord, "offline"))
            continue
        if (
            allow_empty_online
            and empty_online_frame_supported
            and coord in available_empty_online_coords
        ):
            relay_cores.append(EmptyRelayCore(coord, "online"))
            continue
        return None
    return tuple(relay_cores)


def _producer_choices_by_root_suffix(
    producer: OutputProducer,
) -> dict[tuple[CoordXY, tuple[CoordXY, ...]], _ProducerPathChoice]:
    offsets = candidate_offsets(producer.coord, producer.target_coord)
    if not offsets:
        raise OutputCompletionNoFeasiblePlanError(
            "No legal output DATA route offset from "
            f"({producer.coord.x},{producer.coord.y}) to "
            f"({producer.target_coord.x},{producer.target_coord.y})."
        )

    shortest_len = route_len(offsets[0])
    choices: dict[tuple[CoordXY, tuple[CoordXY, ...]], _ProducerPathChoice] = {}
    for offset in offsets:
        path = route_coord_path(producer.coord, offset)
        for index, point in enumerate(path[:-1]):
            if not _is_in_global_route_grid(point):
                continue
            suffix = path[index:]
            key = (point, suffix)
            path_len = route_len(offset)
            choice = _ProducerPathChoice(
                offset, path, path_len, path_len - shortest_len
            )
            current = choices.get(key)
            if current is None or (choice.path_len, choice.offset.to_tuple()) < (
                current.path_len,
                current.offset.to_tuple(),
            ):
                choices[key] = choice

    return choices


def _candidate_root_suffixes(
    producers: tuple[OutputProducer, ...],
) -> dict[tuple[CoordXY, tuple[CoordXY, ...]], tuple[_ProducerPathChoice, ...]]:
    producer_choices = [
        _producer_choices_by_root_suffix(producer) for producer in producers
    ]
    common_keys = set(producer_choices[0])
    for choices in producer_choices[1:]:
        common_keys &= set(choices)

    def sort_key(
        root_suffix: tuple[CoordXY, tuple[CoordXY, ...]],
    ) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
        root, suffix = root_suffix
        return (root.x, root.y), tuple((coord.x, coord.y) for coord in suffix)

    return {
        key: tuple(choices[key] for choices in producer_choices)
        for key in sorted(common_keys, key=sort_key)
    }


def select_output_completion_plan(
    producers: list[OutputProducer],
    used_core_coords: set[CoordXY],
    available_empty_offline_coords: set[CoordXY],
    available_empty_online_coords: set[CoordXY] | None = None,
    *,
    allow_empty_online: bool = False,
    empty_online_frame_supported: bool = False,
) -> OutputCompletionPlan:
    """Choose output DATA routes and a completion root with a shared suffix.

    The selected global signal root lies on every output DATA route. The
    complete/control route from that root to CPU is exactly the common suffix of
    those DATA routes, so a complete frame cannot overtake already-issued DATA
    frames after the convergence point.
    """
    available_empty_online_coords = available_empty_online_coords or set()
    diagnostics: list[str] = []
    if allow_empty_online and not empty_online_frame_supported:
        diagnostics.append(
            "empty online relay candidates disabled: backendv2 online frame1 "
            "export is not wired."
        )

    if not producers:
        root = min(
            used_core_coords,
            key=lambda coord: (coord.x, coord.y),
            default=TEST_DEST_CORE,
        )
        control_offset = candidate_offsets(root, TEST_DEST_CORE)[0]
        return OutputCompletionPlan(
            output_routes=(),
            control_offset=control_offset,
            ingress_side=terminal_route_side(control_offset),
            global_signal_root=root,
            root_kind="used" if root in used_core_coords else "empty_offline",
            relay_cores=(),
            data_penalty=0,
            global_signal_tree_max_depth=0,
            global_signal_tree_total_edges=0,
            complete_path_len=route_len(control_offset),
            diagnostics=tuple(diagnostics),
        )

    producers_tuple = tuple(
        OutputProducer(
            producer.coord, producer.target_coord, max(int(producer.weight), 1)
        )
        for producer in producers
    )
    candidates: list[_RootSuffixCandidate] = []
    for (root, suffix), choices in _candidate_root_suffixes(producers_tuple).items():
        if root == TEST_DEST_CORE:
            continue
        control_offset = _control_offset_for_suffix(
            root, producers_tuple[0].target_coord, suffix
        )
        if control_offset is None:
            continue
        root_kind = _root_kind(
            root,
            used_core_coords,
            available_empty_offline_coords,
            available_empty_online_coords,
            allow_empty_online,
            empty_online_frame_supported,
        )
        if root_kind is None:
            continue

        global_signal_tree = solve_global_signal_tree(
            list(used_core_coords), root=root, verbose=False
        )
        relay_cores = _classify_relay_cores(
            global_signal_tree.added,
            used_core_coords,
            available_empty_offline_coords,
            available_empty_online_coords,
            allow_empty_online,
            empty_online_frame_supported,
        )
        if relay_cores is None:
            continue

        data_penalty = sum(
            producer.weight * choice.penalty
            for producer, choice in zip(producers_tuple, choices, strict=True)
        )
        candidates.append(
            _RootSuffixCandidate(
                root=root,
                suffix=suffix,
                choices=choices,
                control_offset=control_offset,
                root_kind=root_kind,
                relay_cores=relay_cores,
                data_penalty=data_penalty,
                global_signal_tree_max_depth=global_signal_tree.max_depth,
                global_signal_tree_total_edges=global_signal_tree.total_edges,
            )
        )

    if not candidates:
        raise OutputCompletionNoFeasiblePlanError(
            "Cannot find a global signal root on a common output DATA suffix. "
            "producers="
            f"{[(p.coord.x, p.coord.y, p.weight) for p in producers_tuple]}, "
            f"used={sorted((c.x, c.y) for c in used_core_coords)}, "
            "empty_online_enabled="
            f"{allow_empty_online and empty_online_frame_supported}"
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.data_penalty,
            1 if candidate.root_kind != "used" else 0,
            1 if candidate.root_kind == "empty_online" else 0,
            candidate.global_signal_tree_max_depth,
            candidate.global_signal_tree_total_edges,
            len(candidate.suffix) - 1,
            candidate.root.x,
            candidate.root.y,
        ),
    )

    if selected.data_penalty > 0:
        diagnostics.append(
            "selected minimum-penalty fallback because no zero-penalty common "
            f"DATA suffix root was feasible; data_penalty={selected.data_penalty}."
        )

    output_routes = tuple(
        OutputRouteDecision(producer.coord, producer.target_coord, choice.offset)
        for producer, choice in zip(producers_tuple, selected.choices, strict=True)
    )
    return OutputCompletionPlan(
        output_routes=output_routes,
        control_offset=selected.control_offset,
        ingress_side=terminal_route_side(selected.control_offset),
        global_signal_root=selected.root,
        root_kind=selected.root_kind,
        relay_cores=selected.relay_cores,
        data_penalty=selected.data_penalty,
        global_signal_tree_max_depth=selected.global_signal_tree_max_depth,
        global_signal_tree_total_edges=selected.global_signal_tree_total_edges,
        complete_path_len=len(selected.suffix) - 1,
        diagnostics=tuple(diagnostics),
    )
