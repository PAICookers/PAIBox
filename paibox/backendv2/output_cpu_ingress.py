from dataclasses import dataclass
from typing import Literal

from paicorelib import CoordXY, CoordZXYOffset, find_coordxy_shortest_path

from .core_config import TEST_DEST_CORE
from .route_solver import G_X_MAX, G_X_MIN

CpuIngressSide = tuple[Literal["x", "y", "xy", "local"], int]


class CpuIngressError(RuntimeError):
    """Base error for output DATA and control-frame CPU-ingress alignment."""


class CpuIngressNoFeasiblePlanError(CpuIngressError):
    """Raised when no output/control targets can share one CPU ingress side."""


@dataclass(frozen=True)
class OutputRouteEndpoint:
    producer_coord: CoordXY
    target_coord: CoordXY


@dataclass(frozen=True)
class OutputCpuIngressPlan:
    output_target: CoordXY | None
    control_target: CoordXY
    ingress_side: CpuIngressSide


def terminal_cpu_ingress_side(offset: CoordZXYOffset) -> CpuIngressSide:
    """Return the destination-side ingress used by a routed ZXY packet.

    The v2.5 packet walker consumes route fields in Z, then X, then Y order.
    The last non-zero leg is therefore the side from which the packet reaches
    its destination port.
    """
    if offset.y != 0:
        return ("y", 1 if offset.y > 0 else -1)
    if offset.x != 0:
        return ("x", 1 if offset.x > 0 else -1)
    if offset.z != 0:
        return ("xy", 1 if offset.z > 0 else -1)
    return ("local", 0)


def _cpu_io_target_candidates(preferred: list[CoordXY]) -> list[CoordXY]:
    """Return deterministic candidate cores on the CPU-facing chip side."""
    candidates: list[CoordXY] = []
    seen: set[tuple[int, int]] = set()

    def add(coord: CoordXY) -> None:
        key = (coord.x, coord.y)
        if key not in seen:
            seen.add(key)
            candidates.append(coord)

    for coord in preferred:
        add(coord)

    # Offline compute cores are placed at y >= 2. The y=0/1 rows are the
    # CPU-facing side available for DATA and control/test-frame collection.
    for y in (0, 1):
        for x in range(G_X_MIN, G_X_MAX + 1):
            add(CoordXY(x, y))

    return candidates


def _route_ingress_sides(
    endpoints: list[OutputRouteEndpoint], override_target: CoordXY | None = None
) -> set[CpuIngressSide]:
    ingress_sides: set[CpuIngressSide] = set()
    for endpoint in endpoints:
        target = (
            override_target if override_target is not None else endpoint.target_coord
        )
        offset, _ = find_coordxy_shortest_path(target, endpoint.producer_coord)
        ingress_sides.add(terminal_cpu_ingress_side(offset))
    return ingress_sides


def _choose_control_target_for_ingress_side(
    root_coord: CoordXY, ingress_side: CpuIngressSide, preferred: list[CoordXY]
) -> CoordXY | None:
    for target in _cpu_io_target_candidates(preferred):
        offset, _ = find_coordxy_shortest_path(target, root_coord)
        if terminal_cpu_ingress_side(offset) == ingress_side:
            return target
    return None


def select_output_cpu_ingress_plan(
    root_coord: CoordXY, endpoints: list[OutputRouteEndpoint]
) -> OutputCpuIngressPlan:
    """Choose output DATA & completion targets with matching CPU ingress.

    The CPU can observe a completion frame before all DATA frames if the two
    streams enter the CPU-facing side from different ports. Prefer keeping the
    existing output target and changing only the completion/test target. Retarget
    output collection only when DATA producers do not already share one ingress.
    """
    if not endpoints:
        return OutputCpuIngressPlan(None, TEST_DEST_CORE, ("local", 0))

    current_targets = [endpoint.target_coord for endpoint in endpoints]
    current_ingress_sides = _route_ingress_sides(endpoints)
    if len(current_ingress_sides) == 1:
        ingress_side = next(iter(current_ingress_sides))
        control_target = _choose_control_target_for_ingress_side(
            root_coord, ingress_side, current_targets
        )
        if control_target is not None:
            return OutputCpuIngressPlan(None, control_target, ingress_side)

    for output_target in _cpu_io_target_candidates(current_targets):
        ingress_sides = _route_ingress_sides(endpoints, override_target=output_target)
        if len(ingress_sides) != 1:
            continue

        ingress_side = next(iter(ingress_sides))
        control_target = _choose_control_target_for_ingress_side(
            root_coord, ingress_side, [output_target]
        )
        if control_target is not None:
            return OutputCpuIngressPlan(output_target, control_target, ingress_side)

    details = []
    for endpoint in endpoints:
        offset, _ = find_coordxy_shortest_path(
            endpoint.target_coord, endpoint.producer_coord
        )
        details.append(
            f"producer=({endpoint.producer_coord.x},{endpoint.producer_coord.y}) "
            f"target=({endpoint.target_coord.x},{endpoint.target_coord.y}) "
            f"offset=({offset.z},{offset.x},{offset.y}) "
            f"ingress_side={terminal_cpu_ingress_side(offset)}"
        )
    raise CpuIngressNoFeasiblePlanError(
        "Cannot find a CPU-side target that aligns output DATA and completion "
        "control frames to the same ingress side:\n" + "\n".join(details)
    )
