from paicorelib import CoordXY, CoordZXYOffset, find_coordxy_shortest_path

from .core_config import TEST_DEST_CORE
from .output_route_offsets import (
    RouteSide,
    candidate_offsets,
    route_len,
    terminal_route_side,
)
from .output_routes import (
    OutputCpuIngressPlan,
    OutputRouteDecision,
    OutputRouteEndpoint,
)


class CpuIngressError(RuntimeError):
    """Base error for output DATA and control-frame CPU-ingress alignment."""


class CpuIngressNoFeasiblePlanError(CpuIngressError):
    """Raised when no output/control offsets can share one CPU ingress side."""


def terminal_data_ingress_side(offset: CoordZXYOffset) -> RouteSide:
    """Return the CPU ingress side used by output DATA packets."""
    return terminal_route_side(offset)


def terminal_control_ingress_side(offset: CoordZXYOffset) -> RouteSide:
    """Return the CPU ingress side used by completion/test packets."""
    return terminal_route_side(offset)


def _offsets_by_ingress(
    start_coord: CoordXY, target_coord: CoordXY
) -> dict[RouteSide, CoordZXYOffset]:
    by_side: dict[RouteSide, CoordZXYOffset] = {}
    for offset in candidate_offsets(start_coord, target_coord):
        by_side.setdefault(terminal_route_side(offset), offset)
    return by_side


def _route_detail(
    producer_coord: CoordXY, target_coord: CoordXY, offset: CoordZXYOffset
) -> str:
    return (
        f"producer=({producer_coord.x},{producer_coord.y}) "
        f"target=({target_coord.x},{target_coord.y}) "
        f"offset=({offset.z},{offset.x},{offset.y}) "
        f"ingress_side={terminal_route_side(offset)}"
    )


def select_output_cpu_ingress_plan(
    root_coord: CoordXY, endpoints: list[OutputRouteEndpoint]
) -> OutputCpuIngressPlan:
    """Choose route offsets that align DATA and completion at CPU ingress."""
    if not endpoints:
        control_offset = find_coordxy_shortest_path(TEST_DEST_CORE, root_coord)[0]
        return OutputCpuIngressPlan(
            (), control_offset, terminal_control_ingress_side(control_offset)
        )

    control_offsets = _offsets_by_ingress(root_coord, TEST_DEST_CORE)
    output_offsets = [
        _offsets_by_ingress(endpoint.producer_coord, endpoint.target_coord)
        for endpoint in endpoints
    ]

    common_sides = set(control_offsets)
    for offsets in output_offsets:
        common_sides &= set(offsets)

    if common_sides:
        ingress_side = min(
            common_sides,
            key=lambda side: (
                sum(route_len(offsets[side]) for offsets in output_offsets)
                + route_len(control_offsets[side]),
                str(side),
            ),
        )
        return OutputCpuIngressPlan(
            tuple(
                OutputRouteDecision(
                    endpoint.producer_coord,
                    endpoint.target_coord,
                    offsets[ingress_side],
                )
                for endpoint, offsets in zip(endpoints, output_offsets, strict=True)
            ),
            control_offsets[ingress_side],
            ingress_side,
        )

    details: list[str] = []
    for endpoint, offsets in zip(endpoints, output_offsets, strict=True):
        for offset in offsets.values():
            details.append(
                _route_detail(endpoint.producer_coord, endpoint.target_coord, offset)
            )
    for offset in control_offsets.values():
        details.append(_route_detail(root_coord, TEST_DEST_CORE, offset))
    raise CpuIngressNoFeasiblePlanError(
        "Cannot find fixed-CPU route offsets that align output DATA and "
        "completion control frames to the same ingress side:\n" + "\n".join(details)
    )
