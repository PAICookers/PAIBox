from dataclasses import dataclass

from paicorelib import CoordXY, CoordZXYOffset

from .output_route_offsets import RouteSide


@dataclass(frozen=True)
class OutputRouteEndpoint:
    producer_coord: CoordXY
    target_coord: CoordXY


@dataclass(frozen=True)
class OutputRouteDecision:
    producer_coord: CoordXY
    target_coord: CoordXY
    offset: CoordZXYOffset


@dataclass(frozen=True)
class OutputCpuIngressPlan:
    output_routes: tuple[OutputRouteDecision, ...]
    control_offset: CoordZXYOffset
    ingress_side: RouteSide

    def output_route_offsets(self) -> dict[tuple[CoordXY, CoordXY], CoordZXYOffset]:
        return {
            (route.producer_coord, route.target_coord): route.offset
            for route in self.output_routes
        }
