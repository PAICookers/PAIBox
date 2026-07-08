"""CP-SAT placement solver for backendv2 routing groups."""

import itertools
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from ortools.sat.python import cp_model
from paicorelib import (
    AERPacketZXYCopy,
    CoordXY,
    CoordXYLike,
    CoordXYOffset,
    aer_packet_area,
    aer_packet_copy_offsets,
    to_coordxy,
)

from .route_scope import RouteScope, TargetBoard, get_route_scope
from .routing import InputGroup, OutputGroup, RoutingGroup

__all__ = ["RouteSolver", "route_solve"]


@dataclass(frozen=True, slots=True)
class RouteShape:
    """One AER multicast copy shape, represented as offsets from its base.

    Parameters:
        area: Number of offline cores covered by the shape.
        copy_config: Hardware AER packet copy configuration for the shape.
        offsets: Relative core coordinates covered from a selected base core.
    """

    area: int
    copy_config: AERPacketZXYCopy
    offsets: tuple[CoordXYOffset, ...]


@dataclass(frozen=True, slots=True)
class _Placement:
    area_id: int
    shape: RouteShape
    coords: tuple[CoordXY, ...]
    center_x: int
    center_y: int


@cache
def _copy_candidates(
    scope_name: TargetBoard,
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    scope = get_route_scope(scope_name)
    z_limit, x_limit, y_limit = scope.copy_limits
    offline_capacity = len(scope.offline_core_coords)
    candidates = []
    for copy_tuple in itertools.product(
        range(-z_limit, z_limit + 1),
        range(-x_limit, x_limit + 1),
        range(-y_limit, y_limit + 1),
    ):
        area = aer_packet_area(copy_tuple)
        if area <= offline_capacity:
            candidates.append((area, copy_tuple))
    return tuple(sorted(candidates, key=lambda item: (item[0], item[1])))


@cache
def _smallest_shapes_for_area(
    scope_name: TargetBoard, min_area: int
) -> tuple[RouteShape, ...]:
    scope = get_route_scope(scope_name)
    selected_area = None
    shapes: list[RouteShape] = []

    for area, copy_tuple in _copy_candidates(scope_name):
        if area < min_area:
            continue
        if selected_area is not None and area > selected_area:
            break

        offsets = tuple(
            CoordXYOffset(off.x, off.y) for off in aer_packet_copy_offsets(copy_tuple)
        )
        if area != len(offsets) or not _shape_has_placement(scope, offsets):
            continue

        selected_area = area
        copy_config = AERPacketZXYCopy(*copy_tuple)
        shapes.append(RouteShape(area, copy_config.copy(), offsets))

    if selected_area is None:
        raise RuntimeError(f"No shape found for area size>={min_area}.")

    return tuple(sorted(shapes, key=_shape_key))


def _shape_key(shape: RouteShape) -> tuple[int, tuple[int, int, int]]:
    return shape.area, shape.copy_config.to_tuple()


def _shape_has_placement(scope: RouteScope, offsets: tuple[CoordXYOffset, ...]) -> bool:
    """Return whether one AER shape can live wholly on offline compute cores."""
    offline = scope.offline_core_coords
    return any(all(base + off in offline for off in offsets) for base in offline)


class RouteSolver:
    """Assign routing-group multicast shapes to one selected board scope.

    The solver models offline-core exclusivity, one placement per route area,
    and legal successor AER expansion as hard constraints. The optimization
    objective only biases valid placements toward shorter logical edges and
    input/output areas near a selected CPU endpoint.
    """

    def __init__(
        self,
        areas: Sequence[int] | None = None,
        next_area_id: Mapping[int, Sequence[int]] | None = None,
        scope: RouteScope | None = None,
        io_bias_coord: CoordXYLike | None = None,
        input_area_ids: Sequence[int] | None = None,
        output_area_ids: Sequence[int] | None = None,
        *,
        feasibility_only: bool = False,
        max_time_in_seconds: float = 120.0,
        max_memory_in_mb: int = 4096,
    ) -> None:
        """Initialize the route placement solver.

        Args:
            areas: Required offline-core count for each route area. Defaults to
                one area requiring one core.
            next_area_id: Area-level successor graph, keyed by source area id.
            scope: Board route scope that defines offline cores, CPU endpoints,
                and the global route grid. Defaults to the single-chip scope.
            io_bias_coord: Route coordinate or CPU endpoint used by the soft
                input/output placement bias. Defaults to the scope's default
                CPU coordinate.
            input_area_ids: Area ids that receive external input and should be
                softly biased toward `io_bias_coord`.
            output_area_ids: Area ids that produce external output and should be
                softly biased toward `io_bias_coord`.
            feasibility_only: Whether to stop after the first feasible solution
                and skip soft objective construction.
            max_time_in_seconds: CP-SAT solve time limit. Defaults to ``120.0``.
            max_memory_in_mb: CP-SAT memory limit. Defaults to ``4096``.

        Raises:
            ValueError: If area sizes, area ids, or IO bias coordinate are
                invalid for the selected scope.
        """
        self.scope = scope or get_route_scope("single")
        self.areas = tuple((1,) if areas is None else areas)
        self.num_areas = len(self.areas)
        self.next_area_id = {
            int(area_id): tuple(next_ids)
            for area_id, next_ids in (next_area_id or {}).items()
        }
        self.io_bias_coord = self._resolve_io_bias_coord(io_bias_coord)
        self.input_area_ids = tuple(input_area_ids or ())
        self.output_area_ids = tuple(output_area_ids or ())
        self.feasibility_only = feasibility_only
        self.max_time_in_seconds = max_time_in_seconds
        self.max_memory_in_mb = max_memory_in_mb

        self.offline_coords = tuple(
            sorted(self.scope.offline_core_coords, key=lambda coord: (coord.x, coord.y))
        )
        self.offline_index = {
            coord: idx for idx, coord in enumerate(self.offline_coords)
        }
        self.model = cp_model.CpModel()
        self.placements: list[_Placement] = []
        self.x: list[cp_model.IntVar] = []
        self.area_to_placements: list[list[int]] = []
        self.cx: list[cp_model.IntVar] = []
        self.cy: list[cp_model.IntVar] = []

        self._validate_inputs()

    def solve(self) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
        """Solve route placement.

        Returns:
            Pair of multicast copy configs and absolute offline-core
            coordinates, ordered by route area id.

        Raises:
            RuntimeError: If no legal placement candidates exist, CP-SAT cannot
                find a solution, or the solve times out without a solution.
        """
        if self.num_areas == 0:
            return [], []

        self._generate_placements()
        self._build_variables()
        self._add_cell_constraints()
        self._add_uniqueness_constraints()
        if any(self.next_area_id.values()):
            self._add_successor_route_constraints()
        if not self.feasibility_only:
            # Feasibility mode needs only hard hardware legality. Centers and
            # distances are soft placement-quality terms, so building them would
            # enlarge the CP-SAT model without changing feasible/invalid routes.
            self._build_center_variables()
            self._add_placement_binding_constraints()
            self._set_objective()
        return self._run_solver()

    def _resolve_io_bias_coord(self, io_bias_coord: CoordXYLike | None) -> CoordXY:
        if io_bias_coord is None:
            return self.scope.default_cpu.coord
        return to_coordxy(io_bias_coord)

    def _validate_inputs(self) -> None:
        invalid_sizes = [
            (area_id, area) for area_id, area in enumerate(self.areas) if area <= 0
        ]
        if invalid_sizes:
            raise ValueError(f"Area sizes must be positive: {invalid_sizes}.")

        area_ids = set(self.next_area_id)
        area_ids.update(
            next_id for next_ids in self.next_area_id.values() for next_id in next_ids
        )
        area_ids.update(self.input_area_ids)
        area_ids.update(self.output_area_ids)
        invalid_ids = sorted(
            area_id for area_id in area_ids if not 0 <= area_id < self.num_areas
        )
        if invalid_ids:
            raise ValueError(
                f"Area ids out of range for {self.num_areas} areas: {invalid_ids}."
            )

        if not (
            self.scope.is_global_route_coord(self.io_bias_coord)
            or self.io_bias_coord in self.scope.cpu_coords
        ):
            raise ValueError(
                "io_bias_coord must be a route coordinate or CPU endpoint in "
                f"scope {self.scope.name!r}: {self.io_bias_coord}."
            )

    def _generate_placements(self) -> None:
        for area_id, area in enumerate(self.areas):
            shapes = self._find_shapes_for_area(area_id, area)
            for shape in shapes:
                self._try_place_shape(area_id, shape)

    def _find_shapes_for_area(self, area_id: int, area: int) -> tuple[RouteShape, ...]:
        try:
            return _smallest_shapes_for_area(self.scope.name, area)
        except RuntimeError as exc:
            raise RuntimeError(
                f"No shape found for area_id={area_id} size>={area}."
            ) from exc

    def _try_place_shape(self, area_id: int, shape: RouteShape) -> None:
        """Create only placements whose copied cores fit in offline silicon."""
        for base in self.offline_coords:
            coords = tuple(base + offset for offset in shape.offsets)
            if not all(coord in self.scope.offline_core_coords for coord in coords):
                continue

            center_x = round(sum(coord.x for coord in coords) / len(coords))
            center_y = round(sum(coord.y for coord in coords) / len(coords))
            placement = _Placement(area_id, shape, coords, center_x, center_y)
            self.placements.append(placement)

    def _build_variables(self) -> None:
        """Create one Boolean variable per legal placement candidate."""
        self.x = [
            self.model.new_bool_var(f"x_{i}") for i in range(len(self.placements))
        ]
        self.area_to_placements = [[] for _ in range(self.num_areas)]
        for i, placement in enumerate(self.placements):
            self.area_to_placements[placement.area_id].append(i)

        empty_area_ids = [
            area_id
            for area_id, area_placements in enumerate(self.area_to_placements)
            if not area_placements
        ]
        if empty_area_ids:
            raise RuntimeError(
                f"No placement candidates for area ids {empty_area_ids}."
            )

    def _build_center_variables(self) -> None:
        """Create selected-area center coordinates for soft distance costs."""
        x_min, x_max, y_min, y_max = self.scope.offline_bounds
        self.cx = [
            self.model.new_int_var(x_min, x_max, f"cx_{a}")
            for a in range(self.num_areas)
        ]
        self.cy = [
            self.model.new_int_var(y_min, y_max, f"cy_{a}")
            for a in range(self.num_areas)
        ]

    def _io_distance(self, placement: _Placement) -> int:
        return abs(placement.center_x - self.io_bias_coord.x) + abs(
            placement.center_y - self.io_bias_coord.y
        )

    def _add_cell_constraints(self) -> None:
        """Hard constraint: each offline core is used at most once.

        In one compile artifact, an offline core is a unique physical resource:
        its config, neuron state, weights, axons, and output bindings belong to
        one routing group placement. RouteSolver does not model temporal
        multiplexing or context switching, so overlapping routing groups would
        alias hardware state. Multi-chip scopes only enlarge the absolute-core
        coordinate set; uniqueness still applies per absolute offline core.
        """
        cell_to_xs = [[] for _ in range(len(self.offline_coords))]
        for i, placement in enumerate(self.placements):
            for coord in placement.coords:
                cell_to_xs[self.offline_index[coord]].append(self.x[i])

        for covering_xs in cell_to_xs:
            if len(covering_xs) > 1:
                self.model.add_at_most_one(covering_xs)

    def _add_uniqueness_constraints(self) -> None:
        """Hard constraint: every routing area chooses exactly one placement.

        A routing group must have one concrete AER copy shape and one base
        coordinate. Zero placements drops the computation; multiple placements
        duplicate the same logical group on hardware.
        """
        for area_placements in self.area_to_placements:
            self.model.add_exactly_one([self.x[i] for i in area_placements])

    def _add_placement_binding_constraints(self) -> None:
        """Bind center variables to the exactly-one selected placement.

        The one-hot placement variables make the weighted sum equal to the
        chosen placement center. These centers are not hardware resources; they
        are the small proxy used by the soft distance objective.
        """
        for area_id, area_placements in enumerate(self.area_to_placements):
            self.model.add(
                self.cx[area_id]
                == sum(self.placements[i].center_x * self.x[i] for i in area_placements)
            )
            self.model.add(
                self.cy[area_id]
                == sum(self.placements[i].center_y * self.x[i] for i in area_placements)
            )

    def _add_successor_route_constraints(self) -> None:
        """Hard constraint: every producer-to-successor expansion stays routable.

        For edge i -> j, each source core of i may emit an AER packet using j's
        copy shape. Every expanded coordinate must stay inside the selected
        board's global route grid; CPU tiles are endpoints, not pass-through
        route cells.
        """
        for src_id, next_ids in self.next_area_id.items():
            for dst_id in next_ids:
                self._ban_invalid_successor_pairs(src_id, dst_id)

    def _ban_invalid_successor_pairs(self, src_id: int, dst_id: int) -> None:
        """Forbid producer placements incompatible with each consumer shape.

        The hardware legality check depends on the producer coords and the
        consumer AER copy offsets, not on the consumer base coordinate. Grouping
        destination placements by copy shape therefore expresses the same logic
        as pairwise forbidden placement tuples: when shape S is selected, every
        producer placement that would expand outside the global route grid under
        S must be false.

        This deliberately uses a grouped `add_at_most_one` instead of
        `add_forbidden_assignments`. OR-Tools expands negated table constraints
        into a large SAT encoding on medium route cases; the grouped form keeps
        the model Boolean-native and much smaller.
        """
        dst_by_shape: dict[
            tuple[int, int, int], tuple[list[int], tuple[CoordXYOffset, ...]]
        ] = {}
        for dst_i in self.area_to_placements[dst_id]:
            dst_shape = self.placements[dst_i].shape
            shape_key = dst_shape.copy_config.to_tuple()
            dst_placement_ids, _ = dst_by_shape.setdefault(
                shape_key, ([], dst_shape.offsets)
            )
            dst_placement_ids.append(dst_i)

        for dst_placement_ids, dst_offsets in dst_by_shape.values():
            invalid_src_xs = [
                self.x[src_i]
                for src_i in self.area_to_placements[src_id]
                if not self._successor_pair_valid(
                    self.placements[src_i].coords, dst_offsets
                )
            ]
            if invalid_src_xs:
                self.model.add_at_most_one(
                    [
                        *(self.x[dst_i] for dst_i in dst_placement_ids),
                        *invalid_src_xs,
                    ]
                )

    def _successor_pair_valid(
        self, src_coords: tuple[CoordXY, ...], dst_offsets: tuple[CoordXYOffset, ...]
    ) -> bool:
        """Check the hardware route grid for one producer placement and shape."""
        for src_coord in src_coords:
            for offset in dst_offsets:
                routed_coord = src_coord + offset
                if not self.scope.is_global_route_coord(routed_coord):
                    return False
        return True

    def _set_objective(self) -> None:
        """Soft objective: prefer shorter logical edges and CPU-adjacent IO."""
        self.model.minimize(self._build_distance_terms())

    def _build_distance_terms(self):
        """Build Manhattan distance terms over selected placement centers.

        This does not decide hardware legality; hard constraints already did
        that. It only biases among valid placements toward shorter inter-area
        routes and input/output areas near the selected CPU endpoint. It is not
        a hard CPU ingress/egress path-validity constraint.
        """
        total_distance = 0
        x_min, x_max, y_min, y_max = self.scope.offline_bounds
        dx_bound = x_max - x_min
        dy_bound = y_max - y_min
        for src_id, next_ids in self.next_area_id.items():
            for dst_id in next_ids:
                dx = self.model.new_int_var(0, dx_bound, f"dx_{src_id}_{dst_id}")
                dy = self.model.new_int_var(0, dy_bound, f"dy_{src_id}_{dst_id}")
                self.model.add_abs_equality(dx, self.cx[dst_id] - self.cx[src_id])
                self.model.add_abs_equality(dy, self.cy[dst_id] - self.cy[src_id])
                # TODO(route): weight this edge by measured traffic, e.g. the
                # number of source elems targeting dst or their output-bit sum.
                # This should remain objective-only, so feasibility is unchanged.
                total_distance += dx + dy

        for area_id in itertools.chain(self.input_area_ids, self.output_area_ids):
            total_distance += sum(
                self._io_distance(self.placements[i]) * self.x[i]
                for i in self.area_to_placements[area_id]
            )

        return total_distance

    def _run_solver(self) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
        solver = cp_model.CpSolver()
        num_cpu_threads = os.cpu_count() or 2
        solver.parameters.num_workers = min(8, max(1, num_cpu_threads // 2))
        solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        solver.parameters.max_memory_in_mb = self.max_memory_in_mb
        if self.feasibility_only:
            solver.parameters.stop_after_first_solution = True

        status = solver.solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"No solution found. Solver status: {solver.StatusName(status)}."
            )

        copy_configs: list[AERPacketZXYCopy] = []
        coords: list[list[CoordXY]] = []
        for area_placements in self.area_to_placements:
            for placement_id in area_placements:
                if solver.value(self.x[placement_id]) != 1:
                    continue
                placement = self.placements[placement_id]
                copy_configs.append(placement.shape.copy_config.copy())
                coords.append(list(placement.coords))
                break

        return copy_configs, coords


def _route_io_area_ids(
    routing_groups: Sequence[RoutingGroup], input_groups: Sequence[InputGroup]
) -> tuple[list[int], list[int]]:
    """Derive route area ids that should stay near the selected IO endpoint."""
    area_by_rg = {rg: area_id for area_id, rg in enumerate(routing_groups)}

    input_area_ids: set[int] = set()
    for in_grp in input_groups:
        for elem in in_grp.raw_elems:
            dest_group = in_grp.get_dest(elem)
            if isinstance(dest_group, RoutingGroup):
                input_area_ids.add(area_by_rg[dest_group])

    output_area_ids: set[int] = set()
    for rg, area_id in area_by_rg.items():
        for elem in rg.raw_elems:
            if isinstance(rg.get_dest(elem), OutputGroup):
                output_area_ids.add(area_id)
                break

    return sorted(input_area_ids), sorted(output_area_ids)


def route_solve(
    routing_groups: Sequence[RoutingGroup],
    next_area_id: Mapping[int, Sequence[int]],
    input_groups: Sequence[InputGroup],
    scope: RouteScope | None = None,
    io_bias_coord: CoordXYLike | None = None,
    *,
    feasibility_only: bool = False,
    max_time_in_seconds: float = 120.0,
    max_memory_in_mb: int = 4096,
) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
    """Solve route placement for backendv2 routing groups.

    Args:
        routing_groups: Topologically ordered backendv2 routing groups to
            place.
        next_area_id: Area-level successor graph using indexes into
            `routing_groups`.
        input_groups: Backend input groups used to derive input-side route area
            ids for IO placement bias.
        scope: Board route scope for placement and route-grid legality.
            Defaults to the single-chip scope.
        io_bias_coord: Route coordinate or CPU endpoint used by the soft
            input/output placement bias.
        feasibility_only: Whether to stop after the first feasible solution and
            skip soft objective construction.
        max_time_in_seconds: CP-SAT solve time limit. Defaults to ``120.0``.
        max_memory_in_mb: CP-SAT memory limit. Defaults to ``4096``.

    Returns:
        Pair of multicast copy configs and absolute offline-core coordinates,
        ordered to match `routing_groups`.

    Raises:
        ValueError: If required offline cores exceed the selected scope's
            offline capacity.
    """
    resolved_scope = scope or get_route_scope("single")
    areas = [rg.n_core_required for rg in routing_groups]
    if sum(areas) > len(resolved_scope.offline_core_coords):
        raise ValueError(
            f"Total cores needed {sum(areas)} exceeds the "
            f"{len(resolved_scope.offline_core_coords)} offline cores "
            f"available on target_board={resolved_scope.name!r}."
        )

    input_area_ids, output_area_ids = _route_io_area_ids(routing_groups, input_groups)
    solver = RouteSolver(
        areas,
        next_area_id,
        resolved_scope,
        io_bias_coord,
        input_area_ids,
        output_area_ids,
        feasibility_only=feasibility_only,
        max_time_in_seconds=max_time_in_seconds,
        max_memory_in_mb=max_memory_in_mb,
    )
    return solver.solve()
