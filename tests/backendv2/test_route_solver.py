import json
import os
import time

import pytest
from paicorelib import (
    CoordXY,
    CoordXYOffset,
    aer_packet_copy_offsets,
    find_coordxy_shortest_path,
)

from paibox.backendv2 import route_solver as route_solver_module
from paibox.backendv2.coreplacement import OfflineCorePlacementV2
from paibox.backendv2.route_scope import get_route_scope
from paibox.backendv2.route_solver import RouteSolver
from paibox.backendv2.routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup

RUN_ROUTE_SOLVER_PERF = "PAIBOX_RUN_ROUTE_SOLVER_PERF"
SINGLE_SCOPE = get_route_scope("single")


def _solve(**kwargs):
    return RouteSolver(max_time_in_seconds=30.0, **kwargs).solve()


def _capture_route_solver_init(monkeypatch):
    captured = {}

    class FakeRouteSolver:
        def __init__(
            self,
            areas=None,
            next_area_id=None,
            scope=None,
            io_bias_coord=None,
            input_area_ids=None,
            output_area_ids=None,
            **kwargs,
        ):
            captured.update(
                {
                    "areas": areas,
                    "next_area_id": next_area_id,
                    "scope": scope,
                    "io_bias_coord": io_bias_coord,
                    "input_area_ids": input_area_ids,
                    "output_area_ids": output_area_ids,
                    **kwargs,
                }
            )

        def solve(self):
            return [], []

    monkeypatch.setattr(route_solver_module, "RouteSolver", FakeRouteSolver)
    return captured


def _flatten(coords: list[list[CoordXY]]) -> list[CoordXY]:
    return [coord for area_coords in coords for coord in area_coords]


def _assert_valid_coords(coords: list[list[CoordXY]], scope=SINGLE_SCOPE) -> None:
    flat = _flatten(coords)
    assert len(flat) == len(set(flat))
    assert all(coord in scope.offline_core_coords for coord in flat)


def _assert_valid_successor_expansion(
    coords: list[list[CoordXY]],
    copy_configs,
    next_area_id: dict[int, list[int]],
    scope=SINGLE_SCOPE,
) -> None:
    for src_id, dst_ids in next_area_id.items():
        for dst_id in dst_ids:
            offsets = tuple(
                CoordXYOffset(offset.x, offset.y)
                for offset in aer_packet_copy_offsets(copy_configs[dst_id])
            )
            for src_coord in coords[src_id]:
                expanded = [src_coord + offset for offset in offsets]
                assert all(scope.is_global_route_coord(coord) for coord in expanded)


def _center(area_coords: list[CoordXY]) -> tuple[int, int]:
    return (
        round(sum(coord.x for coord in area_coords) / len(area_coords)),
        round(sum(coord.y for coord in area_coords) / len(area_coords)),
    )


def test_route_solve_feasibility_only_returns_solution():
    copy_configs, coords = _solve(
        areas=[1, 1], next_area_id={0: [1]}, feasibility_only=True
    )

    assert len(copy_configs) == 2
    assert len(coords) == 2
    _assert_valid_coords(coords)
    _assert_valid_successor_expansion(coords, copy_configs, {0: [1]})


def test_route_solve_feasibility_only_skips_distance_model():
    solver = RouteSolver(
        areas=[1, 1],
        next_area_id={0: [1]},
        feasibility_only=True,
        max_time_in_seconds=30.0,
    )

    solver.solve()

    variable_names = [variable.name for variable in solver.model.Proto().variables]
    assert not any(
        name.startswith(("cx_", "cy_", "dx_", "dy_")) for name in variable_names
    )
    assert len(solver.model.Proto().objective.vars) == 0


def test_route_solver_successor_constraints_avoid_table_expansion():
    solver = RouteSolver(
        areas=[2, 3, 2],
        next_area_id={0: [1], 1: [2]},
        feasibility_only=True,
        max_time_in_seconds=30.0,
    )

    solver.solve()

    constraints = solver.model.Proto().constraints
    assert not any(constraint.has_table() for constraint in constraints)
    assert any(constraint.has_at_most_one() for constraint in constraints)


def test_route_solver_defaults_to_one_area_solution():
    copy_configs, coords = _solve()

    assert len(copy_configs) == 1
    assert len(coords) == 1
    _assert_valid_coords(coords)


def test_route_solve_optimized_chain_returns_valid_solution():
    next_area_id = {0: [1], 1: [2]}
    copy_configs, coords = _solve(
        areas=[2, 3, 2], next_area_id=next_area_id, feasibility_only=False
    )

    assert len(copy_configs) == 3
    assert len(coords) == 3
    _assert_valid_coords(coords)
    _assert_valid_successor_expansion(coords, copy_configs, next_area_id)


@pytest.mark.parametrize("io_kw", ["input_area_ids", "output_area_ids"])
def test_route_solve_fixed_io_distance_prefers_nearest_offline_core(io_kw):
    copy_configs, coords = _solve(
        areas=[1],
        next_area_id={},
        io_bias_coord=(0, 0),
        feasibility_only=False,
        **{io_kw: [0]},
    )

    assert len(copy_configs) == 1
    _assert_valid_coords(coords)
    assert _center(coords[0]) == (0, 2)


def test_route_solve_raises_when_no_shape_can_cover_area():
    with pytest.raises(RuntimeError, match="No shape found"):
        _solve(areas=[len(SINGLE_SCOPE.offline_core_coords) + 1], next_area_id={})


def test_route_solve_array_2x2_can_bias_to_second_cpu():
    scope = get_route_scope("array_2x2")
    _, coords = _solve(
        areas=[1],
        scope=scope,
        io_bias_coord=scope.cpu_endpoints[1].coord,
        input_area_ids=[0],
    )

    _assert_valid_coords(coords, scope)
    assert _center(coords[0]) == (9, 2)


def test_route_solve_input_placements_have_exact_packet_targets():
    copy_configs, coords = _solve(areas=[38], input_area_ids=[0], feasibility_only=True)

    offset, _ = find_coordxy_shortest_path(coords[0][0], start=CoordXY(0, 0))
    audit = SINGLE_SCOPE.audit_aer_packet(CoordXY(0, 0), offset, copy_configs[0])
    assert audit.valid
    assert set(audit.actual_local) == set(coords[0])


def test_route_solve_derives_io_area_ids(monkeypatch):
    input_a = object()
    input_b = object()
    output_a = object()
    output_b = object()
    input_group = InputGroup([input_a, input_b])
    rg = RoutingGroup([output_a, output_b], [input_a, input_b])
    output_group = OutputGroup([output_a, output_b])

    input_group.dests[input_a] = rg
    input_group.dests[input_b] = rg
    rg.dests[output_a] = output_group
    rg.dests[output_b] = output_group
    rg.core_placements = [OfflineCorePlacementV2()]
    captured = _capture_route_solver_init(monkeypatch)

    route_solver_module.route_solve([rg], {0: []}, [input_group], SINGLE_SCOPE)

    assert captured["areas"] == [1]
    assert captured["next_area_id"] == {0: []}
    assert captured["scope"] is SINGLE_SCOPE
    assert captured["input_area_ids"] == [0]
    assert captured["output_area_ids"] == [0]


def test_route_solve_derives_input_area_through_remap_group(monkeypatch):
    input_elem = object()
    input_group = InputGroup([input_elem])
    rg = RoutingGroup([], [input_elem])
    remap_group = RemapGroup.__new__(RemapGroup)

    def remap_dest(_elem):
        return rg

    remap_group.remap_dest = remap_dest
    input_group.dests[input_elem] = remap_group
    rg.core_placements = [OfflineCorePlacementV2()]
    captured = _capture_route_solver_init(monkeypatch)

    route_solver_module.route_solve([rg], {0: []}, [input_group], SINGLE_SCOPE)

    assert captured["input_area_ids"] == [0]
    assert captured["output_area_ids"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"next_area_id": {2: []}},
        {"next_area_id": {0: [2]}},
        {"input_area_ids": [2]},
        {"output_area_ids": [2]},
    ],
)
def test_route_solve_rejects_out_of_range_area_ids(kwargs):
    with pytest.raises(ValueError, match="Area ids out of range"):
        _solve(areas=[1, 1], **kwargs)


@pytest.mark.perf
def test_route_solver_perf_profile():
    if os.environ.get(RUN_ROUTE_SOLVER_PERF) != "1":
        pytest.skip(f"set {RUN_ROUTE_SOLVER_PERF}=1 to run")

    cases = [
        ("tiny_chain", [1, 1, 1], {0: [1], 1: [2]}),
        ("medium_chain", [2, 3, 2, 4, 3], {0: [1], 1: [2], 2: [3], 3: [4]}),
        ("unroll_like_chain", [1, 2, 1, 2, 1, 2, 1, 2], {i: [i + 1] for i in range(7)}),
    ]
    results = []

    for case_name, areas, next_area_id in cases:
        for feasibility_only in (True, False):
            start = time.perf_counter()
            _, coords = _solve(
                areas=areas,
                next_area_id=next_area_id,
                feasibility_only=feasibility_only,
            )
            results.append(
                {
                    "case": case_name,
                    "areas_count": len(areas),
                    "total_area": sum(areas),
                    "feasibility_only": feasibility_only,
                    "seconds": time.perf_counter() - start,
                    "coords_count": len(coords),
                    "total_cells": len(_flatten(coords)),
                }
            )

    print(json.dumps(results, indent=2, sort_keys=True))
