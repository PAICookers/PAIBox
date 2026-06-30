import json
import os
import time

import pytest
from paicorelib.coordinate import CoordXY
from paicorelib.routing_hexa import AERPacket, aer_packet_walk

from paibox.backendv2.route_solver import (
    MAX_AREA,
    OFFLINE_CORE_COORDS,
    is_global_route_coord,
    route_solve,
)

RUN_ROUTE_SOLVER_PERF = "PAIBOX_RUN_ROUTE_SOLVER_PERF"


def _solve(**kwargs):
    return route_solve(max_time_in_seconds=30.0, **kwargs)


def _flatten(coords: list[list[CoordXY]]) -> list[CoordXY]:
    return [coord for area_coords in coords for coord in area_coords]


def _assert_valid_coords(coords: list[list[CoordXY]]) -> None:
    flat = _flatten(coords)
    assert len(flat) == len(set(flat))
    assert all(coord in OFFLINE_CORE_COORDS for coord in flat)


def _assert_valid_successor_expansion(
    coords: list[list[CoordXY]], copy_configs, next_area_id: dict[int, list[int]]
) -> None:
    for src_id, dst_ids in next_area_id.items():
        for dst_id in dst_ids:
            packet = AERPacket(ncopy=copy_configs[dst_id])
            offsets = aer_packet_walk(packet)
            for src_coord in coords[src_id]:
                expanded = [
                    CoordXY(src_coord.x + offset.x, src_coord.y + offset.y)
                    for offset in offsets
                ]
                assert all(is_global_route_coord(coord) for coord in expanded)


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
        io_target=(0, 0),
        feasibility_only=False,
        **{io_kw: [0]},
    )

    assert len(copy_configs) == 1
    _assert_valid_coords(coords)
    assert _center(coords[0]) == (0, 2)


def test_route_solve_raises_when_no_shape_can_cover_area():
    with pytest.raises(RuntimeError, match="No shape found"):
        _solve(areas=[MAX_AREA + 1], next_area_id={})


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
            _, coords = route_solve(
                areas=areas,
                next_area_id=next_area_id,
                feasibility_only=feasibility_only,
                max_time_in_seconds=30.0,
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
