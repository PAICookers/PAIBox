import itertools
import os
from collections import defaultdict

from ortools.sat.python import cp_model
from paicorelib.coordinate import CoordXY
from paicorelib.routing_hexa import (
    AERPacket,
    AERPacketZXYCopy,
    aer_packet_area,
    aer_packet_walk,
)

ROW_START = 2
ROW_END = 9
COL_START = 0
COL_END = 9

HIVE = set()
for i in range(ROW_START, ROW_END):
    for j in range(COL_START, COL_END):
        HIVE.add((i, j))

HIVE_LIST = list(HIVE)
HIVE_INDEX = {h: i for i, h in enumerate(HIVE_LIST)}

# CANONICAL_SECTOR = set()
# for i in range(0, 5):
#     for j in range(2, 6):
#         CANONICAL_SECTOR.add((i, j))


def get_shapes_by_area() -> (
    tuple[dict[int, list[list[CoordXY]]], dict[int, list[AERPacketZXYCopy]]]
):
    # Some area may not be covered
    shapes = defaultdict(list)
    copy_configs = defaultdict(list)

    for coord in itertools.product(range(0, 5), range(0, 5), range(0, 5)):
        n = aer_packet_area(coord)
        copy_config = AERPacketZXYCopy(*coord)
        copy_configs[n].append(copy_config.copy())
        packet = AERPacket(ncopy=copy_config)
        covered = aer_packet_walk(packet)
        n_covered = len(covered)

        assert n == n_covered
        shapes[n].append(covered)

    return shapes, copy_configs


def test_get_shapes_by_area():
    shapes, copy_configs = get_shapes_by_area()

    for k, v in shapes.items():
        print(f"Area {k}: {len(v)} shapes")


def print_route_result(
    solver: cp_model.CpSolver,
    N: int,
    num_areas: int,
    placements: list[dict],
    x: list[cp_model.IntVar],
    cx: list[cp_model.IntVar],
    cy: list[cp_model.IntVar],
    dx: dict[tuple[int, int], cp_model.IntVar],
    dy: dict[tuple[int, int], cp_model.IntVar],
    next_area_id: dict[int, list[int]],
):
    used = []
    covered = set()
    grid_map = {}
    placed_area_ids = set()
    for i in range(N):
        if solver.Value(x[i]) == 1:
            used.append(i)
            p = placements[i]
            area_id = p["area_id"]
            placed_area_ids.add(area_id)
            for abs_coord in p["absolute_coords"]:
                grid_map[abs_coord] = str(area_id)
            covered.update(p["cells"])

    print("Covered cells:", len(covered), f"/ {len(HIVE_LIST)}")
    print("Utilization:", len(covered) / len(HIVE_LIST))
    print("Number of areas placed:", len(placed_area_ids), f"/ {num_areas}")
    print("Placed Area IDs:", sorted(list(placed_area_ids)))

    print("\nPlacements used:")
    for i in used:
        p = placements[i]
        abs_coords = p["absolute_coords"]
        print(f"  Area {p['area_id']} Shape {p['shape_id']}: {abs_coords}")

    print("\nVisualization (Numbers = Area ID, . = empty):")
    grid = [["." for _ in range(COL_END)] for _ in range(ROW_END)]
    for (r, c), area_id_str in grid_map.items():
        grid[r][c] = area_id_str
    for r_idx, row in reversed(list(enumerate(grid))):
        print(f"{r_idx:2d}: {''.join(row)}")

    print("\nFinal center coordinates for placed areas:")
    for i in sorted(placed_area_ids):
        print(f"  Area {i}: (r={solver.Value(cx[i])}, c={solver.Value(cy[i])})")

    print("\nCalculated distances between placed adjacent areas:")
    for i, next_list in next_area_id.items():
        for j in next_list:
            dx_val = solver.Value(dx[i, j])
            dy_val = solver.Value(dy[i, j])
            print(
                f"  Area {i} to Area {j}: dx={dx_val}, dy={dy_val}, total={dx_val + dy_val}"
            )


SHAPES_BY_AREA, COPY_CONFIGS = get_shapes_by_area()
MAX_AREA = max(SHAPES_BY_AREA.keys())


def route_solve(
    areas=[1], next_area_id={}, io_target=(0, 0), input_area_ids=[], output_area_ids=[]
):

    num_areas = len(areas)
    placements = []
    placement_cells = []
    placement_area = []

    for area_id, area in enumerate(areas):

        shapes = []
        for selected_area in range(area, MAX_AREA + 1):
            if selected_area in SHAPES_BY_AREA:
                shapes = SHAPES_BY_AREA[selected_area]
                break

        for shape_id, shape in enumerate(shapes):
            for h in HIVE:
                shifted = [
                    (h[0] + s.x, h[1] + s.y) for s in shape
                ]  # (h_row + s_row, h_col + s_col)
                if all(c in HIVE for c in shifted):
                    placement_dict = {
                        "actual_area": selected_area,
                        "area_id": area_id,
                        "shape_id": shape_id,
                        "cells": [HIVE_INDEX[c] for c in shifted],
                        "area": len(shifted),
                        "absolute_coords": shifted,
                        "center_r": round(
                            sum(coord[0] for coord in shifted) / len(shifted)
                        ),
                        "center_c": round(
                            sum(coord[1] for coord in shifted) / len(shifted)
                        ),
                    }
                    placements.append(placement_dict)
                    placement_cells.append(placement_dict["cells"])
                    placement_area.append(placement_dict["area"])

    N = len(placements)

    model = cp_model.CpModel()

    # x present whether each placement is used
    x = [model.NewBoolVar(f"x_{i}") for i in range(N)]

    # y present whether each hive cell is covered
    y = [model.NewBoolVar(f"y_{i}") for i in range(len(HIVE_LIST))]

    # sum up x[i] if placement i covers hive cell h_idx
    for h_idx in range(len(HIVE_LIST)):
        model.Add(
            sum(x[i] for i in range(N) if h_idx in placement_cells[i]) == y[h_idx]
        )

    # each hive cell can be covered at most once
    for h_idx in range(len(HIVE_LIST)):
        model.Add(y[h_idx] <= 1)

    # each area is placed exactly once
    for area_id in range(num_areas):
        model.Add(
            sum(x[i] for i, p in enumerate(placements) if p["area_id"] == area_id) == 1
        )

    # break symmetry: the hive cell is a rectangle, so we can enforce an order on area placements
    # model.Add(sum(y[HIVE_INDEX[h]] for h in CANONICAL_SECTOR) >= 1)

    cx = [model.NewIntVar(ROW_START, ROW_END - 1, f"cx_{i}") for i in range(num_areas)]
    cy = [model.NewIntVar(COL_START, COL_END - 1, f"cy_{i}") for i in range(num_areas)]

    # assign center coordinate for each area based on selected placement
    for i, p in enumerate(placements):
        p = placements[i]
        model.Add(cx[p["area_id"]] == p["center_r"]).OnlyEnforceIf(x[i])
        model.Add(cy[p["area_id"]] == p["center_c"]).OnlyEnforceIf(x[i])

    # dictionary of noc distances between adjacent areas
    dx = {}
    dy = {}

    total_distance = 0
    for i, next_list in next_area_id.items():
        for j in next_list:
            dx[i, j] = model.NewIntVar(ROW_START, ROW_END - 1, f"dx_{i}_{j}")
            dy[i, j] = model.NewIntVar(COL_START, COL_END - 1, f"dy_{i}_{j}")
            model.AddAbsEquality(dx[i, j], cx[j] - cx[i])
            model.AddAbsEquality(dy[i, j], cy[j] - cy[i])
            total_distance += dx[i, j] + dy[i, j]

    for input_area_id in input_area_ids:
        dx_input = model.NewIntVar(ROW_START, ROW_END - 1, f"dx_input_{input_area_id}")
        dy_input = model.NewIntVar(COL_START, COL_END - 1, f"dy_input_{input_area_id}")
        model.AddAbsEquality(dx_input, cx[input_area_id] - io_target[0])
        model.AddAbsEquality(dy_input, cy[input_area_id] - io_target[1])
        total_distance += dx_input + dy_input

    for output_area_id in output_area_ids:
        dx_output = model.NewIntVar(
            ROW_START, ROW_END - 1, f"dx_output_{output_area_id}"
        )
        dy_output = model.NewIntVar(
            COL_START, COL_END - 1, f"dy_output_{output_area_id}"
        )
        model.AddAbsEquality(dx_output, cx[output_area_id] - io_target[0])
        model.AddAbsEquality(dy_output, cy[output_area_id] - io_target[1])
        total_distance += dx_output + dy_output

    # Objective: maximize covered cells and minimize total distance
    max_possible_distance = (ROW_END + COL_END) * num_areas * num_areas

    weight_for_distance = 1
    weight_for_placement = max_possible_distance + 1

    model.Maximize(
        sum(x[i] for i in range(N)) * weight_for_placement
        - total_distance * weight_for_distance
    )

    solver = cp_model.CpSolver()

    num_cpu_threads = os.cpu_count() or 2
    solver.parameters.num_search_workers = num_cpu_threads // 2
    solver.parameters.max_time_in_seconds = 120.0
    # solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    copy_configs: list[AERPacketZXYCopy] = []
    coords: list[list[CoordXY]] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            if solver.Value(x[i]) == 1:
                p = placements[i]
                copy_config = COPY_CONFIGS[p["actual_area"]][p["shape_id"]]
                copy_configs.append(copy_config)
                coords.append([CoordXY(r, c) for r, c in p["absolute_coords"]])
    else:
        print("No solution found or solver timed out.")
        print(f"Solver status: {solver.StatusName(status)}")
        raise RuntimeError("No solution found or solver timed out.")

    return copy_configs, coords


route_solve()
