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

# ---------------- 网格定义 ----------------
# HIVE: area 实际可放置的格子范围
X_START, X_END = 0, 9
Y_START, Y_END = 2, 9

# G: 数据包合法可达的全局网格 (用于新约束)
G_X_MIN, G_X_MAX = 0, 8  # x ∈ [0, 8]
G_Y_MIN, G_Y_MAX = 0, 8  # y ∈ [0, 8]

HIVE = set()
for i in range(X_START, X_END):
    for j in range(Y_START, Y_END):
        HIVE.add((i, j))

HIVE_LIST = list(HIVE)
HIVE_INDEX = {h: i for i, h in enumerate(HIVE_LIST)}

# CPU is outside the configurable-core pools. The row bands below match the
# hardware split used by backendv2 planning: y >= 2 are offline cores, while
# y in {0, 1} (except the CPU) are online cores.
CPU_COORD = CoordXY(0, 0)
OFFLINE_CORE_COORDS = frozenset(CoordXY(x, y) for x, y in HIVE)
ONLINE_CORE_COORDS = frozenset(
    CoordXY(x, y)
    for x in range(G_X_MIN, G_X_MAX + 1)
    for y in range(G_Y_MIN, min(G_Y_MAX, 1) + 1)
    if (x, y) != (CPU_COORD.x, CPU_COORD.y)
)


def is_global_route_coord(coord: CoordXY) -> bool:
    return G_X_MIN <= coord.x <= G_X_MAX and G_Y_MIN <= coord.y <= G_Y_MAX


def is_offline_core_coord(coord: CoordXY) -> bool:
    return coord in OFFLINE_CORE_COORDS


def is_online_core_coord(coord: CoordXY) -> bool:
    return coord in ONLINE_CORE_COORDS


def is_configurable_thread_core_coord(coord: CoordXY) -> bool:
    return is_offline_core_coord(coord) or is_online_core_coord(coord)


# ---------------- Shape 枚举 ----------------
def get_shapes_by_area():
    """
    枚举所有合法 AER 复制形状，并按面积分桶。
    额外返回每个 shape 的相对偏移包围盒 (sxmin, sxmax, symin, symax)。
    """
    shapes = defaultdict(list)
    copy_configs = defaultdict(list)
    shape_bboxes = defaultdict(list)

    for coord in itertools.product(range(0, 5), range(0, 5), range(0, 5)):
        n = aer_packet_area(coord)
        copy_config = AERPacketZXYCopy(*coord)
        copy_configs[n].append(copy_config.copy())
        packet = AERPacket(ncopy=copy_config)
        covered = aer_packet_walk(packet)
        n_covered = len(covered)

        assert n == n_covered
        shapes[n].append(covered)

        xs = [s.x for s in covered]
        ys = [s.y for s in covered]
        shape_bboxes[n].append((min(xs), max(xs), min(ys), max(ys)))

    return shapes, copy_configs, shape_bboxes


SHAPES_BY_AREA, COPY_CONFIGS, SHAPE_BBOXES = get_shapes_by_area()
MAX_AREA = max(SHAPES_BY_AREA.keys())


# ---------------- 结果打印 ----------------
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
        print(f"  Area {p['area_id']} Shape {p['shape_id']}: {p['absolute_coords']}")

    print("\nVisualization (Numbers = Area ID, . = empty):")
    grid = [["." for _ in range(Y_END)] for _ in range(X_END)]
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


# ---------------- 主求解函数 ----------------
def route_solve(
    areas=[1],
    next_area_id={},
    io_target=(0, 0),
    input_area_ids=[],
    output_area_ids=[],
):
    num_areas = len(areas)

    # ---- 生成 placement ----
    placements = []
    placement_cells = []

    for area_id, area in enumerate(areas):
        # 找到 ≥ 需求面积的最小可用面积
        shapes = []
        bboxes = []
        chosen_area_size = None
        for selected_area in range(area, MAX_AREA + 1):
            if selected_area in SHAPES_BY_AREA:
                shapes = SHAPES_BY_AREA[selected_area]
                bboxes = SHAPE_BBOXES[selected_area]
                chosen_area_size = selected_area
                break
        if chosen_area_size is None:
            raise RuntimeError(f"No shape found for area_id={area_id} size>={area}")

        for shape_id, shape in enumerate(shapes):
            sb = bboxes[shape_id]  # (sxmin, sxmax, symin, symax)
            for h in HIVE:
                shifted = [(h[0] + s.x, h[1] + s.y) for s in shape]
                if all(c in HIVE for c in shifted):
                    xs = [c[0] for c in shifted]
                    ys = [c[1] for c in shifted]
                    placement_dict = {
                        "actual_area": chosen_area_size,
                        "area_id": area_id,
                        "shape_id": shape_id,
                        "cells": [HIVE_INDEX[c] for c in shifted],
                        "area": len(shifted),
                        "absolute_coords": shifted,
                        "center_r": round(sum(xs) / len(shifted)),
                        "center_c": round(sum(ys) / len(shifted)),
                        "bbox_x_min": min(xs),
                        "bbox_x_max": max(xs),
                        "bbox_y_min": min(ys),
                        "bbox_y_max": max(ys),
                        "shape_bbox": sb,
                    }
                    placements.append(placement_dict)
                    placement_cells.append(placement_dict["cells"])

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

    # ---- 中心坐标 ----
    cx = [model.NewIntVar(X_START, X_END - 1, f"cx_{a}") for a in range(num_areas)]
    cy = [model.NewIntVar(Y_START, Y_END - 1, f"cy_{a}") for a in range(num_areas)]

    # ---- 新增: 每个 area 的 shape bbox 与占位 bbox ----
    # shape bbox 取值范围: shape 偏移 ∈ [0, 8] 这里用 [-8, 8] 安全
    sxmin = [model.NewIntVar(-8, 8, f"sxmin_{a}") for a in range(num_areas)]
    sxmax = [model.NewIntVar(-8, 8, f"sxmax_{a}") for a in range(num_areas)]
    symin = [model.NewIntVar(-8, 8, f"symin_{a}") for a in range(num_areas)]
    symax = [model.NewIntVar(-8, 8, f"symax_{a}") for a in range(num_areas)]

    bx_min = [
        model.NewIntVar(X_START, X_END - 1, f"bxmin_{a}") for a in range(num_areas)
    ]
    bx_max = [
        model.NewIntVar(X_START, X_END - 1, f"bxmax_{a}") for a in range(num_areas)
    ]
    by_min = [
        model.NewIntVar(Y_START, Y_END - 1, f"bymin_{a}") for a in range(num_areas)
    ]
    by_max = [
        model.NewIntVar(Y_START, Y_END - 1, f"bymax_{a}") for a in range(num_areas)
    ]

    # 绑定: 选中的 placement 决定 area 的 bbox 和中心
    for i, p in enumerate(placements):
        a = p["area_id"]
        sb = p["shape_bbox"]
        model.Add(cx[a] == p["center_r"]).OnlyEnforceIf(x[i])
        model.Add(cy[a] == p["center_c"]).OnlyEnforceIf(x[i])
        model.Add(sxmin[a] == sb[0]).OnlyEnforceIf(x[i])
        model.Add(sxmax[a] == sb[1]).OnlyEnforceIf(x[i])
        model.Add(symin[a] == sb[2]).OnlyEnforceIf(x[i])
        model.Add(symax[a] == sb[3]).OnlyEnforceIf(x[i])
        model.Add(bx_min[a] == p["bbox_x_min"]).OnlyEnforceIf(x[i])
        model.Add(bx_max[a] == p["bbox_x_max"]).OnlyEnforceIf(x[i])
        model.Add(by_min[a] == p["bbox_y_min"]).OnlyEnforceIf(x[i])
        model.Add(by_max[a] == p["bbox_y_max"]).OnlyEnforceIf(x[i])

    # ---- 新增约束 A: 通信约束 i -> j ----
    # area i 中每个点按 area j 的 shape 展开都必须落在 G 内
    for i, succs in next_area_id.items():
        for j in succs:
            model.Add(bx_min[i] + sxmin[j] >= G_X_MIN)
            model.Add(bx_max[i] + sxmax[j] <= G_X_MAX)
            model.Add(by_min[i] + symin[j] >= G_Y_MIN)
            model.Add(by_max[i] + symax[j] <= G_Y_MAX)

    # ---- 新增约束 B: 源 area (无入边) 以 (0,0) 为基点展开必须在 G 内 ----
    has_incoming = {j for succs in next_area_id.values() for j in succs}
    sources = [a for a in range(num_areas) if a not in has_incoming]

    for a in sources:
        model.Add(0 + sxmin[a] >= G_X_MIN)
        model.Add(0 + sxmax[a] <= G_X_MAX)
        model.Add(0 + symin[a] >= G_Y_MIN)
        model.Add(0 + symax[a] <= G_Y_MAX)

    # ---- 距离建模 ----
    dx, dy = {}, {}
    total_distance = 0
    for i, next_list in next_area_id.items():
        for j in next_list:
            dx[i, j] = model.NewIntVar(0, X_END - 1, f"dx_{i}_{j}")
            dy[i, j] = model.NewIntVar(0, Y_END - 1, f"dy_{i}_{j}")
            model.AddAbsEquality(dx[i, j], cx[j] - cx[i])
            model.AddAbsEquality(dy[i, j], cy[j] - cy[i])
            total_distance += dx[i, j] + dy[i, j]

    for input_area_id in input_area_ids:
        dxi = model.NewIntVar(0, X_END - 1, f"dx_in_{input_area_id}")
        dyi = model.NewIntVar(0, Y_END - 1, f"dy_in_{input_area_id}")
        model.AddAbsEquality(dxi, cx[input_area_id] - io_target[0])
        model.AddAbsEquality(dyi, cy[input_area_id] - io_target[1])
        total_distance += dxi + dyi

    for output_area_id in output_area_ids:
        dxo = model.NewIntVar(0, X_END - 1, f"dx_out_{output_area_id}")
        dyo = model.NewIntVar(0, Y_END - 1, f"dy_out_{output_area_id}")
        model.AddAbsEquality(dxo, cx[output_area_id] - io_target[0])
        model.AddAbsEquality(dyo, cy[output_area_id] - io_target[1])
        total_distance += dxo + dyo

    # Objective: maximize covered cells and minimize total distance
    max_possible_distance = (X_END + Y_END) * num_areas * num_areas

    weight_for_distance = 1
    weight_for_placement = max_possible_distance + 1

    model.Maximize(
        sum(x[i] for i in range(N)) * weight_for_placement
        - total_distance * weight_for_distance
    )

    solver = cp_model.CpSolver()

    num_cpu_threads = os.cpu_count() or 2
    solver.parameters.num_search_workers = max(1, num_cpu_threads // 2)
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


if __name__ == "__main__":
    route_solve()
