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

# ---------------- 主求解函数 ----------------
def route_solve(
    areas=[1],
    next_area_id={},
    io_target=(0, 0),
    input_area_ids=[],
    output_area_ids=[],
    *,
    feasibility_only: bool = False,
    max_time_in_seconds: float = 120.0,
    max_memory_in_mb: int = 4096,
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
    x = [model.new_bool_var(f"x_{i}") for i in range(N)]

    cell_to_xs = [[] for _ in range(len(HIVE_LIST))]
    area_to_xs = [[] for _ in range(num_areas)]
    area_to_placements = [[] for _ in range(num_areas)]
    for i, p in enumerate(placements):
        area_to_xs[p["area_id"]].append(x[i])
        area_to_placements[p["area_id"]].append(i)
        for h_idx in placement_cells[i]:
            cell_to_xs[h_idx].append(x[i])

    # each hive cell can be covered at most once
    for covering_xs in cell_to_xs:
        if len(covering_xs) > 1:
            model.add_at_most_one(covering_xs)

    # each area is placed exactly once
    for area_xs in area_to_xs:
        model.add_exactly_one(area_xs)

    # ---- 中心坐标 ----
    cx = [model.new_int_var(X_START, X_END - 1, f"cx_{a}") for a in range(num_areas)]
    cy = [model.new_int_var(Y_START, Y_END - 1, f"cy_{a}") for a in range(num_areas)]

    # ---- 新增: 每个 area 的 shape bbox 与占位 bbox ----
    # shape bbox 取值范围: shape 偏移 ∈ [0, 8] 这里用 [-8, 8] 安全
    sxmin = [model.new_int_var(-8, 8, f"sxmin_{a}") for a in range(num_areas)]
    sxmax = [model.new_int_var(-8, 8, f"sxmax_{a}") for a in range(num_areas)]
    symin = [model.new_int_var(-8, 8, f"symin_{a}") for a in range(num_areas)]
    symax = [model.new_int_var(-8, 8, f"symax_{a}") for a in range(num_areas)]

    bx_min = [
        model.new_int_var(X_START, X_END - 1, f"bxmin_{a}") for a in range(num_areas)
    ]
    bx_max = [
        model.new_int_var(X_START, X_END - 1, f"bxmax_{a}") for a in range(num_areas)
    ]
    by_min = [
        model.new_int_var(Y_START, Y_END - 1, f"bymin_{a}") for a in range(num_areas)
    ]
    by_max = [
        model.new_int_var(Y_START, Y_END - 1, f"bymax_{a}") for a in range(num_areas)
    ]

    # Exactly-one placement lets us bind selected attributes with one weighted sum.
    for a, area_placements in enumerate(area_to_placements):
        model.add(
            cx[a] == sum(placements[i]["center_r"] * x[i] for i in area_placements)
        )
        model.add(
            cy[a] == sum(placements[i]["center_c"] * x[i] for i in area_placements)
        )
        model.add(
            sxmin[a]
            == sum(placements[i]["shape_bbox"][0] * x[i] for i in area_placements)
        )
        model.add(
            sxmax[a]
            == sum(placements[i]["shape_bbox"][1] * x[i] for i in area_placements)
        )
        model.add(
            symin[a]
            == sum(placements[i]["shape_bbox"][2] * x[i] for i in area_placements)
        )
        model.add(
            symax[a]
            == sum(placements[i]["shape_bbox"][3] * x[i] for i in area_placements)
        )
        model.add(
            bx_min[a]
            == sum(placements[i]["bbox_x_min"] * x[i] for i in area_placements)
        )
        model.add(
            bx_max[a]
            == sum(placements[i]["bbox_x_max"] * x[i] for i in area_placements)
        )
        model.add(
            by_min[a]
            == sum(placements[i]["bbox_y_min"] * x[i] for i in area_placements)
        )
        model.add(
            by_max[a]
            == sum(placements[i]["bbox_y_max"] * x[i] for i in area_placements)
        )

    # ---- 新增约束 A: 通信约束 i -> j ----
    # area i 中每个点按 area j 的 shape 展开都必须落在 G 内
    for i, succs in next_area_id.items():
        for j in succs:
            model.add(bx_min[i] + sxmin[j] >= G_X_MIN)
            model.add(bx_max[i] + sxmax[j] <= G_X_MAX)
            model.add(by_min[i] + symin[j] >= G_Y_MIN)
            model.add(by_max[i] + symax[j] <= G_Y_MAX)

    # ---- 新增约束 B: 源 area (无入边) 以 (0,0) 为基点展开必须在 G 内 ----
    has_incoming = {j for succs in next_area_id.values() for j in succs}
    sources = [a for a in range(num_areas) if a not in has_incoming]

    for a in sources:
        model.add(0 + sxmin[a] >= G_X_MIN)
        model.add(0 + sxmax[a] <= G_X_MAX)
        model.add(0 + symin[a] >= G_Y_MIN)
        model.add(0 + symax[a] <= G_Y_MAX)

    # ---- 距离建模 ----
    total_distance = 0
    for i, next_list in next_area_id.items():
        for j in next_list:
            dx = model.new_int_var(0, X_END - 1, f"dx_{i}_{j}")
            dy = model.new_int_var(0, Y_END - 1, f"dy_{i}_{j}")
            model.add_abs_equality(dx, cx[j] - cx[i])
            model.add_abs_equality(dy, cy[j] - cy[i])
            total_distance += dx + dy

    for area_id in itertools.chain(input_area_ids, output_area_ids):
        total_distance += sum(
            (
                abs(placements[i]["center_r"] - io_target[0])
                + abs(placements[i]["center_c"] - io_target[1])
            )
            * x[i]
            for i in area_to_placements[area_id]
        )

    model.minimize(total_distance)

    solver = cp_model.CpSolver()

    num_cpu_threads = os.cpu_count() or 2
    solver.parameters.num_search_workers = min(8, max(1, num_cpu_threads // 2))
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    solver.parameters.max_memory_in_mb = max_memory_in_mb
    if feasibility_only:
        # feasibility_only keeps the objective but skips proving optimality.
        solver.parameters.stop_after_first_solution = True
    # solver.parameters.log_search_progress = True
    status = solver.solve(model)

    copy_configs: list[AERPacketZXYCopy] = []
    coords: list[list[CoordXY]] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            if solver.value(x[i]) == 1:
                p = placements[i]
                copy_config = COPY_CONFIGS[p["actual_area"]][p["shape_id"]]
                copy_configs.append(copy_config)
                coords.append([CoordXY(r, c) for r, c in p["absolute_coords"]])
    else:
        print("No solution found or solver timed out.")
        print(f"Solver status: {solver.StatusName(status)}")
        raise RuntimeError("No solution found or solver timed out.")

    return copy_configs, coords
