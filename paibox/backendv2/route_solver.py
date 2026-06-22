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
X_START, X_END = 0, 9
Y_START, Y_END = 2, 9

G_X_MIN, G_X_MAX = 0, 8
G_Y_MIN, G_Y_MAX = 0, 8

HIVE = {(i, j) for i in range(X_START, X_END) for j in range(Y_START, Y_END)}
HIVE_LIST = list(HIVE)
HIVE_INDEX = {h: i for i, h in enumerate(HIVE_LIST)}


# ---------------- Shape 枚举 ----------------
def _enumerate_shapes():
    """枚举所有合法 AER 复制形状，按面积分桶返回。"""
    shapes = defaultdict(list)
    copy_configs = defaultdict(list)
    shape_bboxes = defaultdict(list)

    for coord in itertools.product(range(0, 5), range(0, 5), range(0, 5)):
        n = aer_packet_area(coord)
        copy_config = AERPacketZXYCopy(*coord)
        copy_configs[n].append(copy_config.copy())
        packet = AERPacket(ncopy=copy_config)
        covered = aer_packet_walk(packet)
        assert n == len(covered)
        shapes[n].append(covered)

        xs = [s.x for s in covered]
        ys = [s.y for s in covered]
        shape_bboxes[n].append((min(xs), max(xs), min(ys), max(ys)))

    return shapes, copy_configs, shape_bboxes


SHAPES_BY_AREA, COPY_CONFIGS, SHAPE_BBOXES = _enumerate_shapes()
MAX_AREA = max(SHAPES_BY_AREA.keys())


class RouteSolver:
    """面向对象的路由求解器，封装 CP-SAT 模型的构建与求解过程。"""

    def __init__(
        self,
        areas: list[int],
        next_area_id: dict[int, list[int]],
        io_target: tuple[int, int] = (0, 0),
        input_area_ids: list[int] | None = None,
        output_area_ids: list[int] | None = None,
    ):
        self.areas = areas
        self.num_areas = len(areas)
        self.next_area_id = next_area_id
        self.io_target = io_target
        self.input_area_ids = input_area_ids or []
        self.output_area_ids = output_area_ids or []

        self.model = cp_model.CpModel()
        self.placements: list[dict] = []
        self.placement_cells: list[list[int]] = []

        # 决策变量 (在 _build_variables 中初始化)
        self.x: list[cp_model.IntVar] = []
        self.y: list[cp_model.IntVar] = []
        self.cx: list[cp_model.IntVar] = []
        self.cy: list[cp_model.IntVar] = []
        self.sxmin: list[cp_model.IntVar] = []
        self.sxmax: list[cp_model.IntVar] = []
        self.symin: list[cp_model.IntVar] = []
        self.symax: list[cp_model.IntVar] = []
        self.bx_min: list[cp_model.IntVar] = []
        self.bx_max: list[cp_model.IntVar] = []
        self.by_min: list[cp_model.IntVar] = []
        self.by_max: list[cp_model.IntVar] = []

    def solve(self) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
        """执行完整求解流程，返回 (copy_configs, coords)。"""
        self._generate_placements()
        self._build_variables()
        self._add_coverage_constraints()
        self._add_uniqueness_constraints()
        self._add_placement_binding_constraints()
        self._add_communication_constraints()
        self._add_source_constraints()
        self._set_objective()
        return self._run_solver()

    # ----------------------------------------------------------------
    # Placement 生成
    # ----------------------------------------------------------------

    def _generate_placements(self):
        """为每个 area 枚举所有可行放置方案。"""
        for area_id, area in enumerate(self.areas):
            chosen_area_size, shapes, bboxes = self._find_shapes_for_area(area_id, area)
            for shape_id, shape in enumerate(shapes):
                sb = bboxes[shape_id]
                self._try_place_shape(area_id, chosen_area_size, shape_id, shape, sb)

    def _find_shapes_for_area(self, area_id: int, area: int) -> tuple[int, list, list]:
        """找到 >= 需求面积的最小可用形状集合。"""
        for selected_area in range(area, MAX_AREA + 1):
            if selected_area in SHAPES_BY_AREA:
                return (
                    selected_area,
                    SHAPES_BY_AREA[selected_area],
                    SHAPE_BBOXES[selected_area],
                )
        raise RuntimeError(f"No shape found for area_id={area_id} size>={area}")

    def _try_place_shape(
        self,
        area_id: int,
        chosen_area_size: int,
        shape_id: int,
        shape: list,
        shape_bbox: tuple,
    ):
        """尝试将形状放置到 HIVE 的每个合法位置。"""
        for h in HIVE:
            shifted = [(h[0] + s.x, h[1] + s.y) for s in shape]
            if not all(c in HIVE for c in shifted):
                continue
            xs = [c[0] for c in shifted]
            ys = [c[1] for c in shifted]
            placement = {
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
                "shape_bbox": shape_bbox,
            }
            self.placements.append(placement)
            self.placement_cells.append(placement["cells"])

    # ----------------------------------------------------------------
    # 变量构建
    # ----------------------------------------------------------------

    def _build_variables(self):
        """创建所有 CP-SAT 决策变量。"""
        N = len(self.placements)
        m = self.model

        self.x = [m.NewBoolVar(f"x_{i}") for i in range(N)]
        self.y = [m.NewBoolVar(f"y_{i}") for i in range(len(HIVE_LIST))]

        na = self.num_areas
        self.cx = [m.NewIntVar(X_START, X_END - 1, f"cx_{a}") for a in range(na)]
        self.cy = [m.NewIntVar(Y_START, Y_END - 1, f"cy_{a}") for a in range(na)]

        self.sxmin = [m.NewIntVar(-8, 8, f"sxmin_{a}") for a in range(na)]
        self.sxmax = [m.NewIntVar(-8, 8, f"sxmax_{a}") for a in range(na)]
        self.symin = [m.NewIntVar(-8, 8, f"symin_{a}") for a in range(na)]
        self.symax = [m.NewIntVar(-8, 8, f"symax_{a}") for a in range(na)]

        self.bx_min = [m.NewIntVar(X_START, X_END - 1, f"bxmin_{a}") for a in range(na)]
        self.bx_max = [m.NewIntVar(X_START, X_END - 1, f"bxmax_{a}") for a in range(na)]
        self.by_min = [m.NewIntVar(Y_START, Y_END - 1, f"bymin_{a}") for a in range(na)]
        self.by_max = [m.NewIntVar(Y_START, Y_END - 1, f"bymax_{a}") for a in range(na)]

    # ----------------------------------------------------------------
    # 约束添加
    # ----------------------------------------------------------------

    def _add_coverage_constraints(self):
        """每个 HIVE 格子最多被覆盖一次，且覆盖状态与 placement 选择一致。"""
        N = len(self.placements)
        for h_idx in range(len(HIVE_LIST)):
            covering = [self.x[i] for i in range(N) if h_idx in self.placement_cells[i]]
            self.model.Add(sum(covering) == self.y[h_idx])
            self.model.Add(self.y[h_idx] <= 1)

    def _add_uniqueness_constraints(self):
        """每个 area 恰好被放置一次。"""
        N = len(self.placements)
        for area_id in range(self.num_areas):
            self.model.Add(
                sum(
                    self.x[i]
                    for i, p in enumerate(self.placements)
                    if p["area_id"] == area_id
                )
                == 1
            )

    def _add_placement_binding_constraints(self):
        """选中的 placement 决定对应 area 的中心坐标和边界框。"""
        for i, p in enumerate(self.placements):
            a = p["area_id"]
            sb = p["shape_bbox"]
            lit = self.x[i]
            self.model.Add(self.cx[a] == p["center_r"]).OnlyEnforceIf(lit)
            self.model.Add(self.cy[a] == p["center_c"]).OnlyEnforceIf(lit)
            self.model.Add(self.sxmin[a] == sb[0]).OnlyEnforceIf(lit)
            self.model.Add(self.sxmax[a] == sb[1]).OnlyEnforceIf(lit)
            self.model.Add(self.symin[a] == sb[2]).OnlyEnforceIf(lit)
            self.model.Add(self.symax[a] == sb[3]).OnlyEnforceIf(lit)
            self.model.Add(self.bx_min[a] == p["bbox_x_min"]).OnlyEnforceIf(lit)
            self.model.Add(self.bx_max[a] == p["bbox_x_max"]).OnlyEnforceIf(lit)
            self.model.Add(self.by_min[a] == p["bbox_y_min"]).OnlyEnforceIf(lit)
            self.model.Add(self.by_max[a] == p["bbox_y_max"]).OnlyEnforceIf(lit)

    def _add_communication_constraints(self):
        """通信约束: area i 中每个点按 area j 的 shape 展开都必须落在全局网格 G 内。"""
        for i, succs in self.next_area_id.items():
            for j in succs:
                self.model.Add(self.bx_min[i] + self.sxmin[j] >= G_X_MIN)
                self.model.Add(self.bx_max[i] + self.sxmax[j] <= G_X_MAX)
                self.model.Add(self.by_min[i] + self.symin[j] >= G_Y_MIN)
                self.model.Add(self.by_max[i] + self.symax[j] <= G_Y_MAX)

    def _add_source_constraints(self):
        """源 area (无入边) 以 (0,0) 为基点展开必须在 G 内。"""
        has_incoming = {j for succs in self.next_area_id.values() for j in succs}
        for a in range(self.num_areas):
            if a in has_incoming:
                continue
            self.model.Add(0 + self.sxmin[a] >= G_X_MIN)
            self.model.Add(0 + self.sxmax[a] <= G_X_MAX)
            self.model.Add(0 + self.symin[a] >= G_Y_MIN)
            self.model.Add(0 + self.symax[a] <= G_Y_MAX)

    # ----------------------------------------------------------------
    # 目标函数
    # ----------------------------------------------------------------

    def _set_objective(self):
        """设置优化目标: 最大化放置数量，最小化总曼哈顿距离。"""
        total_distance = self._build_distance_terms()

        N = len(self.placements)
        max_possible_distance = (X_END + Y_END) * self.num_areas * self.num_areas
        weight_for_distance = 1
        weight_for_placement = max_possible_distance + 1

        self.model.Maximize(
            sum(self.x[i] for i in range(N)) * weight_for_placement
            - total_distance * weight_for_distance
        )

    def _build_distance_terms(self):
        """构建 area 间距离以及 IO 距离的线性表达式。"""
        m = self.model
        total_distance = 0

        for i, next_list in self.next_area_id.items():
            for j in next_list:
                dx = m.NewIntVar(0, X_END - 1, f"dx_{i}_{j}")
                dy = m.NewIntVar(0, Y_END - 1, f"dy_{i}_{j}")
                m.AddAbsEquality(dx, self.cx[j] - self.cx[i])
                m.AddAbsEquality(dy, self.cy[j] - self.cy[i])
                total_distance += dx + dy

        for aid in self.input_area_ids:
            dxi = m.NewIntVar(0, X_END - 1, f"dx_in_{aid}")
            dyi = m.NewIntVar(0, Y_END - 1, f"dy_in_{aid}")
            m.AddAbsEquality(dxi, self.cx[aid] - self.io_target[0])
            m.AddAbsEquality(dyi, self.cy[aid] - self.io_target[1])
            total_distance += dxi + dyi

        for aid in self.output_area_ids:
            dxo = m.NewIntVar(0, X_END - 1, f"dx_out_{aid}")
            dyo = m.NewIntVar(0, Y_END - 1, f"dy_out_{aid}")
            m.AddAbsEquality(dxo, self.cx[aid] - self.io_target[0])
            m.AddAbsEquality(dyo, self.cy[aid] - self.io_target[1])
            total_distance += dxo + dyo

        return total_distance

    # ----------------------------------------------------------------
    # 求解与结果提取
    # ----------------------------------------------------------------

    def _run_solver(self) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
        """执行求解并提取结果。"""
        solver = cp_model.CpSolver()
        num_cpu_threads = os.cpu_count() or 2
        solver.parameters.num_search_workers = max(1, num_cpu_threads // 2)
        solver.parameters.max_time_in_seconds = 120.0

        status = solver.Solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"No solution found or solver timed out. "
                f"Status: {solver.StatusName(status)}"
            )

        copy_configs: list[AERPacketZXYCopy] = []
        coords: list[list[CoordXY]] = []

        for i in range(len(self.placements)):
            if solver.Value(self.x[i]) == 1:
                p = self.placements[i]
                copy_configs.append(COPY_CONFIGS[p["actual_area"]][p["shape_id"]])
                coords.append([CoordXY(r, c) for r, c in p["absolute_coords"]])

        return copy_configs, coords


# ---------------- 兼容性入口 ----------------
def route_solve(
    areas=[1],
    next_area_id={},
    io_target=(0, 0),
    input_area_ids=[],
    output_area_ids=[],
) -> tuple[list[AERPacketZXYCopy], list[list[CoordXY]]]:
    """模块级兼容函数，委托给 RouteSolver 类。"""
    solver = RouteSolver(
        areas=areas,
        next_area_id=next_area_id,
        io_target=io_target,
        input_area_ids=input_area_ids,
        output_area_ids=output_area_ids,
    )
    return solver.solve()


if __name__ == "__main__":
    route_solve()
