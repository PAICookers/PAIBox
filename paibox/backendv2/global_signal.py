from collections import defaultdict

from paicorelib import CoordXY, CoordZXYOffset, find_coordxy_shortest_path

from .coreplacement import CorePlacement, EmptyOfflineCorePlacementV2

DIRS = [(1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
RESERVE_DIRS = {
    (1, 1): (-1, -1),
    (-1, -1): (1, 1),
    (1, 0): (-1, 0),
    (-1, 0): (1, 0),
    (0, 1): (0, -1),
    (0, -1): (0, 1),
}
DIRS_OFFSET = {
    (1, 1): 5,  # +xy
    (-1, -1): 4,  # -xy
    (1, 0): 3,  # +x
    (-1, 0): 2,  # -x
    (0, 1): 1,  # +y
    (0, -1): 0,  # -y
}
DIRS_NAME = {
    (1, 1): "+xy",
    (-1, -1): "-xy",
    (1, 0): "+x",
    (-1, 0): "-x",
    (0, 1): "+y",
    (0, -1): "-y",
}


# ---------- 并查集 ----------
class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


# ---------- 正确的最少步数:方向集 {(1,1),(1,0),(0,1)} 及其反向 ----------
def hex_steps(a: tuple[int, int], b: tuple[int, int]):
    dx, dy = b[0] - a[0], b[1] - a[1]
    # 同号时 (1,1)/(-1,-1) 可同时消耗,取 max;异号时只能分别消耗,取和
    # 统一写法: max(|dx|, |dy|, |dx - dy|)
    return max(abs(dx), abs(dy), abs(dx - dy))


# ---------- 生成 A->B 的最短路径上的整数点(含端点) ----------
def path_points(a: tuple[int, int], b: tuple[int, int]):
    steps = hex_steps(a, b)
    if steps == 0:
        return [a]
    pts = [a]
    cx, cy = a
    bx, by = b
    for _ in range(steps):
        remaining_steps = hex_steps((cx, cy), (bx, by))
        chosen = None
        for ddx, ddy in DIRS:
            nx, ny = cx + ddx, cy + ddy
            if hex_steps((nx, ny), (bx, by)) == remaining_steps - 1:
                chosen = (ddx, ddy)
                break
        if chosen is None:
            raise RuntimeError(f"No progress from {(cx, cy)} towards {b}")
        cx, cy = cx + chosen[0], cy + chosen[1]
        pts.append((cx, cy))
    return pts


# ---------- 主流程 ----------
def solve(
    raw_points: list[tuple[int, int]],
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    dict[tuple[int, int], list[tuple[int, int]]],
]:
    # 1) 连通分量
    points = set(raw_points)
    dsu = DSU(points)
    for p in points:
        for dx, dy in DIRS:
            q = (p[0] + dx, p[1] + dy)
            if q in points:
                dsu.union(p, q)
    comps = defaultdict(list[tuple[int, int]])
    for p in points:
        comps[dsu.find(p)].append(p)
    comp_list: list[list[tuple[int, int]]] = list(comps.values())
    print(f"components({len(comp_list)}):")
    for comp in comp_list:
        print(f"  {comp}")

    # 2) MST 连接各分量,补最少点
    added = []
    if len(comp_list) > 1:
        edges = []
        for i in range(len(comp_list)):
            for j in range(i + 1, len(comp_list)):
                best = None
                for pa in comp_list[i]:
                    for pb in comp_list[j]:
                        c = hex_steps(pa, pb)
                        if best is None or c < best[0]:
                            best = (c, pa, pb)
                if best is None:
                    raise RuntimeError(f"No path between components {i} and {j}")
                edges.append((best[0], i, j, best[1], best[2]))
        edges.sort(key=lambda e: e[0])

        comp_dsu = DSU(range(len(comp_list)))
        for cost, i, j, pa, pb in edges:
            if comp_dsu.union(i, j):
                mids = path_points(pa, pb)[1:-1]
                for m in mids:
                    if m not in points:
                        points.add(m)
                        added.append(m)

    # 3) 邻接表
    adj = defaultdict(list)
    for p in points:
        for dx, dy in DIRS:
            q = (p[0] + dx, p[1] + dy)
            if q in points:
                adj[p].append((q, (dx, dy)))

    # 4) 从 x+y 最小的点出发 DFS,记录方向
    start = min(points, key=lambda p: (p[0] + p[1], p[0], p[1]))
    visited = set()
    move_dirs = defaultdict(list)
    order = []

    # 用显式栈,避免点很多时递归爆栈
    stack = [(start, iter(adj[start]))]
    visited.add(start)
    order.append(start)
    while stack:
        u, it = stack[-1]
        advanced = False
        for v, d in it:
            if v not in visited:
                visited.add(v)
                order.append(v)
                move_dirs[u].append(d)
                stack.append((v, iter(adj[v])))
                advanced = True
                break
        if not advanced:
            stack.pop()

    return order, added, dict(move_dirs)


def print_solution(order, added, moves):
    print("visiting order:", order)
    print("added empty cores:", added)
    print("send dirctions:")
    for p, ds in moves.items():
        print(f"  {p} -> {[DIRS_NAME[d] + f':{d}' for d in ds]}")


def set_global_signal(
    coreplacements: list[CorePlacement],
) -> tuple[list[CorePlacement], dict[int, CoordZXYOffset]]:
    # global signal 目前只包含 weight 地址范围,且所有 coreplacement 共用
    # only one thread now, add support for multiple threads later if needed

    points: list[tuple[int, int]] = []
    cp_dict: dict[tuple[int, int], CorePlacement] = {}
    for cp in coreplacements:
        point = (cp.coord.x, cp.coord.y)
        points.append(point)
        cp_dict[point] = cp
    order, added, send_info = solve(points)
    # print_solution(order, added, send_info)
    for p in added:
        empty_cp = EmptyOfflineCorePlacementV2()
        empty_cp._coord = CoordXY(*p)
        cp_dict[p] = empty_cp
        coreplacements.append(empty_cp)
        print(f"Added global signal core at {p}")

    receive_info = defaultdict(list)

    for p, send_directions in send_info.items():
        for d in send_directions:
            dest = (p[0] + d[0], p[1] + d[1])
            receive_info[dest].append(RESERVE_DIRS[d])

    print("\nSend Direction:")
    for p, send_dirs in send_info.items():
        print(f"    {p}: {[DIRS_NAME[d] + f':{d}' for d in send_dirs]}")

    print("\nReceive Direction:")
    for p, recv_dirs in receive_info.items():
        print(f"    {p}: {[DIRS_NAME[d] + f':{d}' for d in recv_dirs]}")

    for p, cp in cp_dict.items():
        send_directions = send_info.get(p, [])
        recv_directions = receive_info.get(p, [])
        if p in added:
            global_send = 0
        else:
            global_send = 1 << 6  # send to local core
        global_receive = 0
        for d in send_directions:
            global_send |= 1 << DIRS_OFFSET[d]
        for d in recv_directions:
            global_receive |= 1 << DIRS_OFFSET[d]
        cp_dict[p].auto_core_config.global_send = global_send
        cp_dict[p].auto_core_config.global_receive = global_receive
        print(
            f"Core at {p} global_send: {global_send:07b}, global_receive: {global_receive:07b}"
        )
    start_coord = CoordXY(*order[0])
    print(f"Global signal start from {start_coord}")
    start_coord_offset, _ = find_coordxy_shortest_path(start_coord)
    print(f"Global signal relative offset: {start_coord_offset}")
    return coreplacements, {0: start_coord_offset}
