from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from paicorelib import (
    CoordXY,
    CoordXYUnitVec,
    CoordZXYOffset,
    find_coordxy_shortest_path,
    global_signal_direction_mask,
    global_signal_direction_name,
    global_signal_directions,
    global_signal_opposite_direction,
)

from .coreplacement import (
    CorePlacement,
    EmptyOfflineCorePlacementV2,
    EmptyOnlineCorePlacementV2,
)

T = TypeVar("T", int, CoordXY)
EmptyRelayCoreKind = Literal["offline", "online"]

GlobalSignalEdge = tuple[CoordXY, CoordXYUnitVec]
GlobalSignalAdjacency = dict[CoordXY, list[GlobalSignalEdge]]
GlobalSignalUnitVecMap = dict[CoordXY, list[CoordXYUnitVec]]


@dataclass(frozen=True)
class GlobalSignalTree:
    root: CoordXY
    order: list[CoordXY]
    added: list[CoordXY]
    send_directions: GlobalSignalUnitVecMap
    max_depth: int
    total_edges: int


class DSU(Generic[T]):
    def __init__(self, items: Iterable[T]) -> None:
        self.parent: dict[T, T] = {item: item for item in items}

    def find(self, x: T) -> T:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: T, b: T) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def hex_steps(a: CoordXY, b: CoordXY) -> int:
    diff = b - a
    # Direction set is (+xy, +x, +y) and its reverse directions.
    return max(abs(diff.x), abs(diff.y), abs(diff.x - diff.y))


def neighbor_coord(coord: CoordXY, direction: CoordXYUnitVec) -> CoordXY:
    return coord + direction.value


def path_coords(a: CoordXY, b: CoordXY) -> list[CoordXY]:
    steps = hex_steps(a, b)
    if steps == 0:
        return [a]
    pts = [a]
    current = a
    for _ in range(steps):
        remaining_steps = hex_steps(current, b)
        chosen = None
        for direction in global_signal_directions():
            neighbor = neighbor_coord(current, direction)
            if hex_steps(neighbor, b) == remaining_steps - 1:
                chosen = direction
                break
        if chosen is None:
            raise RuntimeError(f"No progress from {current} towards {b}")
        current = neighbor_coord(current, chosen)
        pts.append(current)
    return pts


def bfs_distances(
    start: CoordXY, adj: GlobalSignalAdjacency
) -> tuple[dict[CoordXY, int], int]:
    dist: dict[CoordXY, int] = {start: 0}
    queue = deque([start])
    ecc = 0
    while queue:
        u = queue.popleft()
        for v, _ in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                if dist[v] > ecc:
                    ecc = dist[v]
                queue.append(v)
    return dist, ecc


def solve_global_signal_tree(
    raw_points: list[CoordXY], root: CoordXY | None = None, verbose: bool = False
) -> GlobalSignalTree:
    # 1) connected components
    points = set(raw_points)
    if root is not None:
        points.add(root)

    dsu = DSU(points)
    for p in points:
        for direction in global_signal_directions():
            q = neighbor_coord(p, direction)
            if q in points:
                dsu.union(p, q)
    comps = defaultdict(list[CoordXY])
    for p in points:
        comps[dsu.find(p)].append(p)
    comp_list: list[list[CoordXY]] = list(comps.values())
    if verbose:
        print(f"components({len(comp_list)}):")
        for comp in comp_list:
            print(f"  {comp}")

    # 2) connect components with a minimum spanning tree over shortest bridges
    added: list[CoordXY] = []
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
        for _, i, j, pa, pb in edges:
            if comp_dsu.union(i, j):
                mids = path_coords(pa, pb)[1:-1]
                for m in mids:
                    if m not in points:
                        points.add(m)
                        added.append(m)

    # 3) adjacency table. Keep empty entries for isolated single-core trees.
    adj: defaultdict[CoordXY, list[GlobalSignalEdge]] = defaultdict(list)
    for p in points:
        adj[p]
        for direction in global_signal_directions():
            q = neighbor_coord(p, direction)
            if q in points:
                adj[p].append((q, direction))

    # 4) choose root. If planner supplied one, honor it.
    if root is None:
        best_ecc = None
        start = None
        for p in points:
            _, ecc = bfs_distances(p, adj)
            key = (ecc, p.x + p.y, p.x, p.y)
            if best_ecc is None or key < best_ecc:
                best_ecc = key
                start = p
        assert start is not None and best_ecc is not None, "no points to broadcast"
        max_depth = best_ecc[0]
    else:
        start = root
        _, max_depth = bfs_distances(start, adj)
    if verbose:
        print(f"BFS start: {start}, eccentricity (tree depth): {max_depth}")

    # 5) BFS broadcast tree from the selected root
    visited = {start}
    send_directions: defaultdict[CoordXY, list[CoordXYUnitVec]] = defaultdict(list)
    order = [start]

    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, d in adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
                send_directions[u].append(d)
                queue.append(v)

    return GlobalSignalTree(
        start, order, added, dict(send_directions), max_depth, max(len(order) - 1, 0)
    )


def print_solution(
    order: list[CoordXY], added: list[CoordXY], moves: GlobalSignalUnitVecMap
) -> None:
    print("visiting order:", order)
    print("added empty cores:", added)
    print("send directions:")
    for p, ds in moves.items():
        print(f"  {p} -> {[global_signal_direction_name(d) + f':{d}' for d in ds]}")


def set_global_signal(
    coreplacements: list[CorePlacement],
    global_signal_root: CoordXY | None = None,
    relay_core_kinds: Mapping[CoordXY, EmptyRelayCoreKind] | None = None,
    cpu_coord: CoordXY = CoordXY(0, 0),
    verbose: bool = False,
) -> tuple[list[CorePlacement], dict[int, CoordZXYOffset]]:
    """Configure global-signal routes for one backendv2 thread.

    Args:
        coreplacements: Existing configured cores that must receive the global
            signal.
        global_signal_root: Optional root coordinate for the global signal tree.
            When omitted, the solver chooses a root from `coreplacements`.
        relay_core_kinds: Empty relay-core kinds keyed by coordinate. Missing
            relay coordinates default to offline empty cores.
        cpu_coord: CPU endpoint coordinate used to encode the global-signal
            start offset. Defaults to ``CoordXY(0, 0)``.
        verbose: Whether to print the selected tree and start route.

    Returns:
        Updated core placements and per-placement global-signal start offsets.
    """
    # Global signal currently covers one thread and one shared weight range.
    points: list[CoordXY] = []
    cp_dict: dict[CoordXY, CorePlacement] = {}
    for cp in coreplacements:
        points.append(cp.coord)
        cp_dict[cp.coord] = cp

    root = global_signal_root
    global_signal_tree = solve_global_signal_tree(points, root, verbose)
    order = global_signal_tree.order
    added = global_signal_tree.added
    send_info = global_signal_tree.send_directions
    relay_core_kinds = relay_core_kinds or {}

    missing_points = list(added)
    if root is not None and root not in cp_dict:
        missing_points.insert(0, root)

    for p in missing_points:
        if p in cp_dict:
            continue
        relay_kind = relay_core_kinds.get(p, "offline")
        if relay_kind == "online":
            empty_cp = EmptyOnlineCorePlacementV2()
        else:
            empty_cp = EmptyOfflineCorePlacementV2()
        empty_cp._coord = p
        cp_dict[p] = empty_cp
        coreplacements.append(empty_cp)
        if verbose:
            print(f"Added {relay_kind} global signal core at {p}")

    receive_info: defaultdict[CoordXY, list[CoordXYUnitVec]] = defaultdict(list)
    for p, send_directions in send_info.items():
        for d in send_directions:
            dest = neighbor_coord(p, d)
            receive_info[dest].append(global_signal_opposite_direction(d))

    if verbose:
        print("\nSend Direction:")
        for p, send_dirs in send_info.items():
            print(
                f"    {p}: "
                f"{[global_signal_direction_name(d) + f':{d}' for d in send_dirs]}"
            )

        print("\nReceive Direction:")
        for p, recv_dirs in receive_info.items():
            print(
                f"    {p}: "
                f"{[global_signal_direction_name(d) + f':{d}' for d in recv_dirs]}"
            )

    added_points = set(added)
    for p, cp in cp_dict.items():
        send_directions = send_info.get(p, [])
        recv_directions = receive_info.get(p, [])
        global_send = global_signal_direction_mask(
            send_directions, include_local=p not in added_points
        )
        global_receive = global_signal_direction_mask(recv_directions)
        cp.auto_core_config.global_send = global_send
        cp.auto_core_config.global_receive = global_receive
        if verbose:
            print(
                f"Core at {p} global_send: {global_send:07b}, "
                f"global_receive: {global_receive:07b}"
            )

    start_coord = order[0]
    start_coord_offset, _ = find_coordxy_shortest_path(start_coord, start=cpu_coord)
    if verbose:
        print(f"Global signal start from {start_coord}")
        print(f"Global signal relative offset: {start_coord_offset}")
    return coreplacements, {0: start_coord_offset}
