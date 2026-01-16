import itertools
import typing
from collections import defaultdict, deque
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any, TypeVar

from paibox.exceptions import GraphHasCycleError, GraphNotSupportedError

from .group import MergedGroup
from .placement import CoreBlock
from .sub_utils import sub_node_overlap
from .types import EdgeAttr, NodeDegree, NodeName, NodeType

if typing.TYPE_CHECKING:
    from .routing import RoutingGroup

_NT = TypeVar("_NT", CoreBlock, NodeName, "RoutingGroup", MergedGroup)
_T = TypeVar("_T")


def toposort(directed_edges: Mapping[_NT, Iterable[_NT]]) -> list[_NT]:
    """
    Topological sort algorithm by Kahn [1]_.

    Complexity is O(nodes + vertices).

    Parameters
    ----------
    edges : dict
        dict of the form {a: {b, c}} where b and c depend on a

    Returns
    -------
    An ordered list of nodes that satisfy the dependencies of ``edges``

    Examples
    --------

    .. testcode::

       from nengo.utils.graphs import toposort

       print(toposort({1: {2, 3}, 2: {3}, 3: set()}))

    .. testoutput::

       [1, 2, 3]

    Notes
    -----
    Closely follows the wikipedia page [2]_.

    References
    ----------
    .. [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
       Communications of the ACM
    .. [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_edges(directed_edges)
    vertices = set(
        v for v in directed_edges if v not in incoming_edges or not incoming_edges[v]
    )
    ordered = []

    while vertices:
        n = vertices.pop()
        ordered.append(n)

        if n not in directed_edges:
            raise RuntimeError("Graph changed during iteration")

        for m in directed_edges[n]:
            try:
                incoming_edges[m].remove(n)
            except KeyError as e:
                raise RuntimeError("Graph changed during iteration") from e

            if not incoming_edges[m]:
                vertices.add(m)

    if any(incoming_edges.get(v, None) for v in directed_edges):
        raise GraphHasCycleError("the graph with cycles is not supported.")

    return ordered


def _toposort(
    directed_edges: Mapping[_NT, Iterable[_NT]],
) -> Generator[list[_NT], Any, None]:
    incoming_edges = reverse_edges(directed_edges)
    zero_indegree = [
        v for v in directed_edges if v not in incoming_edges or not incoming_edges[v]
    ]

    while zero_indegree:
        this_gene = zero_indegree
        zero_indegree = []

        for n in this_gene:
            if n not in directed_edges:
                raise RuntimeError("Graph changed during iteration")

            for m in directed_edges[n]:
                try:
                    incoming_edges[m].remove(n)
                except KeyError as e:
                    raise RuntimeError("Graph changed during iteration") from e

                if not incoming_edges[m]:
                    zero_indegree.append(m)

        yield this_gene

    if any(incoming_edges.get(v, None) for v in directed_edges):
        raise GraphHasCycleError("the graph with cycles is not supported.")


def iter_toposort(
    directed_edges: Mapping[_NT, Iterable[_NT]],
) -> Generator[_NT, Any, None]:
    for generation in _toposort(directed_edges):
        yield from generation


def reverse_edges(directed_edges: Mapping[_NT, Iterable[_NT]]) -> dict[_NT, list[_NT]]:
    """
    Reverses direction of dependence dict.

    Parameters
    ----------
    directed_edges : dict
        dict of the form {a: {b, c}, b: set(), c: set()} where b and c depend
        on a.

    Returns
    -------
    dict of the form {a: set(), b: {a}, c: {a}} where b and c depend on a.
    """
    reversed = {k: list() for k in directed_edges}
    for key in directed_edges:
        for val in directed_edges[key]:
            if key in reversed[val]:
                raise ValueError(f"edge {key} -> {val} is repeated.")

            reversed[val].append(key)

    return reversed


def reverse_edges2(
    directed_edges: Mapping[_NT, Mapping[_NT, _T]],
) -> dict[_NT, dict[_NT, _T]]:
    reversed = {k: dict() for k in directed_edges}
    for key in directed_edges:
        for val, edge in directed_edges[key].items():
            if key in reversed[val]:
                raise ValueError(f"edge {key} -> {val} is repeated.")

            reversed[val][key] = edge

    return reversed


def get_node_degrees(
    succ_edges: Mapping[_NT, Sequence[_NT] | Mapping[_NT, Any]],
) -> dict[_NT, NodeDegree]:
    degree = defaultdict(NodeDegree)
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)

    for node, succ_nodes in succ_edges.items():
        out_degrees[node] = len(succ_nodes)

        for succ_node in succ_nodes:
            in_degrees[succ_node] += 1

    for node in succ_edges:
        degree[node] = NodeDegree(in_degrees[node], out_degrees[node])

    return degree


def degree_check(
    degree_of_nodes: Mapping[_NT, NodeDegree], succ_dg: Mapping[_NT, Sequence[_NT]]
) -> None:
    """Filter out such network structure, which is currently not supported."""
    for node in filter(lambda node: degree_of_nodes[node].out_degree > 1, succ_dg):
        for succ_node in succ_dg[node]:
            if degree_of_nodes[succ_node].in_degree > 1:
                _node_repr = (
                    succ_node.name
                    if isinstance(succ_node, CoreBlock)
                    else str(succ_node)
                )
                raise GraphNotSupportedError(
                    f"If out-degree of a node is greater than 1, the in-degree of its sucessors must be 1. "
                    f"However, in-degree of {_node_repr} is {degree_of_nodes[succ_node].in_degree}."
                )


def find_cycles(directed_edges: Mapping[_NT, Sequence[_NT]]) -> list[list[_NT]]:
    """Find all cycles in a directed graph where [0] is the source of the cycle.

    Return an empty list if there is no cycle.
    """
    cycles: list[list[_NT]] = []
    visited: set[_NT] = set()
    stack: list[_NT] = []
    stack_set: set[_NT] = set()

    def dfs(node: _NT) -> None:
        if node in stack_set:
            cycle_start_index = stack.index(node)
            cycles.append(stack[cycle_start_index:])
            return None

        if node in visited:
            return None

        visited.add(node)
        stack.append(node)
        stack_set.add(node)

        for neighbor in directed_edges.get(node, []):
            dfs(neighbor)

        stack.pop()
        stack_set.remove(node)

    for node in directed_edges:
        if node not in visited:
            dfs(node)

    return cycles


def merge_cycles(merged_sgrps: list[MergedGroup]) -> list[MergedGroup]:
    """Detects cycles among merged successor groups & merges them into a minimal set of     \
        disjoint groups.

    Args:
        merged_sgrps (list[MergedSuccGroup]): A list of already merged successor groups to  \
            be analyzed for cycles.

    Returns:
        out (list[MergedSuccGroup]): A new list of merged successor groups with detected    \
            cycles resolved.
    """
    succ_merged_grps: dict[MergedGroup, list[MergedGroup]] = defaultdict(list)

    for cur_m, next_m in itertools.combinations(merged_sgrps, 2):
        # (cur_m, (m2, m3, ...)), (m2, (m3, m4, ...)), ...
        if not cur_m.nodes.isdisjoint(next_m.inputs):
            succ_merged_grps[cur_m].append(next_m)

        if not next_m.nodes.isdisjoint(cur_m.inputs):
            succ_merged_grps[next_m].append(cur_m)

    cycles = find_cycles(succ_merged_grps)
    merged_cycles = merge_overlapping_sets(cycles)

    merged: list[MergedGroup] = []
    # remaining = set(merged_sgrps)
    # for mc in merged_cycles:
    #     merged.append(MergedGroup.merge(mc))
    #     remaining.difference_update(mc)

    # merged.extend(remaining)
    # return merged
    all_merged = set()
    for mc in merged_cycles:
        merged_mg = MergedGroup.merge(mc)
        merged.append(merged_mg)
        all_merged.update(mc)
    remaining = [mg for mg in merged_sgrps if mg not in all_merged]
    merged.extend(remaining)
    return merged


def merge_overlapping_sets(sets: Sequence[Sequence[_NT]]) -> list[list[_NT]]:
    """Merges overlapping sets into a minimal set of disjoint sets using Union-Find algorithm.

    Args:
        sets (Sequence[Sequence[_NT]]): A list of lists, each inner list represents a set of    \
            elements.

    Returns:
        out (list[list[_NT]]): A list of lists containing the merged, disjoint sets.
    """
    parent: dict[_NT, _NT] = defaultdict()

    def find(x: _NT) -> _NT:
        if parent[x] != x:
            parent[x] = find(parent[x])

        return parent[x]

    def union(x: _NT, y: _NT) -> None:
        rootx = find(x)
        rooty = find(y)
        if rootx != rooty:
            parent[rooty] = rootx

    for group in sets:
        for elem in group:
            parent[elem] = elem

    for group in sets:
        first_elem = group[0]
        for elem in group[1:]:
            union(first_elem, elem)

    mgrps: dict[_NT, list[_NT]] = defaultdict(list)
    for elem in parent:
        root = find(elem)
        mgrps[root].append(elem)

    return list(mgrps.values())


def prune_disconn_graph(
    succ_dg: Mapping[_NT, Iterable[_NT]],
    start_nodes: Sequence[_NT],
    forward_only: bool = False,
) -> tuple[dict[_NT, Iterable[_NT]], set[_NT]]:
    """Remove all nodes & their associated edges from the computation graph that are not connected to any   \
        of the given start nodes.

    Args:
        succ_dg (dict): The computation graph represented as a dictionary, where keys are nodes & values    \
            are lists of successor nodes. `succ_dg` contains all nodes in the graph.
        start_nodes (list): A collection of start nodes.
        forward_only (bool): If True, prune the predecessors of the start nodes even if they are connected  \
            to them.

    Returns:
        out (dict): The updated computation graph containing only nodes & edges that are connected to at    \
            least one start node.
    """

    def bfs(
        graph: Mapping[_NT, Iterable[_NT]], start_node: _NT, visited: set[_NT]
    ) -> None:
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

    if len(start_nodes) < 1:
        raise ValueError("Start nodes list is empty.")

    for n in start_nodes:
        if n not in succ_dg:
            raise ValueError(f"Start node {n} is not in the computation graph.")

    connected_nodes: set[_NT] = set()
    for n in start_nodes:
        bfs(succ_dg, n, connected_nodes)

    if not forward_only:
        connected_nodes2: set[_NT] = set()
        pred_dg = reverse_edges(succ_dg)
        for n in start_nodes:
            bfs(pred_dg, n, connected_nodes2)

        connected_nodes |= connected_nodes2

    all_nodes = set(succ_dg.keys())
    non_connected_nodes = all_nodes - connected_nodes

    new_succ_dg = {
        node: neighbors
        for node, neighbors in succ_dg.items()
        if node not in non_connected_nodes
    }

    return new_succ_dg, non_connected_nodes


def get_longest_path(
    edges_with_d: dict[_NT, dict[_NT, EdgeAttr]],
    ordered_nodes: list[_NT],
) -> tuple[list[_NT], int]:
    """Get the longest path in the DAG.

    Args:
        - edges_with_d: a dictionary of directed edges with distance.
        - ordered_nodes: nodes in topological sorting order.

    Return:
        A tuple containing the longest path in the graph and its distance.
    """
    distances: dict[_NT, int] = {node: 0 for node in ordered_nodes}
    pred_nodes: dict[_NT, _NT] = dict()

    for node in ordered_nodes:
        for neighbor, edge_attr in edges_with_d[node].items():
            d = edge_attr.distance
            if distances[node] + d > distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = max(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    distance = distances[node]
    path = [node]

    # Construct the longest path by following the predecessors
    while path[-1] in pred_nodes:
        path.append(pred_nodes[path[-1]])

    path.reverse()
    return path, distance


MAX_DISTANCE = 999  # I don't like float('inf')


def get_shortest_path(
    edges_with_d: dict[_NT, dict[_NT, EdgeAttr]],
    ordered_nodes: list[_NT],
    input_nodes: list[_NT],
) -> tuple[list[_NT], int]:
    """Get the shortest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.
        - input_nodes: input nodes.

    Return: the shortest distance in the graph.
    """
    distances: dict[_NT, int] = defaultdict(lambda: MAX_DISTANCE)
    pred_nodes: dict[_NT, _NT] = dict()

    # set initial value for all inputs nodes. If there is no input node,
    # the first node after topological sorting will be used as the starting node.
    if input_nodes:
        for inode in input_nodes:
            distances[inode] = 0
    else:
        distances[ordered_nodes[0]] = 0

    for node in ordered_nodes:
        for neighbor, edge_attr in edges_with_d[node].items():
            d = edge_attr.distance
            if distances[node] + d < distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = min(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    distance = distances[node]
    path = [node]

    # Construct the shortest path by following the predecessors
    while path[-1] in pred_nodes:
        path.append(pred_nodes[path[-1]])

    path.reverse()
    return path, distance


def get_succ_cb_by_node(
    node: NodeType, core_blocks: Sequence[CoreBlock]
) -> list[CoreBlock]:
    return [cb for cb in core_blocks if sub_node_overlap(node, cb.ordered_axons)]


def get_pred_cb_by_succ_cb(
    succ_cb: dict[CoreBlock, list[CoreBlock]],
) -> dict[CoreBlock, list[CoreBlock]]:
    return reverse_edges(succ_cb)


def get_pred_cb_by_node(
    node: NodeType, core_blocks: Sequence[CoreBlock]
) -> list[CoreBlock]:
    return [cb for cb in core_blocks if node in cb.dest]


def get_pred_dg_by_succ_dg(
    succ_dg: dict[NodeName, dict[NodeName, _T]],
) -> dict[NodeName, dict[NodeName, _T]]:
    return reverse_edges2(succ_dg)


def get_pred_nodes_by_succ_dg(
    node: NodeType, succ_dg: dict[NodeName, dict[NodeName, EdgeAttr]]
) -> list[NodeName]:
    pred_nodes = []

    for pred, succ_nodes in succ_dg.items():
        if node in succ_nodes:
            pred_nodes.append(pred)

    return pred_nodes
