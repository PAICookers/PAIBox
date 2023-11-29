from collections import defaultdict
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
)

from paibox.exceptions import NotSupportedError

Node = TypeVar("Node")
Edge = TypeVar("Edge")


class NodeCharacter(Enum):
    """Charactor of a node in the directed graph."""

    MEMBER = auto()
    """As a member layer."""

    INPUT = auto()
    """As an input node."""

    OUTPUT = auto()
    """As an output node."""


class GraphInfo(TypedDict, total=False):
    input: Dict[str, Any]
    output: Dict[str, Any]
    members: Dict[str, Any]
    extras: Dict[str, Any]


class Degree(NamedTuple):
    in_degree: int = 0
    out_degree: int = 0


def toposort(directed_edges: Mapping[Node, Iterable[Node]]) -> List[Node]:
    """
    Topological sort algorithm by Kahn [1]_.

    Complexity is O(nodes + vertices).

    Parameters
    ----------
    edges : dict
        Dict of the form {a: {b, c}} where b and c depend on a

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
        for m in directed_edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                vertices.add(m)

    if any(incoming_edges.get(v, None) for v in directed_edges):
        raise NotSupportedError("The graph with cycles is not supported yet.")

    return ordered


def reverse_edges(
    directed_edges: Mapping[Node, Iterable[Node]]
) -> Dict[Node, Set[Node]]:
    """
    Reverses direction of dependence dict.

    Parameters
    ----------
    directed_edges : dict
        Dict of the form {a: {b, c}, b: set(), c: set()} where b and c depend
        on a.

    Returns
    -------
    Dict of the form {a: set(), b: {a}, c: {a}} where b and c depend on a.
    """
    reversed = {k: set() for k in directed_edges}
    for key in directed_edges:
        for val in directed_edges[key]:
            reversed[val].add(key)

    return reversed


def get_node_degrees(succ_edges: Dict[Node, Dict[Node, Any]]) -> Dict[Node, Degree]:
    degree = defaultdict(Degree)
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)

    for node, succ_nodes in succ_edges.items():
        out_degrees[node] = len(succ_nodes)

        for succ_node in succ_nodes:
            in_degrees[succ_node] += 1

    for node in succ_edges:
        degree[node] = Degree._make((in_degrees[node], out_degrees[node]))

    return degree


def _find_pred_edges(
    succ_edges: Dict[Node, Dict[Node, Edge]], target_node: Node
) -> Set[Edge]:
    pred = set()

    for succ_node in filter(lambda node: target_node in node, succ_edges.values()):
        pred.add(succ_node[target_node])

    return pred


def group_edges(
    edges: List[Edge],
    succ_edges: Dict[Node, Dict[Node, Edge]],
    degree: Dict[Node, Degree],
    *,
    ordered_nodes: Optional[List[Node]] = None,
) -> List[Set[Edge]]:
    """Group all edges according to a certain rule.

    Args:
        - edges: a list of edges.
        - succ_edges: a dictionary recording previous nodes and edges.
        - degree: the in/out-degree of nodes.
        - ordered_nodes: nodes in topological sorting. Optional.

    Returns:
        - A list of set of grouped edges.
    """
    gathered = []
    edges_set = set(edges)

    if isinstance(ordered_nodes, list):
        # In topological sorting.
        ordered = ordered_nodes
    else:
        # Without sorting.
        ordered = list(succ_edges.keys())

    for node in ordered:
        if degree[node].in_degree > 1:
            edge_group = _find_pred_edges(succ_edges, node)
            edge_group_copy = edge_group.copy()

            for ed in edge_group:
                if ed not in edges_set:
                    edge_group_copy.remove(ed)

            edges_set.difference_update(edge_group_copy)
            gathered.append(edge_group_copy)

        if degree[node].out_degree > 1:
            edge_group = set(succ_edges[node].values())

            if edge_group not in gathered:
                edges_set.difference_update(edge_group)
                gathered.append(edge_group)

        elif degree[node].out_degree > 0:
            succ_node = list(succ_edges[node].keys())[0]
            # Check the in-degree of the only following node.
            if degree[succ_node].in_degree == 1:
                gathered.append({succ_edges[node][succ_node]})
        else:
            # out-degree = 0, do nothing.
            continue

    return gathered


def get_longest_path(
    edges_with_d: Dict[Node, Dict[Node, int]], ordered_nodes: List[Node]
) -> Tuple[List[Node], int]:
    """Get the longest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.

    Return: the longest distance in the graph.
    """
    distances: Dict[Node, int] = defaultdict(int)  # init value = 0
    pred_nodes: Dict[Node, Optional[Node]] = defaultdict()

    for node in ordered_nodes:
        for neighbor in edges_with_d[node]:
            d = edges_with_d[node][neighbor]
            if distances[node] + d > distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = max(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    # Add the distance of last node to outside(1)
    distance = distances[node] + 1

    path = [node]
    while node := pred_nodes.get(node, ()):
        path.append(node)

    # Reverse the path and return
    return path[::-1], distance


MAX_DISTANCE = 999  # I don't like float('inf')


def get_shortest_path(
    edges_with_d: Dict[Node, Dict[Node, int]],
    ordered_nodes: List[Node],
    input_nodes: List[Node],
) -> Tuple[List[Node], int]:
    """Get the shortest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.
        - input_nodes: input nodes.

    Return: the shortest distance in the graph.
    """
    distances: Dict[Node, int] = defaultdict(lambda: MAX_DISTANCE)
    pred_nodes: Dict[Node, Optional[Node]] = defaultdict()

    # Set initial value for all inputs nodes.
    for inode in input_nodes:
        distances[inode] = 0

    for node in ordered_nodes:
        for neighbor in edges_with_d[node]:
            d = edges_with_d[node][neighbor]
            if distances[node] + d < distances[neighbor]:
                distances[neighbor] = distances[node] + d
                pred_nodes[neighbor] = node

    # When there are more than one output nodes
    # with same distance, choose the first one.
    node = min(
        filter(lambda node: len(edges_with_d[node]) == 0, distances),
        key=lambda node: distances.get(node, 0),
    )

    # Add the distance of last node to outside(1)
    distance = distances[node] + 1

    path = [node]
    while node := pred_nodes.get(node, ()):
        path.append(node)

    # Reverse the path and return
    return path[::-1], distance
