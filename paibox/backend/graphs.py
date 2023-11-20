from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Set, TypeVar

from paibox.exceptions import NotSupportedError

Node = TypeVar("Node")
Edge = TypeVar("Edge")


class Degree(NamedTuple):
    in_degree: int = 0
    out_degree: int = 0


def toposort(edges: Dict[Node, Set[Node]], is_strict: bool = True) -> List[Node]:
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
    incoming_edges = reverse_edges(edges)
    vertices = set(v for v in edges if v not in incoming_edges or not incoming_edges[v])
    ordered = []

    while vertices:
        n = vertices.pop()
        ordered.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                vertices.add(m)

    if any(incoming_edges.get(v, None) for v in edges):
        if is_strict:
            raise NotSupportedError("The graph with cycles is not supported yet.")

    return ordered


def reverse_edges(edges: Dict[Node, Set[Node]]) -> Dict[Node, Set[Node]]:
    """
    Reverses direction of dependence dict.

    Parameters
    ----------
    edges : dict
        Dict of the form {a: {b, c}, b: set(), c: set()} where b and c depend
        on a.

    Returns
    -------
    Dict of the form {a: set(), b: {a}, c: {a}} where b and c depend on a.
    """
    result = {k: set() for k in edges}
    for key in edges:
        for val in edges[key]:
            result[val].add(key)

    return result


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


def _find_prev_edges(
    succ_edges: Dict[Node, Dict[Node, Edge]], target_node: Node
) -> Set[Edge]:
    prev = set()

    for succ_nodes in succ_edges.values():
        if target_node in succ_nodes:
            prev.add(succ_nodes[target_node])

    return prev


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

    Returns:
        - A list of set of grouped edges.
    """
    gathered = []
    edges_set = set(edges)

    if isinstance(ordered_nodes, list):
        # In topological sorting
        ordered = ordered_nodes
    else:
        ordered = list(succ_edges.keys())

    for node in ordered:
        if degree[node].in_degree > 1:
            edge_group = _find_prev_edges(succ_edges, node)
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
