from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Set, Tuple, TypeVar

Node = TypeVar("Node")
Edge = TypeVar("Edge")


class Degree(NamedTuple):
    in_degree: int = 0
    out_degree: int = 0


def toposort(edges: Dict[Node, Set[Node]], cycle_strict: bool = True) -> List[Node]:
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
        if cycle_strict:
            raise ValueError("Input graph has cycles.")

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

    for node in in_degrees:
        degree[node] = Degree._make((in_degrees[node], out_degrees[node]))

    return degree


def find_prev_nodes(
    succ_edges: Dict[Node, Dict[Node, Any]], target_node: Node
) -> List[Node]:
    prev = []

    for node, succ_nodes in succ_edges.items():
        if target_node in succ_nodes:
            prev.append(node)

    return prev


def group_edges_proto(
    nodes: List[Node],
    edges: List[Edge],
    succ_edges: Dict[Node, Dict[Node, Edge]],
) -> Tuple[Dict[Node, Degree], List[Set[Edge]]]:
    """Only for varification."""

    def _reverse_edges(
        edges: Dict[Node, Dict[Node, Edge]]
    ) -> Dict[Node, Dict[Node, Edge]]:
        result = {k: dict() for k in edges}
        for key in edges:
            for val in edges[key]:
                result[val].update({key: edges[key][val]})

        return result

    pred_edges = _reverse_edges(succ_edges)

    # Calculate the in- & out-degree of each node.
    degree = defaultdict(Degree)
    gathered: List[Set[Edge]] = []
    edges_set = set(edges)

    for node in nodes:
        degree[node] = Degree(
            len(pred_edges[node]),
            len(succ_edges[node]),
        )

    for node in degree:
        if degree[node].out_degree > 1:
            # Out-degree of node > 1.
            edge_group = list(succ_edges[node].values())

            for succ_node in succ_edges[node]:
                if degree[succ_node].in_degree > 1:
                    edge_group += list(pred_edges[succ_node].values())

            if set(edge_group) not in gathered:
                gathered.append(set(edge_group))
                edges_set.difference_update(edge_group)

        if degree[node].in_degree > 1:
            # In-degree of node > 1.
            edge_group = list(pred_edges[node].values())

            for prev_node in pred_edges[node]:
                if degree[prev_node].out_degree > 1:
                    edge_group += list(succ_edges[prev_node].values())

            if set(edge_group) not in gathered:
                gathered.append(set(edge_group))
                edges_set.difference_update(edge_group)

    for edge_remained in edges_set:
        gathered.append({edge_remained})

    return degree, gathered


def group_edges(
    succ_edges: Dict[Node, Dict[Node, Edge]],
    edges: List[Edge],
    degree: Dict[Node, Degree],
) -> List[Set[Edge]]:
    """Group all edges according to a certain rule.

    Args:
        - succ_edges: a dictionary recording previous nodes and edges.
        - edges: a list of edges.
        - degree: the in/out-degree of nodes.

    Returns:
        - A list of set of grouped edges.
    """
    gathered = []
    edges_set = set(edges)

    for node in succ_edges:
        if degree[node].in_degree > 1:
            edge_group = find_prev_nodes(succ_edges, node)

            if set(edge_group) not in gathered:
                gathered.append(set(edge_group))
                edges_set.difference_update(edge_group)

        if degree[node].out_degree > 1:
            edge_group = list(succ_edges[node].values())

            if set(edge_group) not in gathered:
                gathered.append(set(edge_group))
                edges_set.difference_update(edge_group)

    # Break the topological order.
    for edge_remained in edges_set:
        gathered.append({edge_remained})

    return gathered
