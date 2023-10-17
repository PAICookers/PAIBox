from typing import Dict, FrozenSet, List, Set, Tuple, TypeVar

Node = TypeVar("Node")
Edge = TypeVar("Edge")


def toposort(edges: Dict[Node, Set[Node]]) -> List[Node]:
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


INDEGREE_IDX = 0
OUTDEGREE_IDX = 1 - INDEGREE_IDX


def group_edges(
    succ_edges: Dict[Node, Dict[Node, Edge]],
    pred_edges: Dict[Node, Dict[Node, Edge]],
    degree: Dict[Node, Tuple[int, int]],
) -> Set[FrozenSet[Edge]]:
    gathered: Set[FrozenSet[Edge]] = set()
    edges: Set[Edge] = set()

    for v in succ_edges.values():
        edges.update(k for k in v.values())

    for node in degree:
        if degree[node][OUTDEGREE_IDX] > 1:
            # Out-degree of node > 1.
            succ_nodes = list(succ_edges[node].keys())
            edge_group = set(edge for edge in succ_edges[node].values())

            for succ_node in succ_nodes:
                if degree[succ_node][INDEGREE_IDX] > 1:
                    edge_group.update(syn for syn in pred_edges[succ_node].values())

            gathered.add(frozenset(edge_group))
            edges -= edge_group

        if degree[node][INDEGREE_IDX] > 1:
            # In-degree of node > 1.
            prev_nodes = list(pred_edges[node].keys())
            edge_group = set(edge for edge in pred_edges[node].values())

            for prev_node in prev_nodes:
                if degree[prev_node][OUTDEGREE_IDX] > 1:
                    edge_group.update(syn for syn in succ_edges[prev_node].values())

            gathered.add(frozenset(edge_group))
            edges -= edge_group

    return gathered
