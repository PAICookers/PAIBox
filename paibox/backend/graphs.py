import sys
from collections import defaultdict
from dataclasses import dataclass, field
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
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paibox.base import NeuDyn
from paibox.collector import Collector
from paibox.exceptions import BuildError, NotSupportedError
from paibox.network import DynSysGroup
from paibox.projection import InputProj
from paibox.synapses import SynSys

NodeName: TypeAlias = str
EdgeName: TypeAlias = str


class NodePosition(Enum):
    """Charactor of a node in the directed graph."""

    MEMBER = auto()
    """As a member layer."""
    INPUT = auto()
    """As an input node."""
    OUTPUT = auto()
    """As an output node."""


class NodeDegree(NamedTuple):
    in_degree: int = 0
    out_degree: int = 0


class NodeAttr(NamedTuple):
    obj: NeuDyn
    position: NodePosition
    degree: NodeDegree


class EdgeAttr(NamedTuple):
    edge: EdgeName
    distance: int


class GraphInfo(TypedDict, total=False):
    input: Dict[str, Any]
    output: Dict[str, Any]
    members: Dict[str, Any]
    extras: Dict[str, Any]


@dataclass
class PAIGraph:
    """Directed graph of PAIBox. We treat networks as one whole graph. \
        In the graph, synapses are edges while neurons are nodes.
    """

    networks: Tuple[DynSysGroup, ...] = field(default_factory=tuple)
    """All networks are seen as one graph."""
    nodes: Dict[NodeName, NodeAttr] = field(default_factory=dict)
    """General nodes in the graph."""
    edges: Collector = field(default_factory=Collector)
    """General edges in the graph."""

    inodes: Collector = field(default_factory=Collector)
    """Input nodes in the graph."""
    onodes: Collector = field(default_factory=Collector)
    """Output nodes in the graph."""

    ordered_nodes: List[NodeName] = field(default_factory=list)
    """Ordered topologically nodes."""

    succ_dg: Dict[NodeName, Dict[NodeName, EdgeAttr]] = field(default_factory=dict)
    """Successor edges & nodes of every node in the graph."""

    degree_of_nodes: Dict[NodeName, NodeDegree] = field(default_factory=dict)
    """A dictionary of in/out-degree tuple of nodes."""

    """Status options"""
    has_built: bool = field(default=False)

    def clear(self) -> None:
        self.has_built = False

        self.networks = ()
        self.nodes.clear()
        self.edges.clear()
        self.inodes.clear()
        self.onodes.clear()
        self.ordered_nodes.clear()
        self.succ_dg.clear()
        self.degree_of_nodes.clear()

    def build(self, *networks: DynSysGroup, **options) -> None:
        self.clear()

        self.networks = networks

        _nodes = Collector()

        for network in networks:
            sub_nodes = network.nodes(level=1, include_self=False)
            _nodes += sub_nodes.include(InputProj, NeuDyn).unique()
            self.edges += sub_nodes.subset(SynSys).unique()

        # Add all nodes in the graph. DO NOT REMOVE!
        for node in _nodes:
            self.succ_dg[node] = dict()

        for syn in self.edges.values():
            u, v = syn.source.name, syn.dest.name
            self.succ_dg[u][v] = EdgeAttr(edge=syn.name, distance=1)

        self.degree_of_nodes = get_node_degrees(self.succ_dg)

        # `InputProj` nodes are input nodes definitely.
        self.inodes = _nodes.subset(InputProj)

        # By default, nodes with out-degree = 0 are considered as output nodes.
        self.onodes = _nodes.key_on_condition(
            lambda node: self.degree_of_nodes[node].out_degree == 0
        )

        for node in _nodes:
            if node in self.inodes:
                pos = NodePosition.INPUT
            elif node in self.onodes:
                pos = NodePosition.OUTPUT
            else:
                pos = NodePosition.MEMBER

            self.nodes[node] = NodeAttr(
                obj=_nodes[node], position=pos, degree=self.degree_of_nodes[node]
            )

        self.has_built = True

        self._graph_check(**options)

    def _graph_check(self, **options) -> None:
        """Preprocess of the directed graph. Because there are currently    \
            many limitations on the networks that can be processed, checks  \
            are performed at this stage.

        Limitation:
            # For a node with in-degree > 1, the out-degree of all its      \
            #   forward nodes = 1.
            - For a node with out-degree > 1, the in-degree of all its      \
                backward node = 1.
            - Only support the in-degree of backward node of input node is 1.
        """
        # Filter the DG with cycles.
        self.ordered_nodes = toposort(self.succ_dg)

        for node in filter(
            lambda node: self.degree_of_nodes[node].out_degree > 1, self.nodes
        ):
            if any(
                self.degree_of_nodes[succ_node].in_degree > 1
                for succ_node in self.succ_dg[node]
            ):
                raise NotSupportedError(
                    "This structure of network is not supported yet."
                )

        # Only support the in-degree of backward node of input node is 1.
        for inode in self.inodes:
            if any(
                self.degree_of_nodes[succ_node].in_degree > 1
                for succ_node in self.succ_dg[inode]
            ):
                raise NotSupportedError(
                    "Only input nodes are supported as the only input of a node."
                )

    def build_check(self) -> None:
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

    @property
    def graph_name_repr(self) -> str:
        _str = f"Graph_of_{self.networks[0].name}"

        for network in self.networks[1:]:
            _str += f"_and_{network.name}"

        return _str


def toposort(directed_edges: Mapping[NodeName, Iterable[NodeName]]) -> List[NodeName]:
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
    directed_edges: Mapping[NodeName, Iterable[NodeName]]
) -> Dict[NodeName, Set[NodeName]]:
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


def get_node_degrees(
    succ_edges: Mapping[NodeName, Mapping[NodeName, Any]]
) -> Dict[NodeName, NodeDegree]:
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


def _find_pred_edges(
    succ_edges: Dict[NodeName, Dict[NodeName, EdgeAttr]], target_node: NodeName
) -> Set[EdgeName]:
    pred = set()

    for succ_node in filter(lambda node: target_node in node, succ_edges.values()):
        pred.add(succ_node[target_node].edge)

    return pred


def group_edges(
    edges: List[EdgeName],
    succ_edges: Dict[NodeName, Dict[NodeName, EdgeAttr]],
    degree: Dict[NodeName, NodeDegree],
    *,
    ordered_nodes: Optional[List[NodeName]] = None,
) -> List[Set[EdgeName]]:
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
            edge_group = set(e.edge for e in succ_edges[node].values())

            if edge_group not in gathered:
                edges_set.difference_update(edge_group)
                gathered.append(edge_group)

        elif degree[node].out_degree > 0:
            succ_node = list(succ_edges[node].keys())[0]
            # Check the in-degree of the only following node.
            if degree[succ_node].in_degree == 1:
                gathered.append({succ_edges[node][succ_node].edge})
        else:
            # out-degree = 0, do nothing.
            continue

    return gathered


def get_longest_path(
    edges_with_d: Dict[NodeName, Dict[NodeName, EdgeAttr]],
    ordered_nodes: List[NodeName],
) -> Tuple[List[NodeName], int]:
    """Get the longest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.

    Return: the longest distance in the graph.
    """
    distances: Dict[NodeName, int] = defaultdict(int)  # init value = 0
    pred_nodes: Dict[NodeName, Optional[NodeName]] = defaultdict()

    for node in ordered_nodes:
        for neighbor in edges_with_d[node]:
            d = edges_with_d[node][neighbor].distance
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
    edges_with_d: Dict[NodeName, Dict[NodeName, EdgeAttr]],
    ordered_nodes: List[NodeName],
    input_nodes: List[NodeName],
) -> Tuple[List[NodeName], int]:
    """Get the shortest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.
        - input_nodes: input nodes.

    Return: the shortest distance in the graph.
    """
    distances: Dict[NodeName, int] = defaultdict(lambda: MAX_DISTANCE)
    pred_nodes: Dict[NodeName, Optional[NodeName]] = defaultdict()

    # Set initial value for all inputs nodes.
    for inode in input_nodes:
        distances[inode] = 0

    for node in ordered_nodes:
        for neighbor in edges_with_d[node]:
            d = edges_with_d[node][neighbor].distance
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
