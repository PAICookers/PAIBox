from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)
from paicorelib import HwConfig
from paibox.base import NeuDyn
from paibox.collector import Collector
from paibox.exceptions import BuildError, NotSupportedError
from paibox.network import DynSysGroup
from paibox.projection import InputProj
from paibox.synapses import SynSys

from .constrs import GraphNodeConstrs
from .graphs_types import *
from .placement import CoreBlock
from .routing import RoutingGroup

T = TypeVar("T", CoreBlock, NodeName)


@dataclass
class PAIGraph:
    """Directed graph of PAIBox. We treat networks as one whole graph. \
        In the graph, synapses are edges while neurons are nodes.
    """

    networks: Tuple[DynSysGroup, ...] = field(default_factory=tuple)
    """All networks are seen as one graph."""
    raw_nodes: Collector = field(default_factory=Collector)
    """Raw nodes in the networks."""

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

    # node_constrs: GraphNodeConstrs = field(default_factory=GraphNodeConstrs)

    def clear(self) -> None:
        """Clear the PAIGraph."""
        self.has_built = False

        self.networks = ()
        self.raw_nodes.clear()
        self.nodes.clear()
        self.edges.clear()
        self.inodes.clear()
        self.onodes.clear()
        self.ordered_nodes.clear()
        self.succ_dg.clear()
        self.degree_of_nodes.clear()

        # self.node_constrs.clear()

    def build(
        self,
        *networks: DynSysGroup,
        # bounded_nodes: Sequence[Sequence[NeuDyn]] = (),
        # conflicted_nodes: Dict[NodeName, Sequence[NeuDyn]] = {},
    ) -> None:
        self.clear()
        self.networks = networks

        _nodes = Collector()

        for network in networks:
            sub_nodes = network.nodes(level=1, include_self=False)
            _nodes += sub_nodes.include(InputProj, NeuDyn).unique()
            self.edges += sub_nodes.subset(SynSys).unique()

        self.raw_nodes = _nodes

        # Add all nodes in the graph.
        for node in _nodes:
            self.succ_dg[node] = dict()

        for syn in self.edges.values():
            u, v = syn.source.name, syn.dest.name
            # TODO tick_relative = 1 in default here.
            self.succ_dg[u][v] = EdgeAttr(edge=syn.name, distance=1)

        self.degree_of_nodes = get_node_degrees(self.succ_dg)

        # `InputProj` nodes are input nodes definitely.
        self.inodes = _nodes.subset(InputProj)

        # By default, nodes with out-degree = 0 are considered as output nodes.
        self.onodes = _nodes.key_on_condition(
            lambda node: self.degree_of_nodes[node].out_degree == 0
        )

        # _bounded_nodes_check(bounded_nodes)

        for node in _nodes:
            self.nodes[node] = NodeAttr(
                obj=_nodes[node],
                position=self._node_pos(node),
                degree=self.degree_of_nodes[node],
            )

        self.has_built = True

        self._graph_supported_check()

    def _graph_supported_check(self) -> None:
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

        # Filter the DG with certain structure.
        _degree_check(self.degree_of_nodes, self.succ_dg)

        # Only support the in-degree of backward node of input node is 1.
        for inode in self.inodes:
            if any(
                self.degree_of_nodes[succ_node].in_degree > 1
                for succ_node in self.succ_dg[inode]
            ):
                raise NotSupportedError(
                    "Only input nodes as the only input of a node are supported."
                )

        # Only support output nodes with <= 1152 neurons.
        if any(
            onode.num_out > HwConfig.N_FANIN_PER_DENDRITE_MAX
            for onode in self.onodes.values()
        ):
            raise NotSupportedError(
                f"Only output nodes with no more than {HwConfig.N_FANIN_PER_DENDRITE_MAX}"
                f" neurons are supported."
            )

    def _node_pos(self, node: NodeName) -> NodePosition:
        if node in self.inodes:
            return NodePosition.INPUT
        elif node in self.onodes:
            return NodePosition.OUTPUT
        else:
            return NodePosition.MEMBER

    def build_check(self) -> None:
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

    def group_edges(self) -> List[FrozenSet[EdgeName]]:
        """Group all edges according to a certain rule.

        Args:
            - edges: a list of edges.
            - succ_edges: a dictionary recording previous nodes and edges.
            - degree: the in/out-degree of nodes.
            - ordered_nodes: nodes in topological sorting. Optional.

        Returns:
            - A list of set of grouped edges.
        """
        self.build_check()

        gathered: List[frozenset[EdgeName]] = []
        seen_edges: Set[EdgeName] = set()  # Check if all edges are traversed

        for node in self.ordered_nodes:
            if self.degree_of_nodes[node].in_degree > 1:
                edge_group = self._find_pred_edges(self.succ_dg, node)
                # Get the edges traversed for the first time
                comming_edges = edge_group.difference(seen_edges)

                seen_edges.update(comming_edges)
                gathered.append(frozenset(comming_edges))

            if self.degree_of_nodes[node].out_degree > 1:
                """Consider the constraints to the nodes."""
                succ_edges = [e.edge for e in self.succ_dg[node].values()]
                succ_nodes = [self.raw_nodes[n] for n in self.succ_dg[node]]

                # Get the subgroup of indices, like [[0, 1], [2], [3, 4]]
                idx_of_sg = GraphNodeConstrs.tick_wait_attr_constr(succ_nodes)

                if len(idx_of_sg) > 0:
                    for idx in idx_of_sg:
                        succ_edges_sg = frozenset([succ_edges[i] for i in idx])
                        if succ_edges_sg not in gathered:
                            seen_edges.update(succ_edges_sg)
                            gathered.append(succ_edges_sg)
                        else:
                            # FIXME Will this happen?
                            raise NotSupportedError
                else:
                    succ_edges_sg = frozenset(succ_edges)
                    if succ_edges_sg not in gathered:
                        seen_edges.update(succ_edges_sg)
                        gathered.append(succ_edges_sg)
                    else:
                        # FIXME Will this happen?
                        raise NotSupportedError

            elif self.degree_of_nodes[node].out_degree == 1:
                succ_node = list(self.succ_dg[node].keys())[0]
                # Check the in-degree of the only following node.
                if self.degree_of_nodes[succ_node].in_degree == 1:
                    gathered.append(frozenset({self.succ_dg[node][succ_node].edge}))
                else:
                    # This edge is waiting to be processed when
                    # traversing the following node `succ_node`.
                    pass
            else:
                # out-degree = 0, do nothing.
                continue

        return gathered

    @staticmethod
    def _find_pred_edges(
        succ_edges: Dict[NodeName, Dict[NodeName, EdgeAttr]], target_node: NodeName
    ) -> Set[EdgeName]:
        pred = set()
        for succ_node in filter(lambda node: target_node in node, succ_edges.values()):
            pred.add(succ_node[target_node].edge)

        return pred

    @property
    def graph_name_repr(self) -> str:
        _str = f"Graph_of_{self.networks[0].name}"

        for network in self.networks[1:]:
            _str += f"_and_{network.name}"

        return _str


def _degree_check(
    degree_of_nodes: Mapping[T, NodeDegree], succ_dg: Mapping[T, NodeName]
) -> None:
    """Filter out such network structure, which is currently not supported."""
    for node in filter(lambda node: degree_of_nodes[node].out_degree > 1, succ_dg):
        if any(degree_of_nodes[succ_node].in_degree > 1 for succ_node in succ_dg[node]):
            raise NotSupportedError("This structure of network is not supported yet.")


def convert2routing_groups(
    succ_dg_of_cb: Dict[CoreBlock, List[CoreBlock]],
    degrees_of_cb: Dict[CoreBlock, NodeDegree],
) -> List[RoutingGroup]:
    ordered_core_blocks = toposort(succ_dg_of_cb)
    seen_cb = set()
    routing_groups = []

    _degree_check(degrees_of_cb, succ_dg_of_cb)

    for cb in ordered_core_blocks:
        # Check whether it has been traversed
        if cb not in seen_cb:
            seen_cb.add(cb)
            routing_groups.append(RoutingGroup(cb))

        # If the out-degree > 1, treat the following core blocks as one routing group.
        if degrees_of_cb[cb].out_degree > 1:
            succ_cbs = succ_dg_of_cb[cb]
            seen_cb.update(succ_cbs)
            routing_groups.append(RoutingGroup(*succ_cbs))

    return routing_groups


def toposort(directed_edges: Mapping[T, Sequence[T]]) -> List[T]:
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


def reverse_edges(directed_edges: Mapping[T, Sequence[T]]) -> Dict[T, Set[T]]:
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


# def _bounded_nodes_check(constrs: Sequence[Sequence[NeuDyn]]) -> None:
#     seen = set()

#     for bounded in constrs:
#         for node in bounded:
#             if node in seen:
#                 raise ValueError(f"Node {node} is repeated in the list of constraints.")

#             seen.add(node)


# def _bounded_by(node: NodeName, constrs: Sequence[Sequence[NeuDyn]]) -> List[NodeName]:
#     for constr in constrs:
#         for bounded_node in constr:
#             if node == bounded_node.name:
#                 return list(n.name for n in set(constr))

#     return []


# def _conflicted_by(
#     node: NodeName, constrs: Dict[NodeName, Sequence[NeuDyn]]
# ) -> List[NodeName]:
#     """Find all the conflicted nodes of node.

#     Example: {"1": {"2", "3"}, "4": {"1"}}. For node 1, return ["2", "3", "4"].
#     """
#     c = set(constrs.get(node, []))

#     for k, v in constrs.items():
#         for conf_node in v:
#             if node == conf_node.name:
#                 c.add(k)

#     return list(n.name for n in c)


def get_node_degrees(succ_edges: Mapping[T, Mapping[T, Any]]) -> Dict[T, NodeDegree]:
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


def get_longest_path(
    edges_with_d: Dict[T, Dict[T, EdgeAttr]],
    ordered_nodes: List[T],
) -> Tuple[List[T], int]:
    """Get the longest path in the DAG.

    Args:
        - edges_with_d: a dictionary of directed edges with distance.
        - ordered_nodes: nodes in topological sorting order.

    Return:
        A tuple containing the longest path in the graph and its distance.
    """
    distances: Dict[T, int] = defaultdict(int)  # init value = 0
    pred_nodes: Dict[T, Optional[T]] = defaultdict()

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
    edges_with_d: Dict[T, Dict[T, EdgeAttr]],
    ordered_nodes: List[T],
    input_nodes: List[T],
) -> Tuple[List[T], int]:
    """Get the shortest path in the DAG.

    Args:
        - edges_with_d: a list of directed edges with distance.
        - ordered_nodes: nodes in topological sorting.
        - input_nodes: input nodes.

    Return: the shortest distance in the graph.
    """
    distances: Dict[T, int] = defaultdict(lambda: MAX_DISTANCE)
    pred_nodes: Dict[T, Optional[T]] = defaultdict()

    # Set initial value for all inputs nodes.
    for inode in input_nodes:
        distances[inode] = 0

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
