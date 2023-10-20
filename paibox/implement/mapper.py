from collections import defaultdict
from typing import FrozenSet, Dict, List, Set, Union

from paibox._types import FrozenOrderedSet
from paibox.base import NeuDyn, PAIBoxObject
from paibox.network import DynSysGroup
from paibox.projection import InputProj
from paibox.synapses import SynSys

from .graphs import *
from .grouping import GroupedSyn, GroupedSynOnCore
from .placement import RoutingRoot

NodeNameType = str
NodeSynDictType = Dict[str, str]  # key-value for node & synapse.
SuccGroupedSynOnCoreDictType = Dict[str, List[GroupedSynOnCore]]


class Mapper:
    """Mapping the network in the cores."""

    routing_tree = RoutingRoot(tag="L5")
    """The routing tree root."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Union[NeuDyn, InputProj]] = dict()
        """Nodes in the network. Structure:
        {
            node1.name: node1,
            node2.name: node2
        }
        """

        self._ordered_nodes: List[str] = []
        """Ordered topologically nodes."""

        self._succ_nodes: Dict[str, Set[str]] = defaultdict(set)
        self._pred_nodes: Dict[str, Set[str]] = defaultdict(set)

        self._edges: Dict[str, SynSys] = dict()
        """Edges in the network. Structure:
        {
            edge1.name: edge1,
            edge2.name: edge2
        }
        """

        self._edges_grouped: FrozenOrderedSet[FrozenSet[str]] = FrozenOrderedSet()
        """Grouped edges in the network. Structure:
        {
            {edge1.name, edge2.name},
            {edge3.name}
        }
        """

        self._pred_dg: Dict[str, NodeSynDictType] = defaultdict(dict)
        """Pre-synapses of nodes. Structure:
        {
            node.name: {
                pre-node1.name: pre-syn1.name,
                pre-node2.name: pre-syn2.name
            }
        }
        """

        self._succ_dg: Dict[str, NodeSynDictType] = defaultdict(dict)
        """Post-synapses of nodes. Structure:
        {
            node.name: {
                post-node1.name: post-syn1.name,
                post-node2.name: post-syn2.name
            }
        }
        """

        self._degree_of_nodes: Dict[str, Tuple[int, int]] = defaultdict(tuple)
        """A dictionary of in/out-degree tuple of nodes. Structure:
        {
            node.name: (in-degree, out-degree)
        }
        """

        self._gsyns: List[GroupedSyn] = []
        """A list for grouped synapses in topological order."""

        # self._pred_gsyns: Dict[str, Dict[str, GroupedSyn]] = defaultdict(dict)
        """Grouped synapses. Structure: {
            node1.name: grouped pre-syn1,
            node2.name: grouped pre-syn2
        }
        """

        # self._succ_gsyns: Dict[str, Dict[str, GroupedSyn]] = defaultdict(dict)
        """Grouped post-synapses of nodes. Structure:
        {
            node1.name: {
                post-node1.name: grouped post-syn1,
                post-node2.name: grouped post-syn2
            }
        }
        """

        self._gsyns_on_core: List[List[GroupedSynOnCore]] = []

        # self._pred_gsyn_on_core: Dict[str, List[GroupedSynOnCore]] = defaultdict(list)
        """Grouped pre-synapse on core of nodes. Structure:
        {
            node1.name: list1 of grouped pre-synapses,
            node2.name: list2 of grouped pre-synapses
        }
        """

        # self._succ_gsyn_on_core: Dict[
        #     str, Dict[str, List[GroupedSynOnCore]]
        # ] = defaultdict(dict)

        self.is_built = False

    def clear(self) -> None:
        self.is_built = False

        self._nodes.clear()
        self._ordered_nodes.clear()
        self._edges.clear()
        self._edges_grouped.clear()

        self._pred_dg.clear()
        self._succ_dg.clear()

        self._degree_of_nodes.clear()
        self._gsyns.clear()
        # self._pred_gsyns.clear()
        # self._succ_gsyns.clear()
        # self._pred_gsyn_on_core.clear()
        # self._succ_gsyn_on_core.clear()

    def build_graph(self, network: DynSysGroup) -> None:
        """Build the directed graph based on a given network.

        Arguments:
            - network: a `DynSysGroup`.
        """
        self.clear()

        self.network = network
        self._nodes = (
            network.nodes(level=1, include_self=False)
            .include(InputProj, NeuDyn)
            .unique()
        )
        self._edges = network.nodes(level=1, include_self=False).subset(SynSys).unique()

        # Add all nodes in the network. DO NOT REMOVE!
        for node in self._nodes:
            self._pred_dg[node] = dict()
            self._succ_dg[node] = dict()
            self._succ_nodes[node] = set()

        for syn in self._edges.values():
            u, v = syn.source.name, syn.dest.name
            self._pred_dg[v][u] = syn.name
            self._succ_dg[u][v] = syn.name
            self._succ_nodes[u].add(v)

        self._pred_nodes = reverse_edges(self._succ_nodes)

        self.is_built = True

        self._graph_preprocess()

    def do_grouping(self) -> None:
        """
        Prerequisites:
        0. Global config:
            - All weights is 1-bit
            - Pure SNN mode. SNN_EN enbales, input and spike width are 1-bit.

        1. Simplified situation.
            Level 0:
            - Every layer is considered seperately.
            - The number of a group of neurons <= 512.
            - The LCN extension of every layer is LCN_1X.

        2. TODO Level 1:pass
            - Every layer is considered seperately.
            - The number of a group of neurons may > 512.
            - The LCN extension of every layer is LCN_1X.
        """
        if not self.is_built:
            # TODO
            raise Exception

        """1. Group synapses."""
        self._group_synapses()

        """2. Adjust the LCN extension of each layer for target LCN."""
        self._lcn_ex_adjustment()

        """3. Build the grouped synapse on each core."""
        self._build_gsyn_on_core()

        """4. Do core coordinate assignment."""
        self._coord_assignment()

    def _graph_preprocess(self) -> None:
        """Preprocess of the graph.
        
        # Currently, we are unable to cope with arbitrary \
        #     network structures. Therefore, in this stage, the \
        #     input of specific network structures will be restricted.
        
        # However, with development in progress, this limitation \
        #     can be solved.
            
        # Limitation:
        #     For a node with in-degree > 1, the out-degree of \
        #         all its forward nodes = 1.
        #     Or for a node with out-degree > 1, the in-degree of \
        #         all its backward node = 1.
        """
        # TODO Only allow the graph with no cycle.
        self._ordered_nodes = toposort(self._succ_nodes)

        self._degree_of_nodes, self._edges_grouped = group_edges(
            self._ordered_nodes, list(self.edges.keys()), self.pred_dg, self.succ_dg
        )

    def _group_synapses(self) -> None:
        """Group synapses based on grouped edges.
        
        Description:
            After combining all synapses into groups, iterate through \
            each combination synapses to build a `GroupedSyn`.
            
            Then do sorting in ascending order.
        """
        for syns_set in self._edges_grouped:
            syns = [self.edges[syn] for syn in syns_set]
            self._gsyns.append(GroupedSyn.build(*syns))

        """
        Sort in ascending order according to the minimum value of the \
        index of source nodes in the topological order.
        """
        self._gsyns.sort(
            key=lambda gsyn: min(
                self._ordered_nodes.index(src.name) for src in gsyn.source
            )
        )

        # Create `_succ_gsyns` dictionary
        # for node_name in self.nodes:
        #     self._succ_gsyns[node_name] = {}
        #     succ_nodes = self.succ_dg[node_name]

        #     for succ_node_name in succ_nodes:
        #         gsyn = self._pred_gsyns[succ_node_name]
        #         if len(succ_nodes) > 1 or gsyn.n_core > 1:
        #             gsyn.need_broadcast = True

        #         self._succ_gsyns[node_name][succ_node_name] = gsyn

    def _lcn_ex_adjustment(self) -> None:
        """Adjust the LCN extension for each target LCN.

        And indicate wether need to broadcast.
        """
        pass

    def _build_gsyn_on_core(self) -> None:
        """Build the grouped synapse on each core.

        The order of `_gsyns_on_core` is the same as `grouped_syns`.
        """
        for gsyn in self.grouped_syns:
            self._gsyns_on_core.append(gsyn.build_syn_on_core())

    def _coord_assignment(self) -> None:
        """Assign the coordinate of layer.
        
        Steps:
            - 1. Sort all the nodes.
            - 2. Traverse all the post gsyns of nodes in order.
        
        Problem: See Q23100901. Now, For a node with an in-degree > 1, \
            the out-degree of all its forward nodes = 1. -> Moved to grouping phase.
        """

        # for node_name in self._ordered_nodes:
        #     # Traverse all the grouped post-synapse of nodes in order.
        #     if node_name in self._succ_gsyn_on_core:
        #         gsyns_on_core = self._succ_gsyn_on_core[node_name]

        #         # A `gsyn_on_core` is a list of grouped post-synapses on core of a node.
        #         # Consider all the succ gsyn_on_core together:
        #         # For every succ node:
        #         for succ_node_name in gsyns_on_core:
        #             gsyn = self._succ_gsyns[node_name][succ_node_name]
        #             gsyn.is_assigned = True

        print("Coord assignment OK.")

    """Utilities"""

    def _find_obj_by_name(self, name: str) -> PAIBoxObject:
        """Return the object in the network given the name of it.

        If not found, raise `ValueError`.
        """
        objs = (
            self.network.nodes(level=1, include_self=False)
            .subset(PAIBoxObject)
            .unique()
        )
        for v in objs.values():
            if name is v.name:
                return v

        raise ValueError(f"Name {name} not found in the network.")

    def _find_same_lcn_syns(self, pre_node_name: str):
        """Find the grouped synapses with the same LCN extension \
            given a previous node.

        TODO
        Auto find the layers that have the same LCN in the network.
        Check whether all the layers are been traversed.

        If the indegree of a grouped syn == 1:
            If outdegree of its previous node == 1:
                just consider itself, lock its LCN.
            Else, find all the children grouped syn of the previous node, \
                then self._succ_gsyns[node]?
        Else, ...
        """
        pass
        # node_parents = []
        # grouped_syns = self._succ_gsyns[pre_node_name]

        # for syns in grouped_syns.values():
        #     # The source is the parents of the synapse.
        #     node_parents.extend(syns.source)

        # children = []

        # for node in set(node_parents):
        #     children.extend(list(self._succ_gsyns[node.name].values()))

        # return set(children)

    @property
    def nodes(self):
        if self.is_built:
            return self._nodes

        # TODO
        raise ValueError

    @property
    def edges(self):
        if self.is_built:
            return self._edges

        raise ValueError

    @property
    def pred_dg(self):
        if self.is_built:
            return self._pred_dg

        # TODO
        raise ValueError

    @property
    def succ_dg(self):
        if self.is_built:
            return self._succ_dg

        # TODO
        raise ValueError

    @property
    def grouped_syns(self) -> List[GroupedSyn]:
        return self._gsyns


def group_by(_dict: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in _dict.values():
        d[keyfunc(item)].append(item)

    return d
