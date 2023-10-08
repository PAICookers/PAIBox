from collections import defaultdict
from typing import Dict, List, Set, Union

from paibox.base import NeuDyn, PAIBoxObject
from paibox.network import DynSysGroup
from paibox.synapses import SynSys
from paibox.projection import InputProj

from .graphs import toposort
from .grouping import GroupedSyn, GroupedSynOnCore
from .placement import RouterTreeRoot

PredSynDictType = Dict[str, SynSys]
SuccSynDictType = Dict[str, SynSys]
SuccGroupedSynDictType = Dict[str, GroupedSyn]


class Mapper:
    """Mapping the network in the cores."""

    router_tree = RouterTreeRoot()
    """The router tree."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Union[NeuDyn, InputProj]] = defaultdict()
        """Nodes in the network.
        
        Structure:
        {
            node1.name: node1,
            node2.name: node2
        }
        """

        self._pred_dg: Dict[str, PredSynDictType] = defaultdict(dict)
        """Pre-synapses of nodes.
        
        Structure:
        {
            node.name: {
                pre-node1.name: pre-syn1,
                pre-node2.name: pre-syn2
            }
        }
        """
        self._succ_dg: Dict[str, SuccSynDictType] = defaultdict(dict)
        """Post-synapses of nodes.
        
        Structure:
        {
            node.name: {
                post-node1.name: post-syn1,
                post-node2.name: post-syn2
            }
        }
        """

        self._pred_gsyns: Dict[str, GroupedSyn] = defaultdict()
        """Grouped pre-synapses of nodes.
        
        Structure:
        {
            node1.name: grouped pre-syn1,
            node2.name: grouped pre-syn2
        }
        """

        self._succ_gsyns: Dict[str, SuccGroupedSynDictType] = defaultdict()
        """Grouped post-synapses of nodes.
        
        Structure:
        {
            node1.name: {
                post-node1.name: grouped post-syn1,
                post-node2.name: grouped post-syn2
            }
        }
        """

        self._pred_gsyn_on_core: Dict[str, List[GroupedSynOnCore]] = defaultdict()
        """Grouped pre-synapse on core of nodes.
        
        Structure:
        {
            node1.name: list1 of grouped pre-synapses,
            node2.name: list2 of grouped pre-synapses
        }
        """

        self.is_built = False

    def clear(self) -> None:
        self.is_built = False
        self._pred_dg.clear()
        self._succ_dg.clear()
        self._nodes.clear()
        self._pred_gsyns.clear()
        self._succ_gsyns.clear()
        self._pred_gsyn_on_core.clear()

    def build_graph(self, network: DynSysGroup) -> None:
        """Build the directed graph given a network.

        Arguments:
            - network: a `DynSysGroup`.

        For a level-one layer network:
            INPUT -> S0 -> N1 -> S1 -> N2
                            |          -> S3 ->
                            ------------> S2 -> N3 -> OUTPUT

            `_succ_dg`: the synapses of each node.
                {
                    <INPUT>.name: {<N1>.name: <SynSys0>},
                    <N1>.name: {
                        <N2>.name: <SynSys1>,
                        <N3>.name: <SynSys2>
                    },
                    <N2>.name: {<N3>.name: <SynSys3>},
                    <N3>.name: {}
                }

            `_pred_dg`: the predecessors of each node.
                {
                    <INPUT>.name: {},
                    <N1>.name: {<INPUT>.name: <SynSys0>},
                    <N2>.name: {<N1>.name: <SynSys1>},
                    <N3>.name: {
                        <N1>.name: <SynSys2>,
                        <N2>.name: <SynSys3>,
                    }
                }
        """
        self.clear()

        self.network = network
        self._nodes = (
            network.nodes(level=1, include_self=False)
            .include(InputProj, NeuDyn)
            .unique()
        )

        # Add nodes
        for node in self._nodes:
            self._pred_dg[node] = dict()
            self._succ_dg[node] = dict()

        syns = network.nodes(level=1, include_self=False).subset(SynSys).unique()

        # Add edges
        for syn in syns.values():
            u, v = syn.source.name, syn.dest.name
            self._succ_dg[u][v] = syn
            self._pred_dg[v][u] = syn

        self.is_built = True

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

        2. TODO Level 1:
            - Every layer is considered seperately.
            - The number of a group of neurons may > 512.
            - The LCN extension of every layer is LCN_1X.
        """

        """1. Group every synapse at first."""
        self._group_syns()

        """2. Adjust the LCN extension of each layer for target LCN.

        the name of node A: {
            the following node name B : the grouped synapse of A to B.
        }
        """
        # self._lcn_ex_adjustment()

        """3. Build the grouped synapse on each core."""
        self._build_gsyn_on_core()

        """4. Do core coordinate assignment."""
        self._coord_assignment()

    def _group_syns(self) -> None:
        """Group every synapse in the DG."""
        for name in self.nodes:
            # Consider ALL the pre-synapses of a node.
            pre_syns = self.pred_dg[name]
            if pre_syns:
                self._pred_gsyns[name] = GroupedSyn.build(list(pre_syns.values()))

    def _lcn_ex_adjustment(self) -> None:
        """Adjust the LCN extension for each target LCN."""
        for name in self.nodes:
            self._succ_gsyns[name] = {}
            succ_nodes = self.succ_dg[name]

            if succ_nodes:
                for succ_node_name, syn in succ_nodes.items():
                    self._succ_gsyns[name][succ_node_name] = self._in_grouped_syns(syn)

    def _build_gsyn_on_core(self) -> None:
        """Build the grouped synapse on each core."""
        for name, gsyn in self._pred_gsyns.items():
            self._pred_gsyn_on_core[name] = gsyn.build_syn_on_core()

    def _coord_assignment(self) -> None:
        ordered_nodes = toposort(self.succ_dg)

        for node_name in ordered_nodes:
            if isinstance(self.nodes[node_name], InputProj):
                continue

            # Traverse the grouped pre-synapse in order.
            if node_name in self._pred_gsyns:
                gsyns_on_core = self._pred_gsyn_on_core[node_name]

                for gsyn_on_core in gsyns_on_core:
                    if not self.router_tree.insert_gsyn_on_core(gsyn_on_core):
                        raise ValueError
                    
        print("Coord aasignment OK.")

    """Utilities"""

    def _find_obj_by_name(self, name: str) -> PAIBoxObject:
        """Return the object in the network given the name of it."""
        objs = (
            self.network.nodes(level=1, include_self=False)
            .subset(PAIBoxObject)
            .unique()
        )
        for v in objs.values():
            if name is v.name:
                return v

        raise ValueError(f"Name {name} not found in the network.")

    def _in_grouped_syns(self, syn: SynSys) -> GroupedSyn:
        """Find which the grouped synapses the synapse is in."""
        for name in self._pred_gsyns:
            if syn.dest.name == name:
                return self._pred_gsyns[name]

        raise ValueError

    def _find_same_lcn_syns(self, pre_node_name: str) -> Set[NeuDyn]:
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
        node_parents = []
        grouped_syns = self._succ_gsyns[pre_node_name]

        for syns in grouped_syns.values():
            # The source is the parents of the synapse.
            node_parents.extend(syns.source)

        children = []

        for node in set(node_parents):
            children.extend(list(self._succ_gsyns[node.name].values()))

        return set(children)

    @property
    def nodes(self):
        if self.is_built:
            return self._nodes

        # TODO
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


def group_by(_dict: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in _dict.values():
        d[keyfunc(item)].append(item)

    return d
