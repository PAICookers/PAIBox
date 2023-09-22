from collections import defaultdict
from typing import Dict

from paibox.base import DynamicSys, NeuDyn, PAIBoxObject
from paibox.network import DynSysGroup
from paibox.synapses import SynSys

from .grouping import GroupedLayer, GroupedSyn

PredSynDictType = Dict[str, SynSys]
SuccSynDictType = Dict[str, SynSys]
SuccGroupedSynDictType = Dict[str, GroupedSyn]


class Mapper:
    def __init__(self) -> None:
        """Mapper.
        
        Private attributes:
        
            - _pre_grouped_syns: a dictionary of previous \
                grouped synapses.
                {
                    <N1.name>: the combined and grouped \
                        pre-synapses of node `N1`.
                }
            - _succ_grouped_syns: a dictionary of following \
                grouped synapses and nodes.
                {
                    <N1.name>: {
                        <N2.name>: the combined and grouped \
                            pre-synapses of node `N2`,
                        <N3.name>: the combined and grouped \
                            pre-synapses of node `N3`.
                    }
                }

        """
        self._nodes: Dict[str, NeuDyn] = defaultdict()
        self._pred_dg: Dict[str, PredSynDictType] = defaultdict(dict)
        self._succ_dg: Dict[str, SuccSynDictType] = defaultdict(dict)

        self._pre_grouped_syns: Dict[str, GroupedSyn] = defaultdict()
        self._succ_grouped_syns: Dict[str, SuccGroupedSynDictType] = defaultdict()
        self.grouped_layers: Dict[str, GroupedLayer] = defaultdict()

        self.is_built = False

    def clear(self) -> None:
        self.is_built = False
        self._pred_dg.clear()
        self._succ_dg.clear()
        self._nodes.clear()
        self._pre_grouped_syns.clear()
        self._succ_grouped_syns.clear()
        self.grouped_layers.clear()

    def build_graph(self, network: DynSysGroup) -> None:
        """Build the directed graph given a network.

        Arguments:
            - network: a DynSysGroup.

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
        self._nodes = network.nodes(level=1, include_self=False).subset(NeuDyn).unique()

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
        for name in self.nodes:
            # Consider ALL the pre-synapses of a node.
            pre_syns = self.pred_dg[name]
            if pre_syns:
                self._pre_grouped_syns[name] = GroupedSyn.build(list(pre_syns.values()))

        for name in self.nodes:
            self._succ_grouped_syns[name] = {}
            succ_nodes = self.succ_dg[name]

            if succ_nodes:
                for node, syn in succ_nodes.items():
                    self._succ_grouped_syns[name][node] = self._in_grouped_syns(syn)

        """2. Re limit the LCN extension for target LCN."""

        """2. Build placement for each grouped layer."""
        for name, syns in self._pre_grouped_syns.items():
            self.grouped_layers[name] = GroupedLayer.build(self.nodes[name], syns)

    def _find_component(self, component: PAIBoxObject) -> str:
        """Return the name of the component if in the network."""
        for v in self.network.__dict__.values():
            if component is v:
                return v.name

        raise ValueError(f"Component {component} not found in the network.")

    def _find_name(self, name: str) -> PAIBoxObject:
        """Return the object of the name if in the network."""
        for k, v in self.network.__dict__.items():
            if name is v.name:
                return v

        raise ValueError(f"Name {name} not found in the network.")

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

    def _find_src_syns(self, node: DynamicSys):
        name = self._find_component(node)

        return self._pred_dg[name]

    def _find_dest_syns(self, node: DynamicSys):
        name = self._find_component(node)

        return self._succ_dg[name]

    def _in_grouped_syns(self, syn: SynSys) -> GroupedSyn:
        for name in self._pre_grouped_syns:
            if syn.dest.name == name:
                return self._pre_grouped_syns[name]

        raise ValueError


def group_by(_dict: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in _dict.values():
        d[keyfunc(item)].append(item)

    return d
