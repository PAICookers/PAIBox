from collections import defaultdict
from typing import Dict, List

from paibox.base import DynamicSys, NeuDyn, PAIBoxObject
from paibox.network import DynSysGroup
from paibox.synapses import SynSys

from .grouping import GroupedLayer, GroupedSyn
from .placement import Placement


class Mapper:
    def __init__(self) -> None:
        self._nodes: Dict[str, NeuDyn] = defaultdict()
        self._pred_dg: Dict[str, Dict[str, SynSys]] = defaultdict(dict)
        self._succ_dg: Dict[str, Dict[str, SynSys]] = defaultdict(dict)
        self._syn_of_nodes: Dict[str, Dict[str, List[SynSys]]] = defaultdict(dict)

        self.grouped_syns: List[GroupedSyn] = []
        self.grouped_layers: Dict[str, GroupedLayer] = dict()
        self.placement_group = dict()

        self.is_built = False

    def clear(self) -> None:
        self.is_built = False
        self._pred_dg.clear()
        self._succ_dg.clear()
        self._nodes.clear()
        self.grouped_syns.clear()
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

        """1. Group by layer"""
        # Do nothing

        """2. Group neurons into cores using LCN extension optimization."""
        self._grouping_lcn_opt()

        """3. Build placement for each layer."""
        for name, layer in self.grouped_layers.items():
            self.placement_group[name] = Placement.build(layer)

    def _grouping_lcn_opt(self) -> None:
        """LCN extension optimization for grouping a layer.

        Description:
            Find a minimum LCN extension for each synapses connecting this layer, \
            so that to deploy to core(s).

        S1 -> N, the connectivity of S1 is [A1*N]
        S2 -> N, the connectivity of S2 is [A2*N]

        Now consider A1 & A2 seperately.
        A1 -> C0 + C1
        A2 -> C2
        """
        # 1. Group every synapse at first.
        for name in self.nodes:
            pre_syns = self.pred_dg[name]

            for pre_syn in pre_syns.values():
                self.grouped_syns.append(GroupedSyn.build(pre_syn))

        # 2. Generate the grouped layer for every node.
        for name, node in self.nodes.items():
            self.grouped_layers[name] = GroupedLayer.build(node, self.grouped_syns)

    # def _grouping_greedy_lcn_opt(self, groups: Dict[str, Any]) -> List[GroupedNeuron]:
    #     """LCN extension optimization strategy for grouping neurons.

    #     Description:
    #         Always looking for the minimum LCN extension that the group of axons is **ALL** satisfied.

    #     TODO Summary the constant in this function such as 1152, 512, etc.
    #     """
    #     axons_in_layer: np.ndarray = groups["axons_each"]
    #     n_neuron: int = groups["neuron_num"]

    #     indices = np.argsort(axons_in_layer, kind="heapsort")
    #     axons_in_layer.sort(kind="heapsort")

    #     lcn_each_in_layer = []
    #     axons_grouped = []

    #     for i in range(n_neuron):
    #         lcn_ex = int((axons_in_layer[i] - 1) / 1152)
    #         if lcn_ex > LCN_EX.LCN_64X:
    #             # TODO
    #             raise ValueError

    #         lcn_each_in_layer.append(LCN_EX(lcn_ex))

    #     # Traverse the lcn and put them in core
    #     def _get_core_limit(idx: int) -> Tuple[LCN_EX, int, int]:
    #         """Get the LCN extension limit, maximum number of neurons,
    #         & maximum number of axons in a core.
    #         """
    #         lcn_ex_max_in_core = lcn_each_in_layer[idx]
    #         n_max_in_core = int(512 >> lcn_ex_max_in_core)
    #         axons_max_in_core = 1152 * (lcn_ex_max_in_core - LCN_EX.LCN_1X + 1)

    #         return lcn_ex_max_in_core, n_max_in_core, axons_max_in_core

    #     i = i_start = 0

    #     while i_start < n_neuron:
    #         axons_group_sum = axons_in_layer[i_start]
    #         l_limit, n_limit, a_limit = _get_core_limit(i_start)

    #         if i != n_neuron:
    #             i += 1

    #         while True:
    #             if i == n_neuron:
    #                 break

    #             if not (
    #                 (i - i_start) + 1 <= n_limit
    #                 and axons_group_sum + axons_in_layer[i] <= a_limit
    #             ):
    #                 break

    #             axons_group_sum += axons_in_layer[i]
    #             l_limit, n_limit, a_limit = _get_core_limit(i)
    #             i += 1

    #         axons_grouped.append(
    #             # Slice the array [i_last: i+1], which means [i_last, i].
    #             GroupedNeuron(l_limit, axons_in_layer[i_start:i], indices[i_start:i])
    #         )
    #         i_start = i

    #     return axons_grouped

    def _find_component(self, component: PAIBoxObject) -> str:
        """Return the name of the component if in the network."""
        for k, v in self.network.__dict__.items():
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

    def _find_src_syn(self, node: DynamicSys):
        name = self._find_component(node)

        return self._pred_dg[name]

    def _find_dest_syn(self, node: DynamicSys):
        name = self._find_component(node)

        return self._succ_dg[name]


def group_by(_dict: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in _dict.values():
        d[keyfunc(item)].append(item)

    return d
