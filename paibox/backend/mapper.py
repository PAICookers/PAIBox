from collections import defaultdict
from typing import Dict, List, Set

from paibox.base import NeuDyn
from paibox.collector import Collector
from paibox.exceptions import BuildError, ResourceError
from paibox.libpaicore import Coord, HwConfig
from paibox.network import DynSysGroup
from paibox.projection import InputProj
from paibox.synapses import SynSys

from .config_template import CoreConfigDict
from .graphs import *
from .placement import CoreBlock, max_lcn_of_cb
from .routing import RoutingRoot

NodeNameType = str
NodeSynDictType = Dict[str, str]  # key-value for node & synapse.


class Mapper:
    """Mapping the network in the cores."""

    routing_tree = RoutingRoot(tag="L5")
    """The routing tree root."""

    def __init__(self) -> None:
        self._nodes = Collector()
        """Nodes in the network. Structure:
        {
            node1.name: node1,
            node2.name: node2
        }
        """
        self.inodes = Collector()
        self.onodes = Collector()

        self._ordered_nodes: List[str] = []
        """Ordered topologically nodes."""

        self._succ_nodes: Dict[str, Set[str]] = defaultdict(set)
        # self._pred_nodes: Dict[str, Set[str]] = defaultdict(set)

        self._edges: Dict[str, SynSys] = dict()
        """Edges in the network. Structure:
        {
            edge1.name: edge1,
            edge2.name: edge2
        }
        """

        self._edges_grouped: List[Set[str]] = []
        """Grouped edges in the network. Structure:
        [
            {edge1.name, edge2.name},
            {edge3.name}
        ]
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

        self._degree_of_nodes: Dict[str, Degree] = defaultdict(Degree)
        """A dictionary of in/out-degree tuple of nodes. Structure:
        {
            node.name: (in-degree, out-degree)
        }
        """

        self.core_blocks: List[CoreBlock] = []
        """A list for core blocks in topological order."""

        self.succ_core_blocks: Dict[CoreBlock, List[CoreBlock]] = defaultdict(list)
        """Grouped post-synapses of nodes. Structure:
        {
            node1.name: {
                post-node1.name: grouped post-syn1,
                post-node2.name: grouped post-syn2
            }
        }
        """

        self.core_params: Dict[Coord, CoreConfigDict] = dict()
        """The dictionary of core parameters. Structure:
        {
            address of core: {
                parameters...
            }
        }
        """

        self.core_plm_config = dict()

        self.clear()

    def clear(self) -> None:
        self.has_built = False
        self.routing_tree.clear()

        self._nodes.clear()
        self._ordered_nodes.clear()
        self._edges.clear()
        self._edges_grouped.clear()

        self._pred_dg.clear()
        self._succ_dg.clear()

        self._degree_of_nodes.clear()
        self.core_blocks.clear()
        self.succ_core_blocks.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

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

        # self._pred_nodes = reverse_edges(self._succ_nodes)

        self.has_built = True

        # `InputProj` nodes are input nodes definitely.
        self.inodes = self.nodes.subset(InputProj)
        # By default, nodes with out-degree = 0 are considered output nodes.
        self.onodes = self.nodes.on_condition(lambda node: len(self.succ_dg[node]) == 0)

        self._graph_check()

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
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        """1. Build core blocks."""
        self.build_core_blocks()

        """2. Adjust the LCN extension of each layer for target LCN."""
        self.lcn_ex_adjustment()

        """3. Do core coordinate assignment."""
        self.coord_assign()

        """4. Allocate the grouped synapses to the cores."""
        self.core_allocation()

        """5. Export parameters."""
        self.config_export()

    def _graph_check(self) -> None:
        """Preprocess of the directed graph. Because there are currently    \
            many limitations on the networks that can be processed, checks  \
            are performed at this stage.

        Limitation:
            # For a node with in-degree > 1, the out-degree of \
            #     all its forward nodes = 1.
            - For a node with out-degree > 1, the in-degree of \
                all its backward node = 1.
        """
        self._ordered_nodes = toposort(self._succ_nodes, is_strict=True)

        self._degree_of_nodes = get_node_degrees(self.succ_dg)

        for node in self.nodes:
            if self._degree_of_nodes[node].out_degree > 1:
                succ_nodes = list(self.succ_dg[node].keys())

                if any(
                    self._degree_of_nodes[succ_node].in_degree > 1
                    for succ_node in succ_nodes
                ):
                    raise NotSupportedError("Not support this structure of network.")

        # Moved to `build_core_blocks`
        # if do_edges_grouping:
        #     self._degree_of_nodes, self._edges_grouped = group_edges(
        #         self._ordered_nodes, list(self.edges.keys()), self.pred_dg, self.succ_dg
        #     )
        # else:
        #     self._edges_grouped = list({e} for e in self.edges)

    def build_core_blocks(self) -> None:
        """Build core blocks based on grouped edges.

        Description:
            After combining all synapses into groups, iterate through \
            each combination synapses to build `CoreBlock`.

            # Then do sorting in ascending order.
        """
        self._edges_grouped = group_edges(
            list(self.edges.keys()),
            self.succ_dg,
            self._degree_of_nodes,
            ordered_nodes=self._ordered_nodes,
        )

        for syns_set in self._edges_grouped:
            syns = [self.edges[syn] for syn in syns_set]
            self.core_blocks.append(CoreBlock.build(*syns))

        # Check the total core consumption.
        if (
            n_core_required_total := sum(cb.n_core_required for cb in self.core_blocks)
            > HwConfig.N_CORE_OFFLINE
        ):
            # TODO
            raise ResourceError(
                f"#N of total core required out of {HwConfig.N_CORE_OFFLINE}: {n_core_required_total}"
            )

        # """
        #     Sort in ascending order according to the minimum value of \
        #     the index of source nodes in the topological order.
        # """
        # self.core_blocks.sort(
        #     key=lambda cb: min(self._ordered_nodes.index(src.name) for src in cb.source)
        # )

        """
            Traverse the destination node of each grouped synapse, \
            and check if it is the source of other grouped synapses.\
        """
        for cb in self.core_blocks:
            other_cb = self.core_blocks.copy()
            other_cb.remove(cb)

            for dest_node in cb.dest:
                succ_cb = [
                    gs for gs in other_cb if self.nodes[dest_node.name] in gs.source
                ]
                self.succ_core_blocks[cb].extend(succ_cb)

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN extension for each core block. Make sure  \
            that all destination LCNs are equal.

        If the out-degree of `CoreBlock` > 1, the LCN of all its    \
        following core blocks needs to be adjusted. So that the     \
        `target_LCN` can be set.

        The LCN after adjustment = max(LCN_1, LCN_2, ..., LCN_N)
        """
        for cb in self.core_blocks:
            succ_cb = self.succ_core_blocks[cb]

            if len(succ_cb) > 1:
                max_lcn_ex = max_lcn_of_cb(succ_cb)
                for g in succ_cb:
                    g.set_lcn_ex(max_lcn_ex)

                cb.target_lcn = max_lcn_ex
            elif len(succ_cb) > 0:
                cb.target_lcn = succ_cb[0].lcn_ex
                cb.lcn_locked = True
            else:
                cb.lcn_locked = True

    def coord_assign(self) -> None:
        """Assign the coordinate for each `CorePlacement`."""
        for cb in self.core_blocks:
            self.routing_tree.insert_coreblock(cb)

    def core_allocation(self) -> None:
        """Allocate the core blocks to the physical cores.

        The order of `core_plms` is the same as `core_blocks`.
        """
        for cb in self.core_blocks:
            cb.core_plm_alloc()

    def config_export(self) -> None:
        """Export parameters of cores & neurons inside.

        Steps:
            - 1. Export the parameters(PARAMETER_REG, including \
                RANDOM_SEED & Weight RAM) of cores.
            - 2. Export the parameters(Neuron RAM) of neurons inside.
        """
        for cb in self.core_blocks:
            self.core_params |= CoreBlock.export_core_plm_config(cb)
            """
            Traverse all the core placements in core blocks, then find \
                the following core blocks where the axons at.

            If found, get the coordinate of the core placment, all the \
                coordinates of axons(for broadcasting).
            """
            for core_plm in cb.core_placements.values():
                for neu_seg in core_plm.neu_segs:
                    # Find the axons dest
                    dests = [
                        cb for cb in self.core_blocks if neu_seg.parent in cb.source
                    ]

                    if not dests:
                        continue

                    # TODO Necessary to make this condition a premise?
                    assert len(dests) == 1  # ?
                    core_plm.export_neu_config(neu_seg, dests)

                self.core_plm_config[core_plm.coord] = core_plm.export_core_config()

    @property
    def nodes(self):
        if self.has_built:
            return self._nodes

        raise BuildError(f"The graph hasn't been built yet")

    @property
    def n_inode(self) -> int:
        """The #N of input nodes"""
        if self.has_built:
            return len(self.inodes)

        raise BuildError(f"The graph hasn't been built yet")

    @property
    def n_onode(self) -> int:
        """The #N of output nodes"""
        if self.has_built:
            return len(self.onodes)

        raise BuildError(f"The graph hasn't been built yet")

    @property
    def edges(self):
        if self.has_built:
            return self._edges

        raise BuildError(f"The graph hasn't been built yet")

    @property
    def pred_dg(self):
        if self.has_built:
            return self._pred_dg

        raise BuildError(f"The graph hasn't been built yet")

    @property
    def succ_dg(self):
        if self.has_built:
            return self._succ_dg

        raise BuildError(f"The graph hasn't been built yet")


def group_by(dict_: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in dict_.values():
        d[keyfunc(item)].append(item)

    return d
