from collections import defaultdict
from typing import Dict, List, Set

from paibox.base import NeuDyn
from paibox.collector import Collector
from paibox.exceptions import BuildError, ResourceError
from paibox.libpaicore import Coord, CoordOffset, HwConfig, get_replication_id
from paibox.network import DynSysGroup
from paibox.projection import InputProj
from paibox.synapses import SynSys

from .config_template import CoreConfig, NeuronConfig, NeuronDest
from .context import _BACKEND_CONTEXT
from .graphs import *
from .placement import CoreBlock, aligned_coords, max_lcn_of_cb
from .routing import RoutingRoot

NodeName = str
EdgeName = str


__all__ = ["Mapper"]


class Mapper:
    """Mapping the network in the cores."""

    routing_tree = RoutingRoot(tag="L5")
    """The routing tree root."""

    def __init__(self) -> None:
        self.networks = ()
        """Sub networks in the graph."""

        self._nodes = Collector()
        """Nodes in the network. Structure:
        {
            node1.name: node1,
            node2.name: node2
        }
        """
        self.inodes = Collector()
        self.onodes = Collector()

        self._ordered_nodes: List[NodeName] = []
        """Ordered topologically nodes."""

        self._succ_nodes: Dict[NodeName, Set[NodeName]] = defaultdict(set)
        # self._pred_nodes: Dict[str, Set[str]] = defaultdict(set)

        self._edges = Collector()
        """Edges in the network. Structure:
        {
            edge1.name: edge1,
            edge2.name: edge2
        }
        """
        self._succ_edges: Dict[NodeName, Dict[NodeName, int]] = defaultdict(dict)

        self._edges_grouped: List[Set[EdgeName]] = []
        """Grouped edges in the network. Structure:
        [
            {edge1.name, edge2.name},
            {edge3.name}
        ]
        """

        self._pred_dg: Dict[NodeName, Dict[NodeName, EdgeName]] = defaultdict(dict)
        """Pre-synapses of nodes. Structure:
        {
            node.name: {
                pre-node1.name: pre-syn1.name,
                pre-node2.name: pre-syn2.name
            }
        }
        """

        self._succ_dg: Dict[NodeName, Dict[NodeName, EdgeName]] = defaultdict(dict)
        """Post-synapses of nodes. Structure:
        {
            node.name: {
                post-node1.name: post-syn1.name,
                post-node2.name: post-syn2.name
            }
        }
        """

        self._degree_of_nodes: Dict[NodeName, Degree] = defaultdict(Degree)
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

        self.core_params: Dict[Coord, CoreConfig] = dict()
        """The dictionary of core parameters. Structure:
        {
            address of core: {
                parameters...
            }
        }
        """

        self.core_plm_config = dict()
        self.graph_info: GraphInfo

        """Backend contexts"""
        self.env = _BACKEND_CONTEXT

        self.clear()

    def clear(self) -> None:
        self.has_built = False
        self.routing_tree.clear()

        self._nodes.clear()
        self._ordered_nodes.clear()
        self._edges.clear()
        self._succ_edges.clear()
        self._edges_grouped.clear()

        self._pred_dg.clear()
        self._succ_dg.clear()

        self._degree_of_nodes.clear()
        self.core_blocks.clear()
        self.succ_core_blocks.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

    def build_graph(self, *networks: DynSysGroup) -> None:
        """Build the directed graph based on given networks.    \
            More than one networks in one graph is supported.

        Args:
            - networks: one or many `DynSysGroup`.

        TODO verify the following phases when more than one sub  \
            network is given.
        """
        self.clear()

        self.networks = networks

        for network in networks:
            nodes = network.nodes(level=1, include_self=False)
            sub_nodes = nodes.include(InputProj, NeuDyn).unique()
            sub_edges = nodes.subset(SynSys).unique()
            self._nodes += sub_nodes
            self._edges += sub_edges

        # Add all nodes in the graph. DO NOT REMOVE!
        for node in self._nodes:
            self._succ_edges[node] = dict()
            self._pred_dg[node] = dict()
            self._succ_dg[node] = dict()
            self._succ_nodes[node] = set()

        for syn in self._edges.values():
            u, v = syn.source.name, syn.dest.name
            self._succ_edges[u][v] = 1
            self._pred_dg[v][u] = syn.name
            self._succ_dg[u][v] = syn.name
            self._succ_nodes[u].add(v)

        # self._pred_nodes = reverse_edges(self._succ_nodes)

        self.has_built = True

        # `InputProj` nodes are input nodes definitely.
        self.inodes = self.nodes.subset(InputProj)
        # By default, nodes with out-degree = 0 are considered output nodes.
        self.onodes = self.nodes.key_on_condition(
            lambda node: len(self.succ_dg[node]) == 0
        )

        self._graph_check()

    def main_phases(self) -> GraphInfo:
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
        graph_info = self.config_export()

        return graph_info

    def _graph_check(self) -> None:
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
        self._ordered_nodes = toposort(self._succ_nodes)

        self._degree_of_nodes = get_node_degrees(self.succ_dg)

        branch_nodes = filter(
            lambda node: self._degree_of_nodes[node].out_degree > 1, self.nodes
        )
        for node in branch_nodes:
            if any(
                self._degree_of_nodes[succ_node].in_degree > 1
                for succ_node in self.succ_dg[node]
            ):
                raise NotSupportedError(
                    "This structure of network is not supported yet."
                )

        # Only support the in-degree of backward node of input node is 1.
        for inode in self.inodes:
            if any(
                self._degree_of_nodes[succ_node].in_degree > 1
                for succ_node in self.succ_dg[inode]
            ):
                raise NotSupportedError(
                    "Only input nodes are supported as the only input of a node."
                )

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
            raise ResourceError(
                f"#N of total cores required out of {HwConfig.N_CORE_OFFLINE}: {n_core_required_total}"
            )

        # """
        #     Sort in ascending order according to the minimum value of \
        #     the index of source nodes in the topological order.
        # """
        # self.core_blocks.sort(
        #     key=lambda cb: min(self._ordered_nodes.index(src.name) for src in cb.source)
        # )

        """
            Get the following core blocks for each core block.
        """
        for cb in self.core_blocks:
            succ_cbs = list(
                filter(
                    lambda succ_cb: any(d for d in cb.dest if d in succ_cb.source),
                    self.core_blocks,
                )
            )
            self.succ_core_blocks[cb].extend(succ_cbs)

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
            elif len(succ_cb) == 1:
                cb.target_lcn = succ_cb[0].lcn_ex
                cb.lcn_locked = True
            else:
                cb.lcn_locked = True

    def coord_assign(self) -> None:
        """Assign the coordinate for each `CorePlacement`."""
        for cb in self.core_blocks:
            if not RoutingRoot.insert_coreblock(self.routing_tree, cb):
                raise RuntimeError(
                    f"Insert core block {cb.name} into routing tree failed."
                )

    def core_allocation(self) -> None:
        """Allocate the core blocks to the physical cores.

        The order of `core_plms` is the same as `core_blocks`.
        """
        for cb in self.core_blocks:
            cb.core_plm_alloc()

    def config_export(self) -> GraphInfo:
        """Export parameters of cores & neurons inside.

        Steps:
            - 1. Export the parameters(PARAMETER_REG, including \
                RANDOM_SEED & Weight RAM) of cores.
            - 2. Export the parameters(Neuron RAM) of neurons inside.
        """
        input_nodes_info = self._inpproj_config_export()

        # The destination info of the output nodes is
        # a subset of the info of the network members.
        output_dest_info: Dict[NodeName, Dict[int, NeuronConfig]] = defaultdict(dict)

        _ocoord_update_flag = False
        ocoord = _BACKEND_CONTEXT["output_core_addr"]

        for cb in self.core_blocks:
            self.core_params |= CoreBlock.export_core_plm_config(cb)
            """
            Traverse all the core placements in core blocks, then find \
                the following core blocks where the axons at.

            If found, get the coordinate of the core placment, all the \
                coordinates of axons(for broadcasting).
            """
            output_axon_offset = 0

            for core_plm in cb.core_placements.values():
                for neu_seg in core_plm.neu_segs:
                    # neu_seg is an output
                    if neu_seg.parent.name in self.onodes:
                        # Update the offset of axon
                        output_axon_offset = core_plm.export_neu_config(
                            neu_seg,
                            output_core_coord=ocoord,
                            axon_addr_offset=output_axon_offset,
                        )
                        output_dest_info[neu_seg.parent.name][
                            core_plm.coord.address
                        ] = core_plm.neu_config[neu_seg.parent]
                        _ocoord_update_flag = True
                    else:
                        # Find the axon destinations
                        dests = list(
                            filter(
                                lambda cb: neu_seg.parent in cb.source, self.core_blocks
                            )
                        )

                        # TODO Necessary to make this condition a premise?
                        assert len(dests) == 1
                        core_plm.export_neu_config(neu_seg, dests)

                self.core_plm_config[core_plm.coord] = core_plm.export_core_config()

            if _ocoord_update_flag:
                # Coord.x += 1
                ocoord += CoordOffset(1, 0)
                _ocoord_update_flag = False

        self.graph_info = GraphInfo(
            input=input_nodes_info, output=output_dest_info, members={}
        )

        return self.graph_info

    def _inpproj_config_export(self) -> Dict[NodeName, NeuronDest]:
        # Get the input core blocks for all input nodes.
        input_core_blocks = filter(
            lambda cb: any(s for s in cb.source if s.name in self.inodes),
            self.core_blocks,
        )

        input_nodes_info: Dict[NodeName, NeuronDest] = defaultdict()

        for input_cb in input_core_blocks:
            dest_coords = input_cb.core_coords
            dest_rid = get_replication_id(dest_coords)

            # Only traverse input nodes in this core block.
            for inode in filter(lambda s: s in self.inodes.values(), input_cb.source):
                axon_coords = aligned_coords(
                    slice(0, input_cb.n_axon_of(input_cb.source.index(inode)), 1),
                    input_cb.axon_segments[inode],
                )

                neuron_dest = NeuronDest(
                    dest_coords,
                    [coord.tick_relative for coord in axon_coords],
                    [coord.addr_axon for coord in axon_coords],
                    dest_coords[0].x,
                    dest_coords[0].y,
                    dest_rid.x,
                    dest_rid.y,
                    _BACKEND_CONTEXT["local_chip_addr"].x,
                    _BACKEND_CONTEXT["local_chip_addr"].y,
                )

                input_nodes_info[inode.name] = neuron_dest

        return input_nodes_info

    @property
    def inherent_timestep(self) -> int:
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        _, distance = get_longest_path(self._succ_edges, self._ordered_nodes)

        return distance

    @property
    def nodes(self):
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return self._nodes

    @property
    def n_inode(self) -> int:
        """The #N of input nodes"""
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return len(self.inodes)

    @property
    def n_onode(self) -> int:
        """The #N of output nodes"""
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return len(self.onodes)

    @property
    def edges(self):
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return self._edges

    @property
    def pred_dg(self):
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return self._pred_dg

    @property
    def succ_dg(self):
        if not self.has_built:
            raise BuildError(f"The graph hasn't been built yet")

        return self._succ_dg


def group_by(dict_: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in dict_.values():
        d[keyfunc(item)].append(item)

    return d
