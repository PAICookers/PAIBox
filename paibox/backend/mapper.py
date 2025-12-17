import itertools
import logging
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Literal, cast

from paicorelib import ChipCoord, Coord, CoordOffset, HwConfig

from paibox import _logging
from paibox.base import SynSys
from paibox.components import Neuron
from paibox.exceptions import CompileError, ConfigInvalidError, ResourceError
from paibox.network import DynSysGroup

from .conf_exporting import (
    export_core_params_json,
    export_graph_info,
    export_neuron_phy_loc,
    gen_config_frames_by_coreconf,
    get_clk_en_L2_dict,
    get_neuron_phy_loc,
)
from .conf_types import (
    CoreConf,
    CorePlmConf,
    FrameArrayType,
    GraphInfo,
    InputNeuronDest,
    InputNodeConf,
    OutputDestConf,
)
from .context import _BACKEND_CONTEXT, set_cflag
from .graph_utils import get_node_degrees, get_succ_cb_by_node, merge_cycles
from .graphs import PAIGraph
from .placement import (
    CoreBlock,
    OnlineCoreBlock,
    SourceDest,
    get_replication_id,
    max_lcn_of_cb,
)
from .routing import RoutingGroup, RoutingManager
from .sub_utils import SubNeuron, sub_node_overlap
from .types import (
    DendriteSegment,
    DestNodeType,
    NodeDegree,
    NodeType,
    SourceNodeType,
    _coord_to_bin_str,
)

__all__ = ["Mapper"]

log = logging.getLogger(__name__)
build_cb_log = _logging.get_artifact_logger(__name__, "build_core_blocks")
lcn_adj_log = _logging.get_artifact_logger(__name__, "lcn_ex_adjustment")
cb_axon_grp_log = _logging.get_artifact_logger(__name__, "cb_axon_grouping")
coord_asg_log = _logging.get_artifact_logger(__name__, "coord_assign")
ndest_collect = _logging.get_artifact_logger(__name__, "collect_neuron_dest")


class Mapper:
    graph: PAIGraph
    graph_info: GraphInfo
    routing_mgr: RoutingManager

    def __init__(self) -> None:
        self.graph = PAIGraph()
        self.core_blocks: list[CoreBlock] = []
        """List for core blocks in the network."""
        self.succ_core_blocks: dict[CoreBlock, list[CoreBlock]] = defaultdict(list)
        self.input_core_blocks: dict[SourceNodeType, list[CoreBlock]] = defaultdict(
            list
        )
        """List of input core blocks for each input node."""

        self.degrees_of_cb: dict[CoreBlock, NodeDegree] = defaultdict(NodeDegree)

        self.core_plm_config: CorePlmConf = defaultdict(dict)
        self.core_params: CoreConf = defaultdict(dict)
        """The dictionary of core parameters."""

        self.n_core_required = 0
        self.n_core_occupied = 0
        self.routing_mgr = RoutingManager(
            chip_list=_BACKEND_CONTEXT["target_chip_addr"]
        )
        self.neuron_dest: dict[SourceNodeType, SourceDest] = defaultdict(SourceDest)
        """The dictionary of destinations for input or neuron nodes."""

        # Status variables during compilation. Make sure to clear them after each compilation.
        self._core_estimate_only = False
        """Wether this compilation is for core estimation only. If so, no core will be assigned."""

        self.clear()

    def clear(self) -> None:
        self.graph.clear()
        self.routing_mgr.clear()

        self.core_blocks.clear()
        self.succ_core_blocks.clear()
        self.input_core_blocks.clear()

        self.degrees_of_cb.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

        self.n_core_required = 0
        self.n_core_occupied = 0

        self.neuron_dest.clear()

        # Status variables
        self._core_estimate_only = False

        # Set default cflags
        _BACKEND_CONTEXT.cflags.clear()
        set_cflag(enable_wp_opt=True)
        set_cflag(grouping_optim_target="both")
        set_cflag(no_twisted_branch=True)

    def build(self, *networks: DynSysGroup, **build_options) -> None:
        """Build the directed graph based on given networks. More than one networks in one graph is supported.

        Args:
            - networks: one or many `DynSysGroup`.

        TODO verify the following phases when more than one sub network is given.
        """
        self.clear()

        # Filter & check the constraints to nodes.
        self.graph.build(*networks, **build_options)

    def compile(
        self,
        *,
        core_estimate_only: bool = False,
        weight_bit_optimization: bool = True,
        grouping_optim_target: Literal["latency", "core", "both"] = "both",
        no_twisted_branch: bool = False,
        **kwargs,
    ) -> GraphInfo:
        """Compile the network with optimization options.

        Args:
            core_estimate_only (bool): only do the core estimation, without allocation. Default is false.
            weight_bit_optimization (bool): whether to optimize weight width. For example, weights declared as  \
                INT8 are treated as smaller width based on their actual values (when the weight are all between \
                [-8, 7], they can be treated as INT4).
                This option is not applicable to online cores since their weights will be updated during learning.
                By default, it is specified by the corresponding compile option in the backend configuration item.
            grouping_optim_target ("latency", "core", "both"): specify the optimization goal of neuron grouping,\
                which can be `latency`, `core` or `both` which respectively represent the optimization goal of  \
                delay/throughput, occupied cores, or both. The default is specified by the corresponding        \
                compilation option in the backend configuration item. Default is 'both'.
            no_twisted_branch (bool): only for advanced use. when parsing the network topology, whether or not  \
                to prohibit intersecting branch structures will cause such structures to be processed.          \
                For example:

                I -> A -> B -> C
                       ------>

                The out-degree of node A is > 1, and its successor node C has an in-degree > 1. If true, A will \
                be copied & denoted as A', whose forward connection is preserved.

                I -> A -> B -> C
                  -> A'------>

        Return: network information after compilation in dictionary format.
        """
        set_cflag(enable_wp_opt=weight_bit_optimization)
        set_cflag(grouping_optim_target=grouping_optim_target)
        set_cflag(no_twisted_branch=no_twisted_branch)

        self._core_estimate_only = core_estimate_only

        # Preperation:
        # 1. Check whether the PAIGraph has built.
        # 2. Set global compilation flags.
        self._build_check()
        self._set_global_cflags()

        # Untwist the branch nodes if the flag is on.
        if no_twisted_branch:
            self.untwist_branch_nodes()

        self.graph.topo_support_check()  # not used for now

        # Build core blocks
        self.build_core_blocks()

        # Adjust the LCN extension of each core block
        self.lcn_ex_adjustment()

        # Group the axons of core block
        self.cb_axon_grouping()

        # Coordinates assignment
        self.coord_assign(self._core_estimate_only)

        if self._core_estimate_only:
            return GraphInfo(
                name=self.graph.graph_name_repr,
                input={},
                output={},
                members={},
                inherent_timestep=self.graph.get_global_t_1st_vld(),
                output_flow_format=self.graph.get_output_flow_format(),
                n_core_required=self.n_core_required,
                n_core_occupied=0,
            )

        # Collect the neuron destinations for input or neuron nodes.
        self.collect_neuron_dest()

        # Allocate the routing groups to the core placements level.
        self.core_allocation()

        # Export configurations and return. This step does not modify any data.
        return self.config_export()

    def untwist_branch_nodes(self) -> None:
        self.graph.untwist_branch_nodes()

    def handle_copy_in_core_blocks(self) -> None:
        copy_map: dict[NodeType, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for cb in self.core_blocks:
            for ax in cb.ordered_axons:
                if ax.contain_copy:
                    for idx in ax.index:
                        if idx.copy_id > 0:
                            copy_map[ax.target][idx.index].append(idx.copy_id)

        # log.info("################################### Copy Map ###################################")
        # for source_node, node_copy_map in copy_map.items():
        #     log.info(f"source node {source_node.name} copy map: {node_copy_map}")

        for cb in self.core_blocks:
            for sub_edge in cb.obj:
                sub_edge.handle_copy(copy_map[sub_edge.dest.target])

    def build_core_blocks(self) -> None:
        """Build core blocks based on partitioned edges."""
        # Graph partitioning
        merged_grps = self.graph.graph_partition()

        for merged_grp in merged_grps:
            log.info(merged_grp)

        merged_grps = merge_cycles(merged_grps)
        log.info(
            "################################### merge_cycles ###################################"
        )
        for merged_grp in merged_grps:
            log.info(merged_grp)

        # Build routing groups
        raw_rgrps: list[RoutingGroup] = []
        for mgrp in merged_grps:
            raw_rgrps.append(RoutingGroup.build(mgrp, is_root=True))

        # Record the optimized routing groups in the routing manager
        self.routing_mgr.optimize_rgrps(raw_rgrps)

        for rg in self.routing_mgr.routing_grps:
            rg.set_target_chip()

        for rg in self.routing_mgr.routing_grps:
            rg.dump()

        log.info(
            "################################### Routing Group Builded, N_Core_Required Not Set Yet ###################################"
        )

        # Retrive the core blocks from routing groups
        for rg in self.routing_mgr.routing_grps:
            self.core_blocks += rg.core_blocks

        self.handle_copy_in_core_blocks()
        log.info(
            "################################### Copy Handled in Core Blocks ###################################"
        )
        for rg in self.routing_mgr.routing_grps:
            rg.dump()

        log.info(
            "################################### Succ CoreBlock Set ###################################"
        )
        # Build the successor graph of core blocks.
        self._build_cb_graph(no_cb_cycle=True)
        # Collect the input core blocks for each input node.
        self._collect_input_core_blocks()

        # Record the degrees of core blocks for later use.
        self.degrees_of_cb = get_node_degrees(self.succ_core_blocks)

        # Build the successor graph of routing groups.
        self.routing_mgr.build_rg_graph(self.succ_core_blocks)

    def _build_cb_graph(self, no_cb_cycle: bool = True) -> None:
        """Build the successor graph of core blocks.

        Args:
            no_cb_cycle (bool): whether to prohibit core blocks forming a cycle. Default is True. This  \
                situation has been solved in the previous steps.
        """
        # Impossible that the sucessor of one core block is itself (as a loop).
        assert all(
            not sub_node_overlap(cb.dest, cb.ordered_axons) for cb in self.core_blocks
        )

        # Use `combinations` to traverse the core blocks pairs without duplication.
        # Generate (c1, (c2, c3, c4,...)), (c2, (c3, c4, c5,...)), (c3, (c4, c5, c6,...)), etc.
        for cb in self.core_blocks:
            self.succ_core_blocks[cb] = []

        for cur_cb, next_cb in itertools.combinations(self.core_blocks, 2):
            _ol_c2n = sub_node_overlap(cur_cb.dest, next_cb.ordered_axons)
            _ol_n2c = sub_node_overlap(next_cb.dest, cur_cb.ordered_axons)

            if no_cb_cycle:
                assert not (_ol_c2n and _ol_n2c)  # cannot be a cycle.

            if _ol_c2n:
                self.succ_core_blocks[cur_cb].append(next_cb)
            if _ol_n2c:
                self.succ_core_blocks[next_cb].append(cur_cb)

        for cur_cb, succ_cbs in self.succ_core_blocks.items():
            build_cb_log.debug(f"\n{cur_cb.name} Succ:")
            for cb in succ_cbs:
                build_cb_log.debug(f"\t{cb.name}")

    def _collect_input_core_blocks(self) -> None:
        """Collect the input core blocks for each input node."""
        # Record the input core blocks for each input node.
        for inode in self.graph.inodes.values():
            # TODO How to prevent this situation: there is input node & predecessor nodes
            # in a certain core blocks.

            # Disconnected input nodes will not be recorded.
            succ_cb = get_succ_cb_by_node(inode, self.core_blocks)
            if len(succ_cb) > 0:
                self.input_core_blocks[inode] = succ_cb

            build_cb_log.debug(f"\ninput core block of {inode.name}:")
            for cb in succ_cb:
                build_cb_log.debug(f"\t{cb.name}")

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN of each core block & set the target LCN.

        NOTE: The LCN of all successor core blocks of any core block must be the same. Meanwhile,   \
            the `target_lcn` of the core block is equal to that LCN.
        """
        # the core in the same neighbor list should have the same lcn_ex
        neighbor_lists: list[list[CoreBlock]] = list()
        for input_cbs in self.input_core_blocks.values():
            neighbor_lists.append(input_cbs)
        for cb in self.core_blocks:
            neighbor_list = self.succ_core_blocks[cb].copy()
            if cb.online:
                # the online core's lcn is same with target_lcn
                # so it should have the same lcn_ex with its succ
                neighbor_list.append(cb)
            if len(neighbor_list) > 1:
                neighbor_lists.append(neighbor_list)

        # merge neighbors
        merged_lists: list[set[CoreBlock]] = []
        visited = set()

        for i, neighbor_set in enumerate(neighbor_lists):
            if i in visited:
                continue
            # 当前合并集合
            merged = set(neighbor_set)
            changed = True
            visited.add(i)

            while changed:
                changed = False
                for j in range(len(neighbor_lists)):
                    if j in visited:
                        continue
                    if not merged.isdisjoint(neighbor_lists[j]):
                        merged.update(neighbor_lists[j])
                        visited.add(j)
                        changed = True  # 有新合并就继续 while

            merged_lists.append(merged)

        # set lcn_ex for merged neighbors
        for merged_list in merged_lists:
            max_lcn_ex = max_lcn_of_cb(list(merged_list))
            for cb in merged_list:
                # online core's lcn_ex limit check
                # happends in cb.lcn_ex setter
                cb.lcn_ex = max_lcn_ex

        # Set the target LCN of each core block
        for cb in self.core_blocks:
            succ_cbs = self.succ_core_blocks[cb]
            if len(succ_cbs) > 0:
                # the lcn_ex of the successor core blocks have been adjusted to the same
                # use the first successor core block's lcn_ex as the target_lcn
                # the online core's lcn_ex is same with target_lcn
                # this limit also satisfied by the above code
                cb.target_lcn = succ_cbs[0].lcn_ex

            cb._lcn_locked = True

        log.info(
            "################################### LCN Adjustment Finished ###################################"
        )
        for cb in self.core_blocks:
            lcn_adj_log.debug(f"{cb.name}: LCN = {cb.lcn_ex}")

    def cb_axon_grouping(self) -> None:
        """Group the axons after the LCN is modified & locked. The destination axon of the neurons that need to be  \
            multicast needs to be consistent. Check the inputs of all core blocks in the same routing group. If     \
            there are overlapping parts, set the same axon for the overlapping parts.
        """
        for cb in self.core_blocks:
            cb.group_axons()

        log.info(
            "################################### Axon Grouping Finished ###################################"
        )
        for cb in self.core_blocks:
            cb_axon_grp_log.debug(f"cb: {cb.name}:")
            for source, ax_seg in cb.axon_segments.items():
                cb_axon_grp_log.debug(f"\t{source}: {ax_seg}")

    def coord_assign(self, core_estimate_only: bool) -> None:
        """Assign the coordinate of each `CorePlacement`.

        NOTE: The neurons in each core block must be grouped first to determine the \
            #N of cores required, and then the routing coordinates can be assigned.
        """
        for rg in self.routing_mgr.routing_grps:
            for cb in rg.iter_nested_cb():
                cb.group_neurons(
                    optim_target=_BACKEND_CONTEXT.cflags["grouping_optim_target"]
                )

            rg.set_core_required()

        log.info(
            "################################### Neuron Grouping Finished ###################################"
        )
        for cb in self.core_blocks:
            coord_asg_log.debug(cb)

        log.info(
            "################################### Required Cores Set ###################################"
        )

        for rg in self.routing_mgr.ordered_rgrps:
            rg.dump()

        # Optimize the order of routing groups
        # self.routing_grps = reorder_routing_groups(self.succ_rgrps)
        # self.ordered_rgrps = toposort(self.succ_rgrps)

        # Calculate the consumption of required physical cores.
        n_avail_offline_cores = (
            HwConfig.N_CORE_OFFLINE * _BACKEND_CONTEXT.n_target_chips
        )
        n_avail_online_cores = HwConfig.N_CORE_ONLINE * _BACKEND_CONTEXT.n_target_chips
        n_offline_core_required = sum(
            cb.n_core_required if not cb.online else 0 for cb in self.core_blocks
        )
        n_online_core_required = sum(
            cb.n_core_required if cb.online else 0 for cb in self.core_blocks
        )

        self.n_core_required = n_offline_core_required + n_online_core_required

        # If only estimate the core usage, the rest of the steps are not performed.
        if core_estimate_only:
            return None

        if n_offline_core_required > n_avail_offline_cores:
            raise ResourceError(
                OUT_OF_CORE_RESOURCE_TEXT.format(
                    n_offline_core_required, n_avail_offline_cores
                )
            )

        if n_online_core_required > n_avail_online_cores:
            raise ResourceError(
                OUT_OF_CORE_RESOURCE_TEXT.format(
                    n_online_core_required, n_avail_online_cores
                )
            )

        for rg in self.routing_mgr.ordered_rgrps:
            self.routing_mgr.place_routing_group(rg)

        log.info(
            "################################### Assignment Finished ###################################"
        )
        for rg in self.routing_mgr.ordered_rgrps:
            rg.dump_routing_result()

        # Online cores are not counted in the number of occupied cores.
        self.n_core_occupied = self.routing_mgr.n_core_occupied

    def collect_neuron_dest(self) -> None:
        """Collect the destination details for neuron slices in each core block."""
        # Traverse all source node slices & their corresponding axon segments on the input axon side of core blocks.
        for cb in self.core_blocks:
            for sub_source, axon_seg in cb.axon_segments.items():
                self.neuron_dest[sub_source.target].add_dest(sub_source, axon_seg, cb)

        inhi_dest_coords: dict[OnlineCoreBlock, list[Coord]] = defaultdict(list)
        online_cbs: list[OnlineCoreBlock] = list()
        for cb in self.core_blocks:
            if not cb.online:
                continue
            online_cbs.append(cast(OnlineCoreBlock, cb))

        for cb_source in online_cbs:
            for cb_dest in online_cbs:
                # a coreblock must inhi itself
                if not cb_source.laterl_inhi_target.isdisjoint(
                    cb_dest.laterl_inhi_source
                ):
                    inhi_dest_coords[cb_source].extend(cb_dest.core_coords)

        for cb, dest_coords in inhi_dest_coords.items():
            _, rid = get_replication_id(dest_coords)
            cb.inhi_rid = rid

        log.info(
            "################################ Neuron Dest Info Collected ################################"
        )

        for source, dest in self.neuron_dest.items():
            dest.set_dest_rid()
            dest.sort_dest_info()

            ndest_collect.debug(f"source: {source.name}")
            ndest_collect.debug(dest)

    def core_allocation(self) -> None:
        """Allocate the routing groups to core placements level in topological order."""
        self.routing_mgr.allocate_cp()

    def config_export(self) -> GraphInfo:
        """Export parameters of cores & neurons inside.

        Steps:
            1. Export the parameters(PARAMETER_REG, including RANDOM_SEED & Weight RAM) of cores.
            2. Export the parameters(Neuron RAM) of neurons inside.
        """
        if (
            ochip_coord := _BACKEND_CONTEXT.output_chip_addr
        ) in _BACKEND_CONTEXT.target_chip_addr:
            raise ConfigInvalidError(
                f"the output chip address {ochip_coord} should not overlap with the "
                f"target chip addresses, but got {_BACKEND_CONTEXT._target_chip_addr_repr()}."
            )

        input_nodes_info = self._inpproj_config_export()
        output_dest_info = self._member_cb_and_onode_config_export()

        _graph_info = GraphInfo(
            name=self.graph.graph_name_repr,
            input=input_nodes_info,
            output=output_dest_info,
            members=self.core_plm_config,  # The configuration of physical cores is in `core_plm_config`
            inherent_timestep=self.graph.get_global_t_1st_vld(),
            output_flow_format=self.graph.get_output_flow_format(),
            n_core_required=self.n_core_required,
            n_core_occupied=self.n_core_occupied,
            misc={
                "clk_en_L2": get_clk_en_L2_dict(
                    _BACKEND_CONTEXT.target_chip_addr,
                    self.routing_mgr.used_L2_clusters,
                ),
                "target_chip_list": _BACKEND_CONTEXT.target_chip_addr,
            },
        )

        self.graph_info = _graph_info

        return _graph_info

    def _set_global_cflags(self) -> None:
        # NOTE: learnable synapses are not configured for weight width optimization.
        SynSys.CFLAG_ENABLE_WP_OPTIMIZATION = _BACKEND_CONTEXT.cflags["enable_wp_opt"]

    def _inpproj_config_export(self) -> InputNodeConf:
        """Export the configuration of input projections.

        Json exchange file format for input nodes:
        {
            "inp1_1": { # as input node #1 without dest info
                "addr_core_x": 0,
                "addr_core_y": 0,
                "addr_core_x_ex": 1,
                "addr_core_y_ex": 3,
                "addr_chip_x": 0,
                "addr_chip_y": 0,
                "lcn": 1 << lcn_ex,
                "tick_relative": [...],
                "addr_axon": [...]
            },
            "inp2_1": {...} # as input node #2
        }
        """
        input_nodes_info: InputNodeConf = dict()

        for inode in self.graph.inodes.values():
            if inode not in self.neuron_dest:
                continue

            dest = self.neuron_dest[inode]
            # TODO Input nodes can also be sliced, so additional information needs to be saved in the dictionary
            dest_core_info, axon_coords = dest.get_undivided_dest()

            inp_neuron_dest = InputNeuronDest(
                [coord.tick_relative for coord in axon_coords],
                [coord.addr_axon for coord in axon_coords],
                dest_core_info.base_coord.x,
                dest_core_info.base_coord.y,
                dest_core_info.rid.x,
                dest_core_info.rid.y,
                dest_core_info.dest_chip_coord.x,
                dest_core_info.dest_chip_coord.y,
                dest_core_info.timeslot,  # 1 << lcn_ex
            )

            input_nodes_info[inode.name] = inp_neuron_dest

        return input_nodes_info

    def _member_cb_and_onode_config_export(self) -> OutputDestConf:
        """Export configuration & output destinations inormation for core blocks.

        Description:
            Traverse core placements in core blocks, find the following core    \
            blocks where the axons at. Get the coordinate of the core placement \
            & coordinates of axons(for multicasting).

        Json exchange file format for output nodes:
        {
            "n3": { # as output node #1 & required two physical cores
                "4": { # as output core #1 of node #1
                    "tick_relative": [0, 0, 0, 0, 0],
                    "addr_axon": [0, 1, 2, 3, 4, 5],
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0
                },
                "5": {...} # as output core #2 of node #1
            }
            "n4": {...} # as output node #2
        }
        """
        output_dest_info: OutputDestConf = defaultdict(dict)
        # Shallow copy
        ocoord = copy(_BACKEND_CONTEXT["output_core_addr_start"])
        o_nodes = list(self.graph.onodes.values())

        for rg in self.routing_mgr.ordered_rgrps:
            for member_cb in rg.core_blocks:
                self.core_params[rg.chip_coord] |= member_cb.export_core_plm_config()

                for core_plm in member_cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs_of_cplm:
                        # The destination of `neu_seg` is on the chips.
                        if neu_seg.target in self.neuron_dest:
                            target_dest = self.neuron_dest[neu_seg.target]
                            core_plm.export_neu_config(neu_seg, target_dest)

                        # Otherwise, `neu_seg` is an output node & the destination is not on the chips.
                        elif neu_seg.target in o_nodes:
                            # For the destination allocation of output nodes, in order to enable the hardware platform
                            # to distinguish & decode the output data of different output nodes, an allocation method
                            # needs to be agreed upon artificially, as described below:
                            # 1. All output nodes are output to an external chip (recorded in the CP already).
                            # 2. Starting from the `output_core_addr_start`(=c) in `_BACKEND_CONTEXT`, each output node is
                            # output to cores c, c+1, c+2, etc. in turn.
                            # 3. Since we only leverage the axon coordinate attributes in the output working frames and
                            # do not use the `tick_relative` attribute, the number of outputs of each output node cannot
                            # be greater than `N_FANIN_PER_DENDRITE_MAX`(=1152). TODO Can be adjusted later.
                            offset_idx = o_nodes.index(neu_seg.target)
                            cur_ocoord = ocoord + CoordOffset.from_offset(offset_idx)
                            core_plm.export_neu_config(
                                neu_seg, output_core_coord=cur_ocoord
                            )
                            output_dest_info[neu_seg.target.name][core_plm.coord] = (
                                core_plm.neu_configs[neu_seg.target].neuron_dest_info
                            )

                        else:
                            raise ValueError(
                                f"find destination of member {neu_seg} failed."
                            )

                for coord, core_plm in member_cb.core_placements.items():
                    self.core_plm_config[rg.chip_coord][
                        coord
                    ] = core_plm.export_core_plm_config()

            # Generate default configurations for wasted core placements of the routing group
            self.core_plm_config[rg.chip_coord].update(rg.get_wasted_cplm_config())

        return output_dest_info

    def export(
        self,
        write_to_file: bool = True,
        *,
        fp: str | Path | None = None,
        format: Literal["txt", "bin", "npy"] = "bin",
        read_voltage: str | Neuron | Sequence[str] | Sequence[Neuron] | None = None,
        split_by_chip: bool = False,
        export_clk_en_L2: bool = False,
        use_hw_sim: bool = True,
    ) -> dict[ChipCoord, list[FrameArrayType]]:
        """Generate configuration frames & export to file.

        Args:
            write_to_file (bool): whether to write configuration frames into file.
            fp (str, Path): specify the output path for the config file (if `write_to_file` is true) & json files.
            format (str): `txt`, `bin`, or `npy`. `bin` is recommended. If `write_to_file` is false, this argument is   \
                ignored.
            read_voltage (Neuron, Sequence[Neuron]): specify the neuron(s) to read its voltage. Their physical locations\
                on chips will be exported for hardware platform to read.
            split_by_chip (bool): whether to split the generated frames file by the chips.
            export_used_L2 (bool): whether to export the serial port data of the L2 cluster clocks.
            use_hw_sim (bool): whether to use hardware simulator. If used, '.bin' will be exported. If `write_to_file`  \
                is false, this argument is ignored.

        Return: total configurations in dictionary format.
        """
        if self._core_estimate_only:
            raise CompileError(
                "the current compilation is only for core estimation. "
                "Please disable 'core_estimate_only' and compile again before exporting."
            )

        if write_to_file:
            if format not in ("bin", "npy", "txt"):
                raise ValueError(f"format {format} is not supported.")

            formats = [format]
        else:
            formats = []

        if write_to_file:
            if use_hw_sim and "bin" not in formats:
                formats.append("bin")

        _fp = _fp_check(fp)
        config_dict = gen_config_frames_by_coreconf(
            self.graph_info["members"], write_to_file, _fp, formats, split_by_chip
        )

        # Export the parameters of occupied cores
        export_core_params_json(self.core_params, _fp)

        # Export the graph information
        export_graph_info(self.graph_info, _fp, export_clk_en_L2)

        # Retrieve the neuron's physical locations if specified
        if read_voltage is not None:

            def _convert_to_neuron(_neu: str | DestNodeType) -> DestNodeType:
                if isinstance(_neu, DestNodeType):
                    return _neu

                if (neu := self.graph.get_neu_by_name(_neu)) is None:
                    raise ValueError(f"neuron {_neu} not found in the graph.")

                return neu

            if isinstance(read_voltage, (str, Neuron)):
                to_read = (read_voltage,)
            else:
                to_read = read_voltage

            reading_targets = [_convert_to_neuron(n) for n in to_read]
            phy_locations = get_neuron_phy_loc(self.core_blocks, reading_targets)
            export_neuron_phy_loc(phy_locations, _fp)

        return config_dict

    def find_neuron(self, neuron: Neuron | SubNeuron, *, verbose: int = 0) -> None:
        self._build_check()
        sub_neu = neuron if isinstance(neuron, SubNeuron) else SubNeuron(neuron)
        name = sub_neu.target.name

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if sub_node_overlap(sub_neu, cb.dest):
                # NL_overlap(, cb.dest):
                print(f"neurons {name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X")
                for core_plm in cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs_of_cplm:
                        if (
                            neuron is neu_seg.target
                            and sub_neu.custom_index_set.intersection(neu_seg.index)
                        ):
                            print(
                                f"{name} placed in {_coord_to_bin_str(core_plm.coord)}\n"
                                f"N:        {neu_seg.n_neuron}\n"
                                f"Address:  {neu_seg._occupied_addr_repr}"
                            )

    def find_axon(self, neuron: Neuron, *, verbose: int = 0) -> None:
        self._build_check()
        dest = self.neuron_dest[neuron]
        print(f"{neuron.name} destinations:")
        print(dest)

    def _build_check(self) -> None:
        return self.graph.build_check()

    def _find_dest_cb_by_nseg(
        self, neu_seg: DendriteSegment, cb: CoreBlock
    ) -> list[CoreBlock]:
        succ_cbs = self.succ_core_blocks[cb]
        dest_cb_of_nseg = [cb for cb in succ_cbs if neu_seg.target in cb.ordered_axons]

        return dest_cb_of_nseg


def group_by(dict_: dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in dict_.values():
        d[keyfunc(item)].append(item)

    return d


def _cb_routable(
    routing_group: list[RoutingGroup], core_blocks: list[CoreBlock]
) -> bool:
    if len(core_blocks) == 1:
        return True

    for rg in routing_group:
        if core_blocks[0] in rg.core_blocks:
            return all(cb in rg.iter_nested_cb() for cb in core_blocks)

    return False


def _fp_check(fp: str | Path | None = None) -> Path:
    if fp is not None:
        _fp = Path(fp)
    else:
        _fp = _BACKEND_CONTEXT.output_dir

    if not _fp.is_dir():
        _fp.mkdir(parents=True, exist_ok=True)

    return _fp


def _calculate_core_consumption(order_rgs: list[RoutingGroup]) -> int:
    n_core_consumption: int = 0
    rg_consumption: list[int] = [
        1 << (rg.n_core_required - 1).bit_length() for rg in order_rgs
    ]
    rg_wasted: list[int] = [
        rg_consum - rg.n_core_required
        for rg, rg_consum in zip(order_rgs, rg_consumption)
    ]
    for wasted, consumption in zip(rg_wasted, rg_consumption):
        if consumption > HwConfig.N_CORE_OFFLINE:
            raise ValueError(
                "The number of required cores is out of range {0} ({1}).".format(
                    HwConfig.N_CORE_OFFLINE, consumption
                )
            )
        if n_core_consumption % consumption != 0:
            n_core_consumption = (
                n_core_consumption + consumption - n_core_consumption % consumption
            )
        temp_consumption = n_core_consumption + consumption
        temp_consumption = temp_consumption % HwConfig.N_CORE_MAX_INCHIP
        temp_consumption = (
            temp_consumption if temp_consumption != 0 else HwConfig.N_CORE_MAX_INCHIP
        )
        if temp_consumption - wasted > HwConfig.N_CORE_OFFLINE:
            n_core_consumption = (
                n_core_consumption
                + HwConfig.N_CORE_MAX_INCHIP
                - n_core_consumption % HwConfig.N_CORE_MAX_INCHIP
            )
        n_core_consumption += consumption
    return n_core_consumption


def reorder_routing_groups(
    graph: dict[RoutingGroup, list[RoutingGroup]],
) -> list[RoutingGroup]:
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for successor in graph[node]:
            in_degree[successor] += 1
    best_order = []
    min_core_consumption = HwConfig.N_CORE_MAX_INCHIP * _BACKEND_CONTEXT.n_target_chips

    # 辅助函数，用于生成所有可能的拓扑排序
    def backtrack(current_order: list[RoutingGroup]):
        nonlocal best_order, min_core_consumption
        if len(current_order) == len(graph):
            current_cost = _calculate_core_consumption(current_order)
            # print("current_order", current_order)
            # print("current_cost", current_cost)
            if current_cost < min_core_consumption:
                best_order = current_order.copy()
                min_core_consumption = current_cost
            return
        for node in graph:
            if in_degree[node] == 0 and node not in current_order:
                current_order.append(node)
                for successor in graph[node]:
                    in_degree[successor] -= 1
                backtrack(current_order)
                current_order.pop()
                for successor in graph[node]:
                    in_degree[successor] += 1

    backtrack([])
    print("best_order", best_order)
    print("min_cost", min_core_consumption)
    return best_order


OUT_OF_CORE_RESOURCE_TEXT = "the number of required cores is out of range {0} ({1})."
