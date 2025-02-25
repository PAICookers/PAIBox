import itertools
import logging
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Literal, Optional, Union

from paicorelib import ChipCoord, CoordOffset, HwConfig

from paibox import _logging
from paibox.base import SynSys
from paibox.components import Neuron
from paibox.exceptions import CompileError, ConfigInvalidError, ResourceError
from paibox.network import DynSysGroup

from ._slice import NeuronSlice, node_sl_lst_overlap, sl_overlap
from .conf_exporting import *
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
from .graph_utils import (
    find_cycles,
    get_node_degrees,
    get_succ_cb_by_node,
    merge_overlapping_sets,
)
from .graphs import PAIGraph
from .placement import CoreBlock, SourceDest, aligned_coords, max_lcn_of_cb
from .routing import RoutingGroup, RoutingManager
from .succ_group import *
from .types import NeuSegment, NodeDegree, NodeType, SourceNodeType, is_iw8

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
        self.routing_manager = RoutingManager(
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
        self.routing_manager.clear()

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
        set_cflag(multicast_optim=False)
        set_cflag(multicast_optim_nodes=())

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
        multicast_optim: Union[bool, Sequence[NodeType]] = False,
        **kwargs,
    ) -> GraphInfo:
        """Compile the network with optimization options.

        Args:
            weight_bit_optimization (bool): whether to optimize weight precision. For example, weights declared \
                as INT8 are treated as smaller precision based on their actual values (when the weight are all  \
                between [-8, 7], they can be treated as INT4). By default, it is specified by the corresponding \
                compile option in the backend configuration item. Default is true.
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

            multicast_optim (bool, Sequence[NodeType]): whether to perform multicast optimization. If true, the \
                optimization is performed on all nodes in the network. If passing a node list, the optimization \
                is attempted on the specified nodes only. Default is false.
                TODO A description of it is to be added

        Return: network information after compilation in dictionary format.
        """
        set_cflag(enable_wp_opt=weight_bit_optimization)
        set_cflag(grouping_optim_target=grouping_optim_target)
        set_cflag(no_twisted_branch=no_twisted_branch)

        # True, to optimize all nodes. A sequence, to optimize specified nodes
        if isinstance(multicast_optim, bool):
            set_cflag(multicast_optim=multicast_optim)
        elif isinstance(multicast_optim, Sequence):
            _mul_optim_nodes = tuple(node.name for node in multicast_optim)

            if any(node not in self.graph.nodes for node in _mul_optim_nodes):
                raise ValueError("not all specified nodes are in the graph.")

            set_cflag(multicast_optim=True)
            set_cflag(multicast_optim_nodes=_mul_optim_nodes)

        self._core_estimate_only = core_estimate_only

        # Preperation:
        # 1. Check whether the PAIGraph has built.
        # 2. Set global compilation flags.
        self._build_check()
        self._set_global_cflags()

        # Untwist the branch nodes if the flag is on.
        if no_twisted_branch:
            self.untwist_branch_nodes()

        self.graph.topo_support_check()

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

    def build_core_blocks(self) -> None:
        """Build core blocks based on partitioned edges."""
        # Graph partitioning
        merged_sgrps = self.graph.graph_partition()
        merged_sgrps = merge_cycles(merged_sgrps)

        # Build routing groups
        raw_rgrps: list[RoutingGroup] = []
        for msgrp in merged_sgrps:
            raw_rgrps.append(RoutingGroup.build(msgrp, is_root=True))

        # Record the optimized routing groups in the routing manager
        self.routing_manager.optimize_rgrps(raw_rgrps)

        for rg in self.routing_manager.routing_grps:
            rg.dump()

        log.info(
            "################################### Routing Group Builded, N_Core_Required Not Set Yet ###################################"
        )

        # Retrive the core blocks from routing groups
        for rg in self.routing_manager.routing_grps:
            self.core_blocks += rg.core_blocks

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
        self.routing_manager.build_rg_graph(self.succ_core_blocks)

    def _build_cb_graph(self, no_cb_cycle: bool = True) -> None:
        """Build the successor graph of core blocks.

        Args:
            no_cb_cycle (bool): whether to prohibit core blocks forming a cycle. Default is True. This  \
                situation has been solved in the previous steps.
        """
        # Impossible that the sucessor of one core block is itself (as a loop).
        assert all(
            not node_sl_lst_overlap(cb.dest, cb.ordered_axons)
            for cb in self.core_blocks
        )

        # Use `combinations` to traverse the core blocks pairs without duplication.
        # Generate (c1, (c2, c3, c4,...)), (c2, (c3, c4, c5,...)), (c3, (c4, c5, c6,...)), etc.
        for cb in self.core_blocks:
            self.succ_core_blocks[cb] = []

        for cur_cb, next_cb in itertools.combinations(self.core_blocks, 2):
            _ol_c2n = node_sl_lst_overlap(cur_cb.dest, next_cb.ordered_axons)
            _ol_n2c = node_sl_lst_overlap(next_cb.dest, cur_cb.ordered_axons)

            if no_cb_cycle:
                assert not (_ol_c2n and _ol_n2c)  # cannot be a cycle.

            if _ol_c2n:
                self.succ_core_blocks[cur_cb].append(next_cb)
            if _ol_n2c:
                self.succ_core_blocks[next_cb].append(cur_cb)

        for cur_cb, succ_cbs in self.succ_core_blocks.items():
            build_cb_log.debug(f"{cur_cb.name} Succ:")
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

            build_cb_log.debug(f"input core block of {inode.name}:")
            for cb in succ_cb:
                build_cb_log.debug(f"\t{cb.name}")

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN of each core block & set the target LCN.

        NOTE: The LCN of all successor core blocks of any core block must be the same. Meanwhile,   \
            the `target_lcn` of the core block is equal to that LCN.
        """
        # Adjust the `lcn_ex` of the input core blocks for each input node
        for input_cbs in self.input_core_blocks.values():
            if len(input_cbs) > 1:
                max_lcn_ex = max_lcn_of_cb(input_cbs)
                for icb in input_cbs:
                    icb.lcn_ex = max_lcn_ex

        for cb in self.core_blocks:
            succ_cbs = self.succ_core_blocks[cb]
            if succ_cbs:
                max_lcn_ex = (
                    max_lcn_of_cb(succ_cbs) if len(succ_cbs) > 1 else succ_cbs[0].lcn_ex
                )
                if len(succ_cbs) > 1:
                    for scb in succ_cbs:
                        scb.lcn_ex = max_lcn_ex

                cb.target_lcn = max_lcn_ex

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
        for rg in self.routing_manager.routing_grps:
            for cb in rg.iter_nested_cb():
                cb.group_neurons(
                    optim_target=_BACKEND_CONTEXT.cflags["grouping_optim_target"]
                )

            rg.set_core_required()
            rg.dump()

        log.info(
            "################################### Neuron Grouping Finished ###################################"
        )
        for cb in self.core_blocks:
            coord_asg_log.debug(cb)

        log.info(
            "################################### Required Cores Set ###################################"
        )

        # Optimize the order of routing groups
        # self.routing_grps = reorder_routing_groups(self.succ_rgrps)
        # self.ordered_rgrps = toposort(self.succ_rgrps)

        # Calculate the consumption of required physical cores.
        n_avail_cores = HwConfig.N_CORE_OFFLINE * _BACKEND_CONTEXT.n_target_chips
        n_core_required = sum(cb.n_core_required for cb in self.core_blocks)

        self.n_core_required = n_core_required

        # If only estimate the core usage, the rest of the steps are not performed.
        if core_estimate_only:
            return None

        if n_core_required > n_avail_cores:
            raise ResourceError(
                OUT_OF_CORE_RESOURCE_TEXT.format(n_avail_cores, n_core_required)
            )

        for rg in self.routing_manager.ordered_rgrps:
            self.routing_manager.place_routing_group(rg)

        log.info(
            "################################### Assignment Finished ###################################"
        )
        for rg in self.routing_manager.ordered_rgrps:
            rg.dump_routing_result()

        # Online cores are not counted in the number of occupied cores.
        self.n_core_occupied = self.routing_manager.get_n_core_occupied()

    def collect_neuron_dest(self) -> None:
        """Collect the destination details for neuron slices in each core block."""
        # Traverse all source node slices & their corresponding axon segments on the input axon side of core blocks.
        for cb in self.core_blocks:
            for source_slice, axon_seg in cb.axon_segments.items():
                self.neuron_dest[source_slice.target].add_dest(
                    source_slice, axon_seg, cb
                )

        log.info(
            "################################ Neuron Dest Info Collected ################################"
        )

        for source, dest in self.neuron_dest.items():
            dest.set_slice_dest_rid()
            dest.sort_slice_dest_pairs()

            ndest_collect.debug(f"source: {source.name}")
            ndest_collect.debug(dest)

    def core_allocation(self) -> None:
        """Allocate the routing groups to core placements level in topological order."""
        self.routing_manager.allocate_cp()

    def config_export(self) -> GraphInfo:
        """Export parameters of cores & neurons inside.

        Steps:
            1. Export the parameters(PARAMETER_REG, including RANDOM_SEED & Weight RAM) of cores.
            2. Export the parameters(Neuron RAM) of neurons inside.
        """
        if (ochip_coord := _BACKEND_CONTEXT["output_chip_addr"]) in _BACKEND_CONTEXT[
            "target_chip_addr"
        ]:
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
                    _BACKEND_CONTEXT["target_chip_addr"],
                    self.routing_manager.used_L2_clusters,
                ),
                "target_chip_list": _BACKEND_CONTEXT.target_chip_addr,
            },
        )

        self.graph_info = _graph_info

        return _graph_info

    def _set_global_cflags(self) -> None:
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
            slice_dest = dest.is_undivided_dest()

            axon_coords = aligned_coords(
                dest.slices[0],
                slice_dest.dest_axon,
                1,
                slice_dest.timeslot,
                is_iw8(slice_dest.rt_mode),
            )

            inp_neuron_dest = InputNeuronDest(
                [coord.tick_relative for coord in axon_coords],
                [coord.addr_axon for coord in axon_coords],
                slice_dest.base_coord.x,
                slice_dest.base_coord.y,
                slice_dest.rid.x,
                slice_dest.rid.y,
                slice_dest.dest_chip_coord.x,
                slice_dest.dest_chip_coord.y,
                slice_dest.timeslot,  # 1 << lcn_ex
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

        for rg in self.routing_manager.ordered_rgrps:
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
        fp: Optional[Union[str, Path]] = None,
        format: Literal["txt", "bin", "npy"] = "bin",
        split_by_chip: bool = False,
        export_core_params: bool = False,
        export_clk_en_L2: bool = False,
        use_hw_sim: bool = True,
    ) -> dict[ChipCoord, list[FrameArrayType]]:
        """Generate configuration frames & export to file.

        Args:
            - write_to_file: whether to write frames into file.
            - fp: If `write_to_file` is `True`, specify the output path.
            - format: `txt`, `bin`, or `npy`. `bin` is recommended.
            - split_by_chip: whether to split the generated frames file by the chips.
            - export_core_params: whether to export the parameters of occupied cores.
            - export_used_L2: whether to export the serial port data of the L2 cluster clocks.
            - use_hw_sim: whether to use hardware simulator. If use, '.bin' will be exported.

        Return: total configurations in dictionary format.
        """
        if self._core_estimate_only:
            raise CompileError(
                "the current compilation is only for core estimation. "
                "Please disable 'core_estimate_only' and compile again before exporting."
            )

        if format not in ("bin", "npy", "txt"):
            raise ValueError(f"format {format} is not supported.")

        formats = [format]
        if use_hw_sim:
            formats.append("bin")

        formats = list(set(formats))

        _fp = _fp_check(fp)
        config_dict = gen_config_frames_by_coreconf(
            self.graph_info["members"],
            write_to_file,
            _fp,
            split_by_chip,
            formats,
        )

        if export_core_params:
            # Export the parameters of occupied cores
            export_core_params_json(self.core_params, _fp)

        # Export the graph information
        export_graph_info(self.graph_info, _fp, export_clk_en_L2)

        return config_dict

    def find_neuron(
        self, neuron: Union[Neuron, NeuronSlice], *, verbose: int = 0
    ) -> None:
        self._build_check()
        neu_slice = neuron if isinstance(neuron, NeuronSlice) else NeuronSlice(neuron)
        name = neu_slice.target.name

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neu_slice.overlap(cb.dest):
                # NL_overlap(, cb.dest):
                print(f"neurons {name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X")
                for core_plm in cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs_of_cplm:
                        if neuron is neu_seg.target and sl_overlap(
                            neu_slice.index, neu_seg.index
                        ):
                            print(
                                f"{name} placed in {core_plm.coord}\n"
                                f"N:        {neu_seg.n_neuron}\n"
                                f"Address:  {neu_seg._addr_ram_repr}"
                            )

    def find_axon(self, neuron: Neuron, *, verbose: int = 0) -> None:
        self._build_check()
        dest = self.neuron_dest[neuron]

        for slice, slice_dest in zip(dest.slices, dest.dests):
            print(
                f"{neuron.name}[{slice}] dest: {slice_dest.base_coord}, {slice_dest.rid}\n"
                f"N:                {slice_dest.dest_axon.n_axon}\n"
                f"Address width:    {slice_dest.dest_axon.addr_width}\n"
                f"Address offset:   {slice_dest.dest_axon.addr_offset}"
            )

        # for cb in self.core_blocks:
        #     # Find neuron in one or more core blocks.
        #     if neuron in cb.ordered_axons:
        #         print(f"axons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X")
        #         axon_segment = cb.axon_segments[neuron]
        #         print(
        #             f"{neuron.name} placed in {cb.core_coords}\n"
        #             f"N:                {axon_segment.n_axon}\n"
        #             f"Address width:    {axon_segment.addr_width}\n"
        #             f"Address offset:   {axon_segment.addr_offset}"
        #         )

    def _build_check(self) -> None:
        return self.graph.build_check()

    def _find_dest_cb_by_nseg(
        self, neu_seg: NeuSegment, cb: CoreBlock
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


def _fp_check(fp: Optional[Union[str, Path]] = None) -> Path:
    if fp is not None:
        _fp = Path(fp)
    else:
        _fp = _BACKEND_CONTEXT["build_directory"]

    if not _fp.is_dir():
        _fp.mkdir(parents=True, exist_ok=True)

    return _fp


def merge_cycles(merged_sgrps: list[MergedSuccGroup]) -> list[MergedSuccGroup]:
    """Detects cycles among merged successor groups & merges them into a minimal set of     \
        disjoint groups.

    Args:
        merged_sgrps (list[MergedSuccGroup]): A list of already merged successor groups to  \
            be analyzed for cycles.

    Returns:
        out (list[MergedSuccGroup]): A new list of merged successor groups with detected    \
            cycles resolved.
    """
    succ_merged_sgrps: dict[MergedSuccGroup, list[MergedSuccGroup]] = defaultdict(list)

    for cur_m, next_m in itertools.combinations(merged_sgrps, 2):
        # (cur_m, (m2, m3, ...)), (m2, (m3, m4, ...)), ...
        if not cur_m.nodes.isdisjoint(next_m.inputs):
            succ_merged_sgrps[cur_m].append(next_m)

        if not next_m.nodes.isdisjoint(cur_m.inputs):
            succ_merged_sgrps[next_m].append(cur_m)

    cycles = find_cycles(succ_merged_sgrps)
    merged_cycles = merge_overlapping_sets(cycles)

    merged: list[MergedSuccGroup] = []
    remaining = set(merged_sgrps)
    for mc in merged_cycles:
        merged.append(MergedSuccGroup.merge(mc))
        remaining.difference_update(mc)

    merged.extend(remaining)
    return merged


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
    graph: dict[RoutingGroup, list[RoutingGroup]]
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
