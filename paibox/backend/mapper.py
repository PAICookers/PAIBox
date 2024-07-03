from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Literal, Optional, Union

from paicorelib import ChipCoord, Coord, CoordOffset, HwConfig, get_replication_id

from paibox.base import SynSys
from paibox.components import Neuron
from paibox.exceptions import ConfigInvalidError, ResourceError
from paibox.network import DynSysGroup

from .conf_template import (
    CoreConf,
    CorePlmConf,
    FrameArrayType,
    GraphInfo,
    InputNeuronDest,
    InputNodeConf,
    OutputDestConf,
    _get_clk_en_L2_dict,
    export_core_params_json,
    export_input_conf_json,
    export_output_conf_json,
    export_used_L2_clusters,
    gen_config_frames_by_coreconf,
)
from .context import _BACKEND_CONTEXT, set_cflag
from .graphs import (
    PAIGraph,
    convert2routing_groups,
    get_node_degrees,
    get_succ_cb_by_node,
    toposort,
)
from .placement import CoreBlock, aligned_coords, max_lcn_of_cb
from .routing import RoutingGroup, RoutingRoot
from .types import NeuSegment, NodeDegree, NodeType, SourceNodeType

__all__ = ["Mapper"]


class Mapper:
    graph = PAIGraph()
    graph_info: GraphInfo

    def __init__(self) -> None:
        self.core_blocks: list[CoreBlock] = []
        """List for core blocks in the network."""
        self.succ_core_blocks: dict[CoreBlock, list[CoreBlock]] = defaultdict(list)
        self.input_core_blocks: dict[SourceNodeType, list[CoreBlock]] = defaultdict(
            list
        )
        """List of input core blocks for each input node."""

        self.degrees_of_cb: dict[CoreBlock, NodeDegree] = defaultdict(NodeDegree)
        self.routing_groups: list[RoutingGroup] = []
        self.succ_routing_groups: dict[RoutingGroup, list[RoutingGroup]] = dict()

        self.core_plm_config: CorePlmConf = defaultdict(dict)
        self.core_params: CoreConf = defaultdict(dict)
        """The dictionary of core parameters."""

        self.n_core_required = 0
        self.n_core_occupied = 0
        self.routing_tree = RoutingRoot(chip_list=_BACKEND_CONTEXT["target_chip_addr"])

        self.clear()

    def clear(self) -> None:
        self.routing_tree.clear()
        self.graph.clear()

        self.core_blocks.clear()
        self.succ_core_blocks.clear()
        self.input_core_blocks.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

        self.n_core_required = 0
        self.n_core_occupied = 0

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
        no_twisted_branch: bool = True,
        multicast_optim: Union[bool, Sequence[NodeType]] = False,
        **kwargs,
    ) -> GraphInfo:
        """Compile the network with optimization options.

        Args:
            - weight_bit_optimization: whether to optimize weight precision. For example, weights declared as   \
                INT8 are treated as smaller precision based on their actual values (when the weight are all     \
                between [-8, 7], they can be treated as INT4). By default, it is specified by the corresponding \
                compile option in the backend configuration item. Default is true.
            - grouping_optim_target: specify the optimization goal of neuron grouping, which can be `latency`,  \
                `core` or `both`, which respectively represent the optimization goal of delay/throughput,       \
                occupied cores, or both. The default is specified by the corresponding compilation option in the\
                backend configuration item. Default is 'both'.
            - no_twisted_branch: when parsing the network topology, whether or not to prohibit intersecting     \
                branch structures will cause such structures to be processed. For example:

                I -> A -> B -> C
                       ------>

                The out-degree of node A is > 1, and its successor node C has an in-degree > 1. If `no_twisted_branch`    \
                is true, A will be copied & denoted as A', whose forward connection is preserved.

                I -> A -> B -> C
                  -> A'------>

                Default is true.

            - multicast_optim (in dev): whether to perform multicast optimization. If true, the optimization is \
                performed on all nodes in the network. If a node list is passed, the optimization is attempted  \
                on the specified nodes only. Default is false.
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

            if any(node not in self.graph._raw_nodes for node in _mul_optim_nodes):
                raise ValueError("not all specified nodes are in the graph.")

            set_cflag(multicast_optim=True)
            set_cflag(multicast_optim_nodes=_mul_optim_nodes)

        """Preperation.
            1. Check whether the PAIGraph has built.
            2. Set global compilation flags.
        """
        self._build_check()
        self._set_global_cflags()

        """Untwist the branch nodes if flag is on."""
        if no_twisted_branch:
            self.untwist_branch_nodes()

        self.graph.topo_support_check()

        """Build core blocks."""
        self.build_core_blocks()

        """Adjust the LCN extension of each core block."""
        self.lcn_ex_adjustment()

        """Group the axons of core block."""
        self.cb_axon_grouping()

        # Convert core blocks to routing groups
        self.routing_groups, self.succ_routing_groups = convert2routing_groups(
            self.succ_core_blocks, self.degrees_of_cb, self.input_core_blocks
        )

        """Core coordinate assignment."""
        self.coord_assign(core_estimate_only)

        if core_estimate_only:
            return GraphInfo(
                input={},
                output={},
                members={},
                inherent_timestep=self.graph.inherent_timestep,
                n_core_required=self.n_core_required,
                n_core_occupied=0,
                misc={"name": self.graph.graph_name_repr},
            )

        """Allocate the core blocks to the core placments."""
        self.core_allocation()

        """Export configurations."""
        return self.config_export()

    def untwist_branch_nodes(self) -> None:
        self.graph.untwist_branch_nodes()

    def build_core_blocks(self) -> None:
        """Build core blocks based on partitioned edges."""
        partitioned_edges = self.graph.graph_partition()

        for part in partitioned_edges:
            self.core_blocks.append(
                CoreBlock.build(*part.edges, seed=0, routing_id=part.rg_id)
            )

        for cur_cb in self.core_blocks:
            succ_cbs = []
            # cur_cb == cb is possible
            for cb in self.core_blocks:
                if any(d for d in cur_cb.dest if d in cb.source):
                    succ_cbs.append(cb)

            self.succ_core_blocks[cur_cb] = succ_cbs

        for inode in self.graph.inodes.values():
            # TODO How to prevent this situation: there is input node & predecessor nodes
            # in a certain core blocks.

            # Disconnected input nodes will not be recorded.
            succ_cb = get_succ_cb_by_node(inode, self.core_blocks)
            if len(succ_cb) > 0:
                self.input_core_blocks[inode] = succ_cb

        self.degrees_of_cb = get_node_degrees(self.succ_core_blocks)

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN of each core block & set target LCN."""
        # In the absence of the above complex situations, the following judgment is useless.
        # But it'd be better to add this lcn adjustment.
        for input_cbs in self.input_core_blocks.values():
            if len(input_cbs) > 1:
                max_lcn_ex = max_lcn_of_cb(input_cbs)
                # Adjust the `lcn_ex` of the input core blocks for each input node
                for g in input_cbs:
                    g.lcn_ex = max_lcn_ex

        for cb in self.core_blocks:
            succ_cbs = self.succ_core_blocks[cb]

            if len(succ_cbs) > 1:
                max_lcn_ex = max_lcn_of_cb(succ_cbs)
                # Adjust the `lcn_ex` of the following core blocks
                for _cb in succ_cbs:
                    _cb.lcn_ex = max_lcn_ex

                # Adjust `target_lcn` of itself & lock
                cb.target_lcn = max_lcn_ex
            elif len(succ_cbs) == 1:
                # Adjust `target_lcn` of itself & lock
                cb.target_lcn = succ_cbs[0].lcn_ex

            cb._lcn_locked = True

    def cb_axon_grouping(self) -> None:
        """The axons are grouped after the LCN has been modified & locked."""
        for cb in self.core_blocks:
            cb.group_axons()

    def graph_optimization(self) -> None:
        optimized = self.graph.graph_optimization(self.core_blocks, self.routing_groups)
        if optimized:
            self.core_blocks.clear()
            self.succ_core_blocks.clear()
            self._build_check()
            self.build_core_blocks()
            self.lcn_ex_adjustment()

    def coord_assign(self, core_estimate_only: bool) -> None:
        """Assign the coordinate of each `CorePlacement`.

        NOTE: The neurons in each core block must be grouped first to determine the \
            #N of cores required, and then the routing coordinates can be assigned.
        """
        for cb in self.core_blocks:
            # Group the neurons, get the #N of cores required.
            cb.group_neurons(
                optim_target=_BACKEND_CONTEXT.cflags["grouping_optim_target"]
            )

        # Optimize the order of routing groups
        # self.routing_groups = reorder_routing_groups(self.succ_routing_groups)
        self.routing_groups = toposort(self.succ_routing_groups)
        # Calculate the consumption of required physical cores.
        n_avail_cores = HwConfig.N_CORE_OFFLINE * _BACKEND_CONTEXT.n_target_chips
        n_core_required = sum(cb.n_core_required for cb in self.core_blocks)

        self.n_core_required = n_core_required

        if core_estimate_only:
            return None
        elif n_core_required > n_avail_cores:
            raise ResourceError(
                OUT_OF_CORE_RESOURCE_TEXT.format(n_avail_cores, n_core_required)
            )

        for rg in self.routing_groups:
            self.routing_tree.place_routing_group(rg)

        # Calculate the consumption of occupied physical cores.
        if (
            n_core_occupied := sum(
                rg.get_n_core_occupied() for rg in self.routing_groups
            )
        ) > n_avail_cores:
            raise ResourceError(
                OUT_OF_CORE_RESOURCE_TEXT.format(n_avail_cores, n_core_occupied)
            )

        self.n_core_occupied = n_core_occupied

    def core_allocation(self) -> None:
        """Allocate the routing groups to core placements level."""
        for rg in self.routing_groups:
            rg.core_block_alloc()

    def config_export(self) -> GraphInfo:
        """Export parameters of cores & neurons inside.

        Steps:
            - 1. Export the parameters(PARAMETER_REG, including RANDOM_SEED \
                & Weight RAM) of cores.
            - 2. Export the parameters(Neuron RAM) of neurons inside.
        """
        if (ochip_coord := _BACKEND_CONTEXT["output_chip_addr"]) in _BACKEND_CONTEXT[
            "target_chip_addr"
        ]:
            raise ConfigInvalidError(
                f"the output chip address {ochip_coord} should not overlap with the "
                f"chip addresses, but got {_BACKEND_CONTEXT._target_chip_addr_repr()}."
            )

        input_nodes_info = self._inpproj_config_export()
        output_dest_info = self._member_cb_and_onode_config_export()

        _graph_info = GraphInfo(
            input=input_nodes_info,
            output=output_dest_info,
            members=self.core_plm_config,  # The configuration of physical cores is in `core_plm_config`
            inherent_timestep=self.graph.inherent_timestep,
            n_core_required=self.n_core_required,
            n_core_occupied=self.n_core_occupied,
            misc={
                "name": self.graph.graph_name_repr,
                "clk_en_L2": _get_clk_en_L2_dict(
                    _BACKEND_CONTEXT["target_chip_addr"],
                    self.routing_tree.used_L2_clusters,
                ),
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
        input_nodes_info = dict()

        # Traverse input core blocks
        for inode, input_cbs in self.input_core_blocks.items():
            dest_coords: list[Coord] = []

            assert all(input_cbs[0].chip_coord == cb.chip_coord for cb in input_cbs)
            for cb in input_cbs:  # Do not use iterative generation.
                dest_coords.extend(cb.core_coords)

            dest_rid = get_replication_id(dest_coords)

            # The arrangement of axons is the same for the rest of `input_cbs`.
            # LCN of `input_cbs` are the same.
            input_cb = input_cbs[0]
            axon_coords = aligned_coords(
                slice(0, input_cb.n_axon_of(input_cb.source.index(inode)), 1),
                input_cb.axon_segments[inode],
                1,
                input_cb.n_timeslot,
            )

            inp_neuron_dest = InputNeuronDest(
                [coord.tick_relative for coord in axon_coords],
                [coord.addr_axon for coord in axon_coords],
                dest_coords[0].x,
                dest_coords[0].y,
                dest_rid.x,
                dest_rid.y,
                input_cb.chip_coord.x,
                input_cb.chip_coord.y,
                input_cb.n_timeslot,  # 1 << lcn_ex
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
        output_dest_info = defaultdict(dict)
        # Shallow copy
        ocoord = copy(_BACKEND_CONTEXT["output_core_addr_start"])

        for rg in self.routing_groups:
            for member_cb in rg:
                self.core_params[rg.chip_coord] |= CoreBlock.export_core_plm_config(
                    member_cb
                )

                if self.degrees_of_cb[member_cb].out_degree == 0:
                    # member_cb is a pure output core block. All neu_segs inside are output neurons.
                    ocoord = self._onode_cb_config_export(
                        member_cb, output_dest_info, ocoord
                    )
                elif any(d in self.graph.onodes.values() for d in member_cb.dest):
                    # member_cb is both a member & output core block.
                    ocoord = self._member_onode_cb_config_export(
                        member_cb, output_dest_info, ocoord
                    )
                else:
                    # member_cb is a pure member.
                    self._member_cb_config_export(member_cb)

                for coord, core_plm in member_cb.core_placements.items():
                    self.core_plm_config[rg.chip_coord][
                        coord
                    ] = core_plm.export_core_plm_config()

            # Generate default configurations for wasted core placements of the routing group
            self.core_plm_config[rg.chip_coord].update(rg.get_wasted_cplm_config())

        return output_dest_info

    def _member_cb_config_export(self, member_cb: CoreBlock) -> None:
        """Export configuration information for core blocks that are pure members."""
        for core_plm in member_cb.core_placements.values():
            for neu_seg in core_plm.neu_segs_of_cplm:
                # Find the axon destinations of neu_seg, not the successor core blocks.
                dest_cb_of_nseg = self._find_dest_cb_by_nseg(neu_seg, member_cb)

                if len(dest_cb_of_nseg) > 0:
                    assert _cb_routable(self.routing_groups, dest_cb_of_nseg)
                    core_plm.export_neu_config(neu_seg, dest_cb_of_nseg)
                else:
                    raise ValueError(f"find destination of member {neu_seg} failed.")

    def _member_onode_cb_config_export(
        self,
        member_onode_cb: CoreBlock,
        output_dest_info: OutputDestConf,
        ocoord: Coord,
    ) -> Coord:
        """Export configuration information for core blocks that are both members & output."""
        cur_ocoord = ocoord
        output_axon_offset = 0
        o_nodes = [d for d in member_onode_cb.dest if d in self.graph.onodes.values()]

        for core_plm in member_onode_cb.core_placements.values():
            for neu_seg in core_plm.neu_segs_of_cplm:
                dest_cb_of_nseg = self._find_dest_cb_by_nseg(neu_seg, member_onode_cb)

                if len(dest_cb_of_nseg) > 0:
                    # The destination of the neuron segment is another core block(s)
                    assert _cb_routable(self.routing_groups, dest_cb_of_nseg)
                    core_plm.export_neu_config(neu_seg, dest_cb_of_nseg)
                else:
                    # The destination of the neuron segment is outside of the chip(s)
                    offset_idx = o_nodes.index(neu_seg.target)
                    cur_ocoord = ocoord + CoordOffset.from_offset(offset_idx)
                    output_axon_offset = core_plm.export_neu_config(
                        neu_seg,
                        output_core_coord=cur_ocoord,
                        axon_addr_offset=output_axon_offset,
                    )
                    output_dest_info[neu_seg.target.name][core_plm.coord.address] = (
                        core_plm.neu_configs[neu_seg.target].neuron_dest_info
                    )

        # Add the offset as the starting coordinate of the next output node
        return cur_ocoord + CoordOffset.from_offset(1)

    def _onode_cb_config_export(
        self, onode_cb: CoreBlock, output_dest_info: OutputDestConf, ocoord: Coord
    ) -> Coord:
        """Export configuration information for core blocks that are pure output."""
        cur_ocoord = ocoord
        output_axon_offset = 0
        o_nodes = [d for d in onode_cb.dest if d in self.graph.onodes.values()]

        for core_plm in onode_cb.core_placements.values():
            for neu_seg in core_plm.neu_segs_of_cplm:
                # Get the output coordinate of this neu_seg
                offset_idx = o_nodes.index(neu_seg.target)
                cur_ocoord = ocoord + CoordOffset.from_offset(offset_idx)
                output_axon_offset = core_plm.export_neu_config(
                    neu_seg,
                    output_core_coord=cur_ocoord,
                    axon_addr_offset=output_axon_offset,
                )
                output_dest_info[neu_seg.target.name][core_plm.coord.address] = (
                    core_plm.neu_configs[neu_seg.target].neuron_dest_info
                )

        # Add the offset as the starting coordinate of the next output node
        return cur_ocoord + CoordOffset.from_offset(1)

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

        # Export the configurations of input nodes
        export_input_conf_json(self.graph_info["input"], _fp)
        # Export the configurations of output destinations
        export_output_conf_json(self.graph_info["output"], _fp)

        # Export the serial port data of the L2 cluster clocks
        if export_clk_en_L2:
            export_used_L2_clusters(self.graph_info["misc"]["clk_en_L2"], _fp)

        return config_dict

    def find_neuron(self, neuron: Neuron, *, verbose: int = 0) -> None:
        self._build_check()

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neuron in cb.dest:
                print(
                    f"neurons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X"
                )
                for core_plm in cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs_of_cplm:
                        if neuron is neu_seg.target:
                            print(
                                f"{neuron.name} placed in {core_plm.coord}\n"
                                f"N:        {neu_seg.n_neuron}\n"
                                f"Address:  {neu_seg.addr_slice}"
                            )

    def find_axon(self, neuron: Neuron, *, verbose: int = 0) -> None:
        self._build_check()

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neuron in cb.source:
                print(f"axons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X")
                axon_segment = cb.axon_segments[neuron]
                print(
                    f"{neuron.name} placed in {cb.core_coords}\n"
                    f"N:                {axon_segment.n_axon}\n"
                    f"Address width:    {axon_segment.addr_width}\n"
                    f"Address offset:   {axon_segment.addr_offset}"
                )

    def _build_check(self) -> None:
        return self.graph.build_check()

    def _find_dest_cb_by_nseg(
        self, neu_seg: NeuSegment, cb: CoreBlock
    ) -> list[CoreBlock]:
        succ_cbs = self.succ_core_blocks[cb]
        dest_cb_of_nseg = [cb for cb in succ_cbs if neu_seg.target in cb.source]

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
        if core_blocks[0] in rg:
            return all(cb in rg for cb in core_blocks)

    return False


def _fp_check(fp: Optional[Union[str, Path]] = None) -> Path:
    if fp is not None:
        _fp = Path(fp)
    else:
        _fp = _BACKEND_CONTEXT["build_directory"]

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
