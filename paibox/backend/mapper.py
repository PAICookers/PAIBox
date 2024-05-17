import sys
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from paicorelib import Coord, CoordOffset, HwConfig, get_replication_id

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import ConfigInvalidError, ResourceError
from paibox.network import DynSysGroup

from .conf_template import (
    CoreConfig,
    CorePlmConf,
    GraphInfo,
    InputNodeConf,
    NeuronDest,
    OutputDestConf,
    export_core_params_json,
    export_input_conf_json,
    export_output_conf_json,
    gen_config_frames_by_coreconf,
)
from .context import _BACKEND_CONTEXT, set_cflag
from .graphs import PAIGraph, convert2routing_groups, get_node_degrees
from .graphs_types import NodeDegree, SourceNodeType
from .placement import CoreBlock, aligned_coords, max_lcn_of_cb
from .routing import RoutingGroup, RoutingRoot
from .segment_utils import NeuSeg

__all__ = ["Mapper"]


class Mapper:
    graph = PAIGraph()
    graph_info: GraphInfo

    def __init__(self) -> None:
        self.core_blocks: List[CoreBlock] = []
        """List for core blocks in the network."""
        self.succ_core_blocks: Dict[CoreBlock, List[CoreBlock]] = defaultdict(list)
        self.input_core_blocks: Dict[SourceNodeType, List[CoreBlock]] = defaultdict(
            list
        )
        """List of input core blocks for each input node."""

        self.degrees_of_cb: Dict[CoreBlock, NodeDegree] = defaultdict(NodeDegree)
        self.routing_groups: List[RoutingGroup] = []

        self.core_plm_config: CorePlmConf = defaultdict(dict)
        self.core_params: Dict[Coord, CoreConfig] = dict()
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
        weight_bit_optimization: Optional[bool] = None,
        grouping_optim_target: Optional[Literal["latency", "core", "both"]] = None,
    ) -> GraphInfo:
        """Compile the network with optimization options.

        Args:
            - weight_bit_optimization: whether to optimize weight precision. For example, weights declared as   \
                INT8 are treated as smaller precision based on their actual values (when the weight are all     \
                between [-8, 7], they can be treated as INT4). By default, it is specified by the corresponding \
                compile option in the backend configuration item (enabled by default).
            - grouping_optim_target: specify the optimization goal of neuron grouping, which can be `latency`,  \
                `core` or `both`, which respectively represent the optimization goal of delay/throughput,       \
                occupied cores, or both. The default is specified by the corresponding compilation option in the\
                backend configuration item (`both` by default).

        Return: network information after compilation in dictionary format.
        """
        if weight_bit_optimization is not None:
            set_cflag(enable_wp_opt=weight_bit_optimization)

        if grouping_optim_target is not None:
            set_cflag(grouping_optim_target=grouping_optim_target)

        """1. Check whether the PAIGraph has built."""
        self._build_check()

        """2. Set global compilation flags."""
        self._set_global_cflags()

        """3. Build core blocks."""
        self.build_core_blocks()

        """4. Adjust the LCN extension of each core block."""
        self.lcn_ex_adjustment()

        """5. Core coordinate assignment."""
        self.coord_assign()

        """6. Allocate the core blocks to the `CorePlacement`."""
        self.core_allocation()

        """7. Export parameters."""
        return self.config_export()

    def build_core_blocks(self) -> None:
        """Build core blocks based on grouped edges.

        Description: Group all edges & build `CoreBlock` based on the grouped edges.
        """
        grouped_edges, routing_groups_id = self.graph.group_edges()

        if sys.version_info >= (3, 10):
            for syns, routing_id in zip(grouped_edges, routing_groups_id, strict=True):
                self.core_blocks.append(
                    CoreBlock.build(*syns, seed=0, routing_id=routing_id)
                )
        else:
            if len(grouped_edges) != len(routing_groups_id):
                raise ValueError(
                    f"the length of grouped edges & routing groups id are not equal, "
                    f"{len(grouped_edges)} != {len(routing_groups_id)}"
                )

            for syns, routing_id in zip(grouped_edges, routing_groups_id):
                self.core_blocks.append(CoreBlock.build(*syns, routing_id=routing_id))

        for cb in self.core_blocks:
            succ_cbs = list(
                filter(
                    lambda succ_cb: any(d for d in cb.dest if d in succ_cb.source),
                    self.core_blocks,
                )
            )
            self.succ_core_blocks[cb].extend(succ_cbs)

        for inode in self.graph.inodes.values():
            # TODO How to prevent this situation: there is input node & predecessor nodes
            # in a certain core blocks.

            # Disconnected input nodes will not be recorded.
            succ_cb = [cb for cb in self.core_blocks if inode in cb.source]
            if len(succ_cb) > 0:
                self.input_core_blocks[inode] = succ_cb

        self.degrees_of_cb = get_node_degrees(self.succ_core_blocks)

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN extension of each core block."""
        # In the absence of the above complex situations, the following judgment is useless.
        # But it'd be better to add this lcn adjustment.
        for input_cbs in self.input_core_blocks.values():
            if len(input_cbs) > 1:
                max_lcn_ex = max_lcn_of_cb(input_cbs)
                # Adjust the `lcn_ex` of the input core blocks for each input node
                for g in input_cbs:
                    g.lcn_ex = max_lcn_ex

        for cb in self.core_blocks:
            succ_cb = self.succ_core_blocks[cb]

            if len(succ_cb) > 1:
                max_lcn_ex = max_lcn_of_cb(succ_cb)
                # Adjust the `lcn_ex` of the following core blocks
                for g in succ_cb:
                    g.lcn_ex = max_lcn_ex

                # Adjust `target_lcn` of itself & lock
                cb.target_lcn = max_lcn_ex
                cb.lcn_locked = True
            elif len(succ_cb) == 1:
                # Adjust `target_lcn` of itself & lock
                cb.target_lcn = succ_cb[0].lcn_ex
                cb.lcn_locked = True
            else:
                # Doesn't have following core blocks
                cb.lcn_locked = True

    def coord_assign(self) -> None:
        """Assign the coordinate of each `CorePlacement`.

        NOTE: The neurons in each core block must be grouped first to determine the \
            #N of cores required, and then the routing coordinates can be assigned.
        """
        for cb in self.core_blocks:
            # Group the neurons, get the #N of cores required.
            cb.group_neurons(
                optim_target=_BACKEND_CONTEXT.cflags["grouping_optim_target"]
            )

        # Calculate the consumption of required physical cores.
        n_avail_cores = HwConfig.N_CORE_OFFLINE * _BACKEND_CONTEXT.n_target_chips
        if (
            n_core_required := sum(cb.n_core_required for cb in self.core_blocks)
        ) > n_avail_cores:
            raise ResourceError(
                CORE_RESOURCE_OUT_OF_RANGE_TEXT.format(n_avail_cores, n_core_required)
            )

        self.n_core_required = n_core_required

        # Generate routing groups by given the list of core blocks.
        routing_groups = convert2routing_groups(
            self.succ_core_blocks, self.degrees_of_cb, self.input_core_blocks
        )
        for rg in routing_groups:
            self.routing_tree.insert_routing_group(rg)

        self.routing_groups = routing_groups

        # Calculate the consumption of occupied physical cores.
        if (
            n_core_occupied := sum(rg.get_n_core_occupied() for rg in routing_groups)
        ) > n_avail_cores:
            raise ResourceError(
                CORE_RESOURCE_OUT_OF_RANGE_TEXT.format(n_avail_cores, n_core_occupied)
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
                f"The output chip address  {ochip_coord} should not overlap with the "
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
            extras={"name": self.graph.graph_name_repr},
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
                "tick_relative": [...],
                "addr_axon": [...]
            },
            "inp2_1": {...} # as input node #2
        }
        """
        input_nodes_info = dict()

        # Traverse input core blocks
        for inode, input_cbs in self.input_core_blocks.items():
            dest_coords: List[Coord] = []

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

            neuron_dest = NeuronDest(
                [coord.tick_relative for coord in axon_coords],
                [coord.addr_axon for coord in axon_coords],
                dest_coords[0].x,
                dest_coords[0].y,
                dest_rid.x,
                dest_rid.y,
                input_cb.chip_coord.x,
                input_cb.chip_coord.y,
            )

            input_nodes_info[inode.name] = neuron_dest

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
                self.core_params.update(
                    CoreBlock.export_core_plm_config(member_cb)
                )  # compatible for py3.8

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
                    assert _cb_routable(self.routing_groups, dest_cb_of_nseg)
                    core_plm.export_neu_config(neu_seg, dest_cb_of_nseg)
                else:
                    offset_idx = o_nodes.index(neu_seg.parent)

                    if hasattr(CoordOffset, "from_offset"):
                        # For paicorelib > 0.0.13
                        cur_ocoord = ocoord + CoordOffset.from_offset(offset_idx)
                    else:
                        # For paicorelib <= 0.0.13
                        cur_ocoord = ocoord + CoordOffset(
                            offset_idx // 32, offset_idx % 32
                        )

                    output_axon_offset = core_plm.export_neu_config(
                        neu_seg,
                        output_core_coord=cur_ocoord,
                        axon_addr_offset=output_axon_offset,
                    )
                    output_dest_info[neu_seg.parent.name][core_plm.coord.address] = (
                        core_plm.neu_configs[neu_seg.parent].neuron_dest_info
                    )

        # Add the offset as the starting coordinate of the next output node
        return cur_ocoord + CoordOffset(1, 0)

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
                offset_idx = o_nodes.index(neu_seg.parent)

                if hasattr(CoordOffset, "from_offset"):
                    # For paicorelib > 0.0.13
                    cur_ocoord = ocoord + CoordOffset.from_offset(offset_idx)
                else:
                    # For paicorelib <= 0.0.13
                    cur_ocoord = ocoord + CoordOffset(offset_idx // 32, offset_idx % 32)

                output_axon_offset = core_plm.export_neu_config(
                    neu_seg,
                    output_core_coord=cur_ocoord,
                    axon_addr_offset=output_axon_offset,
                )
                output_dest_info[neu_seg.parent.name][core_plm.coord.address] = (
                    core_plm.neu_configs[neu_seg.parent].neuron_dest_info
                )

        return cur_ocoord

    def export(
        self,
        write_to_file: bool = True,
        *,
        fp: Optional[Union[str, Path]] = None,
        format: Literal["txt", "bin", "npy"] = "bin",
        split_by_coord: bool = False,
        export_core_params: bool = False,
    ) -> Dict[Coord, Any]:
        """Generate configuration frames & export to file.

        Args:
            - write_to_file: whether to write frames into file.
            - fp: If `write_to_file` is `True`, specify the output path.
            - format: `txt`, `bin`, or `npy`. `bin` is recommended.
            - split_by_coord: whether to split the generated frames file by the core coordinates.
            - export_core_params: whether to export the parameters of occupied cores.

        Return: a dictionary of configurations.
        """
        if format not in ("bin", "npy", "txt"):
            raise ValueError(f"format {format} is not supported.")

        _fp = _fp_check(fp)
        config_dict = gen_config_frames_by_coreconf(
            self.graph_info["members"], write_to_file, _fp, split_by_coord, format
        )

        if export_core_params:
            # Export the parameters of occupied cores
            export_core_params_json(self.core_params, _fp)

        # Export the configurations of input nodes
        export_input_conf_json(self.graph_info["input"], _fp)
        # Export the configurations of output destinations
        export_output_conf_json(self.graph_info["output"], _fp)

        return config_dict

    def find_neuron(self, neuron: NeuDyn, *, verbose: int = 0) -> None:
        self._build_check()

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neuron in cb.dest:
                print(
                    f"neurons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X"
                )
                for core_plm in cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs_of_cplm:
                        if neuron is neu_seg.parent:
                            print(
                                f"{neuron.name} placed in {core_plm.coord}\n"
                                f"N:        {neu_seg.segment.n_neuron}\n"
                                f"Address:  {neu_seg.segment.addr_slice}"
                            )

    def find_axon(self, neuron: NeuDyn, *, verbose: int = 0) -> None:
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

    def _find_dest_cb_by_nseg(self, neu_seg: NeuSeg, cb: CoreBlock) -> List[CoreBlock]:
        succ_cbs = self.succ_core_blocks[cb]
        dest_cb_of_nseg = [cb for cb in succ_cbs if neu_seg.parent in cb.source]

        return dest_cb_of_nseg


def group_by(dict_: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in dict_.values():
        d[keyfunc(item)].append(item)

    return d


def _cb_routable(
    routing_group: List[RoutingGroup], core_blocks: List[CoreBlock]
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


CORE_RESOURCE_OUT_OF_RANGE_TEXT = (
    "the number of required cores is out of range {0} ({1})."
)
