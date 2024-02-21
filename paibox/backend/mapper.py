from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from paicorelib import (
    Coord,
    CoordLike,
    CoordOffset,
    HwConfig,
    get_replication_id,
    to_coord,
)

from paibox.base import NeuDyn
from paibox.exceptions import ResourceError
from paibox.network import DynSysGroup

from .conf_template import (
    CoreConfig,
    CorePlacementInfo,
    GraphInfo,
    InputNodeInfo,
    NeuronDest,
    OutputDestInfo,
    export_core_params_json,
    export_inp_nodes_conf_json,
    export_outp_dests_conf_json,
    gen_config_frames_by_coreconf,
)
from .context import _BACKEND_CONTEXT, set_cflag
from .graphs import PAIGraph, convert2routing_groups, get_node_degrees
from .graphs_types import NodeDegree
from .placement import CoreBlock, aligned_coords, max_lcn_of_cb
from .routing import RoutingGroup, RoutingRoot

__all__ = ["Mapper"]


class Mapper:
    """
        Responsible for integrating all backend operation processes & \
        providing functions for debugging.
        TODO It doesn't collect information during the build process.
    """

    routing_tree = RoutingRoot(tag="L5")
    """The routing tree root."""
    graph = PAIGraph()

    def __init__(self) -> None:
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
        self.degrees_of_cb: Dict[CoreBlock, NodeDegree] = defaultdict(NodeDegree)

        self.core_params: Dict[Coord, CoreConfig] = dict()
        """The dictionary of core parameters. Structure:
        {
            address of core: {
                parameters...
            }
        }
        """
        self.routing_groups: List[RoutingGroup] = []

        self.core_plm_config: CorePlacementInfo = dict()
        self.graph_info: GraphInfo

        self.clear()

    def clear(self) -> None:
        self.routing_tree.clear()
        self.graph.clear()

        self.core_blocks.clear()
        self.succ_core_blocks.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

        # Set default cflags
        _BACKEND_CONTEXT.cflags.clear()
        set_cflag(enable_wp_opt=True)
        set_cflag(grouping_optim_target="both")

    def build(
        self,
        *networks: DynSysGroup,
        # bounded_nodes: Sequence[Sequence[NeuDyn]] = (),
        # conflicted_nodes: Dict[NodeName, Sequence[NeuDyn]] = {},
    ) -> None:
        """Build the directed graph based on given networks.    \
            More than one networks in one graph is supported.

        Args:
            - networks: one or many `DynSysGroup`.

        TODO verify the following phases when more than one sub  \
            network is given.
        """
        self.clear()

        # Filter & check the constraints to nodes.
        self.graph.build(*networks)

    def compile(
        self,
        *,
        weight_bit_optimization: Optional[bool] = None,
        grouping_optim_target: Optional[Literal["latency", "core", "both"]] = None,
    ) -> None:
        if weight_bit_optimization is not None:
            set_cflag(enable_wp_opt=weight_bit_optimization)

        if grouping_optim_target is not None:
            set_cflag(grouping_optim_target=grouping_optim_target)

        """Backend compilation."""
        self._build_check()

        """1. Build core blocks."""
        self.build_core_blocks()

        """2. Adjust the LCN extension of each core block."""
        self.lcn_ex_adjustment()

        """3. Core coordinate assignment."""
        self.coord_assign()

        """4. Allocate the core blocks to the `CorePlacement`."""
        self.core_allocation()

        """5. Export parameters."""
        self.config_export()

    def build_core_blocks(self) -> None:
        """Build core blocks based on grouped edges.

        Description: Group all edges & build `CoreBlock` based on the grouped edges.
        """
        grouped_edges = self.graph.group_edges()

        for syns in grouped_edges:
            self.core_blocks.append(
                CoreBlock.build(
                    *syns,
                    seed=0,
                    enable_wp_opt=_BACKEND_CONTEXT.cflags["enable_wp_opt"],
                )
            )

        for cb in self.core_blocks:
            succ_cbs = list(
                filter(
                    lambda succ_cb: any(d for d in cb.dest if d in succ_cb.source),
                    self.core_blocks,
                )
            )
            self.succ_core_blocks[cb].extend(succ_cbs)

        self.degrees_of_cb = get_node_degrees(self.succ_core_blocks)

    def lcn_ex_adjustment(self) -> None:
        """Adjust the LCN extension of each core block."""
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

        NOTE: The neurons in each core block must be grouped first  \
            to determine the #N of cores required, and then the     \
            routing coordinates can be assigned.
        """
        for cb in self.core_blocks:
            # Group the neurons, get the #N of cores required.
            cb.group_neurons(
                optim_target=_BACKEND_CONTEXT.cflags["grouping_optim_target"]
            )

        # Calculate the consumption of physical cores required.
        if (
            n_core_required := sum(cb.n_core_required for cb in self.core_blocks)
        ) > HwConfig.N_CORE_OFFLINE:
            raise ResourceError(
                f"#N of total cores required out of {HwConfig.N_CORE_OFFLINE} ({n_core_required})."
            )

        self.n_core_required = n_core_required

        # Generate routing groups by given the list of core blocks.
        routing_groups = convert2routing_groups(
            self.succ_core_blocks, self.degrees_of_cb
        )
        for rg in routing_groups:
            if not self.routing_tree.insert_routing_group(rg):
                raise RuntimeError(
                    f"Insert routing group {rg} into the routing tree failed."
                )

        self.routing_groups = routing_groups

    def core_allocation(self) -> None:
        """Allocate the core blocks to the physical cores. \
            The order of `core_plms` is the same as `core_blocks`.
        """
        for cb in self.core_blocks:
            cb.core_plm_alloc()

    def config_export(self) -> GraphInfo:
        """Export parameters of cores & neurons inside.

        Steps:
            - 1. Export the parameters(PARAMETER_REG, including RANDOM_SEED \
                & Weight RAM) of cores.
            - 2. Export the parameters(Neuron RAM) of neurons inside.
        """
        input_nodes_info = self._inpproj_config_export()
        output_dest_info = self._member_cb_and_onode_config_export()

        _graph_info = GraphInfo(
            input=input_nodes_info,
            output=output_dest_info,
            members=self.core_plm_config,  # The configuration of physical cores is in `core_plm_config`
            inherent_timestep=self.graph.inherent_timestep,
            n_core_required=self.n_core_required,
            extras={"name": self.graph.graph_name_repr},
        )

        self.graph_info = _graph_info

        return _graph_info

    def _inpproj_config_export(self) -> InputNodeInfo:
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

        # Traverse input core blocks where input nodes are
        for input_cb in filter(
            lambda cb: any(s for s in cb.source if s in self.graph.inodes.values()),
            self.core_blocks,
        ):
            dest_coords = input_cb.core_coords
            dest_rid = get_replication_id(dest_coords)

            # Traverse input nodes in the input core block only, in case that
            # "other source nodes" are grouped with the input nodes.
            for inode in filter(
                lambda s: s in self.graph.inodes.values(), input_cb.source
            ):
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
                    _BACKEND_CONTEXT["local_chip_addr"].x,
                    _BACKEND_CONTEXT["local_chip_addr"].y,
                )

                input_nodes_info[inode.name] = neuron_dest

        return input_nodes_info

    def _member_cb_and_onode_config_export(self) -> OutputDestInfo:
        """Export the configuration of member core blocks & output destinations.

        Description:
            Traverse core placements in core blocks, find the following core    \
            blocks where the axons at. Get the coordinate of the core placement \
            & coordinates of axons(for broadcasting).

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

        for member_cb in self.core_blocks:
            self.core_params.update(
                CoreBlock.export_core_plm_config(member_cb)
            )  # compatible for py3.8

            output_axon_offset = 0
            for core_plm in member_cb.core_placements.values():
                for neu_seg in core_plm.neu_segs_of_cplm:
                    # Find the axon destinations
                    dest_cb = [
                        cb for cb in self.core_blocks if neu_seg.parent in cb.source
                    ]

                    # FIXME It is necessary to ensure that when there are both output nodes
                    # & member nodes in the same `CoreBlock`, they need to be allocated on
                    # different physical cores, otherwise routing problem will occur.
                    if dest_cb:  # `neu_seg` is memeber neurons
                        # Should not happen
                        assert _cb_routable(self.routing_groups, dest_cb)
                        core_plm.export_neu_config(neu_seg, dest_cb)
                    else:
                        # `neu_seg` is output neurons. Every neuron segment is a output node.
                        # Update the offset of axon
                        output_axon_offset = core_plm.export_neu_config(
                            neu_seg,
                            output_core_coord=ocoord,
                            axon_addr_offset=output_axon_offset,
                        )
                        output_dest_info[neu_seg.parent.name][
                            core_plm.coord.address
                        ] = core_plm.neu_configs[neu_seg.parent].neuron_dest_info

                        # Coord.x += 1 for the destination of the next output node
                        ocoord += CoordOffset(1, 0)

                self.core_plm_config[core_plm.coord] = core_plm.export_core_plm_config()

        return output_dest_info

    def export(
        self,
        write_to_file: bool = True,
        *,
        fp: Optional[Union[str, Path]] = None,
        format: Literal["txt", "bin", "npy"] = "bin",
        split_by_coordinate: bool = False,
        local_chip_addr: Optional[CoordLike] = None,
        export_core_params: bool = False,
    ) -> Dict[Coord, Any]:
        """Generate configuration frames & export to file.

        Args:
            - write_to_file: whether to write frames into file.
            - fp: If `write_to_file` is `True`, specify the output path.
            - format: `txt`, `bin`, or `npy`.`bin` & `npy` are recommended.
            - split_by_coordinate: whether to split the generated frames file by the    \
                core coordinates.
            - local_chip_addr: the address of the local chip. If not specified, the     \
                default value in `_BACKEND_CONTEXT` will be used.
            - export_core_params: whether to export the parameters of occupied cores.

        Return: a dictionary of configurations.
        """
        if format not in ("bin", "npy", "txt"):
            raise ValueError(f"Format {format} is not supported.")

        _fp = _fp_check(fp)

        if local_chip_addr is not None:
            _local_chip_addr = to_coord(local_chip_addr)
        else:
            _local_chip_addr = _BACKEND_CONTEXT["local_chip_addr"]

        config_dict = gen_config_frames_by_coreconf(
            self.graph_info["members"],
            _local_chip_addr,
            write_to_file,
            _fp,
            split_by_coordinate,
            format,
        )

        if export_core_params:
            # Export the parameters of occupied cores
            export_core_params_json(self.core_params, _fp)

        # Export the info of input nodes
        export_inp_nodes_conf_json(self.graph_info["input"], _fp)
        # Export the info of output destinations
        export_outp_dests_conf_json(self.graph_info["output"], _fp)

        return config_dict

    def find_neuron(self, neuron: NeuDyn, *, verbose: int = 0) -> None:
        self._build_check()

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neuron in cb.dest:
                print(
                    f"Neurons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X"
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
                print(f"Axons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X")
                axon_segment = cb.axon_segments[neuron]
                print(
                    f"{neuron.name} placed in {cb.core_coords}\n"
                    f"N:                {axon_segment.n_axon}\n"
                    f"Address width:    {axon_segment.addr_width}\n"
                    f"Address offset:   {axon_segment.addr_offset}"
                )

    def _build_check(self) -> None:
        return self.graph.build_check()


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
