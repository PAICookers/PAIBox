from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Union

from paibox.base import NeuDyn
from paibox.exceptions import BuildError, ResourceError
from paibox.libpaicore import (
    Coord,
    CoordLike,
    CoordOffset,
    HwConfig,
    NeuronDestInfo,
    get_replication_id,
    to_coord,
)
from paibox.network import DynSysGroup

from .config_template import CoreConfig, NeuronDest
from .context import _BACKEND_CONTEXT
from .graphs import *
from .placement import CoreBlock, aligned_coords, max_lcn_of_cb
from .routing import RoutingRoot
from .runtime import OfflineFrameGen

__all__ = ["Mapper"]


class Mapper:
    """Mapping the network in the cores."""

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

        """Local options"""
        self.has_compiled: bool

        self.clear()

    def clear(self) -> None:
        self.has_compiled = False

        self.routing_tree.clear()
        self.graph.clear()

        self.core_blocks.clear()
        self.succ_core_blocks.clear()

        self.core_params.clear()
        self.core_plm_config.clear()

    def build(self, *networks: DynSysGroup) -> None:
        """Build the directed graph based on given networks.    \
            More than one networks in one graph is supported.

        Args:
            - networks: one or many `DynSysGroup`.

        TODO verify the following phases when more than one sub  \
            network is given.
        """
        self.clear()
        self.graph.build(*networks)

    def compile(self, method: str = "catagory"):
        """Backend compilation."""
        self._build_check()

        """1. Build core blocks."""
        self.build_core_blocks()

        """2. Adjust the LCN extension of each layer for target LCN."""
        self.lcn_ex_adjustment()

        """3. Do core coordinate assignment."""
        self.coord_assign(method)

        """4. Allocate the grouped synapses to the cores."""
        self.core_allocation()

        self.has_compiled = True

        """5. Export parameters."""
        graph_info = self.config_export()

        self.graph_info = graph_info

        return graph_info

    def build_core_blocks(self) -> None:
        """Build core blocks based on grouped edges.

        Description:
            After combining all synapses into groups, iterate through \
            each combination synapses to build `CoreBlock`.

            # Then do sorting in ascending order.
        """
        grouped_edges = group_edges(
            list(self.graph.edges),
            self.graph.succ_dg,
            self.graph.degree_of_nodes,
            ordered_nodes=self.graph.ordered_nodes,
        )

        for syns_set in grouped_edges:
            syns = [self.graph.edges[syn] for syn in syns_set]
            self.core_blocks.append(CoreBlock.build(*syns))

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

    def coord_assign(self, method) -> None:
        """Assign the coordinate for each `CorePlacement`.

        NOTE: The neurons in each core block must be grouped first  \
            to determine the #N of cores required, and then the     \
            routing coordinates can be assigned.
        """
        for cb in self.core_blocks:
            # Group the neurons, get the #N of cores required.
            cb.group_neurons(method)

        # Check the total core consumption.
        if (
            n_core_required := sum(cb.n_core_required for cb in self.core_blocks)
        ) > HwConfig.N_CORE_OFFLINE:
            raise ResourceError(
                f"#N of total cores required out of {HwConfig.N_CORE_OFFLINE}: {n_core_required}"
            )

        self.n_core_required = n_core_required

        for cb in self.core_blocks:
            if not RoutingRoot.insert_coreblock(self.routing_tree, cb):
                raise RuntimeError(
                    f"Insert core block {cb.name} into the routing tree failed."
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

        # The destination info of the output nodes is a subset
        # of the info of the network members.
        output_dest_info: Dict[NodeName, Dict[int, NeuronDestInfo]] = defaultdict(dict)

        _ocoord_update_flag = False
        ocoord = _BACKEND_CONTEXT["output_core_addr_start"]

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
                    if neu_seg.parent.name in self.graph.onodes:
                        # Update the offset of axon
                        output_axon_offset = core_plm.export_neu_config(
                            neu_seg,
                            output_core_coord=ocoord,
                            axon_addr_offset=output_axon_offset,
                        )
                        output_dest_info[neu_seg.parent.name][
                            core_plm.coord.address
                        ] = core_plm.neu_configs[neu_seg.parent].neuron_dest_info
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

                self.core_plm_config[core_plm.coord] = core_plm.export_core_plm_config()

            if _ocoord_update_flag:
                # Coord.x += 1
                ocoord += CoordOffset(1, 0)
                _ocoord_update_flag = False

        return GraphInfo(
            input=input_nodes_info,
            output=output_dest_info,
            members=self.core_plm_config,
            extras={
                "name": self.graph.graph_name_repr,
                "timestep": self.get_inherent_timestep(),
            },
        )

    def _inpproj_config_export(self) -> Dict[NodeName, NeuronDest]:
        input_nodes_info: Dict[NodeName, NeuronDest] = dict()

        # Get the input core blocks for all input nodes.
        for input_cb in filter(
            lambda cb: any(s for s in cb.source if s.name in self.graph.inodes),
            self.core_blocks,
        ):
            dest_coords = input_cb.core_coords
            dest_rid = get_replication_id(dest_coords)

            # Only traverse input nodes in this core block.
            for inode in filter(
                lambda s: s in self.graph.inodes.values(), input_cb.source
            ):
                axon_coords = aligned_coords(
                    slice(0, input_cb.n_axon_of(input_cb.source.index(inode)), 1),
                    input_cb.axon_segments[inode],
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

    def export(
        self,
        write_to_file: bool = True,
        *,
        fp: Optional[Union[str, Path]] = None,
        format: Literal["txt", "bin", "npy"] = "npy",
        local_chip_addr: Optional[CoordLike] = None,
    ) -> Dict[Coord, Any]:
        if fp is not None:
            _fp = Path(fp)
        else:
            _fp = _BACKEND_CONTEXT["build_directory"]

        if not _fp.is_dir():
            _fp.mkdir(parents=True, exist_ok=True)

        if local_chip_addr is not None:
            _local_chip_addr = to_coord(local_chip_addr)
        else:
            _local_chip_addr = _BACKEND_CONTEXT["local_chip_addr"]

        config_dict = OfflineFrameGen.gen_config_frames(
            _local_chip_addr,
            self.core_plm_config,
            write_to_file,
            _fp,
            format,
        )

        return config_dict

    def get_inherent_timestep(self) -> int:
        self._build_check()

        _, distance = get_longest_path(self.graph.succ_dg, self.graph.ordered_nodes)

        return distance

    def find_neuron(self, neuron: NeuDyn, *, verbose: int = 0) -> None:
        self._build_check()

        for cb in self.core_blocks:
            # Find neuron in one or more core blocks.
            if neuron in cb.dest:
                print(
                    f"Neurons {neuron.name} placed in {cb.name}, LCN_{1 << cb.lcn_ex}X"
                )
                for core_plm in cb.core_placements.values():
                    for neu_seg in core_plm.neu_segs:
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

    def _compile_check(self) -> None:
        if not self.has_compiled:
            raise BuildError(f"The graph hasn't been compiled yet")


def group_by(dict_: Dict, keyfunc=lambda item: item):
    """Groups the given list or dictionary by the value returned by ``keyfunc``."""
    d = defaultdict(list)

    for item in dict_.values():
        d[keyfunc(item)].append(item)

    return d
