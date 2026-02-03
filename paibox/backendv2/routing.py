from __future__ import annotations

from typing import Optional

import numpy as np
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
    DataWidth,
    FoldType,
    NeuronType,
    OfflineCoreRegV2,
    OfflineNeuDestInfoV2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    WeightCompressType,
    find_coordxy_shortest_path,
)

from .core_config import Backend_Core_Config, Frontend_Core_Config
from .coreplacement import (
    CorePlacement,
    EmptyOfflineCorePlacementV2,
    OfflineCorePlacementV2,
)
from .neuron import InputElem, Neuron, OfflineNeuronPlacement
from .weight import Weight

FANIN_BASE = 512


def get_raw_weights(
    raw_neus: list[Neuron], input_neus: list[Neuron | InputElem]
) -> np.ndarray:
    # emplement your own weight retrieval logic here
    # raw_neu and input_neu are both single neuron at an index of a CoreOpNode or InputNode
    # the shape of the complete core op node or input node can be found in raw_neu.target.shape or input_neu.target.shape
    # the index of the elem in flatten is neu.index.idx or elem.index.idx 
    for neu in raw_neus:
        neu.target.shape
        neu.index.idx
    for elem in input_neus:
        elem.target.shape
        elem.index.idx

    n_output = len(raw_neus)
    n_input = len(input_neus)
    
    # weight's shape should be (n_output, n_input)
    weights = np.zeros((n_output, n_input), dtype=np.int32)
    for i, neu in enumerate(raw_neus):
        for j, elem in enumerate(input_neus):
            # set weight value according to your logic
            # weights[i, j] should be the weight from input_neus[j] to raw_neus[i]
            # which is the weight from elem.target to neu.target
            # at the corresponding indices elem.index.idx and neu.index.idx
            
            # if there is no weight connection, set weights[i, j] to 0
            # for now, we can only consider the neu.target.raw_node is a SeqCoreOp
            # at PAIBox/paibox/fx_converter/core_op.py:208
            
            # if neu.target.predecessors contains elem.target
            # then the weight is decided by neu.target.raw_node.op1
            # op1 is can be linear or conv2d
            # for linear the weight from neu.target to elem.target can be found in op1.weight
            
            # for conv2d the weight from neu.target to elem.target 
            # should be the unfolded weight of the conv2d
            # (you can ask how to convert conv2d to a matrix multiplication for more details)
            
            # the neu.target and elem.target are mostly same in raw_neus and input_neus
            # so you can cache the weight matrix for each unique target node to accelerate the process
            
            # there is a simple test example at PAIBox/test.py
            
            weights[i, j] = -1 # unset weight value  

    return weights


class RoutingGroup:
    _counter = 0

    # core_blocks in the same routing group share the same following properties
    # lcn, input_sign, input_width
    def __init__(self, raw_neus: list[Neuron], input_list: list[Neuron | InputElem]):
        self.id: int = type(self)._counter
        type(self)._counter += 1
        self.name: str = f"RG_{self.id}"

        # set by generate_routing_groups
        self.raw_neus: list[Neuron] = raw_neus
        self.input_list: list[Neuron | InputElem] = (
            input_list  # input_list can be reordered later
        )
        self.input_set: set[Neuron | InputElem] = set(input_list)

        # set by mapper.set_rough_dest()
        self.dests: dict[Neuron, "RoutingGroup"] = {}
        self.lcn: LCN_EX = LCN_EX.LCN_1X

        # self.core_blocks: list[CoreBlock] = []
        self.core_placements: list[CorePlacement] = []
        self.n_core_required: int = -1

        # set by mapper.routing() call self.assign_coord()
        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._multicast_config: Optional[AERPacketZXYCopy] = None
        self._base_coord: Optional[CoordXY] = None

    def set_lcn(self):
        input_widths: set[DataWidth] = set(
            [neu.core_config().input_width for neu in self.raw_neus]
        )
        assert (
            len(input_widths) == 1
        ), "All neurons in the routing group must have the same input width."
        # if input width, input sign weight are different, need to split routing group before calling this function
        input_width = input_widths.pop()
        max_axon_addr = len(self.input_list) * (2**input_width)
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)

    def allocate_neurons(self):
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_neus, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        weights = get_raw_weights(self.raw_neus, self.input_list)

        print(
            f"Allocating neurons for Routing Group {self.name} with {len(self.raw_neus)} neurons."
        )

        # 1. Grouping Phase
        core_groups: dict[
            tuple[Frontend_Core_Config, Backend_Core_Config],
            list[tuple[Neuron, np.ndarray]],
        ] = {}
        for i, neu in enumerate(self.raw_neus):
            frontend_core_conf = neu.core_config()
            backend_core_conf = Backend_Core_Config(
                lcn=self.lcn,
                target_lcn=self.dests[neu].lcn,
            )
            weight_of_neu = weights[i]
            key = (frontend_core_conf, backend_core_conf)
            if key not in core_groups:
                core_groups[key] = []
            core_groups[key].append((neu, weight_of_neu))

        self.core_placements: list[CorePlacement] = []

        # 2. Allocation Phase
        for key, group_items in core_groups.items():
            frontend_core_conf, backend_core_conf = key
            # Initialize the first core for the current group
            current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)

            last_full_attrs = None

            for neu, weight_of_neu in group_items:
                attrs_part2 = neu.attrs_part2()

                # Weight Strategy
                current_weight_width = frontend_core_conf.weight_width
                w_dense = Weight(
                    weight_of_neu, WeightCompressType.DENSE, current_weight_width
                )
                w_sparse = Weight(
                    weight_of_neu, WeightCompressType.SPARSE, current_weight_width
                )

                if w_sparse.n_sram_required() < w_dense.n_sram_required():
                    selected_weight = w_sparse
                    attrs_part2.weight_compress = WeightCompressType.SPARSE
                else:
                    selected_weight = w_dense
                    attrs_part2.weight_compress = WeightCompressType.DENSE

                neuron_type = (
                    NeuronType.HALF
                    if attrs_part2 == last_full_attrs
                    else NeuronType.FULL
                )
                output_type = neu.output_type()
                attrs_part1 = OfflineNeuFullAttrsV2Part1(
                    weight_skew=0,
                    weight_address_start=0,
                    weight_address_end=0,
                    fold_type=FoldType.UNFOLDED,
                    neuron_type=neuron_type,
                    output_type=output_type,
                )
                neu_placement = OfflineNeuronPlacement([neu], attrs_part1, attrs_part2)

                # SRAM Check
                neu_sram_req = neu_placement.n_sram_required()
                weight_sram_req = selected_weight.n_sram_required()
                total_req = neu_sram_req + weight_sram_req

                if total_req > 4096:
                    raise NotImplementedError(
                        f"Neuron {neu} with its weight requires {total_req} SRAM lines."
                    )

                if current_core.n_sram_required() + total_req > 4096:
                    self.core_placements.append(current_core)

                    # Create new core with inheritance
                    current_core = OfflineCorePlacementV2(
                        frontend_core_conf, backend_core_conf
                    )
                    neuron_type = NeuronType.FULL
                    neu_placement.neu_attrs_part1.neuron_type = NeuronType.FULL
                    last_full_attrs = attrs_part2

                elif neuron_type == NeuronType.FULL:
                    last_full_attrs = attrs_part2

                # print(f"Neuron {neu} assigned type {neu_placement.neuron_type}.")
                current_core.neus.append(neu_placement)
                current_core.weights.append(selected_weight)

                idx = len(current_core.neus) - 1
                current_core.neu_weight_map[idx] = idx

            if len(current_core.neus) > 0:
                self.core_placements.append(current_core)

        self.n_core_required = len(self.core_placements)

        for core_placement in self.core_placements:
            core_placement.set_weight_address()

    def assign_coord(self, coords: list[CoordXY], copy_config: AERPacketZXYCopy):
        assert len(coords) >= len(
            self.core_placements
        ), "Not enough coordinates provided to assign."
        for i, coord in enumerate(coords):
            if i < len(self.core_placements):
                self.assigned_cores[coord] = self.core_placements[i]
            else:
                self.assigned_cores[coord] = EmptyOfflineCorePlacementV2()
            self.assigned_cores[coord]._coord = coord
        self._base_coord = coords[0]
        self._multicast_config = copy_config

    def set_detail_dest(self):
        for core_placement in self.core_placements:
            for neu_placement in core_placement.neus:
                # mostly each neu_placement contains only one raw_neu
                # for folded neuron placement, it may contain multiple raw_neus
                # but we only need to set dest_info for the first raw_neu
                main_neu = neu_placement.raw_neus[0]
                dest_routing_group = self.dests[main_neu]

                dest_coord = dest_routing_group.base_coord
                coord_copy = dest_routing_group.multicast_config
                axon_addr_logic = dest_routing_group.input_list.index(main_neu)

                # the input width of all cores in dest routing group should be the same,
                # so we can use the output width of this core as the input width of dest routing group
                axon_addr_count = axon_addr_logic * (2**core_placement.output_width)

                addr_axon = axon_addr_count % FANIN_BASE
                tick_relative = axon_addr_count // FANIN_BASE

                coord_offset, _ = find_coordxy_shortest_path(
                    target=dest_coord, start=core_placement.coord
                )

                neu_placement.dest_info = OfflineNeuDestInfoV2(
                    tick_relative=tick_relative,
                    addr_axon=addr_axon,  # to be set during core allocation
                    addr_core_xy=coord_offset.z,
                    addr_core_x=coord_offset.x,
                    addr_core_y=coord_offset.y,
                    addr_copy_xy=coord_copy.z,
                    addr_copy_x=coord_copy.x,
                    addr_copy_y=coord_copy.y,
                )

    def set_auto_core_config(self):
        for core_placement in self.core_placements:
            core_placement.set_auto_core_config()

    @property
    def multicast_config(self) -> AERPacketZXYCopy:
        if self._multicast_config is None:
            raise ValueError("multicast_config has not been set yet.")
        return self._multicast_config

    @property
    def base_coord(self) -> CoordXY:
        if self._base_coord is None:
            raise ValueError("base_coord has not been set yet.")
        return self._base_coord

    def __hash__(self):
        # use hash of each raw_neu to identify routing group
        return hash(tuple(sorted([hash(neu) for neu in self.raw_neus])))

    def __str__(self) -> str:
        neu_strs = [str(neu) for neu in self.raw_neus]
        # if too many neurons, only show first 3 and last 3
        if len(neu_strs) > 6:
            neu_strs = neu_strs[:3] + ["..."] + neu_strs[-3:]
        # if too many inputs, only show first 3 and last 3
        input_strs = [str(neu) for neu in self.input_list]
        if len(input_strs) > 6:
            input_strs = input_strs[:3] + ["..."] + input_strs[-3:]
        return f"\n{self.name}(\nraw_neus=[{', '.join(neu_strs)}], \ninput_list=[{', '.join(input_strs)}])\n"

    def __repr__(self) -> str:
        return self.__str__()

    def info(self) -> str:
        info_str = f"{self.name}:\n"
        # if too many dests, only show first 3 and last 3
        dest_strs = []
        for neu, dest_rg in self.dests.items():
            dest_strs.append(f"{str(neu)} -> {dest_rg.name}")
        if len(dest_strs) > 6:
            dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
        info_str += f"  Number of Neurons: {len(self.raw_neus)}\n"
        info_str += f"  Number of Inputs: {len(self.input_list)}\n"
        info_str += f"  Dests:\n    " + "\n    ".join(dest_strs) + "\n"
        info_str += f"  LCN: {self.lcn.name}\n"
        info_str += f"  Number of Core Placements: {len(self.core_placements)}\n"
        if self._base_coord is not None:
            info_str += f"  Base Coord: ({self._base_coord.x}, {self._base_coord.y})\n"
        if self._multicast_config is not None:
            info_str += f"  Multicast Config: (Z: {self._multicast_config.z}, X: {self._multicast_config.x}, Y: {self._multicast_config.y})\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"{self.name} Routing Summary:\n"
        for i, core_placement in enumerate(self.core_placements):
            summary_str += f"  Core Placement {i} at {core_placement.coord}:\n"
            summary_str += f"    Number of Neurons: {len(core_placement.neus)}\n"
            summary_str += f"    Number of Weights: {len(core_placement.weights)}\n"
        return summary_str


def toposort_for_rg(
    routing_groups: list[RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    from collections import defaultdict, deque

    indegree = {rg: 0 for rg in routing_groups}
    graph = defaultdict(list)

    for rg in routing_groups:
        for dest_rg in rg.dests.values():
            if dest_rg not in routing_groups:
                continue
            graph[rg].append(dest_rg)
            indegree[dest_rg] += 1

    queue = deque([rg for rg in routing_groups if indegree[rg] == 0])
    sorted_rgs = []

    while queue:
        rg = queue.popleft()
        sorted_rgs.append(rg)
        for neighbor in graph[rg]:
            if neighbor not in routing_groups:
                continue
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    next_rg_id = {}
    for i, rg in enumerate(sorted_rgs):
        next_rg_id[i] = [sorted_rgs.index(dest_rg) for dest_rg in graph[rg]]

    if len(sorted_rgs) != len(routing_groups):
        raise ValueError("Cycle detected in routing groups.")

    return sorted_rgs, next_rg_id
