from __future__ import annotations

from typing import Optional

import numpy as np
from coreplacement import CorePlacement, EmptyOfflineCorePlacementV2
from neuron import Neuron
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
    OfflineNeuDestInfoV2,
    find_coordxy_shortest_path,
)

FANIN_BASE = 512


def get_raw_weights(raw_neus: list[Neuron], input_neus: list[Neuron]) -> np.ndarray:
    n_output = len(raw_neus)
    n_input = len(input_neus)
    # random weights for testing, shape (n_output, n_input)
    weights = np.random.randint(-128, 127, size=(n_output, n_input), dtype=np.int8)
    return weights


class RoutingGroup:
    # core_blocks in the same routing group share the same following properties
    # lcn, input_sign, input_width
    def __init__(self):
        # set by generate_routing_groups
        self.raw_neus: list[Neuron] = []

        # set by mapper.set_rough_dest()
        self.dests: dict[Neuron, "RoutingGroup"] = {}
        self.input_list: list[Neuron] = []  # input_list can be reordered later
        self.lcn: LCN_EX = LCN_EX.LCN_1X

        # self.core_blocks: list[CoreBlock] = []
        self.core_placements: list[CorePlacement] = []
        self.n_core_required: int = -1

        # set by mapper.routing() call self.assign_coord()
        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._muticast_config: Optional[AERPacketZXYCopy] = None
        self._base_coord: Optional[CoordXY] = None

    def set_lcn(self):
        input_widths = set([neu.core_config().input_width for neu in self.raw_neus])
        assert (
            len(input_widths) == 1
        ), "All neurons in the routing group must have the same input width."
        # if input width, input sign weight are different, need to split routing group before calling this function
        input_width = input_widths.pop()
        max_axon_addr = len(self.input_list) * input_width
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)

    def allocate_neurons(self):
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_neus, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        for neu in self.raw_neus:
            attrs = neu.attrs_part2()
            inherited_core_config = neu.core_config()
            target_lcn = self.dests[neu].lcn
            lcn = self.lcn

        # implement your core placement generation logic here
        # you need to create OfflineCorePlamentV2 instances, set their properties including:
        # coreplacement._core_config, coreplacment.neus, coreplacment.weights, coreplacment.neu_weight_map
        # remember the neurons in the same core placement share the same core config, so

        # the number of core placements created should be set to self.n_core_required
        # dest_info in NeuronPlacement should remain unset, it will be set later during routing.
        # auto_core_config in CorePlacement should remain unset, it will be set later during core allocation.

        # each Coreplacement has 4096 sram, you need to make sure the total sram required by neu and weight in each coreplacement does not exceed 4096
        # you can get the sram required by each neuron using OfflineNeuronPlacement.n_sram_required() function
        # you can get the sram required by each weight using Weight.n_sram_required() function

        # you can use get_raw_weights function to get the raw weights between self.raw_neus and self.input_list
        # then you can reorder the input_list as well as the weight's rows according to your core placement strategy

        # you can think about the following optimizations:
        # 1. for neurons in the same Coreplacement with the same attrs_part2, you can use half neuron to save SRAM
        # 2. you can reorder the input_list, to move as many as possible zero weights to the end of each weight row, so that you can reduce the weight storage requirement
        # ... etc.

        raise NotImplementedError("allocate_neurons method is not implemented yet.")

    def assign_coord(self, coords: list[CoordXY], copy_config: AERPacketZXYCopy):
        assert len(coords) >= len(
            self.core_placements
        ), "Not enough coordinates provided to assign."
        for i, coord in enumerate(coords):
            if i < len(self.core_placements):
                self.assigned_cores[coord] = self.core_placements[i]
            else:
                self.assigned_cores[coord] = EmptyOfflineCorePlacementV2()
        self._base_coord = coords[0]
        self._muticast_config = copy_config

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
                axon_addr_count = (
                    axon_addr_logic * core_placement.core_config.output_width
                )

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

    @property
    def multicast_config(self) -> AERPacketZXYCopy:
        if self._muticast_config is None:
            raise ValueError("multicast_config has not been set yet.")
        return self._muticast_config

    @property
    def base_coord(self) -> CoordXY:
        if self._base_coord is None:
            raise ValueError("base_coord has not been set yet.")
        return self._base_coord

    def __hash__(self):
        # use hash of each raw_neu to identify routing group
        return hash(tuple(sorted([hash(neu) for neu in self.raw_neus])))


def toposort_for_rg(
    routing_groups: list[RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    from collections import defaultdict, deque

    indegree = {rg: 0 for rg in routing_groups}
    graph = defaultdict(list)

    for rg in routing_groups:
        for dest_rg in rg.dests.values():
            graph[rg].append(dest_rg)
            indegree[dest_rg] += 1

    queue = deque([rg for rg in routing_groups if indegree[rg] == 0])
    sorted_rgs = []

    while queue:
        rg = queue.popleft()
        sorted_rgs.append(rg)
        for neighbor in graph[rg]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    next_rg_id = {}
    for i, rg in enumerate(sorted_rgs):
        next_rg_id[i] = [sorted_rgs.index(dest_rg) for dest_rg in graph[rg]]

    if len(sorted_rgs) != len(routing_groups):
        raise ValueError("Cycle detected in routing groups.")

    return sorted_rgs, next_rg_id
