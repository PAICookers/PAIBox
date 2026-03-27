from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import numpy as np
from paicorelib import (
    LCN_EX,
    AddPotentialMode,
    AERPacketZXYCopy,
    CoordXY,
    CoordZXYOffset,
    DataWidth,
    FoldType,
    NeuronType,
    OfflineNeuDestInfoV2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    WeightCompressType,
    find_coordxy_shortest_path,
)
from rich.progress import track

from .core_config import Backend_Core_Config, Frontend_Core_Config
from .coreplacement import (
    CorePlacement,
    EmptyOfflineCorePlacementV2,
    OfflineCorePlacementV2,
)
from .get_weight import (
    choose_weight_strategy,
    get_raw_weights,
    group_shift_weights_optimized,
    reorder_by_base_weight,
    weight_info,
)
from .neuron import OfflineNeuronPlacement
from .op_node import (
    CoreOpNode,
    Neuron,
    ReorderElem,
    ReorderNode,
    SourceElem,
    SourceNode,
)

FANIN_BASE = 512


class Group:
    def __init__(self):
        self.dests: dict[Neuron | ReorderElem, "RoutingGroup|ReorderGroup"] = {}

    def set_lcn(self) -> None:
        pass

    def allocate_neurons(self) -> None:
        pass

    @abstractmethod
    def get_lcn(self, _: Neuron | ReorderElem) -> LCN_EX:
        pass


class ReorderGroup(Group):
    reorder_group_counter = 0

    def __init__(
        self,
        raw_neus: list[ReorderElem],
        input_list: list[SourceElem],
        nodes: Optional[set[ReorderNode]] = None,
        input_nodes: Optional[set[SourceNode]] = None,
    ):
        super().__init__()
        self.id: int = type(self).reorder_group_counter
        type(self).reorder_group_counter += 1
        self.name: str = f"ReorderG_{self.id}"

        self.nodes: Optional[set[ReorderNode]] = nodes
        self.input_nodes: Optional[set[SourceNode]] = input_nodes
        self.input_set: set[SourceElem] = set(input_list)
        self.raw_neus: list[ReorderElem] = raw_neus
        self.input_list: list[SourceElem] = (
            input_list  # input_list can be reordered later
        )
        self.input_set: set[SourceElem] = set(input_list)
        self.index_map: dict[SourceElem, int] = {
            elem: i for i, elem in enumerate(input_list)
        }

        # set by mapper.set_rough_dest()
        # self.dests: dict[ReorderElem, "RoutingGroup|ReorderGroup"] = {}

        assert len(raw_neus) == len(
            input_list
        ), "raw_neus and input_list must have the same length for ReorderGroup"
        self.reorder_map: dict[SourceElem, ReorderElem] = dict()
        self.set_reorder_map()

    def set_reorder_map(self) -> None:
        assert self.nodes is not None, "nodes must be provided for ReorderGroup"
        for node in self.nodes:
            reorder_map = node.get_reorder_info()
            self.reorder_map.update(reorder_map)
        assert (
            set(self.reorder_map.keys()) == self.input_set
        ), "reorder_map keys must match input_set"
        assert set(self.reorder_map.values()) == set(
            self.raw_neus
        ), "reorder_map values must match raw_neus"

    def reorder_axon(self, elem: ReorderElem | Neuron) -> SourceElem:
        out_elem = self.reorder_map[elem]
        return self.get_axon(out_elem)

    def get_axon(self, elem: ReorderElem) -> SourceElem:
        dest_group = self.dests[elem]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_axon(elem)
        return elem

    def reorder_dest(self, elem: ReorderElem | Neuron) -> "RoutingGroup":
        out_elem = self.reorder_map[elem]
        return self.get_dest(out_elem)

    def get_dest(self, elem: ReorderElem) -> "RoutingGroup":
        dest_group = self.dests[elem]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_dest(self.reorder_map[elem])
        return dest_group

    def reorder_dest_info(
        self, elem: ReorderElem | Neuron
    ) -> tuple["RoutingGroup", int]:
        out_elem = self.reorder_map[elem]
        return self.get_dest_info(out_elem)

    def get_dest_info(self, elem: ReorderElem | Neuron) -> tuple["RoutingGroup", int]:
        dest_group = self.dests[elem]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_dest_info(elem)
        dest_axon = dest_group.index_map.get(elem, -1)
        assert dest_axon >= 0, f"Neuron {elem} not found in dest_group's index_map"
        return dest_group, dest_axon

    def info(self) -> str:
        info_str = f"{self.name}:\n"
        # if too many dests, only show first 3 and last 3
        dest_strs = []
        for neu in self.raw_neus:
            dest_rg, index = self.get_dest_info(neu)
            dest_strs.append(f"{str(neu)} -> {dest_rg.name}[{index}]")
        if len(dest_strs) > 6:
            dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
        info_str += f"  Number of Neurons: {len(self.raw_neus)}\n"
        info_str += f"  Number of Inputs: {len(self.input_list)}\n"
        info_str += "  Dests:\n    " + "\n    ".join(dest_strs) + "\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"Reorder Group {self.name} No Summary\n"
        return summary_str

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


class RoutingGroup(Group):
    routing_group_counter = 0

    # core_blocks in the same routing group share the same following properties
    # lcn, input_sign, input_width
    def __init__(
        self,
        raw_neus: list[Neuron],
        input_list: list[SourceElem],
        nodes: Optional[set[CoreOpNode]] = None,
        input_nodes: Optional[set[SourceNode]] = None,
    ):
        super().__init__()
        self.id: int = type(self).routing_group_counter
        type(self).routing_group_counter += 1
        self.name: str = f"RG_{self.id}"

        # for better optimization, if nodes and input_nodes are provided,
        # it means the raw_neus and input_list are all neu and input_elem generated from these nodes,
        # so we can directly get the weight matrix for this routing group without checking the connection between each raw_neu and input_elem
        self.nodes = nodes
        self.input_nodes = input_nodes

        # set by generate_routing_groups
        self.raw_neus: list[Neuron] = raw_neus
        self.input_list: list[SourceElem] = (
            input_list  # input_list can not be reordered
        )
        self.input_set: set[SourceElem] = set(input_list)
        self.index_map: dict[SourceElem, int] = {}
        self.set_index_map()
        # set by mapper.set_rough_dest()
        # self.dests: dict[Neuron, "RoutingGroup|ReorderGroup"] = {}
        self.lcn: LCN_EX = LCN_EX.LCN_1X

        # self.core_blocks: list[CoreBlock] = []
        self.last_full_attrs: Optional[OfflineNeuFullAttrsV2Part2] = None
        self.last_dest_group: Optional[RoutingGroup] = None
        self.last_dest_index: Optional[int] = None

        self.core_placements: list[CorePlacement] = []
        self.n_core_required: int = -1

        # set by mapper.routing() call self.assign_coord()
        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._multicast_config: Optional[AERPacketZXYCopy] = None
        self._base_coord: Optional[CoordXY] = None
        self.input_bit_num: int = 0

    def set_index_map(self):
        self.index_map = {elem: i for i, elem in enumerate(self.input_list)}

    def set_lcn(self):
        input_widths: set[DataWidth] = set(
            [neu.core_config().input_width for neu in self.raw_neus]
        )
        add_potentials: set[AddPotentialMode] = set(
            [neu.core_config().add_potential for neu in self.raw_neus]
        )
        assert (
            len(add_potentials) == 1
        ), "All neurons in the routing group must have the same add potential mode."
        add_potential = add_potentials.pop()
        if add_potential == AddPotentialMode.NORMAL:
            assert (
                len(input_widths) == 1
            ), "All neurons in the routing group must have the same input width."
            # if input width, input sign weight are different, need to split routing group before calling this function
            input_width = input_widths.pop()
            self.input_bit_num = 2**input_width
        else:
            # if add potential mode is not normal
            self.input_bit_num = 32  # use 32 bit to transmit potential value
            max_axon_addr = len(self.input_list) * 32

        max_axon_addr = len(self.input_list) * self.input_bit_num
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)

    def get_axon(self, neu: Neuron) -> SourceElem:
        dest_group = self.dests[neu]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_axon(neu)
        return neu

    def get_dest_info(self, neu: Neuron) -> tuple["RoutingGroup", int]:
        dest_group = self.dests[neu]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_dest_info(neu)
        dest_axon = dest_group.index_map.get(neu, -1)
        assert (
            len(dest_group.index_map) != 0
        ), f"Dest group {dest_group.name} has empty index_map, cannot find dest axon for neuron {neu}"
        assert dest_axon >= 0, f"Neuron {neu} not found in dest_group's index_map"
        return dest_group, dest_axon

    def get_dest(self, neu: Neuron) -> "RoutingGroup":
        dest_group = self.dests[neu]
        if isinstance(dest_group, ReorderGroup):
            return dest_group.reorder_dest(neu)
        return dest_group

    def try_store_neuron(
        self,
        neu: Neuron,
        stored_base_weight: dict[int, tuple[int, WeightCompressType]],
        current_core: OfflineCorePlacementV2,
        weight_info: weight_info,
        base_weights: List[np.ndarray],
        frontend_core_conf: Frontend_Core_Config,
        backend_core_conf: Backend_Core_Config,
    ) -> OfflineCorePlacementV2:
        # print("\ttrying to store neuron", neu)
        attrs_part2 = neu.attrs_part2()
        # Weight Strategy
        if weight_info.index not in stored_base_weight:
            # 这个 base weight 还没有存储过，需要存储
            current_weight_width = frontend_core_conf.weight_width
            base_weight = base_weights[weight_info.index]
            current_weight_width = frontend_core_conf.weight_width
            selected_weight, weight_compress = choose_weight_strategy(
                base_weight,
                current_weight_width,
                frontend_core_conf.input_width,
                frontend_core_conf.add_potential,
            )
            weight_sram_req = selected_weight.n_sram_required
            attrs_part2.weight_compress = weight_compress
            if weight_sram_req > 4096:
                raise NotImplementedError(
                    f"Base weight {weight_info.index} requires {weight_sram_req} SRAM lines, which exceeds the limit."
                )
            elif current_core.n_sram_required + weight_sram_req > 4096:
                # 当前 core 放不下了，需要换 core
                if len(current_core.neus) > 0:
                    # print(f"0: current_core({id(current_core)})_sram: {current_core.n_sram_required}")
                    # print(f"allocate a new core")
                    self.core_placements.append(current_core)
                current_core = OfflineCorePlacementV2(
                    frontend_core_conf, backend_core_conf
                )
                self.last_full_attrs = None  # 换 core 了，之前的 neuron attrs 不算了
                stored_base_weight.clear()  # 换 core 了，之前存储的 base weight 不算了
            # print(
            #     f"Storing base weight {weight_info.index} in core {id(current_core)} with compression {weight_compress} cost {weight_sram_req} SRAM lines."
            # )
            current_core.weights.append(selected_weight)
            stored_base_weight[weight_info.index] = (
                len(current_core.weights) - 1,
                weight_compress,
            )  # 存储这个 base weight 的位置和压缩方式

        # 这个 neu 的 base weight 已经存储过了，直接复用
        base_weight_idx, weight_compress = stored_base_weight[weight_info.index]
        attrs_part2.weight_compress = weight_compress
        neuron_type = (
            NeuronType.HALF if attrs_part2 == self.last_full_attrs else NeuronType.FULL
        )
        output_type = neu.output_type()

        input_width = frontend_core_conf.input_width
        weight_skew = weight_info.offset * (2**input_width)
        attrs_part1 = OfflineNeuFullAttrsV2Part1(
            weight_skew=weight_skew,
            weight_address_start=0,  # 之后会统一设置
            weight_address_end=0,  # 之后会统一设置
            fold_type=FoldType.UNFOLDED,
            neuron_type=neuron_type,
            output_type=output_type,
        )
        neu_placement = OfflineNeuronPlacement([neu], attrs_part1, attrs_part2)
        neu_sram_req = neu_placement.n_sram_required
        if neu_sram_req > 4096:
            raise NotImplementedError(
                f"Neuron {neu} requires {neu_sram_req} SRAM lines, which exceeds the limit."
            )
        elif current_core.n_sram_required + neu_sram_req > 4096:
            if len(current_core.neus) == 0:
                raise NotImplementedError(
                    f"Neuron {neu} with its weight cannot fit into an empty core."
                )
            self.core_placements.append(current_core)
            # print(f"1: current_core({id(current_core)})_sram: {current_core.n_sram_required}")
            # print(f"allocate a new core")
            current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)
            self.last_full_attrs = None  # 换 core 了，之前的 neuron attrs 不算了
            stored_base_weight.clear()  # 换 core 了，之前存储的 base weight 不算了
            return self.try_store_neuron(
                neu,
                stored_base_weight,
                current_core,
                weight_info,
                base_weights,
                frontend_core_conf,
                backend_core_conf,
            )
        else:
            if neuron_type == NeuronType.FULL:
                self.last_full_attrs = attrs_part2
            current_core.neus.append(neu_placement)
            # current_core.weights.append(selected_weight)  # weight 已经存储过了，不需要重复存储
            idx = len(current_core.neus) - 1
            current_core.neu_weight_map[idx] = (
                base_weight_idx  # 这个 neu 使用的 weight 是 base_weight_idx
            )
        return current_core

    def place_neurons_optimal(
        self,
        frontend_core_conf: Frontend_Core_Config,
        backend_core_conf: Backend_Core_Config,
        group_items: list[tuple[Neuron, np.ndarray]],
        block_id: int = 0,
    ):
        weights_of_group = [item[1] for item in group_items]
        weight_infos, base_weights = group_shift_weights_optimized(weights_of_group)
        # reorder group_items making the ones with the same base weight together, to improve weight storage efficiency
        reordered_items, reordered_infos = reorder_by_base_weight(
            group_items, weight_infos
        )

        # with open(f"{self.name}_weight_base.txt", "w") as f:
        #     for weight in base_weights:
        #         f.write(" ".join(map(str, weight)) + "\n")
        #     for info in reordered_infos:
        #         f.write(f"index: {info.index}, offset: {info.offset}\n")

        current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)
        self.last_full_attrs = None
        stored_base_weight: dict[int, tuple[int, WeightCompressType]] = (
            {}
        )  # map from base weight index to the index of core's weight list where it's stored

        description = f"allocating {self.name} block [{block_id}]"
        for (neu, weight_of_neu), weight_info in track(
            zip(reordered_items, reordered_infos),
            description=description,
            total=len(reordered_items),  # 明确指定总数，确保进度条计算准确
        ):
            current_core = self.try_store_neuron(
                neu,
                stored_base_weight,
                current_core,
                weight_info,
                base_weights,
                frontend_core_conf,
                backend_core_conf,
            )
        if len(current_core.neus) > 0:
            self.core_placements.append(current_core)

    def place_neurons_raw(
        self,
        frontend_core_conf: Frontend_Core_Config,
        backend_core_conf: Backend_Core_Config,
        group_items: list[tuple[Neuron, np.ndarray]],
    ):
        current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)

        self.last_full_attrs = None

        for neu, weight_of_neu in group_items:
            attrs_part2 = neu.attrs_part2()

            # Weight Strategy
            current_weight_width = frontend_core_conf.weight_width
            selected_weight, attrs_part2.weight_compress = choose_weight_strategy(
                weight_of_neu,
                current_weight_width,
                frontend_core_conf.input_width,
                frontend_core_conf.add_potential,
            )

            neuron_type = (
                NeuronType.HALF
                if attrs_part2 == self.last_full_attrs
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
            neu_sram_req = neu_placement.n_sram_required
            weight_sram_req = selected_weight.n_sram_required
            total_req = neu_sram_req + weight_sram_req

            if total_req > 4096:
                raise NotImplementedError(
                    f"Neuron {neu} with its weight requires {total_req} SRAM lines."
                )

            if current_core.n_sram_required + total_req > 4096:
                if len(current_core.neus) > 0:
                    self.core_placements.append(current_core)

                # Create new core with inheritance
                current_core = OfflineCorePlacementV2(
                    frontend_core_conf, backend_core_conf
                )
                neuron_type = NeuronType.FULL
                neu_placement.neu_attrs_part1.neuron_type = NeuronType.FULL
                self.last_full_attrs = attrs_part2

            elif neuron_type == NeuronType.FULL:
                self.last_full_attrs = attrs_part2

            # print(f"Neuron {neu} assigned type {neu_placement.neuron_type}.")
            current_core.neus.append(neu_placement)
            current_core.weights.append(selected_weight)

            idx = len(current_core.neus) - 1
            current_core.neu_weight_map[idx] = idx

        if len(current_core.neus) > 0:
            self.core_placements.append(current_core)

    def allocate_neurons(self):
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_neus, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        if self.nodes is not None and self.input_nodes is not None:
            node = list(self.nodes)[0]
            # print(f"kernel weight from node {node.name}:\n", node.weights[0])

        weights = get_raw_weights(self.raw_neus, self.input_list)

        # print(f"weight of routing group {self.name}:\n", weights)
        print(f"weight shape of routing group {self.name}: {weights.shape}")

        # print compelet weights into file for debug
        # with open(f"{self.name}_weights.txt", "w") as f:
        #     weights_transposed = weights.T
        #     for row in weights_transposed:
        #         f.write(" ".join(map(str, row)) + "\n")

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
            dest_rg = self.get_dest(neu)
            backend_core_conf = Backend_Core_Config(
                lcn=self.lcn,
                target_lcn=dest_rg.lcn,
            )
            weight_of_neu = weights[i]
            key = (frontend_core_conf, backend_core_conf)
            if key not in core_groups:
                core_groups[key] = []
            core_groups[key].append((neu, weight_of_neu))

        self.core_placements: list[CorePlacement] = []
        print(f"grouping finished, number of core groups: {len(core_groups)}")

        # 2. Allocation Phase
        for i, (key, group_items) in enumerate(core_groups.items()):
            # print(f"\n\nAllocating group with frontend_core_conf={key[0]}")
            # print(f"backend_core_conf={key[1]}")
            # print(f"Neu of this group: {[str(item[0]) for item in group_items]}")
            print(f"Number of neurons in this group: {len(group_items)}")
            frontend_core_conf, backend_core_conf = key
            # Initialize the first core for the current group
            current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)

            self.last_full_attrs = None
            self.place_neurons_optimal(
                frontend_core_conf, backend_core_conf, group_items, i
            )
            print(
                f"Number of cores after allocating this group: {len(self.core_placements)}"
            )

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
        for core_placement in track(
            self.core_placements,
            description=f"Setting Detail Destinations for {self.name}",
            total=len(self.core_placements),
        ):
            for neu_placement in core_placement.neus:
                # mostly each neu_placement contains only one raw_neu
                # for folded neuron placement, it may contain multiple raw_neus
                # but we only need to set dest_info for the first raw_neu
                main_neu = neu_placement.raw_neus[0]

                dest_routing_group, axon_addr_logic = self.get_dest_info(main_neu)
                assert axon_addr_logic >= 0, "axon_addr_logic should be non-negative"
                dest_coord = dest_routing_group.base_coord
                coord_copy = dest_routing_group.multicast_config

                # the input width of all cores in dest routing group should be the same,
                # so we can use the output width of this core as the input width of dest routing group
                axon_addr_count = axon_addr_logic * dest_routing_group.input_bit_num

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
        if len(self.raw_neus) > 6:
            print_neus = self.raw_neus[:3] + self.raw_neus[-3:]
        else:
            print_neus = self.raw_neus

        for i, neu in enumerate(print_neus):
            dest_rg, index = self.get_dest_info(neu)
            dest_strs.append(f"{str(neu)} -> {dest_rg.name}[{index}]")
        if len(self.raw_neus) > 6:
            dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
        info_str += f"  Number of Neurons: {len(self.raw_neus)}\n"
        info_str += f"  Number of Inputs: {len(self.input_list)}\n"
        info_str += "  Dests:\n    " + "\n    ".join(dest_strs) + "\n"
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
            summary_str += (
                f"    Neuron SRAM Required: {core_placement.neuron_sram_required}\n"
            )
            summary_str += (
                f"    Weight SRAM Required: {core_placement.weight_sram_required}\n"
            )
            summary_str += (
                f"    Total SRAM Required: {core_placement.n_sram_required}\n"
            )
        return summary_str


def toposort_for_rg(
    groups: list[ReorderGroup | RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    from collections import defaultdict, deque

    routing_groups = [rg for rg in groups if isinstance(rg, RoutingGroup)]

    indegree = {rg: 0 for rg in routing_groups}

    for rg in indegree.keys():
        print(f"Routing Group {rg.name} has indegree {indegree[rg]} before sorting.")
    graph = defaultdict(list)

    for rg in routing_groups:
        for neu in rg.raw_neus:
            dest_rg = rg.get_dest(neu)
            if dest_rg not in routing_groups or dest_rg == rg:
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
