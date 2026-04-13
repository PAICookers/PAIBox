from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence, Set
from typing import Generic, TypeVar

import numpy as np
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
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
    WeightInfo,
    choose_weight_strategy,
    get_raw_weights,
    group_shift_weights_optimized,
    reorder_by_base_weight,
)
from .neuron import OfflineNeuronPlacement
from .op_node import (
    CoreOpNode,
    InNode,
    InputElem,
    Neuron,
    RemapElem,
    ReorderNode,
    SourceElem,
    SourceNode,
)

FANIN_BASE = 512


class Group:
    group_counter = 0

    def __init__(self):
        self.id: int = type(self).group_counter
        type(self).group_counter += 1
        self.name: str = f"Group_{self.id}"

    def info(self, prefix: str = "") -> str:
        info_str = f"{prefix}{self.name}:"
        return info_str

    def __str__(self) -> str:
        return self.info() + "\n"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(id(self))

SOURCE_ELEM = TypeVar("SOURCE_ELEM", bound=SourceElem)
SOURCE_NODE = TypeVar("SOURCE_NODE", bound=SourceNode)


class DestGroup(Generic[SOURCE_ELEM, SOURCE_NODE]):
    def __init__(
        self,
        input_list: Sequence[SOURCE_ELEM],
        input_nodes: Set[SOURCE_NODE] | None = None,
    ):
        self.input_nodes: set[SOURCE_NODE] | None = (
            set(input_nodes) if input_nodes is not None else None
        )
        self.input_list: list[SOURCE_ELEM] = list(input_list)
        self.input_set: set[SOURCE_ELEM] = set(input_list)
        self.index_map: dict[SOURCE_ELEM, int] = {}
        self.set_index_map()

    def set_index_map(self):
        self.index_map = {elem: i for i, elem in enumerate(self.input_list)}

    def info(self, prefix: str = "") -> str:
        info_str = ""
        input_strs = []
        if len(self.input_list) > 6:
            print_inputs = self.input_list[:3] + self.input_list[-3:]
        else:
            print_inputs = self.input_list

        for input in print_inputs:
            input_strs.append(str(input))
        if len(self.input_list) > 6:
            input_strs = input_strs[:3] + ["..."] + input_strs[-3:]
        info_str += f"\n{prefix}Number of Inputs: {len(self.input_list)}"
        info_str += f"\n{prefix}Inputs:\n{prefix}   "
        info_str += f"\n{prefix}   ".join(input_strs)
        return info_str

    def __str__(self) -> str:
        return "DestGroup: \n" + self.info(prefix="  ") + "\n"


class SourceGroup(Generic[SOURCE_ELEM, SOURCE_NODE]):
    def __init__(
        self,
        raw_neus: Sequence[SOURCE_ELEM],
        nodes: Set[SOURCE_NODE] | None = None,
    ):
        self.nodes: Set[SOURCE_NODE] | None = set(nodes) if nodes is not None else None
        self.raw_elems: list[SOURCE_ELEM] = list(raw_neus)
        self.elem_set: set[SOURCE_ELEM] = set(raw_neus)
        self.dests: dict[SOURCE_ELEM, "RoutingGroup | OutputGroup| RemapGroup"] = {}

    @abstractmethod
    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        pass

    def get_dest_info(
        self, elem: SOURCE_ELEM
    ) -> tuple["RoutingGroup | OutputGroup", int]:
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.reorder_dest_info(elem)
        dest_axon = dest_group.index_map.get(elem, -1)
        assert dest_axon >= 0, f"Neuron {elem} not found in dest_group's index_map"
        return dest_group, dest_axon

    def get_dest(self, elem: SOURCE_ELEM) -> "RoutingGroup | OutputGroup":
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.reorder_dest(elem)
        return dest_group

    def get_axon(self, elem: SOURCE_ELEM) -> SourceElem:
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.reorder_axon(elem)
        return elem

    def info(self, prefix: str = "") -> str:
        info_str = ""
        dest_strs = []
        if len(self.raw_elems) > 6:
            print_elems = self.raw_elems[:3] + self.raw_elems[-3:]
        else:
            print_elems = self.raw_elems
        dest_set: bool = len(self.dests) > 0
        for elem in print_elems:
            if dest_set:
                dest_rg, index = self.get_dest_info(elem)
                dest_strs.append(f"{str(elem)} -> {dest_rg.name}[{index}]")
            else:
                dest_strs.append(f"{str(elem)} -> None")
        if len(self.raw_elems) > 6:
            dest_strs = dest_strs[:3] + ["..."] + dest_strs[-3:]
        info_str += f"\n{prefix}Number of Neurons: {len(self.raw_elems)}"
        info_str += f"\n{prefix}Dests:\n{prefix}   "
        info_str += f"\n{prefix}   ".join(dest_strs)
        return info_str

    def routing_summary(self) -> str:
        summary_str = "    Not Deploy Group\n"
        return summary_str

    def __str__(self) -> str:
        info_str = "SourceGroup: \n" + self.info(prefix="  ")
        return info_str + "\n"

    def get_detail_dest(
        self, elem: SOURCE_ELEM, self_coord: CoordXY = CoordXY(0, 0)
    ) -> OfflineNeuDestInfoV2:
        dest_routing_group = self.get_dest(elem)
        axon_elem = self.get_axon(elem)
        if isinstance(dest_routing_group, RoutingGroup):
            axon_addr_logic = dest_routing_group.index_map.get(axon_elem, -1)
            assert axon_addr_logic >= 0, "axon_addr_logic should be non-negative"
            assert (
                elem.output_bit_num == dest_routing_group.input_bit_num
            ), "Output bit num of elem must match input bit num of dest routing group"
            axon_bit_count = axon_addr_logic * dest_routing_group.input_bit_num
        elif isinstance(dest_routing_group, OutputGroup):
            axon_bit_count = dest_routing_group.axon_bit_allocator.allocate(
                self_coord, elem
            )
            assert axon_bit_count < FANIN_BASE * (
                2**LCN_EX.LCN_128X.value
            ), "Total axon bit count for output group exceeds the maximum supported by LCN_128X"

        dest_coord = dest_routing_group.base_coord
        coord_copy = dest_routing_group.multicast_config

        addr_axon = axon_bit_count % FANIN_BASE
        tick_relative = axon_bit_count // FANIN_BASE

        coord_offset, _ = find_coordxy_shortest_path(
            target=dest_coord, start=self_coord
        )

        return OfflineNeuDestInfoV2(
            tick_relative=tick_relative,
            addr_axon=addr_axon,  # to be set during core allocation
            addr_core_xy=coord_offset.z,
            addr_core_x=coord_offset.x,
            addr_core_y=coord_offset.y,
            addr_copy_xy=coord_copy.z,
            addr_copy_x=coord_copy.x,
            addr_copy_y=coord_copy.y,
        )

    def set_detail_dest(self):
        pass


class RemapGroup(
    Group, DestGroup[SourceElem, SourceNode], SourceGroup[RemapElem, ReorderNode]
):
    reorder_group_counter = 0

    def __init__(
        self,
        raw_elems: Sequence[RemapElem],
        input_list: Sequence[SourceElem],
        nodes: Set[ReorderNode] | None = None,
        input_nodes: Set[SourceNode] | None = None,
    ):
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        SourceGroup.__init__(self, raw_elems, nodes)
        self.name: str = f"ReorderG_{self.id}"
        self.remap_dict: dict[SourceElem, RemapElem] = dict()
        self.source_dict: dict[RemapElem, SourceElem] = dict()
        self.set_remap_dict()

    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        if not isinstance(elem, RemapElem):
            raise TypeError("Only RemapElem can be added to RemapGroup")
        self.raw_elems.append(elem)
        self.elem_set.add(elem)
        raw_elem = elem.origin_elem()
        raw_input = self.source_dict[raw_elem]
        copy_input = raw_input.copy(elem.index.copy_id)
        self.input_list.append(copy_input)
        self.input_set.add(copy_input)
        self.remap_dict[copy_input] = elem
        self.source_dict[elem] = copy_input
        self.nodes = None
        return copy_input

    def set_remap_dict(self) -> None:
        assert self.nodes is not None, "nodes must be provided for ReorderGroup"
        for node in self.nodes:
            reorder_map = node.get_reorder_info()
            self.remap_dict.update(reorder_map)
        assert (
            set(self.remap_dict.keys()) == self.input_set
        ), "reorder_map keys must match input_set"
        assert set(self.remap_dict.values()) == set(
            self.raw_elems
        ), "reorder_map values must match raw_neus"

        for src, dst in self.remap_dict.items():
            self.source_dict[dst] = src

    def reorder_axon(self, elem: SourceElem) -> SourceElem:
        out_elem = self.remap_dict[elem]
        return self.get_axon(out_elem)

    def reorder_dest(self, elem: SourceElem) -> "RoutingGroup|OutputGroup":
        out_elem = self.remap_dict[elem]
        return self.get_dest(out_elem)

    def reorder_dest_info(
        self, elem: SourceElem
    ) -> tuple["RoutingGroup | OutputGroup", int]:
        out_elem = self.remap_dict[elem]
        return self.get_dest_info(out_elem)

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix)
        info_str += DestGroup.info(self, prefix=prefix + "  ")
        info_str += SourceGroup.info(self, prefix=prefix + "  ")
        info_str += "\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"Reorder Group {self.name}:\n"
        summary_str += SourceGroup.routing_summary(self)
        return summary_str

    def __str__(self) -> str:
        info_str = self.info()
        return info_str


class RoutingGroup(
    Group, DestGroup[SourceElem, SourceNode], SourceGroup[Neuron, CoreOpNode]
):
    def __init__(
        self,
        raw_neus: Sequence[Neuron],
        input_list: Sequence[SourceElem],
        nodes: Set[CoreOpNode] | None = None,
        input_nodes: Set[SourceNode] | None = None,
    ):
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        SourceGroup.__init__(self, raw_neus, nodes)
        self.name: str = f"RG_{self.id}"

        self.lcn: LCN_EX = LCN_EX.LCN_1X
        self.input_bit_num: int = 0

        # self.core_blocks: list[CoreBlock] = []
        self.last_full_attrs: OfflineNeuFullAttrsV2Part2 | None = None
        self.last_dest_group: RoutingGroup | None = None
        self.last_dest_index: int | None = None

        self.n_core_required: int = -1
        self.core_placements: list[CorePlacement] = []

        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._multicast_config: AERPacketZXYCopy | None = None
        self._base_coord: CoordXY | None = None

    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        if not isinstance(elem, Neuron):
            raise TypeError("Only Neuron can be added to RoutingGroup")
        self.raw_elems.append(elem)
        self.elem_set.add(elem)
        self.nodes = None
        return None

    def set_lcn(self):
        intput_bit_nums: set[int] = set([neu.input_bit_num for neu in self.raw_elems])
        pred_output_bit_nums: set[int] = set(
            [src.output_bit_num for src in self.input_list]
        )
        assert (
            len(intput_bit_nums) == 1
        ), "All neurons in the routing group must have the same input bit num."
        assert (
            len(pred_output_bit_nums) == 1
        ), "All input elements in the routing group must have the same output bit num."
        print(f"{self.raw_elems[0]}: input_bit_nums: {intput_bit_nums}")
        print(f"{self.input_list[0]}: pred_output_bit_nums: {pred_output_bit_nums}")

        self.input_bit_num = intput_bit_nums.pop()
        assert (
            self.input_bit_num == pred_output_bit_nums.pop()
        ), "Input bit num of neurons must match output bit num of input elements."
        max_axon_addr = len(self.input_list) * self.input_bit_num
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        self.lcn = LCN_EX(lcn)

    def try_store_neuron(
        self,
        neu: Neuron,
        stored_base_weight: dict[int, tuple[int, WeightCompressType]],
        current_core: OfflineCorePlacementV2,
        weight_info: WeightInfo,
        base_weights: list[np.ndarray],
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
        stored_weight_index, weight_compress = stored_base_weight[weight_info.index]
        attrs_part2.weight_compress = weight_compress
        neuron_type = (
            NeuronType.HALF if attrs_part2 == self.last_full_attrs else NeuronType.FULL
        )
        output_type = neu.output_type()

        weight_skew = weight_info.offset * self.input_bit_num
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
                stored_weight_index  # 这个 neu 使用的 weight 是 base_weight_idx
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
        block_id: int = 0,
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
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_elems, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        if self.nodes is not None and self.input_nodes is not None:
            node = list(self.nodes)[0]
            # print(f"kernel weight from node {node.name}:\n", node.weights[0])

        weights = get_raw_weights(self.raw_elems, self.input_list)

        # print(f"weight of routing group {self.name}:\n", weights)
        print(f"weight shape of routing group {self.name}: {weights.shape}")

        # print compelet weights into file for debug
        # with open(f"{self.name}_weights.txt", "w") as f:
        #     weights_transposed = weights.T
        #     for row in weights_transposed:
        #         f.write(" ".join(map(str, row)) + "\n")

        print(
            f"Allocating neurons for Routing Group {self.name} with {len(self.raw_elems)} neurons."
        )

        # 1. Grouping Phase
        core_groups: dict[
            tuple[Frontend_Core_Config, Backend_Core_Config],
            list[tuple[Neuron, np.ndarray]],
        ] = {}
        for i, neu in enumerate(self.raw_elems):
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
                dest_info = self.get_detail_dest(
                    main_neu, self_coord=core_placement.coord
                )
                neu_placement.dest_info = dest_info

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

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix)
        prefix = prefix + "  "
        info_str += DestGroup.info(self, prefix=prefix)
        info_str += SourceGroup.info(self, prefix=prefix)
        info_str += f"\n{prefix}LCN: {self.lcn.name}"
        info_str += f"\n{prefix}Number of Core Placements: {len(self.core_placements)}"
        if self._base_coord is not None:
            info_str += (
                f"\n{prefix}Base Coord: ({self._base_coord.x}, {self._base_coord.y})"
            )
        if self._multicast_config is not None:
            info_str += f"\n{prefix}Multicast Config: (Z: {self._multicast_config.z}, X: {self._multicast_config.x}, Y: {self._multicast_config.y})"
        info_str += "\n"
        return info_str

    def __str__(self) -> str:
        info_str = self.info()
        return info_str

    def routing_summary(self) -> str:
        summary_str = (
            f"{self.name} Routing Summary ({len(self.core_placements)} cores):\n"
        )
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


class InputGroup(Group, SourceGroup[InputElem, InNode]):
    def __init__(
        self,
        raw_elems: Sequence[SourceElem],
        nodes: Set[SourceNode] | None = None,
    ):
        Group.__init__(self)
        SourceGroup.__init__(self, raw_elems, nodes)
        self.name: str = f"InputG_{self.id}"
        self.dest_infos: dict[SourceElem, OfflineNeuDestInfoV2] = {}

    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        if not isinstance(elem, InputElem):
            raise TypeError("Only InputElem can be added to InputGroup")
        self.raw_elems.append(elem)
        self.elem_set.add(elem)
        self.nodes = None
        return None

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix=prefix)
        info_str += SourceGroup.info(self, prefix=prefix + "  ")
        info_str += "\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"Input Group {self.name}:\n"
        summary_str += SourceGroup.routing_summary(self)
        return summary_str

    def __str__(self) -> str:
        info_str = self.info()
        return info_str

    def set_detail_dest(self):
        for elem in self.raw_elems:
            dest_info = self.get_detail_dest(elem)
            self.dest_infos[elem] = dest_info


class OutputAxonAllocator:
    def __init__(self):
        self.axon_infos: dict[CoordXY, list[tuple[int, SourceElem]]] = {}
        self.used_bits: dict[CoordXY, set[int]] = {}
        self.lowest_free_bit: dict[CoordXY, int] = {}

    def get_next_free_bit(self, coord: CoordXY, start: int) -> int:
        while True:
            if start not in self.used_bits.get(coord, set()):
                return start
            start += 1

    def free_to_store_32bit(self, coord: CoordXY, start: int) -> bool:
        for i in range(4):
            if start + i * 8 in self.used_bits.get(coord, set()):
                return False
        return True

    def allocate(self, coord: CoordXY, elem: SourceElem) -> int:
        if coord not in self.axon_infos:
            self.axon_infos[coord] = []
            self.lowest_free_bit[coord] = 0
            self.used_bits[coord] = set()
        if elem.output_bit_num <= 8:
            axon_bit = self.lowest_free_bit[coord]
            self.used_bits[coord].add(axon_bit)
            self.axon_infos[coord].append((axon_bit, elem))
            next_free_bit = self.get_next_free_bit(coord, axon_bit + 1)
            self.lowest_free_bit[coord] = next_free_bit
        elif elem.output_bit_num == 32:
            candidate_bit = self.lowest_free_bit[coord]
            while True:
                if self.free_to_store_32bit(coord, candidate_bit):
                    axon_bit = candidate_bit
                    for i in range(4):
                        self.used_bits[coord].add(candidate_bit + i * 8)
                    self.axon_infos[coord].append((axon_bit, elem))
                    updated_free_bit = self.get_next_free_bit(
                        coord, self.lowest_free_bit[coord]
                    )
                    self.lowest_free_bit[coord] = updated_free_bit
                    break
                else:
                    candidate_bit = self.get_next_free_bit(coord, candidate_bit + 1)
        else:
            raise ValueError(
                f"Unsupported output bit num {elem.output_bit_num} for element {elem}."
            )
        return axon_bit


class OutputGroup(Group, DestGroup[SourceElem, SourceNode]):
    def __init__(
        self,
        input_list: Sequence[SourceElem],
        input_nodes: Set[SourceNode] | None = None,
    ):
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        self.name: str = f"OutputG_{self.id}"
        self._multicast_config: AERPacketZXYCopy = AERPacketZXYCopy(0, 0, 0)
        self._base_coord: CoordXY = CoordXY(0, 0)
        self.axon_bit_allocator = OutputAxonAllocator()
        self.lcn = LCN_EX.LCN_128X
        self.input_bit_num: int = 1

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix=prefix)
        info_str += DestGroup.info(self, prefix=prefix + "  ")
        info_str += "\n"
        return info_str

    def routing_summary(self) -> str:
        summary_str = f"Output Group {self.name}:\n"
        summary_str += "   Output Group is the final destination, no further routing.\n"
        return summary_str

    def __str__(self) -> str:
        info_str = self.info()
        return info_str

    def set_detail_dest(self):
        pass

    @property
    def multicast_config(self) -> AERPacketZXYCopy:
        return self._multicast_config

    @property
    def base_coord(self) -> CoordXY:
        return self._base_coord


def toposort_for_rg(
    groups: list[RemapGroup | RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    from collections import defaultdict, deque

    routing_groups = [rg for rg in groups if isinstance(rg, RoutingGroup)]
    rg_set = set(routing_groups)

    print("Routing Groups before topological sort:")
    for rg in routing_groups:
        print(f"{rg.name}")

    indegree = {rg: 0 for rg in routing_groups}

    for rg in indegree.keys():
        print(f"Routing Group {rg.name} has indegree {indegree[rg]} before sorting.")
    graph: dict[RoutingGroup, list[RoutingGroup]] = defaultdict(list)
    graph_set: dict[RoutingGroup, set[RoutingGroup]] = defaultdict(set)

    for rg in routing_groups:
        for neu in track(
            rg.raw_elems,
            description=f"Processing Routing Group {rg.name} ({len(rg.raw_elems)} neurons)",
            total=len(rg.raw_elems),  # 明确指定总数，确保进度条计算准确
        ):
            dest_rg = rg.get_dest(neu)
            if dest_rg not in rg_set or dest_rg is rg:
                continue
            if dest_rg not in graph_set[rg]:
                graph[rg].append(dest_rg)
                graph_set[rg].add(dest_rg)
                indegree[dest_rg] += 1

    for rg in graph:
        print(
            f"Routing Group {rg.name} has edges to {[dest.name for dest in graph[rg]]}"
        )

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
    if len(sorted_rgs) != len(routing_groups):
        for rg in routing_groups:
            if rg not in sorted_rgs:
                print(f"Routing Group {rg.name} is part of a cycle.")
                sorted_rgs.append(
                    rg
                )  # add the remaining RGs to the end of sorted list, even though they are in cycle

    next_rg_id = {}
    for i, rg in enumerate(sorted_rgs):
        next_rg_id[i] = [sorted_rgs.index(dest_rg) for dest_rg in graph[rg]]

    if len(sorted_rgs) != len(routing_groups):
        raise ValueError("Cycle detected in routing groups.")

    return sorted_rgs, next_rg_id
