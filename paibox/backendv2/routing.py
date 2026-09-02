import warnings
from abc import abstractmethod
from collections import defaultdict, deque
from collections.abc import Sequence, Set
from typing import Generic, TypeVar

import numpy as np
from paicorelib import (
    LCN_EX,
    AERPacketZXYCopy,
    CoordXY,
    CoordXYLike,
    CoordZXYOffset,
    CSCAccelerateMode,
    FoldType,
    NeuronType,
    OfflineNeuDestInfoV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OfflineNeuFullAttrsV2Part1,
    OfflineNeuFullAttrsV2Part2,
    OfflineNeuRegLimV2,
    WeightCompressType,
    find_coordxy_shortest_path,
    to_coordxy,
)
from rich.progress import track

from .core_config import Backend_Core_Config, Frontend_Core_Config
from .coreplacement import (
    CorePlacement,
    EmptyOfflineCorePlacementV2,
    OfflineCorePlacementV2,
)
from .fold_neu import get_fold_info
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
    RemapNode,
    SourceElem,
    SourceNode,
)
from .route_scope import RouteScope, get_route_scope
from .weight import Weight

FANIN_BASE = 512
OutputRouteOffsets = dict[tuple[CoordXY, CoordXY], CoordZXYOffset]
FOLD_WEIGHT_SKEW_MAX = OfflineNeuRegLimV2.FOLD_SKEW_MAX
FOLD_AXON_MAX = OfflineNeuRegLimV2.FOLD_AXON_MAX


def _first_invalid_fold_field(
    skews: list[int], bit_num: int, field_max: int
) -> tuple[int, int] | None:
    field_values = [skew * bit_num for skew in skews]
    if any(not 0 <= field_value <= field_max for field_value in field_values):
        return next(
            (
                (skew, field_value)
                for skew, field_value in zip(skews, field_values)
                if not 0 <= field_value <= field_max
            ),
            None,
        )
    return None


class Group:
    group_counter = 0

    def __init__(self) -> None:
        self.id: int = type(self).group_counter
        type(self).group_counter += 1
        self.name: str = f"Group_{self.id}"

    def info(self, prefix: str = "") -> str:
        return f"{prefix}{self.name}:"

    def __str__(self) -> str:
        return self.info() + "\n"

    __repr__ = __str__

    def __hash__(self):
        return hash(id(self))


SOURCE_ELEM = TypeVar("SOURCE_ELEM", bound=SourceElem)
SOURCE_NODE = TypeVar("SOURCE_NODE", bound=SourceNode)


class DestGroup(Generic[SOURCE_ELEM, SOURCE_NODE]):
    def __init__(
        self,
        input_list: Sequence[SOURCE_ELEM],
        input_nodes: Set[SOURCE_NODE] | None = None,
    ) -> None:
        self.input_nodes: set[SOURCE_NODE] | None = (
            set(input_nodes) if input_nodes is not None else None
        )
        self.input_list: list[SOURCE_ELEM] = list(input_list)
        self.input_set: set[SOURCE_ELEM] = set(input_list)
        self.index_map: dict[SOURCE_ELEM, int] = {}
        self.set_index_map()

    def set_index_map(self) -> None:
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
        return f"{self.__class__.__name__}: \n" + self.info(prefix="  ") + "\n"


class SourceGroup(Generic[SOURCE_ELEM, SOURCE_NODE]):
    def __init__(
        self, raw_neus: Sequence[SOURCE_ELEM], nodes: Set[SOURCE_NODE] | None = None
    ) -> None:
        self.nodes: Set[SOURCE_NODE] | None = set(nodes) if nodes is not None else None
        self.raw_elems: list[SOURCE_ELEM] = list(raw_neus)
        self.used_elems: list[SOURCE_ELEM] = []
        self.elem_set: set[SOURCE_ELEM] = set(raw_neus)
        self.dests: dict[SOURCE_ELEM, "RoutingGroup | OutputGroup| RemapGroup"] = {}

    def update_raw_elems(self) -> None:
        self.raw_elems = self.used_elems

    @abstractmethod
    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        pass

    def get_dest_info(
        self, elem: SOURCE_ELEM
    ) -> tuple["RoutingGroup | OutputGroup", int]:
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.remap_dest_info(elem)
        dest_axon = dest_group.index_map.get(elem, -1)
        assert dest_axon >= 0, f"Neuron {elem} not found in dest_group's index_map"
        return dest_group, dest_axon

    def get_dest(self, elem: SOURCE_ELEM) -> "RoutingGroup | OutputGroup":
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.remap_dest(elem)
        return dest_group

    def get_axon(self, elem: SOURCE_ELEM) -> SourceElem:
        dest_group = self.dests[elem]
        if isinstance(dest_group, RemapGroup):
            return dest_group.remap_axon(elem)
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

    def routing_summary(self, prefix: str = "") -> str:
        summary_str = f"{prefix}    Not Deploy Group\n"
        return summary_str

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: \n" + self.info(prefix="  ") + "\n"

    def get_detail_dest(
        self,
        elems: list[SOURCE_ELEM],
        self_coord: CoordXYLike = CoordXY(0, 0),
        output_route_offsets: OutputRouteOffsets | None = None,
        route_scope: RouteScope | None = None,
    ) -> OfflineNeuDestInfoV2:
        self_coord = to_coordxy(self_coord)
        dest_routing_group = self.get_dest(elems[0])
        axon_elem = self.get_axon(elems[0])
        if isinstance(dest_routing_group, RoutingGroup):
            axon_addr_logic = dest_routing_group.index_map.get(axon_elem, -1)
            assert axon_addr_logic >= 0, "axon_addr_logic should be non-negative"
            assert elems[0].output_bit_num == dest_routing_group.input_bit_num, (
                "Output bit num of elem must match input bit num of dest routing group"
            )
            axon_bit_count = axon_addr_logic * dest_routing_group.input_bit_num
        elif isinstance(dest_routing_group, OutputGroup):
            axon_bit_count = -1
            axon_bit_count = dest_routing_group.input_mapping[axon_elem]
            assert (
                axon_bit_count <= dest_routing_group.axon_bit_allocator.max_axon_bit
            ), (
                "Total axon bit count for output group exceeds the maximum "
                f"supported by {dest_routing_group.lcn.name}"
            )

        dest_coord = dest_routing_group.base_coord
        coord_copy = dest_routing_group.multicast_config

        tick_relative, addr_axon = divmod(axon_bit_count, FANIN_BASE)

        if output_route_offsets is not None and isinstance(
            dest_routing_group, OutputGroup
        ):
            coord_offset = output_route_offsets.get((self_coord, dest_coord))
        else:
            coord_offset = None

        if coord_offset is None:
            coord_offset, _ = find_coordxy_shortest_path(dest_coord, start=self_coord)

        scope = route_scope or get_route_scope("single")
        audit = scope.audit_aer_packet(self_coord, coord_offset, coord_copy)
        if not audit.valid:
            raise ValueError(
                f"Illegal DATA route from {self_coord} to {dest_coord}: "
                f"{audit.failure.message if audit.failure else 'unknown audit failure'}."
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

    def set_detail_dest(self) -> None:
        pass


class RemapGroup(
    Group, DestGroup[SourceElem, SourceNode], SourceGroup[RemapElem, RemapNode]
):
    remap_group_counter = 0

    def __init__(
        self,
        raw_elems: Sequence[RemapElem],
        input_list: Sequence[SourceElem],
        nodes: Set[RemapNode] | None = None,
        input_nodes: Set[SourceNode] | None = None,
    ) -> None:
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        SourceGroup.__init__(self, raw_elems, nodes)
        self.name: str = f"RemapG_{self.id}"
        self.remap_dict: dict[SourceElem, RemapElem] = dict()
        self.source_dict: dict[RemapElem, SourceElem] = dict()
        self.set_remap_dict()

    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        if not isinstance(elem, RemapElem):
            raise TypeError("Only RemapElem can be added to RemapGroup")
        self.raw_elems.append(elem)
        self.elem_set.add(elem)
        raw_elem = elem.origin_elem()
        raw_input = self.source_dict.get(raw_elem, None)
        if raw_input is None:
            return None
        copy_input = raw_input.copy(elem.index.copy_id)
        self.input_list.append(copy_input)
        self.input_set.add(copy_input)
        self.remap_dict[copy_input] = elem
        self.source_dict[elem] = copy_input
        self.nodes = None
        return copy_input

    def set_remap_dict(self) -> None:
        assert self.nodes is not None, "nodes must be provided for RemapGroup"
        for node in self.nodes:
            remap_info = node.get_remap_info()
            self.remap_dict.update(remap_info)
        assert set(self.remap_dict.keys()) == self.input_set, (
            "remap_info keys must match input_set"
        )
        assert set(self.remap_dict.values()).issubset(set(self.raw_elems)), (
            "remap_info values must match raw_neus"
        )

        for src, dst in self.remap_dict.items():
            self.source_dict[dst] = src

    def remap_axon(self, elem: SourceElem) -> SourceElem:
        out_elem = self.remap_dict[elem]
        return self.get_axon(out_elem)

    def remap_dest(self, elem: SourceElem) -> "RoutingGroup|OutputGroup":
        out_elem = self.remap_dict[elem]
        return self.get_dest(out_elem)

    def remap_dest_info(
        self, elem: SourceElem
    ) -> tuple["RoutingGroup | OutputGroup", int]:
        out_elem = self.remap_dict[elem]
        return self.get_dest_info(out_elem)

    def remap_info(self, prefix: str = "") -> str:
        info_str = ""
        remap_strs = []
        if len(self.input_list) > 6:
            print_remaps = self.input_list[:3] + self.input_list[-3:]
        else:
            print_remaps = self.input_list

        for input in print_remaps:
            remap_strs.append(f"{str(input)} -> {str(self.remap_dict[input])}")
        if len(self.input_list) > 6:
            remap_strs = remap_strs[:3] + ["..."] + remap_strs[-3:]
        info_str += f"\n{prefix}Number of Remaps: {len(self.input_list)}"
        info_str += f"\n{prefix}Remaps:\n{prefix}   "
        info_str += f"\n{prefix}   ".join(remap_strs)
        return info_str

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix)
        info_str += DestGroup.info(self, prefix=prefix + "  ")
        info_str += SourceGroup.info(self, prefix=prefix + "  ")
        info_str += self.remap_info(prefix=prefix + "  ")
        info_str += "\n"
        return info_str

    def routing_summary(self, prefix: str = "") -> str:
        summary_str = f"{prefix}Remap Group {self.name}:\n"
        summary_str += SourceGroup.routing_summary(self, prefix=prefix)
        return summary_str

    def __str__(self) -> str:
        return self.info()


class RoutingGroup(
    Group, DestGroup[SourceElem, SourceNode], SourceGroup[Neuron, CoreOpNode]
):
    def __init__(
        self,
        raw_neus: Sequence[Neuron],
        input_list: Sequence[SourceElem],
        nodes: Set[CoreOpNode] | None = None,
        input_nodes: Set[SourceNode] | None = None,
    ) -> None:
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        SourceGroup.__init__(self, raw_neus, nodes)
        self.name: str = f"RG_{self.id}"

        self.lcn: LCN_EX = LCN_EX.LCN_1X
        self.recommand_lcn: LCN_EX | None = None
        self.input_bit_num: int = 0

        # self.core_blocks: list[CoreBlock] = []
        self.last_full_attrs: OfflineNeuFullAttrsV2Part2 | None = None
        self.last_full_stored_weight_index: int | None = None

        self.core_placements: list[CorePlacement] = []

        self.assigned_cores: dict[CoordXY, CorePlacement] = {}
        self._multicast_config: AERPacketZXYCopy | None = None
        self._base_coord: CoordXY | None = None

    @property
    def n_core_required(self) -> int:
        return len(self.core_placements)

    def layer_key(self) -> tuple[tuple[int, ...], str]:
        """Return a stable grouping key for tiled groups from the same source node."""
        targets = {neu.target for neu in self.raw_elems}
        if not targets and self.nodes:
            targets = set(self.nodes)
        if not targets:
            return (id(self),), self.name

        ordered_targets = sorted(targets, key=id)
        return tuple(id(target) for target in ordered_targets), "+".join(
            str(target) for target in ordered_targets
        )

    def add_elem(self, elem: SourceElem) -> SourceElem | None:
        if not isinstance(elem, Neuron):
            raise TypeError("Only Neuron can be added to RoutingGroup")
        self.raw_elems.append(elem)
        self.elem_set.add(elem)
        self.nodes = None
        return None

    def set_lcn(self) -> None:
        intput_bit_nums: set[int] = set([neu.input_bit_num for neu in self.raw_elems])
        pred_output_bit_nums: set[int] = set(
            [src.output_bit_num for src in self.input_list]
        )
        assert len(intput_bit_nums) == 1, (
            "All neurons in the routing group must have the same input bit num."
        )
        assert len(pred_output_bit_nums) == 1, (
            "All input elements in the routing group must have the same output bit num."
        )
        # print(f"{self.raw_elems[0]}: input_bit_nums: {intput_bit_nums}")
        # print(f"{self.input_list[0]}: pred_output_bit_nums: {pred_output_bit_nums}")

        self.input_bit_num = intput_bit_nums.pop()
        assert self.input_bit_num == pred_output_bit_nums.pop(), (
            "Input bit num of neurons must match output bit num of input elements."
        )
        max_axon_addr = len(self.input_list) * self.input_bit_num
        lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
        if self.recommand_lcn is not None and lcn < self.recommand_lcn.value:
            self.lcn = self.recommand_lcn
        else:
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
            base_weight = base_weights[weight_info.index]
            selected_weight, weight_compress = choose_weight_strategy(
                base_weight,
                frontend_core_conf.weight_width,
                frontend_core_conf.input_width,
                frontend_core_conf.add_potential,
            )
            weight_sram_req = selected_weight.n_sram_required
            attrs_part2.weight_compress = weight_compress
            print(
                f"\tno zero elements in base weight {weight_info.index} is {np.count_nonzero(base_weight)}"
            )
            print(
                f"\tbase weight {weight_info.index} requires {weight_sram_req} SRAM lines with compression {weight_compress}."
            )
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
                self.last_full_stored_weight_index = None
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
        if can_use_half_neuron := (attrs_part2 == self.last_full_attrs):
            if (
                current_core.default_core_config.csc_accelerate
                == CSCAccelerateMode.ENABLE
                and weight_compress == WeightCompressType.SPARSE
                and stored_weight_index != self.last_full_stored_weight_index
            ):
                # Half neurons still carry their own weight address range in part1.
                # The unsafe shared field is part2.vjt_initial: with CSC accelerate
                # it mirrors weight_address_start, so different stored sparse weights
                # need different full-neuron part2 records.
                can_use_half_neuron = False

        neuron_type = NeuronType.HALF if can_use_half_neuron else NeuronType.FULL
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
            self.last_full_stored_weight_index = None
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
                self.last_full_stored_weight_index = stored_weight_index

            current_core.neus.append(neu_placement)
            # current_core.weights.append(selected_weight)  # weight 已经存储过了，不需要重复存储
            idx = len(current_core.neus) - 1
            current_core.neu_weight_map[idx] = (
                stored_weight_index  # 这个 neu 使用的 weight 是 base_weight_idx
            )
        return current_core

    def try_to_fold_neuron(
        self,
        base_weights: list[np.ndarray],
        ordered_neus: list[Neuron],
        ordered_infos: list[WeightInfo],
        frontend_core_conf: Frontend_Core_Config,
        backend_core_conf: Backend_Core_Config,
        prefix: str = "",
    ):
        folded_neurons: set[Neuron] = set()
        buckets: dict[int, list[tuple[Neuron, WeightInfo]]] = defaultdict(
            list[tuple[Neuron, WeightInfo]]
        )
        for neu, info in zip(ordered_neus, ordered_infos):
            buckets[info.index].append((neu, info))
        current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)
        stored_base_weight: dict[int, tuple[int, WeightCompressType]] = {}
        weight_strategy_cache: dict[int, tuple[Weight, WeightCompressType]] = {}
        for index, bucket in buckets.items():
            neurons = [neu for neu, _ in bucket]
            neu_nodes = [neu.target for neu in neurons]
            if len(set(neu_nodes)) != 1:
                # 这个 base weight 对应的 neurons 来自不同的 neuron nodes 上，暂时不考虑折叠
                continue
            sub_buckets: dict[
                RoutingGroup | OutputGroup, list[tuple[Neuron, WeightInfo]]
            ] = defaultdict(list[tuple[Neuron, WeightInfo]])
            for neu, info in bucket:
                dest_group = self.get_dest(neu)
                sub_buckets[dest_group].append((neu, info))

            for dest_group, sub_bucket in sub_buckets.items():
                if isinstance(dest_group, OutputGroup):
                    print(
                        f"{prefix} neurons with base weight [{index}]: dest is OutputGroup {dest_group.name}, skip fold."
                    )
                    # OutputGroup 的 axon bit 依赖路由后坐标分配，当前阶段不做 fold
                    continue
                if len(sub_bucket) <= 1:
                    continue
                sub_neurons = [neu for neu, _ in sub_bucket]
                weight_offsets = [info.offset for _, info in sub_bucket]
                axon_addr_offsets = [self.get_dest_info(neu)[1] for neu in sub_neurons]

                fold_info = get_fold_info(weight_offsets, axon_addr_offsets)
                if fold_info is None:
                    continue
                ranges, fold_weight_skews, fold_axon_skews = fold_info
                print(
                    f"{prefix}Find fold {len(sub_neurons)} neurons with base weight {index} "
                    f"to dest group {dest_group.name}:"
                )
                print(f"{prefix}    fold_range: {ranges}")
                print(f"{prefix}    fold_weight_skews: {fold_weight_skews}")
                print(f"{prefix}    fold_axon_skews: {fold_axon_skews}")
                invalid_weight = _first_invalid_fold_field(
                    fold_weight_skews, self.input_bit_num, FOLD_WEIGHT_SKEW_MAX
                )
                if invalid_weight is not None:
                    skew, field_value = invalid_weight
                    print(
                        f"{prefix}Fold weight skew {skew} maps to {field_value}, "
                        f"outside [0, {FOLD_WEIGHT_SKEW_MAX}], skipping fold."
                    )
                    continue

                invalid_axon = _first_invalid_fold_field(
                    fold_axon_skews, dest_group.input_bit_num, FOLD_AXON_MAX
                )
                if invalid_axon is not None:
                    skew, field_value = invalid_axon
                    print(
                        f"{prefix}Fold axon skew {skew} maps to {field_value}, "
                        f"outside [0, {FOLD_AXON_MAX}], skipping fold."
                    )
                    continue

                attrs_part2s = [neu.attrs_part2() for neu in sub_neurons]
                if any(
                    attrs_part2 != attrs_part2s[0] for attrs_part2 in attrs_part2s[1:]
                ):
                    print(
                        f"{prefix}Fold candidates with base weight {index} have "
                        "different full-neuron Part2 attrs, skipping fold."
                    )
                    continue

                if index not in weight_strategy_cache:
                    base_weight = base_weights[index]
                    selected_weight, weight_compress = choose_weight_strategy(
                        base_weight,
                        frontend_core_conf.weight_width,
                        frontend_core_conf.input_width,
                        frontend_core_conf.add_potential,
                    )
                    weight_strategy_cache[index] = (selected_weight, weight_compress)
                selected_weight, weight_compress = weight_strategy_cache[index]
                weight_sram_req = selected_weight.n_sram_required

                attrs_part2 = attrs_part2s[0]
                attrs_part2.weight_compress = weight_compress
                neuron_type = NeuronType.FULL
                output_type = sub_neurons[0].output_type()
                attrs_part1 = OfflineNeuFullAttrsV2Part1(
                    weight_skew=weight_offsets[0] * self.input_bit_num,
                    weight_address_start=0,  # 之后会统一设置
                    weight_address_end=0,  # 之后会统一设置
                    fold_type=FoldType.FOLDED,
                    neuron_type=neuron_type,
                    output_type=output_type,
                )

                if isinstance(dest_group, OutputGroup):
                    # if folded neurons send to output group,
                    # axon skews should be all 1.
                    assert fold_axon_skews == [
                        1,
                        1,
                        1,
                    ], (
                        "Folded neurons sending to output group should have axon skew of 1"
                    )

                fold_attrs_part1 = OfflineNeuFoldedAttrsV2Part1(
                    fold_axon_y=fold_axon_skews[0] * dest_group.input_bit_num,
                    fold_axon_x=fold_axon_skews[1] * dest_group.input_bit_num,
                    fold_axon_xy=fold_axon_skews[2] * dest_group.input_bit_num,
                    fold_skew_y=fold_weight_skews[0] * self.input_bit_num,
                    fold_skew_x=fold_weight_skews[1] * self.input_bit_num,
                    fold_skew_xy=fold_weight_skews[2] * self.input_bit_num,
                    fold_range_y=ranges[0],
                    fold_range_x=ranges[1],
                    fold_range_xy=ranges[2],
                    fold_number=len(sub_neurons),
                )
                num_fold_part2 = (len(sub_neurons) - 1 + 3) // 4
                fold_attrs_part2s = [
                    OfflineNeuFoldedAttrsV2Part2() for _ in range(num_fold_part2)
                ]
                neu_placement = OfflineNeuronPlacement(
                    sub_neurons,
                    attrs_part1,
                    attrs_part2,
                    fold_attrs_part1,
                    fold_attrs_part2s,
                )
                neu_sram_req = neu_placement.n_sram_required

                added_weight_sram_req = (
                    0 if index in stored_base_weight else weight_sram_req
                )
                total_sram_req = neu_sram_req + added_weight_sram_req
                if total_sram_req > 4096:
                    print(
                        f"{prefix}Folding neurons with base weight {index} to dest group "
                        f"{dest_group.name} requires {total_sram_req} SRAM lines, "
                        "which exceeds the limit. Skipping folding."
                    )
                    continue

                if current_core.n_sram_required + total_sram_req > 4096:
                    if len(current_core.neus) > 0:
                        self.core_placements.append(current_core)
                    current_core = OfflineCorePlacementV2(
                        frontend_core_conf, backend_core_conf
                    )
                    stored_base_weight.clear()
                    added_weight_sram_req = weight_sram_req
                    total_sram_req = neu_sram_req + added_weight_sram_req
                    if total_sram_req > 4096:
                        print(
                            f"{prefix}Folding neurons with base weight {index} to dest group "
                            f"{dest_group.name} requires {total_sram_req} SRAM lines, "
                            "which exceeds the limit. Skipping folding."
                        )
                        continue

                remaining_sram = 4096 - current_core.n_sram_required - total_sram_req
                print(
                    f"{prefix}core[{len(self.core_placements)}]:Folded neuron stored with base weight {index} "
                    f"to dest group {dest_group.name}:"
                )
                print(f"{prefix}    num neurons: {len(sub_neurons)}")
                print(f"{prefix}    num sram lines: {neu_sram_req}")
                print(f"{prefix}    weight sram lines: {added_weight_sram_req}")
                print(f"{prefix}    total sram lines: {total_sram_req}")
                print(f"{prefix}    remaining sram lines: {remaining_sram}")

                if index not in stored_base_weight:
                    current_core.weights.append(selected_weight)
                    stored_base_weight[index] = (
                        len(current_core.weights) - 1,
                        weight_compress,
                    )
                stored_weight_index, _ = stored_base_weight[index]
                current_core.neus.append(neu_placement)
                current_core.neu_weight_map[len(current_core.neus) - 1] = (
                    stored_weight_index
                )
                folded_neurons.update(sub_neurons)

        return folded_neurons, current_core, stored_base_weight

    def place_neurons_optimal(
        self,
        frontend_core_conf: Frontend_Core_Config,
        backend_core_conf: Backend_Core_Config,
        group_items: list[tuple[Neuron, np.ndarray]],
        block_id: int = 0,
        prefix: str = "",
    ) -> None:
        weights_of_group = [item[1] for item in group_items]
        weight_infos, base_weights = group_shift_weights_optimized(
            weights_of_group, prefix
        )
        # reorder group_items making the ones with the same base weight together, to improve weight storage efficiency
        reordered_neus, reordered_infos = reorder_by_base_weight(
            group_items, weight_infos
        )

        import os

        if os.environ.get("DUMP_WEIGHTS", None) == "1":
            output_dir = os.environ.get("PAIBOX_OUTPUT_PATH", "./weights")
            with open(f"{output_dir}/weights/{self.name}_weight_base.txt", "w") as f:
                for weight in base_weights:
                    f.write(" ".join(map(str, weight)) + "\n")
                for info in reordered_infos:
                    f.write(f"index: {info.index}, offset: {info.offset}\n")

        print(f"{prefix}trying to fold neurons")
        folded_neurons, current_core, stored_base_weight = self.try_to_fold_neuron(
            base_weights,
            reordered_neus,
            reordered_infos,
            frontend_core_conf,
            backend_core_conf,
            prefix=f"{prefix}    ",
        )

        # current_core = OfflineCorePlacementV2(frontend_core_conf, backend_core_conf)
        self.last_full_attrs = None
        self.last_full_stored_weight_index = None
        description = f"{prefix}place unfolded neurons"
        for neu, weight_info in track(
            zip(reordered_neus, reordered_infos),
            description=description,
            total=len(reordered_neus),  # 明确指定总数，确保进度条计算准确
        ):
            if neu in folded_neurons:
                continue
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

    def allocate_neurons(self) -> None:
        """core placement generation"""
        # you can get neu_attrs_part2, inherited_core_config, target_lcn, lcn for each neuron in self.raw_elems, like:
        # all the attrs in neu_attrs_part2 are valid except weight compress, you should set weight compress according to your weight storage strategy
        # inherited core_config are valid except weight_width, you can set weight_width larger than or equal to the original value for optimization

        print(f"\nAllocating neurons for Routing Group {self.name}...")
        prefix = "    "
        weights = get_raw_weights(self.raw_elems, self.input_list)

        # print(f"weight of routing group {self.name}:\n", weights)
        print(f"{prefix}weight shape of routing group {self.name}: {weights.shape}")

        # print compelet weights into file for debug
        # with open(f"{self.name}_weights.txt", "w") as f:
        #     weights_transposed = weights.T
        #     for row in weights_transposed:
        #         f.write(" ".join(map(str, row)) + "\n")

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
        print(f"{prefix}Number of core blocks: {len(core_groups)}")
        # for key, group_items in core_groups.items():
        #     print(f"\n\tCore block with {len(group_items)}:")
        #     print(f"\t\tfrontend_core_conf={key[0]}")
        #     print(f"\t\tbackend_core_conf={key[1]}")

        # 2. Allocation Phase
        for i, (key, group_items) in enumerate(core_groups.items()):
            # print(f"\n\nAllocating group with frontend_core_conf={key[0]}")
            # print(f"backend_core_conf={key[1]}")
            # print(f"Neu of this group: {[str(item[0]) for item in group_items]}")
            print(f"{prefix}allocating core_block[{i}] ({len(group_items)} neurons)...")
            frontend_core_conf, backend_core_conf = key
            self.last_full_attrs = None
            self.last_full_stored_weight_index = None
            self.place_neurons_optimal(
                frontend_core_conf,
                backend_core_conf,
                group_items,
                i,
                prefix=f"{prefix}    ",
            )
            print(
                f"{prefix}Number of cores of core_block[{i}]: {len(self.core_placements)}"
            )

        for core_placement in self.core_placements:
            core_placement.set_weight_address()

    def assign_coord(self, coords: list[CoordXY], copy_config: AERPacketZXYCopy):
        assert len(coords) >= len(self.core_placements), (
            "Not enough coordinates provided to assign."
        )
        for i, coord in enumerate(coords):
            if i < len(self.core_placements):
                self.assigned_cores[coord] = self.core_placements[i]
            else:
                self.assigned_cores[coord] = EmptyOfflineCorePlacementV2()
            self.assigned_cores[coord]._coord = coord
        self._base_coord = coords[0]
        self._multicast_config = copy_config

    def set_detail_dest(
        self,
        output_route_offsets: OutputRouteOffsets,
        route_scope: RouteScope | None = None,
    ) -> None:
        for core_placement in track(
            self.core_placements,
            description=f"Setting Detail Destinations for {self.name}",
            total=len(self.core_placements),
        ):
            for neu_placement in core_placement.neus:
                # mostly each neu_placement contains only one raw_neu
                # for folded neuron placement, it may contain multiple raw_neus
                # but we only need to set dest_info for the first raw_neu
                dest_info = self.get_detail_dest(
                    neu_placement.raw_neus,
                    core_placement.coord,
                    output_route_offsets,
                    route_scope,
                )
                neu_placement.dest_info = dest_info

    def set_auto_core_config(self, test_dest_core: CoordXY) -> None:
        for cp in self.core_placements:
            test_offset, _ = find_coordxy_shortest_path(test_dest_core, cp.coord)
            cp.set_auto_core_config(test_offset)

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
        return self.info()

    def routing_summary(self, prefix: str = "") -> str:
        summary_str = f"{prefix}{self.name} Routing Summary ({len(self.core_placements)} cores):\n"
        for i, core_placement in enumerate(self.core_placements):
            if core_placement._coord is not None:
                summary_str += (
                    f"{prefix}  Core Placement {i} at {core_placement.coord}:\n"
                )
            else:
                summary_str += f"{prefix}  Core Placement {i} at Unassigned Coord:\n"
            summary_str += f"{prefix}    Number of Neurons: {sum([len(neu_placement.raw_neus) for neu_placement in core_placement.neus])}\n"
            summary_str += (
                f"{prefix}    Number of Weights: {len(core_placement.weights)}\n"
            )
            summary_str += (
                f"{prefix}    Number of Neuron Placements: {len(core_placement.neus)}\n"
            )
            summary_str += f"{prefix}    Neuron SRAM Required: {core_placement.neuron_sram_required}\n"
            summary_str += f"{prefix}    Weight SRAM Required: {core_placement.weight_sram_required}\n"
            summary_str += (
                f"{prefix}    Total SRAM Required: {core_placement.n_sram_required}\n"
            )
            summary_str += f"{prefix}    Total Compute Pressure: {core_placement.get_compute_pressure()}\n"
        return summary_str


class InputGroup(Group, SourceGroup[InputElem, InNode]):
    def __init__(
        self, raw_elems: Sequence[InputElem], nodes: Set[InNode] | None = None
    ) -> None:
        Group.__init__(self)
        SourceGroup.__init__(self, raw_elems, nodes)
        self.name: str = f"InputG_{self.id}"
        self.dest_infos: dict[InputElem, OfflineNeuDestInfoV2] = {}
        self.dest_lcn: dict[InputElem, LCN_EX] = {}
        self.thread_id: int = 0

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

    def routing_summary(self, prefix: str = "") -> str:
        summary_str = f"{prefix}{self.__class__.__name__} {self.name}:\n"
        summary_str += SourceGroup.routing_summary(self, prefix=prefix)
        return summary_str

    def __str__(self) -> str:
        return self.info()

    def set_detail_dest(
        self,
        input_coord: CoordXYLike = CoordXY(0, 0),
        route_scope: RouteScope | None = None,
    ) -> None:
        for elem in self.raw_elems:
            dest_info = self.get_detail_dest(
                [elem], input_coord, route_scope=route_scope
            )
            self.dest_infos[elem] = dest_info
            dest_rg = self.get_dest(elem)
            self.dest_lcn[elem] = dest_rg.lcn


def _max_axon_bit_for_lcn(target_lcn: LCN_EX) -> int:
    """Return the largest flat output axon bit address supported by an LCN."""
    return FANIN_BASE * (1 << target_lcn.value) - 1


MAX_LCN = LCN_EX.LCN_128X
OUTPUT_TIMESTEP_FIELD_BITS = MAX_LCN.value + 1


class OutputAxonAllocator:
    """Assign stable flat output axon bit addresses for one OutputGroup."""

    DEFAULT_TARGET_LCN = MAX_LCN

    def __init__(self, target_lcn: LCN_EX = DEFAULT_TARGET_LCN) -> None:
        self.target_lcn = target_lcn
        self.max_axon_bit = _max_axon_bit_for_lcn(target_lcn)
        # Ordered export view: proto output entries keep this allocation order.
        self.axon_infos: list[tuple[int, SourceElem]] = []
        # Fast idempotency guard for repeated detail-destination generation.
        self.axon_by_elem: dict[SourceElem, int] = {}
        self.used_bits: set[int] = set()
        self.lowest_free_bit: int = 0

    def retarget(self, target_lcn: LCN_EX) -> None:
        """Narrow or widen capacity after allocation without changing addresses."""
        max_axon_bit = _max_axon_bit_for_lcn(target_lcn)
        if self.used_bits and max(self.used_bits) > max_axon_bit:
            raise ValueError(
                f"Cannot retarget output allocator to {target_lcn.name}: "
                f"allocated axon bit {max(self.used_bits)} exceeds "
                f"the maximum supported axon bit {max_axon_bit}."
            )
        self.target_lcn = target_lcn
        self.max_axon_bit = max_axon_bit

    def get_next_free_bit(self, start: int) -> int:
        while True:
            if start > self.max_axon_bit:
                return start
            if start not in self.used_bits:
                return start
            start += 1

    def free_to_store_32bit(self, start: int) -> bool:
        for i in range(4):
            # A 32-bit voltage value occupies 4 byte lanes, 8 bits apart.
            axon_bit = start + i * 8
            if axon_bit > self.max_axon_bit or axon_bit in self.used_bits:
                return False
        return True

    def allocate(self, elem: SourceElem) -> int:
        if elem in self.axon_by_elem:
            return self.axon_by_elem[elem]

        if elem.output_bit_num <= 8:
            axon_bit = self.lowest_free_bit
            if axon_bit > self.max_axon_bit:
                raise ValueError(
                    f"Cannot allocate {elem.output_bit_num}-bit output for "
                    f"element {elem}: output axon space is exhausted."
                )
            self.used_bits.add(axon_bit)
            self.axon_infos.append((axon_bit, elem))
            next_free_bit = self.get_next_free_bit(axon_bit + 1)
            self.lowest_free_bit = next_free_bit
        elif elem.output_bit_num == 32:
            candidate_bit = self.lowest_free_bit
            while True:
                if candidate_bit > self.max_axon_bit:
                    raise ValueError(
                        f"Cannot allocate 32-bit output for element {elem}: "
                        "output axon space is exhausted."
                    )
                # A 32-bit voltage value occupies 4 byte lanes, 8 bits apart.
                if self.free_to_store_32bit(candidate_bit):
                    axon_bit = candidate_bit
                    for i in range(4):
                        self.used_bits.add(candidate_bit + i * 8)
                    self.axon_infos.append((axon_bit, elem))
                    updated_free_bit = self.get_next_free_bit(self.lowest_free_bit)
                    self.lowest_free_bit = updated_free_bit
                    break
                else:
                    candidate_bit = self.get_next_free_bit(candidate_bit + 1)
        else:
            raise ValueError(
                f"Unsupported output bit num {elem.output_bit_num} for element {elem}."
            )
        if axon_bit > self.max_axon_bit:
            raise ValueError(
                f"Axon bit {axon_bit} allocated for element {elem} exceeds "
                f"the maximum supported axon bit {self.max_axon_bit}."
            )

        self.axon_by_elem[elem] = axon_bit
        return axon_bit


class OutputGroup(Group, DestGroup[SourceElem, SourceNode]):
    def __init__(
        self,
        input_list: Sequence[SourceElem],
        input_nodes: Set[SourceNode] | None = None,
    ) -> None:
        Group.__init__(self)
        DestGroup.__init__(self, input_list, input_nodes)
        self.name: str = f"OutputG_{self.id}"
        self._multicast_config: AERPacketZXYCopy = AERPacketZXYCopy(0, 0, 0)
        self._base_coord: CoordXY = CoordXY(0, 0)
        self.lcn = OutputAxonAllocator.DEFAULT_TARGET_LCN
        self.axon_bit_allocator = OutputAxonAllocator(self.lcn)
        self.input_bit_num: int = 1
        self.thread_id: int = 0
        self.input_mapping: dict[SourceElem, int] = {}

    def info(self, prefix: str = "") -> str:
        info_str = Group.info(self, prefix=prefix)
        info_str += DestGroup.info(self, prefix=prefix + "  ")
        info_str += "\n"
        return info_str

    def routing_summary(self, prefix: str = "") -> str:
        summary_str = f"{prefix}{self.__class__.__name__} {self.name}:\n"
        summary_str += f"{prefix}   {self.__class__.__name__} is the final destination, no further routing.\n"
        return summary_str

    def __str__(self) -> str:
        return self.info()

    def set_detail_dest(self) -> None:
        pass

    def _build_axon_allocator(self, target_lcn: LCN_EX) -> OutputAxonAllocator:
        # Build the allocator to completion so capacity checks and final state match.
        allocator = OutputAxonAllocator(target_lcn)
        for elem in self.input_list:
            allocator.allocate(elem)
        return allocator

    def set_lcn(self, required_ts: int) -> None:
        """Select output target LCN from deployed axon addresses.

        ``required_ts`` is the application runtime length. It does not drive the
        selected LCN; it only warns when the chosen output address width leaves
        too few local timestep bits for STREAM decoding.
        """
        if required_ts <= 0:
            raise ValueError(f"required_ts must be positive, got {required_ts}.")

        try:
            max_allocator = self._build_axon_allocator(MAX_LCN)
        except ValueError as exc:
            raise ValueError(
                f"Output axon space is exhausted even with {MAX_LCN.name}."
            ) from exc

        if max_allocator.used_bits:
            max_axon_addr = max(max_allocator.used_bits)
        else:
            max_axon_addr = 0

        # LCN is derived from deployed output axon address width. Runtime step
        # width is considered only to warn when STREAM decode cannot cover T.
        min_tick_relative_bit = (max_axon_addr // FANIN_BASE).bit_length()
        min_ts_bit = max((required_ts - 1).bit_length(), 1)

        if min_tick_relative_bit > MAX_LCN.value:
            raise ValueError(
                f"Max axon address {max_axon_addr} requires at least "
                f"{min_tick_relative_bit} bits, which exceeds the maximum "
                f"supported by {MAX_LCN.name}."
            )
        if min_ts_bit + min_tick_relative_bit > OUTPUT_TIMESTEP_FIELD_BITS:
            warnings.warn(
                "Output timestep bits and axon address bits cannot both fit in "
                f"{OUTPUT_TIMESTEP_FIELD_BITS} bits; runtime decode will require "
                "STEP mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            target_lcn = MAX_LCN
        else:
            target_lcn = LCN_EX(min_tick_relative_bit)

        max_allocator.retarget(target_lcn)
        self.lcn = target_lcn
        self.axon_bit_allocator = max_allocator
        self.input_mapping = max_allocator.axon_by_elem.copy()

    @property
    def multicast_config(self) -> AERPacketZXYCopy:
        return self._multicast_config

    @property
    def base_coord(self) -> CoordXY:
        return self._base_coord

    def set_base_coord(self, coord: CoordXY) -> None:
        """Set the CPU endpoint used as this output group's destination.

        Args:
            coord: CPU endpoint coordinate used by output DATA routes.
        """

        self._base_coord = coord


def toposort_for_rg(
    groups: list[RemapGroup | RoutingGroup],
) -> tuple[list[RoutingGroup], dict[int, list[int]]]:
    """topological sort for routing groups based on their dests"""
    routing_groups = [rg for rg in groups if isinstance(rg, RoutingGroup)]
    rg_set = set(routing_groups)

    indegree = {rg: 0 for rg in routing_groups}

    graph: dict[RoutingGroup, list[RoutingGroup]] = defaultdict(list)
    graph_set: dict[RoutingGroup, set[RoutingGroup]] = defaultdict(set)

    for rg in routing_groups:
        graph[rg] = []
        graph_set[rg] = set()
        for neu in rg.raw_elems:
            dest_rg = rg.get_dest(neu)
            if dest_rg not in rg_set or dest_rg is rg:
                continue
            if dest_rg not in graph_set[rg]:
                graph[rg].append(dest_rg)
                graph_set[rg].add(dest_rg)
                indegree[dest_rg] += 1

    print("\nRouting Group Graph:")
    for rg in graph:
        print(f"\t{rg.name}: {[dest.name for dest in graph[rg]]}")

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
