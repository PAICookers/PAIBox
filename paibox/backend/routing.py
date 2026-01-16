import itertools
import logging
import math
from collections import defaultdict, deque
from collections.abc import Generator, Iterable
from functools import cached_property
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    ONLINE_CORES_BASE_COORD,
    ChipCoord,
    Coord,
    CoreMode,
    HwConfig,
    RoutingCoord,
)
from paicorelib import ROUTING_DIRECTIONS_IDX as DIREC_IDX
from paicorelib import RoutingDirection as Direction
from paicorelib import RoutingLevel as Level
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

from paibox import _logging
from paibox.components import Conv2d, InputProj, MatMul2d
from paibox.components.neuron.base import NEU_TARGET_CHIP_UNSET
from paibox.exceptions import NotSupportedError, ResourceError, RoutingError
from paibox.utils import check_elem_same

from .conf_types import CorePlmConfInChip
from .constrs import GraphNodeConstrs
from .graph_utils import merge_cycles, toposort
from .group import InhiGroup, MergedGroup
from .placement import CoreBlock, EmptyCorePlacement
from .sub_utils import SubEdge, SubSourceType
from .tiling import conv2d_optimize
from .types import EdgeType, NodeType, _1st_core_coord_repr

__all__ = ["RoutingGroup", "RoutingManager"]

rt_grp_log = _logging.get_artifact_logger(__name__, "routing_group_info")


def MatMul2d_slices(mat_mul: MatMul2d) -> tuple[list[slice], list[slice]]:
    shape_in = mat_mul.shape_in
    shape_out = mat_mul.shape_out
    in_slice_len = shape_in[1]
    out_slice_len = shape_out[1]

    input_slices = [
        slice(i * in_slice_len, (i + 1) * in_slice_len) for i in range(shape_in[0])
    ]
    output_slices = [
        slice(i * out_slice_len, (i + 1) * out_slice_len) for i in range(shape_out[0])
    ]
    return input_slices, output_slices


def flatten_last_n_dims(x: np.ndarray, n=3):
    return x.reshape(x.shape[:-n] + (-1,))


def flatten_array(x: NDArray) -> NDArray:
    assert x.ndim == 6
    y = x.reshape(np.prod(x.shape[:3]), np.prod(x.shape[3:]))
    return y


def build_elements(
    merged_sgrp: MergedGroup, online: bool
) -> "list[CoreBlock| RoutingGroup]":
    nodes = list(merged_sgrp.nodes)
    elements: "list[CoreBlock| RoutingGroup]" = []

    mode = cast(CoreMode, nodes[0].mode)
    if any(mode != node.mode for node in nodes):
        raise NotSupportedError("mixed mode is not supported.")

    def simple_build(edges: Iterable[EdgeType]) -> None:
        sub_edges = [SubEdge(edge) for edge in edges]
        elements.append(CoreBlock.build(*sub_edges, rt_mode=mode, online=online))

    print(
        f"build elements for merged group with nodes: {[node.name for node in nodes]}"
    )
    # Optimize weight in single operator, like 'Mat2d', 'Conv2d'.
    if len(nodes) == 1:
        edges = merged_sgrp.outputs[nodes[0]]
        # find edges with divisible weight
        divisible_edge = None
        if len(edges) > 1:
            for edge in edges:
                # only one edge is allowed to have divisible weight
                if isinstance(edge, MatMul2d):
                    divisible_edge = edge
                    break
            # TODO we can judge whether optimization is needed here
            if divisible_edge is None:
                simple_build(edges)
            else:
                input_slices, output_slices = MatMul2d_slices(divisible_edge)
                for input_slice, output_slice in zip(input_slices, output_slices):
                    sub_edges: list[SubEdge] = []
                    in_raw_index = list(range(input_slice.start, input_slice.stop))
                    out_raw_index = list(range(output_slice.start, output_slice.stop))
                    for edge in edges:
                        if edge == divisible_edge:
                            sub_edges.append(
                                SubEdge(
                                    edge,
                                    in_raw_index=in_raw_index,
                                    out_raw_index=out_raw_index,
                                )
                            )
                        else:
                            sub_edges.append(
                                SubEdge(
                                    edge, in_raw_index=None, out_raw_index=out_raw_index
                                )
                            )
                    core_block = CoreBlock.build(
                        *sub_edges, rt_mode=mode, online=online
                    )
                    routing_group = RoutingGroup([core_block], [])
                    elements.append(routing_group)

        else:
            edge = edges[0]
            if isinstance(edge, Conv2d) and not isinstance(edge.source, InputProj):
                est_result, i_tiled_idx_map, o_tiled_idx_map, k_tiles, copy_times = (
                    conv2d_optimize(edge, online)
                )
                input_index_map = flatten_array(i_tiled_idx_map)
                output_index_map = flatten_array(o_tiled_idx_map)

                input_mask = input_index_map != -1
                output_mask = output_index_map != -1

                copy_count = np.zeros(
                    copy_times.size, dtype=copy_times.dtype
                )  # 展平 + 初始化
                sub_edges = []

                for i in range(input_index_map.shape[0]):
                    in_idx = input_index_map[i][input_mask[i]]
                    out_idx = output_index_map[i][output_mask[i]]
                    in_copy_id = copy_count[in_idx].copy()  # 取当前 copy id

                    sub_edge = SubEdge(
                        edge,
                        in_raw_index=in_idx.tolist(),
                        in_copy_id=in_copy_id.tolist(),
                        out_raw_index=out_idx.tolist(),
                    )

                    core_block = CoreBlock.build(sub_edge, rt_mode=mode, online=online)
                    routing_group = RoutingGroup([core_block], [])
                    elements.append(routing_group)
                    np.add.at(copy_count, in_idx, 1)
            else:
                simple_build(edges)

    else:
        # TODO More constraints for nodes can be called here.
        # TODO weight can be optimized between operators.
        idx_in_cbs = GraphNodeConstrs.apply_constrs(nodes, online)
        # if len(idx_of_sg) == 0:
        #     idx_of_sg = [list(range(len(nodes)))]

        for idx_in_cb in idx_in_cbs:
            edges_set: set[EdgeType] = set()
            for i in idx_in_cb:
                edges_set.update(merged_sgrp.outputs[nodes[i]])
            simple_build(edges_set)

    return elements


RoutingElemType = "CoreBlock | RoutingGroup"
OrderedElemsType = list["RoutingGroup"]
UnorderedElemsType = list[RoutingElemType]


def _iter_rg(iter: Iterable) -> Generator["RoutingGroup", None, None]:
    return (elem for elem in iter if isinstance(elem, RoutingGroup))


def _iter_cb(iter: Iterable) -> Generator[CoreBlock, None, None]:
    return (elem for elem in iter if isinstance(elem, CoreBlock))


class RoutingGroup:
    """Each routing group should be able to route by single coord."""

    _debug_id: ClassVar[int] = 0
    """Class counter for debugging."""

    def __init__(
        self,
        unordered_elems: UnorderedElemsType,
        ordered_elems: OrderedElemsType,
        is_root: bool = False,
    ) -> None:
        self.unordered_elems = unordered_elems
        self.ordered_elems = ordered_elems
        self.routing_elems = unordered_elems + ordered_elems
        self.offset: list[int] = []  # TODO Change a name
        self.n_core_required: int = 0
        """The actual number of cores required by the routing group."""
        self.n_tail_waste: int = 0
        """Waste cores at the tail of the routing group."""

        # The following variables maintain the same interface as `CoreBlock`.
        # Unordered axons
        self.axons = set(ax for elem in self.routing_elems for ax in elem.axons)
        # for elem in self.routing_elems:
        #     axons.update(elem.axons)
        # self.axons = list(axons)

        self.dest = set(d for elem in self.routing_elems for d in elem.dest)
        # for elem in self.routing_elems:
        #     dest.update(elem.dest)
        # self.dest = list(dest)

        self.assigned_coords: list[Coord] = []
        """Assigned core coordinates in the routing group"""
        self.wasted_coords: list[Coord] = []
        """Wasted core coordinates in routing group"""
        self.wasted_core_plm: dict[Coord, EmptyCorePlacement] = {}
        """Wasted core placements"""

        self.global_axons: list[SubSourceType] = []
        """Multicast axons inheritted from the parent routing group."""
        self.private_axons: list[SubSourceType] = []
        """Multicast axons valid only within this routing group."""

        # Status options
        self.is_assigned = False
        """Whether the coordinates of chip & cores are assigned."""
        self.is_root = is_root

        self.target_chip_idx: int | None = None
        """The index of the target chip for this routing group."""

        self.online = self.core_blocks[0].online
        """Whether the routing group is in online mode."""

        if is_root:
            self.init_with_multicast_axons()
            self.set_cb_ordered_ax()

        # For debugging
        self._id = RoutingGroup._debug_id
        RoutingGroup._debug_id += 1

    def set_target_chip(self) -> None:
        if not check_elem_same(
            d.target_chip_idx for cb in self.core_blocks for d in cb.dest
        ):
            raise ValueError("Cannot multicast to different target chips.")

        self.target_chip_idx = self.core_blocks[0].dest[0].target_chip_idx

    def init_with_multicast_axons(
        self, multicast_axons: list[SubSourceType] = []
    ) -> None:
        """Initialize the routing group with multicast axons."""
        self.global_axons = multicast_axons
        used_axons: set[SubSourceType] = set()

        for elem in self.routing_elems:
            for ax in elem.axons:
                if ax not in self.global_axons and ax not in self.private_axons:
                    if isinstance(elem, CoreBlock):
                        # All axons in the core blocks of the routing elements need to multicast to the
                        # whole routing group because this routing group is the only one that can access
                        # the core blocks.
                        self.private_axons.append(ax)
                    else:
                        if ax in used_axons:
                            self.private_axons.append(ax)
                        else:
                            used_axons.add(ax)

        for elem in self.iter_elem_rg():
            elem.init_with_multicast_axons(self.global_axons + self.private_axons)

    def set_cb_ordered_ax(self) -> None:
        for elem in self.routing_elems:
            if isinstance(elem, RoutingGroup):
                elem.set_cb_ordered_ax()
            else:
                # The core blocks in a routing group should reserve space for all axons that multicast
                # to the routing group.
                elem.ordered_axons = self.global_axons + self.private_axons

    def set_core_required(self) -> None:
        """Calculate the number of cores required for the routing group iteratively."""
        if not all(cb._neurons_grouped for cb in self.iter_elem_cb()):
            # TODO change the exception type
            raise ValueError(
                "All core blocks should be grouped before calculating the number of cores required."
            )

        for rg in self.iter_elem_rg():
            rg.set_core_required()

        # Record the used cores of the members, but not the actual amount.
        n_core_used = 0

        # Unordered core blocks sorted in descending order, avoiding assigning waste.
        unordered_elem = sorted(
            self.unordered_elems,
            key=lambda x: (isinstance(x, CoreBlock), -x.n_core_required),
        )
        for elem in unordered_elem:
            self.offset.append(n_core_used)
            n_core_used += elem.n_core_required

        # Ordered routing groups should be assgined first.
        for rgrp in self.ordered_elems:
            n_core_assigned = _nearest_multiple_above(n_core_used, rgrp.n_core_required)
            self.offset.append(n_core_assigned)
            n_core_used = n_core_assigned + rgrp.n_core_required

        # Routing elements need satisfy the topological order
        # The order of routing elements is updated.
        self.routing_elems = unordered_elem + self.ordered_elems

        # Due to the chip's NoC architecture, data can only be multicast to cores that are an integer power
        # of 2.
        self.n_core_required = 1 << (n_core_used - 1).bit_length()  # required actually

        # If there are ordered routing groups, the final waste is the tail waste of the last routing group,
        # otherwise it is 0.
        n_tail_waste_by_rg = (
            self.ordered_elems[-1].n_tail_waste if self.ordered_elems else 0
        )
        self.n_tail_waste = self.n_core_required - n_core_used + n_tail_waste_by_rg

    def assign_coord(
        self, chip_coord: Coord, allocated: list[Coord]
    ) -> tuple[list[Coord], list[Coord]]:
        """Assign core coordinates to the routing group."""
        cur_i = 0
        assigned_coords: list[Coord] = []
        wasted_coords: list[Coord] = []

        for elem, offset in zip(self.routing_elems, self.offset):
            if offset > cur_i:
                wasted_coords += allocated[cur_i:offset]

            cur_i = offset
            n = elem.n_core_required
            assigned, wasted = elem.assign_coord(
                chip_coord, allocated[cur_i : cur_i + n]
            )
            assigned_coords += assigned
            wasted_coords += wasted
            cur_i += n

        self.assigned_coords = assigned_coords
        self.wasted_coords = wasted_coords + allocated[cur_i:]
        self.is_assigned = True

        return self.assigned_coords, self.wasted_coords

    def optimize_routing_elems(self) -> list["RoutingGroup"]:
        if self.online:
            return [self]

        # Optimize unordered elements by recursively optimizing sub-routing groups
        optim_unordered: UnorderedElemsType = []
        for elem in self.unordered_elems:
            if isinstance(elem, RoutingGroup):
                optim_unordered += elem.optimize_routing_elems()
            else:
                optim_unordered.append(elem)

        # Optimize ordered elements by recursively optimizing sub-routing groups
        optim_ordered: OrderedElemsType = []
        for elem in self.ordered_elems:
            optim_ordered += elem.optimize_routing_elems()

        # If a sub-routing group does not use the private multicast axons, then make it independent.
        # Any core block in a routing group always uses the private multicast axons. Otherwise, this
        # core block should not be in a routing group.
        unordered_grps, remaining_unordered = self._optimize_unordered_elems(
            optim_unordered
        )
        ordered_grps, remaining_ordered = self._optimize_ordered_elems(optim_ordered)

        optim_grps: list["RoutingGroup"] = []
        if remaining_ordered or remaining_unordered:
            # The remaining routing groups inherit the axons of the current routing group.
            remaining_rgrp = RoutingGroup(remaining_unordered, remaining_ordered)
            remaining_rgrp.global_axons = self.global_axons
            remaining_rgrp.private_axons = self.private_axons
            remaining_rgrp.is_root = self.is_root
            optim_grps.append(remaining_rgrp)

        # Keep the order of combined routing groups
        ordered_optim_grp = unordered_grps + optim_grps + ordered_grps
        if self.is_root:
            for rgrp in ordered_optim_grp:
                rgrp.set_cb_ordered_ax()

        return ordered_optim_grp

    def _optimize_ordered_elems(
        self, ordered_elems: OrderedElemsType
    ) -> tuple[list["RoutingGroup"], OrderedElemsType]:
        ordered_grps: list["RoutingGroup"] = []
        remaining: OrderedElemsType = []
        remaining_inputs: set[SubSourceType] = set()

        for elem in reversed(ordered_elems):
            # One element uses the private axons of the current routing group.
            # To make sure the private axons use unique routable coord, can not be independent.
            if not set(self.private_axons).isdisjoint(elem.axons):
                remaining_inputs.update(elem.axons)
                remaining.insert(0, elem)
            # If one element's output is used by the remaining elements,
            # To satisfy the topological order, can not be independent.
            elif not remaining_inputs.isdisjoint(elem.dest):
                remaining_inputs.update(elem.axons)
                remaining.insert(0, elem)
            else:
                # When making a routing group independent, the private axons do not change.
                # because the element does not use the private axons of the current routing group.
                # so there is no difference for this elem's init_axons with multicast axons is
                # self.global_axons + self.private_axons or self.global_axons.
                elem.global_axons = self.global_axons
                elem.is_root = self.is_root
                ordered_grps.insert(0, elem)

        return ordered_grps, remaining

    def _optimize_unordered_elems(
        self, unordered_elems: UnorderedElemsType
    ) -> tuple[list["RoutingGroup"], UnorderedElemsType]:
        unordered_grps: list["RoutingGroup"] = []
        remaining: UnorderedElemsType = []

        for elem in unordered_elems:
            if isinstance(elem, CoreBlock) or not set(self.private_axons).isdisjoint(
                elem.axons
            ):
                remaining.append(elem)
            else:
                # When making a routing group independent, the private axons do not change.
                elem.global_axons = self.global_axons
                elem.is_root = self.is_root
                unordered_grps.append(elem)

        return unordered_grps, remaining

    @property
    def core_blocks(self) -> list[CoreBlock]:
        """Retrieve all core blocks within the routing group iteratively."""
        cbs = []

        for elem in self.routing_elems:
            if isinstance(elem, CoreBlock):
                cbs.append(elem)
            else:
                cbs += elem.core_blocks

        return cbs

    @classmethod
    def build(cls, merged_grp: MergedGroup, is_root: bool = False) -> "RoutingGroup":
        online_values = {n.online for n in merged_grp.nodes}
        if len(online_values) != 1:
            raise NotSupportedError(
                "Mixed online and offline nodes in a routing group is not supported."
            )
        online = online_values.pop()

        # If an input node in the merged groups is an output node of the merged groups, the node is
        # recorded and called a subordinate node.
        global_nodes = set(merged_grp.nodes)
        raw_inhi_groups: list[set[NodeType]] = []
        raw_data_groups: list[set[NodeType]] = []

        def print_nodes(nodes: set[NodeType]):
            print([node.name for node in nodes])

        def filter_sets(sets: list[set]) -> list[set]:
            """去掉子集集合，并确保任意两集合要么包含要么不相交"""
            # 先检查两两关系是否合法
            for i, a in enumerate(sets):
                for j, b in enumerate(sets):
                    if i >= j:
                        continue
                    # 若部分相交但不是子集关系
                    if not (a.issubset(b) or b.issubset(a) or a.isdisjoint(b)):
                        raise ValueError(
                            f"Can not support inhi {a} and {b} with partial overlap."
                        )

            # 去掉子集（保留最大集）
            filtered: list[set] = []
            for s in sets:
                if not any(s < other for other in sets):
                    exist = False
                    for seen in filtered:
                        if seen.issubset(s) and s.issubset(seen):
                            exist = True
                            break
                    if not exist:
                        filtered.append(s)

            return filtered

        def merge_sets(sets: list[set]) -> list[set]:
            visited: set[int] = set()

            def dfs(i: int, merged_s: set):
                for j, other in enumerate(sets):
                    if j not in visited and not sets[i].isdisjoint(other):
                        visited.add(j)
                        merged_s.update(other)
                        dfs(j, merged_s)

            merged_sets: list[set] = []
            for i, s in enumerate(sets):
                if i not in visited:
                    visited.add(i)
                    merged_s = set(s)
                    dfs(i, merged_s)
                    merged_sets.append(merged_s)

            return merged_sets

        for group in merged_grp:
            if isinstance(group, InhiGroup):
                if global_nodes == set(group.nodes):
                    continue
                raw_inhi_groups.append(set(group.nodes))
            else:
                if group.input in merged_grp.nodes:
                    raw_data_groups.append(set(group.nodes))

        processed_inhi_groups = filter_sets(raw_inhi_groups)
        processed_data_groups = merge_sets(raw_data_groups)

        final_inhi_groups: list[set[NodeType]] = []

        for inhi_group in processed_inhi_groups:
            independent = True
            for data_group in processed_data_groups:
                if not inhi_group.isdisjoint(data_group):
                    data_group.update(inhi_group)
                    independent = False
            if independent:
                final_inhi_groups.append(inhi_group)

        final_data_groups = merge_sets(processed_data_groups)

        if len(final_data_groups) == 1 and final_data_groups[0] == global_nodes:
            raise ValueError(
                f"Cannot make groups {data_group} and {final_inhi_groups} independent."
            )

        remain_nodes = global_nodes.copy()
        for group in final_data_groups:
            remain_nodes -= group

        for group in final_inhi_groups:
            remain_nodes -= group

        data_mgrps = [merged_grp.reserve_node(g) for g in final_data_groups]
        data_mgrps = merge_cycles(data_mgrps)

        inhi_mgrps = [merged_grp.reserve_node(g) for g in final_inhi_groups]

        remain_mgrp = merged_grp.reserve_node(remain_nodes)

        merged_data_grp_graph: dict[MergedGroup, list[MergedGroup]] = defaultdict(list)
        for i in range(len(data_mgrps)):
            cur_node = data_mgrps[i]
            merged_data_grp_graph[cur_node] = []
            for j in range(len(data_mgrps)):
                if j == i:
                    continue
                succ_node = data_mgrps[j]
                if not set(succ_node.inputs).isdisjoint(cur_node.nodes):
                    merged_data_grp_graph[cur_node].append(succ_node)

        data_mgrps = toposort(merged_data_grp_graph)

        # print("after toposort the result is: ")
        # for data_msgrp in data_msgrps:
        #     print(data_msgrp)

        ordered_elems: OrderedElemsType = []
        unordered_elems: UnorderedElemsType = []
        for msgrp in data_mgrps:
            if len(msgrp) > 0:
                data_rgrp = RoutingGroup.build(msgrp)
                ordered_elems.append(data_rgrp)

        for msgrp in inhi_mgrps:
            if len(msgrp.nodes) > 0:
                inhi_rgrp = RoutingGroup.build(msgrp)
                unordered_elems.append(inhi_rgrp)

        if len(remain_mgrp.nodes) > 0:
            unordered_elems.extend(build_elements(remain_mgrp, online))

        return cls(unordered_elems, ordered_elems, is_root)

    def allocate_cp(self) -> None:
        if not self.is_assigned:
            raise ValueError("coordinates are not assigned.")

        for cb in self.core_blocks:
            cb.core_plm_alloc()

        # Allocate empty core placements for the wasted coordinates.
        for coord in self.wasted_coords:
            self.wasted_core_plm[coord] = EmptyCorePlacement.build(coord, self.online)

    def get_wasted_cplm_config(self) -> CorePlmConfInChip:
        return {
            coord: core_plm.export_core_plm_config()
            for coord, core_plm in self.wasted_core_plm.items()
        }

    @property
    def chip_coord(self) -> ChipCoord:
        if not all(
            cb.chip_coord == self.core_blocks[0].chip_coord for cb in self.core_blocks
        ):
            raise RoutingError(
                "chip coordinates in the routing group are not consistent."
            )

        return self.core_blocks[0].chip_coord

    def iter_elem_rg(self) -> Generator["RoutingGroup", None, None]:
        """Return a generator of routing groups in current routing elements."""
        return _iter_rg(self.routing_elems)

    def iter_elem_cb(self) -> Generator[CoreBlock, None, None]:
        """Return a generator of core blocks in current routing elements."""
        return _iter_cb(self.routing_elems)

    def iter_nested_cb(self) -> Generator[CoreBlock, Any, None]:
        """Return a generator of core blocks in all nested routing groups."""
        for elem in self.routing_elems:
            if isinstance(elem, CoreBlock):
                yield elem
            else:
                yield from elem.iter_nested_cb()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self._id}"

    def dump(
        self, indents: int = 0, father_logger: logging.Logger | None = None
    ) -> None:
        _logger = rt_grp_log if father_logger is None else father_logger

        tabs = "\t" * indents

        _logger.debug(
            tabs
            + f"{self}(root: {self.is_root}, target_chip: {self.target_chip_idx}, {self.n_core_required} cores):"
        )
        _logger.debug(
            tabs + f"Global axons: {[str(axon) for axon in self.global_axons]}"
        )
        _logger.debug(
            tabs + f"Private axons: {[str(axon) for axon in self.private_axons]}"
        )
        _logger.debug(tabs + f"Offset: {self.offset}")

        for elem in self.routing_elems:
            elem.dump(indents + 1, father_logger=_logger)

        if indents == 0:
            _logger.debug("")

    def dump_routing_result(
        self, indents: int = 0, father_logger: logging.Logger | None = None
    ) -> None:
        _logger = rt_grp_log if father_logger is None else father_logger

        tabs = "\t" * indents
        ind1 = tabs + "\t"

        _logger.debug(
            tabs + f"{self}(root: {self.is_root}, {self.n_core_required} cores):"
        )
        _logger.debug(tabs + f"Chip coord: {self.chip_coord}")
        _logger.debug(tabs + f"Start core coord: {self._start_core_coord_repr()}")

        for elem in self.routing_elems:
            if isinstance(elem, CoreBlock):
                _logger.debug(ind1 + f"{elem.name} ({elem.n_core_required} cores):")
                _logger.debug(ind1 + f"Chip coord: {elem.chip_coord}")
                _logger.debug(
                    ind1 + f"Start core coord: {elem._start_core_coord_repr()}"
                )
            else:
                elem.dump_routing_result(indents + 1, father_logger=_logger)

        if indents == 0:
            _logger.debug("")

    def _start_core_coord_repr(self) -> str:
        return _1st_core_coord_repr(self.assigned_coords)


class RoutingManager:
    def __init__(self, chip_list: list[ChipCoord], **kwargs) -> None:
        self.chip_list: list[ChipCoord] = chip_list
        self.used_L2_clusters = self._default_used_L2_clusters()
        """Used L2 clusters in each chip. The clocks of unused L2 clusters can be turned off   \
            through the serial port to reduce power consumption.
        """
        self.n_core_total: int = 0
        self.n_core_occupied: int = 0
        self.n_core_per_chip = self._default_n_core_per_chip()

        self.routing_state_stack: deque[dict] = deque()
        self.cur_start = 0
        self.cur_child_size = HwConfig.N_CORE_MAX_INCHIP
        self.cur_child_state = [0] * len(self.chip_list)
        self.cur_end = self.cur_start + self.cur_child_size * len(self.cur_child_state)
        self.online_state = [0] * len(self.chip_list)

        self.routing_grps: list[RoutingGroup] = []
        self.succ_rgrps: dict[RoutingGroup, list[RoutingGroup]] = defaultdict(list)

    def clear(self) -> None:
        self.n_core_total = 0
        self.n_core_occupied = 0
        self.cur_start = 0
        self.cur_child_size = HwConfig.N_CORE_MAX_INCHIP
        self.cur_child_state = [0] * len(self.chip_list)
        self.cur_end = self.cur_start + self.cur_child_size * len(self.cur_child_state)
        self.online_state = [0] * len(self.chip_list)
        self._clear_n_core_per_chip()
        self._clear_used_L2_clusters()
        self.routing_grps.clear()
        self.succ_rgrps.clear()

        # Clear the cached property safely
        if hasattr(self, "ordered_rgrps"):
            del self.ordered_rgrps

    def optimize_rgrps(self, rgrps: list[RoutingGroup]) -> None:
        optimized = []
        for rg in rgrps:
            optimized.extend(rg.optimize_routing_elems())

        self.routing_grps = optimized

    def build_rg_graph(
        self, succ_core_blocks: dict[CoreBlock, list[CoreBlock]]
    ) -> None:
        """Build the successor graph of routing groups."""
        for rg in self.routing_grps:
            self.succ_rgrps[rg] = []  # Record all routing groups to keys.
            rg_succ_cb: set[CoreBlock] = set()
            # Iterate over all core blocks within the routing group.
            for cb in rg.iter_nested_cb():
                rg_succ_cb.update(succ_core_blocks[cb])

            for next_rg in self.routing_grps:
                if next_rg == rg:
                    continue

                for succ_cb in rg_succ_cb:
                    if succ_cb in next_rg.iter_nested_cb():
                        self.succ_rgrps[rg].append(next_rg)
                        break

    def check_valid(
        self,
        n_core_incoming,
        online,
    ) -> bool:
        if online and n_core_incoming > HwConfig.N_CORE_ONLINE:
            raise ResourceError(
                f"the online routing group({n_core_incoming}) exceeds the hardware limit."
            )

        if not online and n_core_incoming > HwConfig.N_CORE_MAX_INCHIP / 2:
            raise ResourceError(
                f"the offline routing group({n_core_incoming}) exceeds the hardware limit."
            )

        return True

    def pop_to_top(self):
        while len(self.routing_state_stack) > 0:
            self.stack_pop()

    def stack_push(self, child_index: int = -1):
        if child_index != -1:
            if child_index < 0 or child_index >= len(self.cur_child_state):
                raise ResourceError(
                    f"the child_index {child_index} is out of range [0, {len(self.cur_child_state)})."
                )
            elif self.cur_child_state[child_index] == 1:
                raise ResourceError(
                    f"the child with {self.cur_child_size} cores at {self.cur_start + self.cur_child_size * child_index} is already occupied."
                )
            child_index = child_index
        elif 0 in self.cur_child_state:
            child_index = self.cur_child_state.index(0)
        else:
            raise ResourceError(
                f"the all children with {self.cur_child_size} cores at {self.cur_start} are all occupied."
            )
        self.routing_state_stack.append(
            {
                "cur_start": self.cur_start,
                "cur_child_size": self.cur_child_size,
                "cur_child_state": self.cur_child_state.copy(),
                "cur_end": self.cur_end,
                "child_index": child_index,
            }
        )
        self.cur_start = self.cur_start + self.cur_child_size * child_index
        self.cur_child_size = self.cur_child_size // HwConfig.N_SUB_ROUTING_NODE
        self.cur_child_state = [0] * HwConfig.N_SUB_ROUTING_NODE
        self.cur_end = self.cur_start + self.cur_child_size * len(self.cur_child_state)

    def stack_pop(self):
        if len(self.routing_state_stack) == 0:
            raise ResourceError(
                "the number of cores required by the routing group exceeds the hardware limit."
            )
        deque_state = self.routing_state_stack.pop()
        empty_core_num = self.available_child_num * self.cur_child_size
        self.n_core_occupied += empty_core_num
        self.n_core_per_chip[self.cur_chip_index] += empty_core_num

        self.cur_start = deque_state["cur_start"]
        self.cur_child_size = deque_state["cur_child_size"]
        self.cur_child_state = deque_state["cur_child_state"]
        self.cur_end = deque_state["cur_end"]
        child_index = deque_state["child_index"]
        self.cur_child_state[child_index] = 1

    @property
    def in_online(self) -> bool:
        return (
            self.cur_start % HwConfig.N_CORE_MAX_INCHIP >= ONLINE_CORES_BASE_COORD
            and ((self.cur_end - 1) % HwConfig.N_CORE_MAX_INCHIP + 1)
            <= ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE
        )

    @property
    def cur_chip_index(self) -> int:
        return self.cur_start // HwConfig.N_CORE_MAX_INCHIP

    @property
    def max_group_size(self) -> int:
        if self.cur_child_size == HwConfig.N_CORE_MAX_INCHIP:
            if 0 in self.cur_child_state:
                return HwConfig.N_CORE_MAX_INCHIP // 2
            else:
                return 0
        elif 1 not in self.cur_child_state:
            return self.cur_child_size * len(self.cur_child_state)
        else:
            if (
                self.cur_child_state[0] == 0
                and self.cur_child_state[1] == 0
                or self.cur_child_state[2] == 0
                and self.cur_child_state[3] == 0
            ):
                return self.cur_child_size * 2
            elif 0 in self.cur_child_state:
                return self.cur_child_size
            else:
                return 0

    @property
    def available_child_num(self) -> int:
        return self.cur_child_state.count(0)

    def insert_incoming(
        self, n_core_incoming: int, target_chip_idx: int, online: bool
    ) -> tuple[int, int, list[Direction]]:
        if self.max_group_size < n_core_incoming:
            # cur level cannot hold the incoming group, try previous level
            self.stack_pop()
            return self.try_get_insert_location(
                n_core_incoming, target_chip_idx, online
            )

        elif self.cur_child_size > n_core_incoming:
            # the child level can hold the incoming group, go deeper
            self.stack_push()
            return self.try_get_insert_location(
                n_core_incoming, target_chip_idx, online
            )

        elif (
            self.cur_child_size == n_core_incoming
            or self.cur_child_size * 2 == n_core_incoming
            or self.cur_child_size * 4 == n_core_incoming
        ):
            if self.cur_child_size == n_core_incoming:
                child_index = self.cur_child_state.index(0)
            elif self.cur_child_size * 2 == n_core_incoming:
                if self.cur_child_state[0] == 0 and self.cur_child_state[1] == 0:
                    child_index = 0
                elif self.cur_child_state[2] == 0 and self.cur_child_state[3] == 0:
                    child_index = 2
                else:
                    raise ResourceError(
                        f"the maximum incoming group size {self.max_group_size} is not correct."
                    )
            elif self.cur_child_size * 4 == n_core_incoming:
                if self.cur_child_state == [0, 0, 0, 0]:
                    child_index = 0
                else:
                    raise ResourceError(
                        f"the maximum incoming group size {self.max_group_size} is not correct."
                    )

            core_loc = self.cur_start + self.cur_child_size * child_index
            core_start = core_loc % HwConfig.N_CORE_MAX_INCHIP
            core_end = core_start + n_core_incoming

            if online:
                if not (
                    core_start >= ONLINE_CORES_BASE_COORD
                    and core_end <= ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE
                ):
                    self.stack_pop()
                    return self.try_get_insert_location(
                        n_core_incoming, target_chip_idx, online
                    )
            else:
                overlap_online = (
                    core_end > ONLINE_CORES_BASE_COORD
                    and core_start < ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE
                )
                if overlap_online:
                    self.stack_pop()
                    return self.try_get_insert_location(
                        n_core_incoming, target_chip_idx, online
                    )

            if (
                target_chip_idx != NEU_TARGET_CHIP_UNSET
                and target_chip_idx != self.cur_chip_index
            ):
                raise ResourceError(
                    f"the target chip {target_chip_idx} is not the current chip {self.cur_chip_index}."
                )

            self.n_core_occupied += n_core_incoming
            self.n_core_per_chip[self.cur_chip_index] += n_core_incoming

            occuried_child_num = n_core_incoming // self.cur_child_size
            for i in range(occuried_child_num):
                self.cur_child_state[child_index + i] = 1

            routing_idx = core_loc % HwConfig.N_CORE_MAX_INCHIP
            # From L0 to L4
            routing_path = []
            for _ in range(MAX_ROUTING_PATH_LENGTH):
                routing_idx, re = divmod(routing_idx, HwConfig.N_SUB_ROUTING_NODE)
                routing_path.append(DIREC_IDX[re])

            return core_loc, self.cur_chip_index, routing_path
        else:
            raise ResourceError(
                f"incoming group size {n_core_incoming} is not supported"
            )

    def move_to_chip(self, target_chip_idx: int):
        if target_chip_idx == NEU_TARGET_CHIP_UNSET:
            return
        elif target_chip_idx < 0 or target_chip_idx >= len(self.chip_list):
            raise ResourceError(
                f"the target chip {target_chip_idx} is out of range [0, {len(self.chip_list)})."
            )
        elif self.cur_chip_index == target_chip_idx:
            return
        else:
            self.pop_to_top()
            self.stack_push(target_chip_idx)
            return

    def insert_online(
        self, n_core_incoming: int, target_chip_idx: int = NEU_TARGET_CHIP_UNSET
    ) -> tuple[int, int, list[Direction]]:
        self.move_to_chip(target_chip_idx)

        # not in online area, try to move to online area
        if not self.in_online:
            cur_chip_index = self.cur_start // HwConfig.N_CORE_MAX_INCHIP
            if self.online_state[cur_chip_index] == 1:
                # the online cores in this chip are already occupied, try other chip
                self.pop_to_top()
                self.stack_push()
                return self.insert_online(n_core_incoming, target_chip_idx)
            else:
                # the online cores in this chip are available, move stack until online cores
                while not self.in_online:
                    cur_end_in_chip = (
                        self.cur_end - 1
                    ) % HwConfig.N_CORE_MAX_INCHIP + 1
                    if cur_end_in_chip <= ONLINE_CORES_BASE_COORD:
                        self.stack_pop()
                    else:
                        online_child_index = (
                            ONLINE_CORES_BASE_COORD
                            - self.cur_start % HwConfig.N_CORE_MAX_INCHIP
                        ) // self.cur_child_size
                        self.stack_push(online_child_index)
                return self.insert_online(n_core_incoming, target_chip_idx)

        else:
            self.online_state[self.cur_start // HwConfig.N_CORE_MAX_INCHIP] = 1
            return self.insert_incoming(n_core_incoming, target_chip_idx, True)

    def insert_offline(
        self, n_core_incoming: int, target_chip_idx: int = NEU_TARGET_CHIP_UNSET
    ) -> tuple[int, int, list[Direction]]:
        self.move_to_chip(target_chip_idx)
        while self.in_online:
            self.stack_pop()
        return self.insert_incoming(n_core_incoming, target_chip_idx, False)

    def try_get_insert_location(
        self,
        n_core_incoming: int,
        target_chip_idx: int = NEU_TARGET_CHIP_UNSET,
        online: bool = False,
    ) -> tuple[int, int, list[Direction]]:
        self.check_valid(n_core_incoming, online)
        if online:
            return self.insert_online(n_core_incoming, target_chip_idx)
        else:
            return self.insert_offline(n_core_incoming, target_chip_idx)

    def get_insert_location(
        self,
        n_core_incoming: int,
        n_core_wasted: int,
        target_chip_idx: int = NEU_TARGET_CHIP_UNSET,
    ) -> tuple[int, int, list[Direction]]:
        """Look for the insertion location for the incoming routing group.

        Args:
            n_core_incoming: #N of cores required by the incoming routing group, including the wasted cores.    \
                It must be an integer power of 2.
            n_core_wasted: #N of wasted cores.
        """
        n_core_wasted = 0

        if n_core_incoming & (n_core_incoming - 1) != 0:
            raise ValueError(
                f"'n_core_incoming' ({n_core_incoming}) is not an integer power of 2."
            )

        n_core_aligned = _nearest_multiple_above(self.n_core_total, n_core_incoming)
        n_core_predicted = n_core_aligned + n_core_incoming
        start_core_inchip = _num_inchip(n_core_aligned)
        end_core_inchip = _num_inchip(n_core_predicted) - n_core_wasted

        # If online cores are hit, start from the first core after the online cores
        if (
            start_core_inchip <= ONLINE_CORES_BASE_COORD
            and end_core_inchip > ONLINE_CORES_BASE_COORD
        ):
            online_end_inchip = ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE
            # The first core after the online cores
            online_end = n_core_aligned - start_core_inchip + online_end_inchip
            n_core_aligned = _nearest_multiple_above(online_end, n_core_incoming)

        core_loc = n_core_aligned
        chip_idx_loc = core_loc // HwConfig.N_CORE_MAX_INCHIP
        if target_chip_idx > NEU_TARGET_CHIP_UNSET and chip_idx_loc != target_chip_idx:
            if chip_idx_loc > target_chip_idx:
                raise ResourceError(
                    f"the target chip {target_chip_idx} is not routable, "
                    f"the routing group should be placed in chip {chip_idx_loc}."
                )
            else:
                core_loc = HwConfig.N_CORE_MAX_INCHIP * target_chip_idx
                chip_idx_loc = target_chip_idx

        if chip_idx_loc >= len(self.chip_list):
            raise ResourceError(
                f"the number of required chips exceeds the limit {len(self.chip_list)} ({chip_idx_loc + 1})."
            )

        self.n_core_total = n_core_aligned + n_core_incoming
        # n_core_aligned % HWConfig.N_CORE_MAX_INCHIP == 0 means the incoming
        # routing group will be placed in a new chip.
        # n_core_aligned != 0 make sure the new chip is not the first chip.
        # In this case, set n_core_per_chip of the last chip properly.
        if n_core_aligned % HwConfig.N_CORE_MAX_INCHIP == 0 and n_core_aligned != 0:
            self.n_core_per_chip[chip_idx_loc - 1] = HwConfig.N_CORE_MAX_INCHIP
        self.n_core_per_chip[chip_idx_loc] = _num_inchip(self.n_core_total)

        routing_idx = core_loc % HwConfig.N_CORE_MAX_INCHIP
        routing_path: list[Direction] = []

        # From L0 to L4
        for _ in range(MAX_ROUTING_PATH_LENGTH):
            routing_idx, re = divmod(routing_idx, HwConfig.N_SUB_ROUTING_NODE)
            routing_path.append(DIREC_IDX[re])

        return core_loc, chip_idx_loc, routing_path

    def place_routing_group(
        self, rgrp: RoutingGroup
    ) -> tuple[list[Coord], list[Coord]]:
        """Place a routing group in the chip list. Assign each core blocks with routing coordinates &   \
            make sure they are routable.

        Returns:
            A tuple of lists of assigned and wasted coordinates.
        """
        n_core_cost = rgrp.n_core_required
        # NOTE: The online cores cannot be in the range of offline-cores-to-offline-cores multicast.
        # So set `n_tail_waste=0` so that the new offline routing group will look for a location
        # after the online cores.
        n_tail_waste = 0
        n_core_req = n_core_cost - n_tail_waste

        # Check whether a single routing group can be placed within a single core.
        # The number of offline cores that can be deployed continuously is `ONLINE_CORES_BASE_COORD`.
        if n_core_req > ONLINE_CORES_BASE_COORD:
            raise ResourceError(
                "the number of cores required by the routing group exceeds the hardware limit, "
                f"{n_core_req} > {ONLINE_CORES_BASE_COORD}."
            )

        if rgrp.target_chip_idx is None:
            raise ValueError("The 'target_chip_idx' of the routing group is not set.")

        core_insert_loc, chip_idx_loc, rpath_start = self.try_get_insert_location(
            n_core_cost, rgrp.target_chip_idx, rgrp.online
        )

        allocated_coords: list[Coord] = []
        for i, rpath in _routing_path_generator(n_core_cost, rpath_start):
            leaf_coord = RoutingCoord(*reversed(rpath))
            # Record the used L2 clusters
            if (core_insert_loc + i) % (HwConfig.N_SUB_ROUTING_NODE**Level.L2) == 0:
                L2_coord = RoutingCoord(*reversed(rpath[Level.L2 :]))
                self.used_L2_clusters[chip_idx_loc].append(L2_coord)

            allocated_coords.append(leaf_coord.to_coord())

        return rgrp.assign_coord(self.chip_list[chip_idx_loc], allocated_coords)

    def allocate_cp(self) -> None:
        """Allocate core placements for all core blocks in all routing groups."""
        for rg in self.ordered_rgrps:
            rg.allocate_cp()

    @cached_property
    def ordered_rgrps(self) -> list[RoutingGroup]:
        """Return a list of routing groups in topological order.

        NOTE: The routing group must be acyclic. Once the property is accessed, the topological order is cached &   \
            will not be recalculated. Use `del self.ordered_rgrps` to clear the cache.
        """
        return toposort(self.succ_rgrps)

    def _default_used_L2_clusters(self) -> list[list[RoutingCoord]]:
        return [[]] * len(self.chip_list)

    def _clear_used_L2_clusters(self) -> None:
        for e in self.used_L2_clusters:
            e.clear()

    def _default_n_core_per_chip(self) -> list[int]:
        return [0] * len(self.chip_list)

    def _clear_n_core_per_chip(self) -> None:
        for i in range(len(self.n_core_per_chip)):
            self.n_core_per_chip[i] = 0


def _nearest_multiple_above(a: int, x: int) -> int:
    """Return the nearest number greater than or equal to `a`, and is an integer multiple of `x`.

    For example, given a=10 & x=3, return 12.
    """
    # (above + multiple - 1) // multiple
    return math.ceil(a / x) * x


def _num_inchip(n: int) -> int:
    return (n - 1) % HwConfig.N_CORE_MAX_INCHIP + 1


def _routing_path_generator(
    n_times: int, rpath: list[Direction]
) -> Generator[tuple[int, list[Direction]], Any, None]:
    for i in range(n_times):
        yield i, rpath

        for lx in range(len(rpath)):
            if rpath[lx] == DIREC_IDX[-1]:
                rpath[lx] = DIREC_IDX[0]
            else:
                rpath[lx] = DIREC_IDX[(DIREC_IDX.index(rpath[lx]) + 1) % len(DIREC_IDX)]
                break


def _all_lx_clusters(lx: Level | int) -> list[RoutingCoord]:
    return [
        RoutingCoord(*path)
        for path in itertools.product(DIREC_IDX, repeat=MAX_ROUTING_PATH_LENGTH - lx)
    ]


def get_unused_lx(
    used_lx: list[RoutingCoord], lx: Level | int = Level.L2
) -> list[RoutingCoord]:
    all_lx = _all_lx_clusters(lx)

    for _lx in set(used_lx):  # make used_lx unduplicated
        all_lx.remove(_lx)  # keep the rest clusters in order

    return all_lx
