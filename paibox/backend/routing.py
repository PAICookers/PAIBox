import itertools
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Generator, Iterable
from functools import cached_property
from typing import Any, ClassVar, Optional, Union, cast

from paicorelib import ONLINE_CORES_BASE_COORD
from paicorelib import ROUTING_DIRECTIONS_IDX as DIREC_IDX
from paicorelib import ChipCoord, Coord, CoreMode, HwConfig, RoutingCoord
from paicorelib import RoutingDirection as Direction
from paicorelib import RoutingLevel as Level
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

from paibox import _logging
from paibox.components import Conv2d, MatMul2d
from paibox.components.neuron.base import NEU_TARGET_CHIP_NOT_SET
from paibox.exceptions import (
    NotSupportedError,
    PAIBoxDeprecationWarning,
    ResourceError,
    RoutingError,
)
from paibox.utils import check_elem_same

from ._slice import *
from .conf_types import CorePlmConfInChip
from .constrs import GraphNodeConstrs
from .graph_utils import toposort
from .placement import CoreBlock, EmptyCorePlacement
from .succ_group import MergedSuccGroup
from .types import EdgeType, NodeType, _1st_core_coord_repr

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


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


def SearchOptimalSplit(shape_in, shape_out) -> tuple[int, int]:
    # Currently, replication is done using multicast instead of replication cores, which may lead to inaccurate expected results on the actual board.
    # The impact of replication cores is temporarily not considered when estimating the number of cores.
    fin = [1152, 2304, 4068, 9216, 18432, 36864, 73728]
    cin, hin, win = shape_in[0], shape_in[1], shape_in[2]
    cout, hout, wout = shape_out[0], shape_out[1], shape_out[2]
    kh = hin - hout + 1
    kw = win - wout + 1
    min_cores = 1e9
    n_cores = 0
    split = 1
    split_start = 1
    split_stop = min(hout, wout)
    for i in range(split_start, split_stop):
        out_slice_len_h = math.ceil(hout / i)
        out_slice_len_w = math.ceil(wout / i)
        in_slice_len_h = out_slice_len_h + kh - 1
        in_slice_len_w = out_slice_len_w + kw - 1
        input_len = in_slice_len_h * in_slice_len_w
        if input_len > 9216:
            continue
        n_fin = next((n for n in fin if n >= input_len), None)
        n_fout = 73728 // n_fin
        n_cores = (
            i * i * cin * math.ceil(cout * out_slice_len_h * out_slice_len_w / n_fout)
        )
        if n_cores < min_cores:
            min_cores = n_cores
            split = i
    return split, split


def Conv2d_slices(conv2d: Conv2d) -> tuple[list[list[slice]], list[list[slice]]]:
    # Currently, the axon segment in the backend requires a contiguous block of data.
    # However, during slicing, it's impossible to fully overlap shared data without changing the total size.
    # Using multiple slices to fetch the data may be the correct approach,
    # but the changes involve many components and are complex, I haven't been able to make it work.
    # For now, axon slicing still follows the original approach due to these unresolved backend issues.

    shape_in = conv2d.shape_in
    shape_out = conv2d.shape_out

    cin = shape_in[0]
    cout = shape_out[0]

    kh = shape_in[1] - shape_out[1] + 1
    kw = shape_in[2] - shape_out[2] + 1
    h_splits, w_splits = SearchOptimalSplit(shape_in, shape_out)

    out_slice_len_h = math.ceil(shape_out[1] / h_splits)
    out_slice_len_w = math.ceil(shape_out[2] / w_splits)
    in_slice_len_h = out_slice_len_h + kh - 1
    in_slice_len_w = out_slice_len_w + kw - 1

    input_slices_c = slice(0, cin)
    input_slices_h = [
        slice(i * out_slice_len_h, (i + 1) * out_slice_len_h + kh - 1)
        for i in range(h_splits)
    ]
    input_slices_w = [
        slice(i * out_slice_len_w, (i + 1) * out_slice_len_w + kw - 1)
        for i in range(w_splits)
    ]

    output_slices_c = slice(0, cout)
    output_slices_h = [
        slice(i * out_slice_len_h, (i + 1) * out_slice_len_h) for i in range(h_splits)
    ]
    output_slices_w = [
        slice(i * out_slice_len_w, (i + 1) * out_slice_len_w) for i in range(w_splits)
    ]

    input_slices = []
    output_slices = []
    for h_slice in input_slices_h:
        for w_slice in input_slices_w:
            slice_start = cin * (
                h_slice.start * shape_in[2] + w_slice.start * in_slice_len_h
            )
            slice_stop = slice_start + cin * in_slice_len_h * in_slice_len_w
            input_slices.append(
                [input_slices_c, h_slice, w_slice, slice(slice_start, slice_stop)]
            )

    for h_slice in output_slices_h:
        for w_slice in output_slices_w:
            slice_start = cout * (
                h_slice.start * shape_out[2] + w_slice.start * out_slice_len_h
            )
            slice_stop = slice_start + cout * out_slice_len_h * out_slice_len_w
            output_slices.append(
                [output_slices_c, h_slice, w_slice, slice(slice_start, slice_stop)]
            )

    return input_slices, output_slices


def build_elements(
    merged_sgrp: MergedSuccGroup,
) -> list[Union[CoreBlock, "RoutingGroup"]]:
    nodes = list(merged_sgrp.nodes)
    elements: list[Union[CoreBlock, "RoutingGroup"]] = []

    mode = cast(CoreMode, nodes[0].mode)
    if any(mode != node.mode for node in nodes):
        raise NotSupportedError("mixed mode is not supported.")

    # Optimize weight in single operator, like 'Mat2d'.
    if len(nodes) == 1:
        edges = merged_sgrp.outputs[nodes[0]]
        # find edges with divisible weight
        divisible_edge = None
        for edge in edges:
            # only one edge is allowed to have divisible weight
            if isinstance(edge, MatMul2d):
                divisible_edge = edge
                break
            if isinstance(edge, Conv2d):
                divisible_edge = edge
                break
        # TODO we can judge whether optimization is needed here
        if divisible_edge is None:
            edge_slices = [EdgeSlice(edge) for edge in edges]
            elements.append(CoreBlock.build(*edge_slices, rt_mode=mode))

        else:
            if isinstance(divisible_edge, MatMul2d):
                input_slices, output_slices = MatMul2d_slices(divisible_edge)
            else:
                input_slices, output_slices = Conv2d_slices(divisible_edge)

            for input_slice, output_slice in zip(input_slices, output_slices):
                edge_slices: list[EdgeSlice] = []
                for edge in edges:
                    if edge == divisible_edge:
                        edge_slices.append(EdgeSlice(edge, input_slice, output_slice))
                    else:
                        edge_slices.append(EdgeSlice(edge, None, output_slice))
                core_block = CoreBlock.build(*edge_slices, rt_mode=mode)
                routing_group = RoutingGroup([core_block], [])
                elements.append(routing_group)
    else:
        # TODO More constraints for nodes can be called here.
        # TODO weight can be optimized between operators.
        idx_in_cbs = GraphNodeConstrs.apply_constrs(nodes)
        # if len(idx_of_sg) == 0:
        #     idx_of_sg = [list(range(len(nodes)))]

        for idx_in_cb in idx_in_cbs:
            edges_set: set[EdgeType] = set()

            for i in idx_in_cb:
                edges_set.update(merged_sgrp.outputs[nodes[i]])

            edge_slices = [EdgeSlice(edge) for edge in edges_set]
            core_block = CoreBlock.build(*edge_slices, rt_mode=mode)
            elements.append(core_block)

    return elements


RoutingElemType = Union[CoreBlock, "RoutingGroup"]
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

        self.global_axons: list[SourceSliceType] = []
        """Multicast axons inheritted from the parent routing group."""
        self.private_axons: list[SourceSliceType] = []
        """Multicast axons valid only within this routing group."""

        # Status options
        self.is_assigned = False
        """Whether the coordinates of chip & cores are assigned."""
        self.is_root = is_root

        self.target_chip_idx: Union[int, None] = None
        """The index of the target chip for this routing group."""

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
        self, multicast_axons: list[SourceSliceType] = []
    ) -> None:
        """Initialize the routing group with multicast axons."""
        self.global_axons = multicast_axons
        used_axons: set[SourceSliceType] = set()

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

        # This is the amount of cores required actually.
        assert n_core_used > 0
        self.n_core_required = 1 << (n_core_used - 1).bit_length()

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
        remaining_inputs: set[SourceSliceType] = set()

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
    def build(
        cls, merged_sgrp: MergedSuccGroup, is_root: bool = False
    ) -> "RoutingGroup":
        msgrp = MergedSuccGroup()
        remaining = MergedSuccGroup()
        sub_nodes: set[NodeType] = set()

        # If an input node in the merged groups is an output node of the merged groups, the node is
        # recorded and called a subordinate node.
        for group in merged_sgrp:
            if group.input in merged_sgrp.nodes:
                sub_nodes.update(group.nodes)

        remaining_nodes = set(merged_sgrp.nodes) - sub_nodes

        for group in merged_sgrp:
            if not sub_nodes.isdisjoint(group.nodes):
                msgrp.add_group(group)
            if not remaining_nodes.isdisjoint(group.nodes):
                remaining.add_group(group)

        # remaining.nodes &= remaining_nodes
        for node in remaining.nodes - remaining_nodes:
            remaining.remove_node(node)

        # msgrp.nodes &= sub_nodes
        for node in msgrp.nodes - sub_nodes:
            msgrp.remove_node(node)

        # Build the subordinate routing groups if there are any subordinate nodes.
        if len(msgrp.nodes) > 0:
            sub_rgrp = RoutingGroup.build(msgrp)
            ordered_rgrp = [sub_rgrp]
        else:
            ordered_rgrp = []

        unordered_elems = build_elements(remaining)

        return cls(unordered_elems, ordered_rgrp, is_root)

    def allocate_cp(self) -> None:
        if not self.is_assigned:
            raise ValueError("coordinates are not assigned.")

        for cb in self.core_blocks:
            cb.core_plm_alloc()

        # Allocate empty core placements for the wasted coordinates.
        for coord in self.wasted_coords:
            self.wasted_core_plm[coord] = EmptyCorePlacement.build(coord)

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
        self, indents: int = 0, father_logger: Optional[logging.Logger] = None
    ) -> None:
        _logger = rt_grp_log if father_logger is None else father_logger

        tabs = "\t" * indents

        _logger.debug(
            tabs + f"{self}(root: {self.is_root}, {self.n_core_required} cores):"
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

    def dump_routing_result(
        self, indents: int = 0, father_logger: Optional[logging.Logger] = None
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
        self.n_core_per_chip = self._default_n_core_per_chip()

        self.routing_grps: list[RoutingGroup] = []
        self.succ_rgrps: dict[RoutingGroup, list[RoutingGroup]] = defaultdict(list)

    def clear(self) -> None:
        self.n_core_total = 0
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

    def get_insert_location(
        self,
        n_core_incoming: int,
        n_core_wasted: int,
        target_chip_idx: int = NEU_TARGET_CHIP_NOT_SET,
    ) -> tuple[int, int, list[Direction]]:
        """Look for the insertion location for the incoming routing group.

        Args:
            n_core_incoming: #N of cores required by the incoming routing group, including the wasted cores.    \
                It must be an integer power of 2.
            n_core_wasted: #N of wasted cores.
        """
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
        if (
            target_chip_idx > NEU_TARGET_CHIP_NOT_SET
            and chip_idx_loc != target_chip_idx
        ):
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
                f"the number of required chips exceeds the limit {len(self.chip_list)} ({chip_idx_loc+1})."
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

        core_insert_loc, chip_idx_loc, rpath_start = self.get_insert_location(
            n_core_cost, n_tail_waste, rgrp.target_chip_idx
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

    def get_n_core_occupied(self) -> int:
        """Get the #N of cores occupied by all routing groups. Online cores are not counted."""
        n_chip_full_used, remaining = divmod(
            self.n_core_total, HwConfig.N_CORE_MAX_INCHIP
        )
        occupied = n_chip_full_used * HwConfig.N_CORE_OFFLINE

        if remaining <= ONLINE_CORES_BASE_COORD:
            occupied += remaining
        elif remaining <= ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE:
            # When the wasted cores of the last routing group in the chip overlap with the online cores.
            occupied += ONLINE_CORES_BASE_COORD
        else:
            # Online cores were all counted incorrectly.
            occupied += remaining - HwConfig.N_CORE_ONLINE

        return occupied

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


@deprecated(
    "'RoutingRoot' is deprecated in version 1.2.0 and will be "
    "removed in version 1.3.0. Use `RoutingManager` instead.",
    category=PAIBoxDeprecationWarning,
)
class RoutingRoot(RoutingManager):
    pass


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


def _all_lx_clusters(lx: Union[Level, int]) -> list[RoutingCoord]:
    return [
        RoutingCoord(*path)
        for path in itertools.product(DIREC_IDX, repeat=MAX_ROUTING_PATH_LENGTH - lx)
    ]


def get_unused_lx(
    used_lx: list[RoutingCoord], lx: Union[Level, int] = Level.L2
) -> list[RoutingCoord]:
    all_lx = _all_lx_clusters(lx)

    for l in set(used_lx):  # make used_lx unduplicated
        all_lx.remove(l)  # keep the rest clusters in order

    return all_lx
