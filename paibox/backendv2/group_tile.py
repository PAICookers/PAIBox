from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
from paicorelib import LCN_EX

from paibox.paiir.nn.pool import SumPool1d, SumPool2d

from ..paiir.ir.add_ops import PotentialAddOp
from ..paiir.ir.op_node import AccumulateOp, StandaloneActOp
from .op_node import (
    CoreOpNode,
    CustomIndex,
    Neuron,
    SourceElem,
    SourceNode,
    get_elem,
)
from .routing import (
    FANIN_BASE,
    InputGroup,
    OutputGroup,
    RemapGroup,
    RoutingGroup,
)

MAX_LCN = LCN_EX.LCN_128X

Conv = torch.nn.Conv1d | torch.nn.Conv2d
Pool = SumPool1d | SumPool2d | torch.nn.MaxPool1d | torch.nn.MaxPool2d
TileComp = Conv | Pool
Comp_1D = torch.nn.Conv1d | SumPool1d | torch.nn.MaxPool1d
Comp_2D = torch.nn.Conv2d | SumPool2d | torch.nn.MaxPool2d
Range1D = tuple[int, int]
TILE_COMP_TYPES = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    SumPool1d,
    SumPool2d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
)
COMP_1D_TYPES = (torch.nn.Conv1d, SumPool1d, torch.nn.MaxPool1d)
COMP_2D_TYPES = (torch.nn.Conv2d, SumPool2d, torch.nn.MaxPool2d)

POTENTIAL_OP_TYPES = (PotentialAddOp, StandaloneActOp, AccumulateOp)


@dataclass(frozen=True)
class AxisTilePlan:
    axis: Literal["h", "w"]
    in_len: int
    max_axis_len: int
    axis_in_ranges: list[Range1D]
    axis_out_ranges: list[Range1D]
    overlap_elems: int


def to_nd_tuple(value: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        if len(value) != ndim:
            raise ValueError(f"Expected a tuple of length {ndim}, but got {value}.")
        return value

    return (value,) * ndim


def build_tile_group(
    in_shape: tuple[int, int, int, int],  # batch, in_channel, height, width
    out_shape: tuple[int, int, int, int],  # batch, out_channel, height, width
    in_node: SourceNode,
    out_node: CoreOpNode,
    kernel_size: int,
    padding: int,
    stride: int,
) -> tuple[list[SourceElem], list[RoutingGroup]]:
    input_bit_num = in_node.output_bit_num
    _, c_in, h_in, w_in = in_shape
    _, c_out, h_out, w_out = out_shape

    n_overlap = kernel_size - stride
    assert (
        n_overlap >= 0
    ), "Stride must be less than or equal to kernel size for tiling."

    print(
        f"parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}, input_bit_num={input_bit_num}"
    )

    max_fanin = FANIN_BASE * (2**MAX_LCN.value)

    def build_tile_ranges_along_axis(
        axis_name: Literal["h", "w"],
        in_len: int,
        out_len: int,
        max_axis_len: int,
    ) -> tuple[list[Range1D], list[Range1D]]:
        if max_axis_len <= 0:
            raise ValueError(
                f"Cannot tile along {axis_name}: max input {axis_name} is non-positive ({max_axis_len})."
            )
        axis_in_ranges: list[Range1D] = []
        axis_out_ranges: list[Range1D] = []
        cur_out_start = 0
        while cur_out_start < out_len:
            in_start = cur_out_start * stride - padding
            cur_in_start = max(in_start, 0)
            cur_in_end = min(in_start + kernel_size, in_len)

            if cur_in_start >= cur_in_end:
                raise ValueError(
                    f"Output index {cur_out_start} on {axis_name} axis does not consume any input. "
                    f"padding={padding} is too large for input {axis_name}={in_len}."
                )
            if cur_in_end - cur_in_start > max_axis_len:
                raise ValueError(
                    f"Single output receptive field exceeds max input {axis_name} for tiling."
                )

            cur_out_end = cur_out_start + 1
            while cur_out_end < out_len:
                next_in_start = cur_out_end * stride - padding
                next_in_valid_start = max(next_in_start, 0)
                next_in_valid_end = min(next_in_start + kernel_size, in_len)
                if next_in_valid_start >= next_in_valid_end:
                    raise ValueError(
                        f"Output index {cur_out_end} on {axis_name} axis does not consume any input. "
                        f"padding={padding} is too large for input {axis_name}={in_len}."
                    )
                if next_in_valid_end - cur_in_start > max_axis_len:
                    break
                cur_in_end = next_in_valid_end
                cur_out_end += 1
            # print(
            #     f"Tile {axis_name} output range: ({cur_out_start}, {cur_out_end}), "
            #     f"input range: ({cur_in_start}, {cur_in_end})"
            # )
            axis_out_ranges.append((cur_out_start, cur_out_end))
            axis_in_ranges.append((cur_in_start, cur_in_end))
            cur_out_start = cur_out_end

        return axis_in_ranges, axis_out_ranges

    axis_plans: list[AxisTilePlan] = []
    axis_errors: list[str] = []
    axis_specs: tuple[tuple[Literal["h", "w"], int, int, int], ...] = (
        ("h", h_in, h_out, w_in),
        ("w", w_in, w_out, h_in),
    )
    for axis_name, in_len, out_len, other_len in axis_specs:
        max_axis_len = max_fanin // (input_bit_num * c_in * other_len)
        try:
            axis_in_ranges, axis_out_ranges = build_tile_ranges_along_axis(
                axis_name, in_len, out_len, max_axis_len
            )
        except ValueError as e:
            axis_errors.append(str(e))
            continue

        overlap_axis = sum(end - start for start, end in axis_in_ranges) - in_len
        overlap_elems = overlap_axis * other_len * c_in
        axis_plans.append(
            AxisTilePlan(
                axis=axis_name,
                in_len=in_len,
                max_axis_len=max_axis_len,
                axis_in_ranges=axis_in_ranges,
                axis_out_ranges=axis_out_ranges,
                overlap_elems=overlap_elems,
            )
        )

    if not axis_plans:
        raise ValueError("Cannot tile on either h/w axis. " + " | ".join(axis_errors))

    axis_plans.sort(
        key=lambda p: (
            -p.in_len,
            p.overlap_elems,
            len(p.axis_in_ranges),
        )
    )
    selected_plan = axis_plans[0]
    tile_axis = selected_plan.axis
    max_axis_len = selected_plan.max_axis_len
    print(f"Selected tile axis: {tile_axis}, max input length on axis: {max_axis_len}")

    if tile_axis == "h":
        tile_h_in_ranges = list(selected_plan.axis_in_ranges)
        tile_h_out_ranges = list(selected_plan.axis_out_ranges)
        tile_w_in_ranges = [(0, w_in)] * len(tile_h_in_ranges)
        tile_w_out_ranges = [(0, w_out)] * len(tile_h_out_ranges)
    else:
        tile_w_in_ranges = list(selected_plan.axis_in_ranges)
        tile_w_out_ranges = list(selected_plan.axis_out_ranges)
        tile_h_in_ranges = [(0, h_in)] * len(tile_w_in_ranges)
        tile_h_out_ranges = [(0, h_out)] * len(tile_w_out_ranges)

    n_tiles = len(tile_h_in_ranges)
    for t in range(n_tiles):
        print(
            f"\tTile {t}: in {selected_plan.axis_in_ranges[t]}, out {selected_plan.axis_out_ranges[t]}"
        )

    tile_input_lists: list[list[SourceElem]] = list()
    tile_output_lists: list[list[Neuron]] = list()

    copied_input_elems: list[SourceElem] = list()
    tiled_groups: list[RoutingGroup] = list()
    last_in_end: int | None = None
    for t in range(n_tiles):
        h_in_start, h_in_end = tile_h_in_ranges[t]
        w_in_start, w_in_end = tile_w_in_ranges[t]
        tile_input_lists.append([])
        for c in range(c_in):
            for h in range(h_in_start, h_in_end):
                for w in range(w_in_start, w_in_end):
                    raw_idx = c * h_in * w_in + h * w_in + w
                    copy_id = 0
                    if tile_axis == "h" and last_in_end is not None and h < last_in_end:
                        copy_id = 1
                    if tile_axis == "w" and last_in_end is not None and w < last_in_end:
                        copy_id = 1
                    elem = get_elem(in_node, raw_idx, copy_id)
                    tile_input_lists[t].append(elem)
                    if copy_id == 1:
                        copied_input_elems.append(elem)
        last_in_end = h_in_end if tile_axis == "h" else w_in_end

        h_out_start, h_out_end = tile_h_out_ranges[t]
        w_out_start, w_out_end = tile_w_out_ranges[t]
        tile_output_lists.append([])
        for c in range(c_out):
            for h in range(h_out_start, h_out_end):
                for w in range(w_out_start, w_out_end):
                    raw_idx = c * h_out * w_out + h * w_out + w
                    tile_output_lists[t].append(Neuron(out_node, CustomIndex(raw_idx)))
        tiled_group = RoutingGroup(
            raw_neus=tile_output_lists[t],
            input_list=tile_input_lists[t],
        )
        tiled_group.recommand_lcn = (
            MAX_LCN  # recommend using max lcn for tiled groups to avoid further tiling
        )
        tiled_groups.append(tiled_group)
    return copied_input_elems, tiled_groups


def try_tile_conv(
    in_node: SourceNode,
    out_node: CoreOpNode,
) -> tuple[list[SourceElem], list[RoutingGroup]]:
    if len(out_node.comps) != 1:
        raise ValueError(
            "Only groups with one output component are supported for tiling."
        )

    comp = out_node.comps[0]
    print(f"try to tile group {in_node} ->({type(comp).__name__}) {out_node}")
    assert isinstance(
        comp, TILE_COMP_TYPES
    ), "Only groups with Conv1d/Conv2d/Pool components are supported for tiling."
    assert not isinstance(
        comp.padding, str
    ), "String padding is not supported for tiling."

    if isinstance(comp, COMP_2D_TYPES):
        _kernel_size = to_nd_tuple(comp.kernel_size, 2)
        _stride = to_nd_tuple(comp.stride, 2)
        _padding = to_nd_tuple(comp.padding, 2)
        b, c_in, h_in, w_in = in_node.shape
        b, c_out, h_out, w_out = out_node.shape
        print(
            f"{type(comp).__name__} input shape: {in_node.shape}, output shape: {out_node.shape}"
        )
        if _stride[0] == _stride[1]:
            stride = _stride[0]
        else:
            raise ValueError("Asymmetric stride is not supported for tiling.")
        if _padding[0] == _padding[1]:
            padding = _padding[0]
        else:
            raise ValueError("Asymmetric padding is not supported for tiling.")

        if _kernel_size[0] == _kernel_size[1]:
            kernel_size = _kernel_size[0]
        else:
            raise ValueError("Only square kernels are supported for tiling.")

        in_shape = (b, c_in, h_in, w_in)
        out_shape = (b, c_out, h_out, w_out)
        copied_input_elems, tiled_groups = build_tile_group(
            in_shape, out_shape, in_node, out_node, kernel_size, padding, stride
        )
    elif isinstance(comp, COMP_1D_TYPES):
        _kernel_size = to_nd_tuple(comp.kernel_size, 1)
        _stride = to_nd_tuple(comp.stride, 1)
        _padding = to_nd_tuple(comp.padding, 1)
        b, c_in, h_in = in_node.shape
        b, c_out, h_out = out_node.shape
        w_in = 1
        w_out = 1
        print(
            f"{type(comp).__name__} input shape: {in_node.shape}, output shape: {out_node.shape}"
        )
        stride = _stride[0]
        padding = _padding[0]
        kernel_size = _kernel_size[0]
        in_shape = (b, c_in, h_in, w_in)
        out_shape = (b, c_out, h_out, w_out)
        copied_input_elems, tiled_groups = build_tile_group(
            in_shape, out_shape, in_node, out_node, kernel_size, padding, stride
        )
    else:
        raise NotImplementedError(
            f"Unsupported component type for tiling: {type(comp)}"
        )
    return copied_input_elems, tiled_groups


def try_tile_potential(
    in_nodes: list[SourceNode],
    out_nodes: list[CoreOpNode],
) -> tuple[list[SourceElem], list[RoutingGroup]]:

    op_numel = out_nodes[0].shape.numel()
    input_bit_num = in_nodes[0].output_bit_num
    if not all(out_node.shape.numel() == op_numel for out_node in out_nodes):
        raise ValueError(
            "All output nodes must have the same number of elements for potential tiling."
        )
    if not all(in_node.shape == out_nodes[0].shape for in_node in in_nodes):
        raise ValueError(
            "Input nodes and output nodes must have the same shape for potential tiling."
        )
    if not all(in_node.output_bit_num == input_bit_num for in_node in in_nodes):
        raise ValueError(
            "All input nodes must have the same output bit num for potential tiling."
        )
    tile_groups: list[RoutingGroup] = []
    max_fanin = FANIN_BASE * (2**MAX_LCN.value)
    max_single_op_numel = max_fanin // (input_bit_num * len(in_nodes))
    for start in range(0, op_numel, max_single_op_numel):
        end = min(start + max_single_op_numel, op_numel)
        tile_input_list = []
        for in_node in in_nodes:
            for idx in range(start, end):
                tile_input_list.append(get_elem(in_node, idx))
        tile_output_list = []
        for out_node in out_nodes:
            for idx in range(start, end):
                tile_output_list.append(Neuron(out_node, CustomIndex(idx)))
        tiled_group = RoutingGroup(
            raw_neus=tile_output_list,
            input_list=tile_input_list,
        )
        tile_groups.append(tiled_group)
    return [], tile_groups


def try_tile_group(
    origin_grp: RoutingGroup,
) -> tuple[list[SourceElem], list[RoutingGroup]]:
    print(f"Trying to tile group out lcn limit {origin_grp.name}...")
    print(f"input nodes {origin_grp.input_nodes} and output nodes {origin_grp.nodes}")
    if origin_grp.input_nodes is None or origin_grp.nodes is None:
        raise ValueError(
            "Input nodes and output nodes of the group must be built before tiling."
        )
    if len(origin_grp.input_nodes) != len(origin_grp.nodes):
        raise ValueError(
            "Only groups with same number of input and output nodes are supported for tiling."
        )

    out_nodes = list(origin_grp.nodes)
    in_nodes = list(origin_grp.input_nodes)

    if len(out_nodes) == 1 and isinstance(out_nodes[0].comps[0], TileComp):
        print("Trying to tile convolution/pooling group...")
        copied_input_elems, tiled_groups = try_tile_conv(in_nodes[0], out_nodes[0])
    elif all(
        isinstance(out_node.raw_node, POTENTIAL_OP_TYPES) for out_node in out_nodes
    ):
        print("Trying to tile potential add group...")
        copied_input_elems, tiled_groups = try_tile_potential(in_nodes, out_nodes)
    else:
        print([isinstance(out_node.comps[0], TileComp) for out_node in out_nodes])
        print(
            [
                isinstance(out_node.raw_node, POTENTIAL_OP_TYPES)
                for out_node in out_nodes
            ]
        )
        raise ValueError(
            "Only groups with one convolution/pooling output node or groups with potential add/standalone act output nodes are supported for tiling."
        )
    print(f"Copied input elements: {len(copied_input_elems)}")
    print(f"Original input elements: {len(origin_grp.input_list)}")
    print(f"Tiled groups (num = {len(tiled_groups)}):")
    for grp in tiled_groups:
        print(f"{grp.info('   ')}")
    return copied_input_elems, tiled_groups


def tile_groups(
    groups: list[RoutingGroup | InputGroup | OutputGroup | RemapGroup],
) -> list[RoutingGroup | InputGroup | OutputGroup | RemapGroup]:
    group_after_tile: list[RoutingGroup | InputGroup | OutputGroup | RemapGroup] = []
    copy_elems: list[SourceElem] = []
    for grp in groups:
        if isinstance(grp, RoutingGroup):
            intput_bit_nums: set[int] = set(
                [neu.input_bit_num for neu in grp.raw_elems]
            )
            pred_output_bit_nums: set[int] = set(
                [src.output_bit_num for src in grp.input_list]
            )
            assert (
                len(intput_bit_nums) == 1
            ), "All neurons in the routing group must have the same input bit num."
            assert (
                len(pred_output_bit_nums) == 1
            ), "All input elements in the routing group must have the same output bit num."
            input_bit_num = intput_bit_nums.pop()
            assert (
                input_bit_num == pred_output_bit_nums.pop()
            ), "Input bit num of neurons must match output bit num of input elements."
            max_axon_addr = len(grp.input_list) * input_bit_num
            lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
            if lcn > MAX_LCN.value:
                copied_elems, tile_conv_groups = try_tile_group(grp)
                copy_elems.extend(copied_elems)
                group_after_tile.extend(tile_conv_groups)
            else:
                group_after_tile.append(grp)
        else:
            group_after_tile.append(grp)

    print(f"Total copied elements after tiling: {len(copy_elems)}")

    copy_elem_deque = deque(copy_elems)

    while copy_elem_deque:
        copy_elem = copy_elem_deque.popleft()
        added = False
        for grp in group_after_tile:
            if isinstance(grp, OutputGroup):
                continue
            raw_elem = copy_elem.origin_elem()
            if raw_elem in grp.elem_set:
                new_elem = grp.add_elem(copy_elem)
                if new_elem is not None:
                    copy_elem_deque.append(new_elem)
                    copy_elems.append(new_elem)
                added = True
                break

        if not added:
            raise ValueError(
                f"Copied element {copy_elem} cannot be added to any group."
            )

    return group_after_tile
