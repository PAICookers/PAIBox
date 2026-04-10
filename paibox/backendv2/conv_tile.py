from __future__ import annotations

from collections import deque

import torch
from paicorelib import LCN_EX

from .op_node import (
    AllNode,
    CustomIndex,
    InputElem,
    Neuron,
    RemapElem,
    SourceElem,
    build_nodes,
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


def build_tiled_conv(
    conv_grp: RoutingGroup,
) -> tuple[list[SourceElem], list[RoutingGroup]]:
    if conv_grp.input_nodes is None or conv_grp.nodes is None:
        raise ValueError(
            "Input nodes and output nodes of the convolution group must be built before tiling."
        )
    if len(conv_grp.input_nodes) != 1 or len(conv_grp.nodes) != 1:
        raise ValueError(
            "Only convolution groups with one input node and one output node are supported for tiling."
        )
    conv_input_node = next(iter(conv_grp.input_nodes))
    conv_output_node = next(iter(conv_grp.nodes))
    conv_input_bit_num = conv_input_node.output_bit_num
    input_len = len(conv_grp.input_list)

    if len(conv_output_node.comps) != 1:
        raise ValueError(
            "Only convolution groups with one output component are supported for tiling."
        )
    comp = conv_output_node.comps[0]
    assert isinstance(
        comp, torch.nn.Conv2d
    ), "Only convolution groups with Conv2d component are supported for tiling."
    b, c_in, h_in, w_in = conv_input_node.shape
    b, c_out, h_out, w_out = conv_output_node.shape
    print(
        f"Conv2d input shape: {conv_input_node.shape}, output shape: {conv_output_node.shape}"
    )
    _kernel_size = comp.kernel_size
    _stride = comp.stride
    _padding = comp.padding
    _dilation = comp.dilation

    assert not isinstance(_padding, str), "String padding is not supported for tiling."

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

    n_overlap = kernel_size - stride

    assert (
        n_overlap >= 0
    ), "Stride must be less than or equal to kernel size for tiling."

    print(
        f"Conv2d parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}"
    )

    max_h_in = FANIN_BASE * (2**MAX_LCN.value) // (conv_input_bit_num * c_in * w_in)
    print(f"Max input height for tiling: {max_h_in}")

    n_tiles = (h_in - n_overlap + max_h_in - n_overlap - 1) // (
        max_h_in - n_overlap
    )  # -1 for overlapping rows, -1 for ceiling division
    print(f"Number of tiles needed: {n_tiles}")

    avg_h_in = (
        h_in + (n_tiles - 1) * n_overlap + n_tiles - 1
    ) // n_tiles  # +n_tiles-1 for ceiling division, +n_tiles-1 for overlapping rows
    print(f"Average input height per tile: {avg_h_in}")

    assert stride == 1, "Only stride 1 is supported for tiling."

    tile_h_in_ranges: list[tuple[int, int]] = []
    tile_h_out_ranges: list[tuple[int, int]] = []
    cur_out_start = 0
    cur_in_start = 0
    for i in range(h_out):
        h_in_start = i * stride - padding
        h_in_end = min(h_in_start + kernel_size, h_in)
        if h_in_end - cur_in_start >= avg_h_in:
            assert (
                h_in_end - cur_in_start <= max_h_in
            ), "Calculated tile input height exceeds max input height."
            if h_in_end != h_in:
                tile_h_out_ranges.append((cur_out_start, i + 1))
                cur_out_start = i + 1
                tile_h_in_ranges.append((cur_in_start, h_in_end))
                cur_in_start = (i + 1) * stride - padding
        if i == h_out - 1:
            assert (
                h_in - cur_in_start <= max_h_in
            ), "Last tile input height exceeds max input height."
            tile_h_out_ranges.append((cur_out_start, h_out))
            tile_h_in_ranges.append((cur_in_start, h_in))
    for t in range(n_tiles):
        print(
            f"Tile {t}: input height range: {tile_h_in_ranges[t]}, output height range: {tile_h_out_ranges[t]}"
        )

    tile_input_lists: list[list[SourceElem]] = list()
    tile_output_lists: list[list[Neuron]] = list()

    copied_input_elems: list[SourceElem] = list()
    tiled_groups: list[RoutingGroup] = list()
    last_in_end = None
    for t in range(n_tiles):
        h_in_start, h_in_end = tile_h_in_ranges[t]
        tile_input_lists.append([])
        for c in range(c_in):
            for h in range(h_in_start, h_in_end):
                for w in range(w_in):
                    raw_idx = c * h_in * w_in + h * w_in + w
                    copy_id = 0
                    if last_in_end is not None and h < last_in_end:
                        copy_id = 1
                    elem = get_elem(conv_input_node, raw_idx, copy_id)
                    tile_input_lists[t].append(elem)
                    if copy_id == 1:
                        copied_input_elems.append(elem)
        last_in_end = h_in_end

        h_out_start, h_out_end = tile_h_out_ranges[t]
        tile_output_lists.append([])
        for c in range(c_out):
            for h in range(h_out_start, h_out_end):
                for w in range(w_out):
                    raw_idx = c * h_out * w_out + h * w_out + w
                    tile_output_lists[t].append(
                        Neuron(conv_output_node, CustomIndex(raw_idx))
                    )
        tiled_group = RoutingGroup(
            raw_neus=tile_output_lists[t],
            input_list=tile_input_lists[t],
        )
        tiled_groups.append(tiled_group)
    # for elem in copied_input_elems:
    #     print(f"Copied input element: {elem}")
    print(f"Total copied input elements: {len(copied_input_elems)}")
    print(f"Total input elements: {len(conv_grp.input_list)}")
    print(f"Total tiled groups: {len(tiled_groups)}")
    for grp in tiled_groups:
        print(f"Tiled group: {grp}")

    return copied_input_elems, tiled_groups


def tile_conv(
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
            print(f"{grp.raw_elems[0]}: input_bit_nums: {intput_bit_nums}")
            print(f"{grp.input_list[0]}: pred_output_bit_nums: {pred_output_bit_nums}")

            input_bit_num = intput_bit_nums.pop()
            assert (
                input_bit_num == pred_output_bit_nums.pop()
            ), "Input bit num of neurons must match output bit num of input elements."
            max_axon_addr = len(grp.input_list) * input_bit_num
            lcn = ((max_axon_addr - 1) // FANIN_BASE).bit_length()
            if lcn > MAX_LCN.value:
                copied_elems, tile_conv_groups = build_tiled_conv(grp)
                copy_elems.extend(copied_elems)
                group_after_tile.extend(tile_conv_groups)
            else:
                group_after_tile.append(grp)
        else:
            group_after_tile.append(grp)

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
