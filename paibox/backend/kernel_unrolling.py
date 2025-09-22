from paibox.components.synapses.conv_types import Size1Type, Size2Type, Size4Type
from paibox.components.synapses.conv_utils import (
    _conv1d_unroll,
    _conv1d_unroll_asymmetric_padding,
    _conv2d_unroll,
    _conv2d_unroll_asymmetric_padding,
)
from paibox.types import WeightType

__all__ = [
    "conv1d_tiled_kernel_unroll",
    "conv2d_tiled_kernel_unroll",
    "conv1d_tiled_kernel_unroll_no_pad",
    "conv2d_tiled_kernel_unroll_no_pad",
    "conv1d_tiled_kernel_unroll_no_pad_multi_grp",
    "conv2d_tiled_kernel_unroll_no_pad_multi_grp",
]


def _get_k_tile_by_groups(
    kernel: WeightType, groups: int, g_start: int, g_tile: int
) -> WeightType:
    _, ci_in_grp = kernel.shape[:2]
    ksize = kernel.shape[2:]
    kernel = kernel.reshape(groups, -1, ci_in_grp, *ksize)
    k_tl = kernel[g_start : g_start + g_tile]
    k_tl = k_tl.reshape(-1, ci_in_grp, *ksize)
    return k_tl


def conv1d_tiled_kernel_unroll(
    kernel: WeightType,
    stride: Size1Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile: Size1Type,
    o_tile: Size1Type,
    i_tile_padding: Size2Type,
) -> WeightType:
    return _conv1d_unroll_asymmetric_padding(
        i_tile,
        o_tile,
        _get_k_tile_by_groups(kernel, groups, g_start, g_tile),
        stride,
        i_tile_padding,
        g_tile,
    )


def conv2d_tiled_kernel_unroll(
    kernel: WeightType,
    stride: Size2Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile: Size2Type,
    o_tile: Size2Type,
    i_tile_padding: Size4Type,
) -> WeightType:
    return _conv2d_unroll_asymmetric_padding(
        i_tile,
        o_tile,
        _get_k_tile_by_groups(kernel, groups, g_start, g_tile),
        stride,
        i_tile_padding,
        g_tile,
    )


def conv1d_tiled_kernel_unroll_no_pad(
    kernel: WeightType, stride: Size1Type, i_tile: Size1Type, o_tile: Size1Type
) -> WeightType:
    return _conv1d_unroll(i_tile, o_tile, kernel, stride, (0,), 1)


def conv2d_tiled_kernel_unroll_no_pad(
    kernel: WeightType, stride: Size2Type, i_tile: Size2Type, o_tile: Size2Type
) -> WeightType:
    return _conv2d_unroll(i_tile, o_tile, kernel, stride, (0, 0), 1)


def conv1d_tiled_kernel_unroll_no_pad_multi_grp(
    kernel: WeightType,
    stride: Size1Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile: Size1Type,
    o_tile: Size1Type,
) -> WeightType:
    return _conv1d_unroll(
        i_tile,
        o_tile,
        _get_k_tile_by_groups(kernel, groups, g_start, g_tile),
        stride,
        (0,),
        g_tile,
    )


def conv2d_tiled_kernel_unroll_no_pad_multi_grp(
    kernel: WeightType,
    stride: Size2Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile_hw: Size2Type,
    o_tile_hw: Size2Type,
) -> WeightType:
    return _conv2d_unroll(
        i_tile_hw,
        o_tile_hw,
        _get_k_tile_by_groups(kernel, groups, g_start, g_tile),
        stride,
        (0, 0),
        g_tile,
    )
