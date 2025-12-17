from collections.abc import Callable, Generator
from typing import overload

import numpy as np

from paibox.backend.kernel_unrolling import (
    conv1d_tiled_kernel_unroll,
    conv1d_tiled_kernel_unroll_no_pad,
    conv1d_tiled_kernel_unroll_no_pad_multi_grp,
    conv2d_tiled_kernel_unroll,
    conv2d_tiled_kernel_unroll_no_pad,
    conv2d_tiled_kernel_unroll_no_pad_multi_grp,
)
from paibox.backend.tiling import (
    INDEX_DTYPE_WITH_INVALID,
    IndexMapArrayType,
    TileSize2d,
    TileSize3d,
    _cast_size2type,
    _cast_size3type,
    _eff_input_tile_size1d,
    _eff_input_tile_size2d,
    _eff_step,
    _invalid_addr_idx_value,
)
from paibox.components.synapses.conv_types import Size1Type, Size2Type, Size3Type
from paibox.components.synapses.conv_utils import (
    _conv1d_oshape,
    _conv1d_unroll,
    _conv2d_oshape,
    _conv2d_unroll,
    group_ch_check,
)
from paibox.types import VOLTAGE_DTYPE, SynOutType, WeightType

__all__ = [
    "conv1d_unroll_tiled_by_tiles",
    "conv2d_unroll_tiled_by_tiles",
    "conv1d_unroll_tiled_from_full_kernel",
    "conv2d_unroll_tiled_from_full_kernel",
    "tiled_vmm_conv_compact",
    "tiled_vmm_conv1d",
    "tiled_vmm_conv2d",
]


def tiled_vmm_conv_compact(
    x: np.ndarray,
    out_shape: Size2Type | Size3Type,
    groups: int,
    conv_tiles: np.ndarray,
) -> SynOutType:
    """Calculate the output of a 1d/2d convolution in compact tile format."""
    assert x.ndim == len(
        out_shape
    ), f"Input shape {x.shape} and output shape {out_shape} must have the same number of dimensions."
    ci = x.shape[0]
    co = out_shape[0]

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    x_gchw = x.reshape(groups, ci_in_grp, *x.shape[1:])
    out = np.zeros((groups, co_in_grp, *out_shape[1:]), dtype=np.int64)

    for i_blk_idx, o_blk_idx, k_tl_unrolled in conv_tiles.flat:
        x_tl = x_gchw[i_blk_idx.slice_range]
        o = x_tl.ravel().astype(np.int64) @ k_tl_unrolled

        out[o_blk_idx.slice_range] = o.reshape(o_blk_idx.shape_ext)

    return out.reshape(out_shape).astype(VOLTAGE_DTYPE)


def conv1d_unroll_tiled_by_tiles(
    in_shape: Size2Type,
    kernel: np.ndarray,
    stride: Size1Type,
    padding: Size1Type,
    groups: int,
    tile_size: TileSize2d | None = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 1d convolution kernel by tiles. Return a generator of unrolled kernels."""
    ci, li = in_shape
    co, ci_in_grp, kl = kernel.shape
    out_shape = _conv1d_oshape((li,), (kl,), stride, padding)
    (lo,) = out_shape

    group_ch_check(ci, co, groups, ci_in_grp)

    if tile_size is None:
        g_tl, lo_tl = groups, lo
    else:
        g_tl, lo_tl = tile_size

    for g_start in range(0, groups, g_tl):
        # Acctual tile length of g dimension
        g_tl_ = _eff_step(g_start, g_tl, groups)

        for lo_start in range(0, lo, lo_tl):
            # Acctual tile length of h dimension
            lo_tl_ = _eff_step(lo_start, lo_tl, lo)

            li_tl, _, paddings = _eff_input_tile_size1d(
                (li,), (kl,), stride, padding, lo_tl_, lo_start
            )
            # Get unrolled kernel for each tile & yield it
            yield conv1d_tiled_kernel_unroll(
                kernel,
                stride,
                groups,
                g_start,
                g_tl_,
                (li_tl,),
                (lo_tl_,),
                paddings,
            )


def conv2d_unroll_tiled_by_tiles(
    in_shape: Size3Type,
    kernel: np.ndarray,
    stride: Size2Type,
    padding: Size2Type,
    groups: int,
    tile_size: TileSize3d | None = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 2d convolution kernel by tiles. Return a generator of unrolled kernels."""
    ci, hi, wi = in_shape
    co, ci_in_grp, kh, kw = kernel.shape
    out_shape = _conv2d_oshape((hi, wi), (kh, kw), stride, padding)
    ho, wo = out_shape

    group_ch_check(ci, co, groups, ci_in_grp)

    if tile_size is None:
        g_tl, ho_tl, wo_tl = groups, ho, wo
    else:
        g_tl, ho_tl, wo_tl = tile_size

    for g_start in range(0, groups, g_tl):
        # Acctual tile length of g dimension
        g_tl_ = _eff_step(g_start, g_tl, groups)

        for ho_start in range(0, ho, ho_tl):
            # Acctual tile length of h dimension
            ho_tl_ = _eff_step(ho_start, ho_tl, ho)

            for wo_start in range(0, wo, wo_tl):
                # Acctual tile length of w dimension
                wo_tl_ = _eff_step(wo_start, wo_tl, wo)

                (hi_tl, wi_tl), _, paddings = _eff_input_tile_size2d(
                    (hi, wi),
                    (kh, kw),
                    stride,
                    padding,
                    (ho_tl_, wo_tl_),
                    (ho_start, wo_start),
                )
                # Get unrolled kernel for each tile & yield it
                yield conv2d_tiled_kernel_unroll(
                    kernel,
                    stride,
                    groups,
                    g_start,
                    g_tl_,
                    (hi_tl, wi_tl),
                    (ho_tl_, wo_tl_),
                    paddings,
                )


def conv1d_unroll_tiled_from_full_kernel(  # Slower
    in_shape: Size2Type,
    kernel: np.ndarray,
    stride: Size1Type,
    padding: Size1Type,
    groups: int,
    o_tile_size: TileSize2d | None = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 1d convolution kernel from the full unrolled kernel matrix. Return a generator  \
        of unrolled kernels.
    """
    ci, li = in_shape
    co, ci_in_grp, kl = kernel.shape
    out_shape = _conv1d_oshape((li,), (kl,), stride, padding)
    (lo,) = out_shape

    group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if o_tile_size is None:
        g_tl = groups
        lo_tl = lo
    else:
        g_tl, lo_tl = o_tile_size

    # Full unrolled kernel matrix
    k_full_ur = _conv1d_unroll((li,), out_shape, kernel, stride, padding, groups)

    for g_start in range(0, groups, g_tl):
        g_tl_ = _eff_step(g_start, g_tl, groups)
        ci_blk = g_tl_ * ci_in_grp
        co_blk = g_tl_ * co_in_grp
        ci_start = g_start * ci_in_grp
        co_start = g_start * co_in_grp
        ci = ci_start + np.arange(ci_blk)
        co = co_start + np.arange(co_blk)

        for lo_start in range(0, lo, lo_tl):
            lo_tl_ = _eff_step(lo_start, lo_tl, lo)

            li_tl, li_start, _ = _eff_input_tile_size1d(
                (li,), (kl,), stride, padding, lo_tl_, lo_start
            )

            len = np.arange(li_start, li_start + li_tl)
            input_idx = (ci[:, np.newaxis] * li + len[np.newaxis, :]).ravel()
            # input_idx = np.ravel_multi_index(np.ix_(ci, l), (ci_blk, li)).ravel()

            lo_pos = lo_start + np.arange(lo_tl_)

            output_idx = (co[:, np.newaxis] * lo + lo_pos[np.newaxis, :]).ravel()
            # output_idx = np.ravel_multi_index(np.ix_(co, lo_pos), (co_blk, lo)).ravel()

            yield k_full_ur[np.ix_(input_idx, output_idx)]


def conv2d_unroll_tiled_from_full_kernel(  # Slower
    in_shape: Size3Type,
    kernel: np.ndarray,
    stride: Size2Type,
    padding: Size2Type,
    groups: int,
    o_tile_size: TileSize3d | None = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 2d convolution kernel from the full unrolled kernel matrix. Return a generator  \
        of unrolled kernels.
    """
    ci, hi, wi = in_shape
    co, ci_in_grp, kh, kw = kernel.shape
    out_shape = _conv2d_oshape((hi, wi), (kh, kw), stride, padding)
    ho, wo = out_shape

    group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if o_tile_size is None:
        g_tl = groups
        ho_tl = ho
        wo_tl = wo
    else:
        g_tl, ho_tl, wo_tl = o_tile_size

    # Full unrolled kernel matrix
    k_full_ur = _conv2d_unroll((hi, wi), out_shape, kernel, stride, padding, groups)

    for g_start in range(0, groups, g_tl):
        g_tl_ = _eff_step(g_start, g_tl, groups)
        ci_blk = g_tl_ * ci_in_grp
        co_blk = g_tl_ * co_in_grp
        ci_start = g_start * ci_in_grp
        co_start = g_start * co_in_grp
        ci = ci_start + np.arange(ci_blk)
        co = co_start + np.arange(co_blk)

        for ho_start in range(0, ho, ho_tl):
            ho_tl_ = _eff_step(ho_start, ho_tl, ho)

            for wo_start in range(0, wo, wo_tl):
                wo_tl_ = _eff_step(wo_start, wo_tl, wo)

                (hi_tl, wi_tl), (hi_start, wi_start), _ = _eff_input_tile_size2d(
                    (hi, wi),
                    (kh, kw),
                    stride,
                    padding,
                    (ho_tl_, wo_tl_),
                    (ho_start, wo_start),
                )

                h = np.arange(hi_start, hi_start + hi_tl)
                w = np.arange(wi_start, wi_start + wi_tl)

                input_idx = (
                    ci[:, np.newaxis, np.newaxis] * hi * wi
                    + h[np.newaxis, :, np.newaxis] * wi
                    + w[np.newaxis, np.newaxis, :]
                ).ravel()
                # input_idx = np.ravel_multi_index(
                #     np.ix_(ci, h, w), (ci_blk, hi, wi)
                # ).ravel()

                ho_pos = ho_start + np.arange(ho_tl_)
                wo_pos = wo_start + np.arange(wo_tl_)

                output_idx = (
                    co[:, np.newaxis, np.newaxis] * ho * wo
                    + ho_pos[np.newaxis, :, np.newaxis] * wo
                    + wo_pos[np.newaxis, np.newaxis, :]
                ).ravel()
                # output_idx = np.ravel_multi_index(
                #     np.ix_(co, ho_pos, wo_pos), (co_blk, ho, wo)
                # ).ravel()

                yield k_full_ur[np.ix_(input_idx, output_idx)]


def _get_valid_idx_map(
    idx_map: IndexMapArrayType, zero_as_invalid_addr: bool = False
) -> IndexMapArrayType:
    return (
        idx_map.astype(INDEX_DTYPE_WITH_INVALID) - 1
        if zero_as_invalid_addr
        else idx_map
    )


def take_tile_from_arr(
    arr: np.ndarray, indices: np.ndarray, zero_as_invalid_addr: bool
) -> np.ndarray:
    mask_value = _invalid_addr_idx_value(zero_as_invalid_addr)
    mask = indices > mask_value
    tile = np.zeros_like(indices, dtype=arr.dtype)

    _indices = _get_valid_idx_map(indices, zero_as_invalid_addr)

    tile[mask] = np.take(arr, _indices[mask])
    return tile


def put_tile_in_arr(
    arr: np.ndarray, indices: np.ndarray, tile: np.ndarray, zero_as_invalid_addr: bool
) -> None:
    assert indices.shape == tile.shape
    mask_value = _invalid_addr_idx_value(zero_as_invalid_addr)
    mask = indices > mask_value

    _indices = _get_valid_idx_map(indices, zero_as_invalid_addr)
    arr.flat[_indices[mask]] = tile[mask]


@overload
def _tiled_vmm_conv(
    x: np.ndarray,
    out_shape: Size2Type,
    i_tiled_idx_map: IndexMapArrayType,
    o_tiled_idx_map: IndexMapArrayType,
    tile_shape: Size2Type,
    o_tile_shape: Size2Type,
    get_k_tl_unrolled_hdlr: Callable[[Size2Type], np.ndarray],
    zero_as_invalid_addr: bool,
) -> SynOutType: ...


@overload
def _tiled_vmm_conv(
    x: np.ndarray,
    out_shape: Size3Type,
    i_tiled_idx_map: IndexMapArrayType,
    o_tiled_idx_map: IndexMapArrayType,
    tile_shape: Size3Type,
    o_tile_shape: Size3Type,
    get_k_tl_unrolled_hdlr: Callable[[Size3Type], np.ndarray],
    zero_as_invalid_addr: bool,
) -> SynOutType: ...


def _tiled_vmm_conv(
    x: np.ndarray,
    out_shape: Size2Type | Size3Type,
    i_tiled_idx_map: IndexMapArrayType,
    o_tiled_idx_map: IndexMapArrayType,
    tile_shape: Size2Type | Size3Type,
    o_tile_shape: Size2Type | Size3Type,
    get_k_tl_unrolled_hdlr: (
        Callable[[Size2Type], np.ndarray] | Callable[[Size3Type], np.ndarray]
    ),
    zero_as_invalid_addr: bool,
) -> SynOutType:
    out = np.zeros(out_shape, dtype=np.int64)

    def inner_vmm(
        i_tl: IndexMapArrayType, o_tl: IndexMapArrayType, k_tl: np.ndarray
    ) -> None:
        x_tl = take_tile_from_arr(x, i_tl, zero_as_invalid_addr)
        o = x_tl.ravel().astype(np.int64) @ k_tl
        put_tile_in_arr(out, o_tl, o.reshape(o_tile_shape), zero_as_invalid_addr)

    for tl_idx in np.ndindex(tile_shape):
        i_tl = i_tiled_idx_map[tl_idx]
        o_tl = o_tiled_idx_map[tl_idx]
        inner_vmm(i_tl, o_tl, get_k_tl_unrolled_hdlr(tl_idx))  # type: ignore

    return out.astype(VOLTAGE_DTYPE)


def tiled_vmm_conv1d(
    x: np.ndarray,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size1Type,
    groups: int,
    i_tiled_idx_map: IndexMapArrayType,
    o_tiled_idx_map: IndexMapArrayType,
    zero_as_invalid_addr: bool = False,
    k_tiles_unrolled: WeightType | None = None,
) -> SynOutType:
    """Calculate the output of a 1d convolution in tile format."""
    assert x.ndim == len(out_shape) == 2
    ci = x.shape[0]
    co = out_shape[0]

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    co_in_grp = co // groups

    assert i_tiled_idx_map.shape[:2] == o_tiled_idx_map.shape[:2]

    tile_shape = _cast_size2type(i_tiled_idx_map.shape[:2])
    i_tile_cl = _cast_size2type(i_tiled_idx_map.shape[-2:])
    i_tile_l = i_tile_cl[-1:]
    o_tile_cl = _cast_size2type(o_tiled_idx_map.shape[-2:])
    o_tile_l = o_tile_cl[-1:]
    g_tl = o_tile_cl[0] // co_in_grp

    if k_tiles_unrolled is None:
        if groups == 1:
            k_tl_one_grp = conv1d_tiled_kernel_unroll_no_pad(
                kernel, stride, i_tile_l, o_tile_l
            )

            def get_k_tl_unrolled(tl_idx):
                return k_tl_one_grp

        else:

            def get_k_tl_unrolled(tl_idx):
                g_start = tl_idx[0] * g_tl
                return conv1d_tiled_kernel_unroll_no_pad_multi_grp(
                    kernel, stride, groups, g_start, g_tl, i_tile_l, o_tile_l
                )

    else:
        if groups == 1:

            def get_k_tl_unrolled(tl_idx):
                return k_tiles_unrolled

        else:

            def get_k_tl_unrolled(tl_idx):
                return k_tiles_unrolled[tl_idx]

    return _tiled_vmm_conv(
        x,
        out_shape,
        i_tiled_idx_map,
        o_tiled_idx_map,
        tile_shape,
        o_tile_cl,
        get_k_tl_unrolled,
        zero_as_invalid_addr,
    )


def tiled_vmm_conv2d(
    x: np.ndarray,
    out_shape: Size3Type,
    kernel: WeightType,
    stride: Size2Type,
    groups: int,
    i_tiled_idx_map: IndexMapArrayType,
    o_tiled_idx_map: IndexMapArrayType,
    zero_as_invalid_addr: bool = False,
    k_tiles_unrolled: WeightType | None = None,
) -> SynOutType:
    """Calculate the output of a 2d convolution in tile format."""
    assert x.ndim == len(out_shape) == 3
    ci = x.shape[0]
    co = out_shape[0]

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    co_in_grp = co // groups

    assert i_tiled_idx_map.shape[:3] == o_tiled_idx_map.shape[:3]

    tile_shape = _cast_size3type(i_tiled_idx_map.shape[:3])
    i_tile_chw = _cast_size3type(i_tiled_idx_map.shape[-3:])
    i_tile_hw = i_tile_chw[-2:]
    o_tile_chw = _cast_size3type(o_tiled_idx_map.shape[-3:])
    o_tile_hw = o_tile_chw[-2:]
    g_tl = o_tile_chw[0] // co_in_grp

    if k_tiles_unrolled is None:
        if groups == 1:
            k_tl_one_grp = conv2d_tiled_kernel_unroll_no_pad(
                kernel, stride, i_tile_hw, o_tile_hw
            )

            def get_k_tl_unrolled(tl_idx):
                return k_tl_one_grp

        else:

            def get_k_tl_unrolled(tl_idx):
                g_start = tl_idx[0] * g_tl
                return conv2d_tiled_kernel_unroll_no_pad_multi_grp(
                    kernel, stride, groups, g_start, g_tl, i_tile_hw, o_tile_hw
                )

    else:
        if groups == 1:

            def get_k_tl_unrolled(tl_idx):
                return k_tiles_unrolled

        else:

            def get_k_tl_unrolled(tl_idx):
                return k_tiles_unrolled[tl_idx]

    return _tiled_vmm_conv(
        x,
        out_shape,
        i_tiled_idx_map,
        o_tiled_idx_map,
        tile_shape,
        o_tile_chw,
        get_k_tl_unrolled,
        zero_as_invalid_addr,
    )
