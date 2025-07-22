from collections.abc import Generator
from dataclasses import dataclass
from enum import IntEnum, unique
import math
import numpy as np

from typing import Optional, Literal, Union

from paicorelib import LCN_EX, HwConfig
from paibox import _logging
from paibox.components.synapses.conv_utils import (
    _assert_max_index,
    _group_ch_check,
    _conv1d_oshape,
    _conv1d_unroll,
    _conv1d_unroll_asymmetric_padding,
    _conv2d_oshape,
    _conv2d_unroll,
    _conv2d_unroll_asymmetric_padding,
    _pair,
    _single,
)
from paibox.components.synapses.conv_types import (
    _Size1Type,
    _Size2Type,
    Size1Type,
    Size2Type,
    Size3Type,
    Size4Type,
)
from paibox.types import VOLTAGE_DTYPE, Shape, SynOutType, WeightType
from paibox.utils import shape2num

tl_optim_log = _logging.get_artifact_logger(__name__, "tiling_optim")

TileSize2d = Size2Type
TileSize3d = Size3Type


class TileSlice2d:
    def __init__(self, x_start: int, x_end: int, y_start: int, y_end: int) -> None:
        assert x_start < x_end and y_start < y_end
        self.x = slice(x_start, x_end, 1)
        self.y = slice(y_start, y_end, 1)

    def to_shape(self):
        raise NotImplementedError

    def to_slice(self):
        raise NotImplementedError

    @property
    def start(self) -> tuple[int, int]:
        return (self.x.start, self.y.start)

    @property
    def end(self) -> tuple[int, int]:
        return (self.x.stop, self.y.stop)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.xlen, self.ylen)

    @property
    def xlen(self) -> int:
        return self.x.stop - self.x.start

    @property
    def ylen(self) -> int:
        return self.y.stop - self.y.start


class TileSlice3d:
    def __init__(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        z_start: int,
        z_end: int,
    ) -> None:
        assert x_start < x_end and y_start < y_end and z_start < z_end
        self.x = slice(x_start, x_end, 1)
        self.y = slice(y_start, y_end, 1)
        self.z = slice(z_start, z_end, 1)

    def to_shape(self):
        raise NotImplementedError

    def to_slice(self):
        raise NotImplementedError

    @property
    def start(self) -> tuple[int, int, int]:
        return (self.x.start, self.y.start, self.z.start)

    @property
    def end(self) -> tuple[int, int, int]:
        return (self.x.stop, self.y.stop, self.z.stop)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.xlen, self.ylen, self.zlen)

    @property
    def xlen(self) -> int:
        return self.x.stop - self.x.start

    @property
    def ylen(self) -> int:
        return self.y.stop - self.y.start

    @property
    def zlen(self) -> int:
        return self.z.stop - self.z.start


class TileSliceConv1d(TileSlice2d):
    def __init__(self, g_start: int, g_end: int, l_start: int, l_end: int) -> None:
        super().__init__(g_start, g_end, l_start, l_end)

    def to_shape(self):
        return (self.xlen, -1, self.ylen)

    def to_slice(self):
        return (self.x, slice(None), self.y)


class TileSliceConv2d(TileSlice3d):
    def __init__(
        self,
        g_start: int,
        g_end: int,
        h_start: int,
        h_end: int,
        w_start: int,
        w_end: int,
    ) -> None:
        super().__init__(g_start, g_end, h_start, h_end, w_start, w_end)

    def to_shape(self):
        return (self.xlen, -1, self.ylen, self.zlen)

    def to_slice(self):
        return (self.x, slice(None), self.y, self.z)


def _eff_step(start: int, step: int, len: int) -> int:
    """Return the effective step."""
    if start + step <= len:
        return step
    else:
        return len - start


def _eff_tile_size2d(
    tile_start_idx: tuple[int, int],
    conv1d_tile_size: TileSize2d,
    range: tuple[int, int],
) -> TileSize2d:
    g, l = map(
        lambda args: _eff_step(*args), zip(tile_start_idx, conv1d_tile_size, range)
    )
    return g, l


def _eff_tile_size3d(
    tile_start_idx: tuple[int, int, int],
    conv2d_tile_size: TileSize3d,
    range: tuple[int, int, int],
) -> TileSize3d:
    g, h, w = map(
        lambda args: _eff_step(*args), zip(tile_start_idx, conv2d_tile_size, range)
    )
    return g, h, w


def _conv_pos_o2i(pos: int, s: int, p: int) -> int:
    return pos * s - p


def _input_pos_conv1d(pos: int, stride: Size1Type, padding: Size1Type) -> int:
    """Compute the input position from the output position."""
    return _conv_pos_o2i(pos, stride[0], padding[0])


def _input_pos_conv2d(
    pos: Size2Type, stride: Size2Type, padding: Size2Type
) -> Size2Type:
    """Compute the input position from the output position."""

    hi, wi = map(lambda args: _conv_pos_o2i(*args), zip(pos, stride, padding))
    return hi, wi


def _conv_len_o2i(len: int, k: int, s: int) -> int:
    return (len - 1) * s + k


def _i_tile_hw_conv1d(o_tile_size: int, ksize: Size1Type, stride: Size1Type) -> int:
    """Compute the input tile size by given the output tile size, kernel size & stride."""
    k = ksize[0]
    s = stride[0]

    return _conv_len_o2i(o_tile_size, k, s)


def _i_tile_hw_conv2d(
    o_tile_size: Size2Type, ksize: Size2Type, stride: Size2Type
) -> Size2Type:
    """Compute the input tile size by given the output tile size, kernel size & stride."""

    hi_tl, wi_tl = map(
        lambda args: _conv_len_o2i(*args), zip(o_tile_size, ksize, stride)
    )
    return hi_tl, wi_tl


def _eff_input_tile_size1d(
    in_shape_l: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
    o_tile_l: int,
    o_tile_l_start: int,
) -> tuple[int, int, Size2Type]:
    (li,) = in_shape_l

    _li_start = _input_pos_conv1d(o_tile_l_start, stride, padding)
    _li_tl = _i_tile_hw_conv1d(o_tile_l, ksize, stride)
    _li_end = _li_start + _li_tl

    pad_l = max(-_li_start, 0)
    pad_r = max(_li_end - li, 0)

    li_start = max(_li_start, 0)

    # Actual block size without padding
    li_tl = _li_tl - pad_l - pad_r

    return li_tl, li_start, (pad_l, pad_r)


def _eff_input_tile_size2d(
    in_shape_hw: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    o_tile_hw: Size2Type,
    o_tile_hw_start: Size2Type,
) -> tuple[Size2Type, Size2Type, Size4Type]:
    hi, wi = in_shape_hw

    _hi_start, _wi_start = _input_pos_conv2d(o_tile_hw_start, stride, padding)
    _hi_tl, _wi_tl = _i_tile_hw_conv2d(o_tile_hw, ksize, stride)
    _hi_end = _hi_start + _hi_tl
    _wi_end = _wi_start + _wi_tl

    pad_t = max(-_hi_start, 0)
    pad_b = max(_hi_end - hi, 0)
    pad_l = max(-_wi_start, 0)
    pad_r = max(_wi_end - wi, 0)

    hi_start = max(_hi_start, 0)
    wi_start = max(_wi_start, 0)

    # Actual block size without padding
    hi_tl = _hi_tl - pad_t - pad_b
    wi_tl = _wi_tl - pad_l - pad_r

    return (hi_tl, wi_tl), (hi_start, wi_start), (pad_t, pad_b, pad_l, pad_r)


@unique
class EstCoreCostStatus(IntEnum):
    """Status of estimating core cost."""

    SUCCESS = 0
    FAN_IN_TOO_LARGE = -1
    """The fan-in of the operator is too large for the computing cores."""
    CORES_MORE_THAN_ONE_CHIP = -2
    """The number of cores needed is more than one chip so it is impossible to be multicasted."""
    FAILED = -99


_EstStatus = EstCoreCostStatus


@dataclass
class EstCoreCostResult:
    status: _EstStatus
    n_core: int
    lcn: LCN_EX


_EstResult = EstCoreCostResult


def optimal_lcn_matmul2d(
    shape_a: Size2Type,
    shape_b: Size2Type,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
) -> _EstResult:
    """Estimate the optimal LCN for 2d matmul tiling. A 2d matmul `Y = A(m*n) * B(n*k)` can be viewed as:
        [Y1, Y2, ..., Ym]^T = [A1, A2, ..., An]^T @ B

        where B is the patch matrix and `m` is the number of patches. `m` B matrix is arranged diagonally.

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
    """
    assert shape_a[1] == shape_b[0]

    est_result = _EstResult(_EstStatus.FAILED, 0, LCN_EX.LCN_1X)
    n_fanin_patch, n_fanout_patch = shape_b
    n_patch = shape_a[0]

    for lcn in LCN_EX.__members__.values():
        fin_capacity = core_n_fanin_base << lcn
        fout_capacity = core_n_fanout_base >> lcn

        n_patch_in_tile = math.floor(fin_capacity / n_fanin_patch)
        if n_patch_in_tile < 1:
            # LCN is too small to allocate 1 patch
            continue

        # Get the #N of cores to allocate for this tile
        n_core_tile = math.ceil(n_patch_in_tile * n_fanout_patch / fout_capacity)
        # if n_core_tile > HwConfig.N_CORE_OFFLINE:
        #     # The #N of cores for a single tile exceeds the maximum number of cores in a single chip.
        #     est_result.status = _EstStatus.CORES_MORE_THAN_ONE_CHIP
        #     break

        # Get the #N of tiles to allocate for this matmul
        n_tile, n_patch_remain = divmod(n_patch, n_patch_in_tile)

        n_core = n_core_tile * n_tile
        if n_patch_remain > 0:
            n_core += math.ceil(n_patch_remain * n_fanout_patch / fout_capacity)

        tl_optim_log.debug(f"Current lcn: {lcn}, estimated cores: {n_core}")

        if n_core > HwConfig.N_CORE_OFFLINE:
            # Exceeds the maximum number of cores in a single chip.
            continue

        # If this is the first successful estimation or the total number of cores is less than
        # the previous one, update the result.
        if est_result.status == _EstStatus.FAILED or n_core < est_result.n_core:
            est_result.status = _EstStatus.SUCCESS
            est_result.n_core = n_core
            est_result.lcn = lcn

    return est_result


def operator_core_cost_estimate(
    in_shape: Shape,
    out_shape: Shape,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
) -> _EstResult:
    """Estimate the number of cores & LCN needed for a given input & output shape of an operator.

    Args:
        in_shape (Shape): the shape of the input tensor.
        out_shape (Shape): the shape of the output tensor.
        core_n_fanin_base (int): the base fan-in of a core.
        core_n_fanout_base (int): the base fan-out of a core.

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
    """
    n_fanin = shape2num(in_shape)
    n_fanout = shape2num(out_shape)

    # Estimate #N of cores needed for this op
    if (_lcn := ((n_fanin - 1) // core_n_fanin_base).bit_length()) > LCN_EX.LCN_64X:
        # Fan-in is too large for the core
        return _EstResult(_EstStatus.FAN_IN_TOO_LARGE, 0, LCN_EX.LCN_64X)

    est_n_core = math.ceil(n_fanout / (core_n_fanout_base >> _lcn))
    if est_n_core > HwConfig.N_CORE_OFFLINE:
        return _EstResult(_EstStatus.CORES_MORE_THAN_ONE_CHIP, est_n_core, LCN_EX(_lcn))

    return _EstResult(_EstStatus.SUCCESS, est_n_core, LCN_EX(_lcn))


def _traverse_tile_size1d(
    lo: int, order: Literal["all", "even"] = "all"
) -> Generator[int, None, None]:
    """Generate a sequence of tile sizes by given the 1d output feature map."""
    if order == "all":
        for l in range(1, lo + 1):
            yield l
    else:
        for l in range(2, lo + 1, 2):
            yield l


def _traverse_tile_size2d(
    ho: int, wo: int, order: Literal["all", "Lshape", "even"] = "all"
) -> Generator[Size2Type, None, None]:
    """Generate a sequence of tile sizes by given the 2d output feature map."""
    if order == "all":
        for h in range(1, ho + 1):
            for w in range(1, wo + 1):
                yield (h, w)
    elif order == "even":
        for h in range(2, ho + 1, 2):
            for w in range(2, wo + 1, 2):
                yield (h, w)
    else:
        for w in range(1, wo + 1):
            yield (1, w)
        for h in range(1, ho + 1):
            yield (h, wo)


def optimal_tiling_conv1d(
    in_shape: Size2Type,
    out_shape: Size2Type,
    ksize: Size1Type,
    stride: Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    *,
    traverse_order: Literal["all", "even"] = "all",
) -> tuple[_EstResult, TileSize2d, TileSize2d]:
    """Tile the 1d convolution & find the optimal block size of the output feature map.

    Args:
        in_shape (tuple[int, int]): the shape of the input feature map in C, L order.
        out_shape (tuple[int, int]): the shape of the output feature map in C, L order.
        ksize (Size1Type): the kernel size.
        stride (Size1Type): the stride.
        groups (int): the number of groups.
        core_n_fanin_base (int): the base fan-in of a core.
        core_n_fanout_base (int): the base fan-out of a core.
        traverse_order (Literal["all", "even"]): the order to traverse the possible optimal output feature map. \
            Defaults to "all".

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
        tile_size_ghw (tuple[int, int, int]): the block size on dimensions g, l.
        n_tile_ghw (tuple[int, int, int]): the number of blocks in each dimension of g, l.
    """
    ci, _ = in_shape
    co, lo = out_shape

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    est_result = _EstResult(_EstStatus.FAILED, 0, LCN_EX.LCN_1X)

    # Tile the conv1d in g, l dimensions (channels will be followed by groups)
    best_n_tile_gl = (0, 0)
    best_tile_size_gl = (0, 0)

    for g_tl in range(1, groups + 1):
        n_tile_g = math.ceil(groups / g_tl)

        for lo_tl in _traverse_tile_size1d(lo, traverse_order):
            o_tile_size_gl = (g_tl, lo_tl)
            li_tl = _i_tile_hw_conv1d(lo_tl, ksize, stride)

            i_tile_size = (g_tl, ci_in_grp, li_tl)
            o_tile_size = (g_tl, co_in_grp, lo_tl)

            est = operator_core_cost_estimate(
                i_tile_size, o_tile_size, core_n_fanin_base, core_n_fanout_base
            )
            if est.status != _EstStatus.SUCCESS:
                continue

            # Even if there are incomplete tile, treat them as a `o_tile_size` block for calculating
            # resource costs.
            n_tile_l = math.ceil(lo / lo_tl)
            n_tile_gl = (n_tile_g, n_tile_l)

            total_n_core = est.n_core * shape2num(n_tile_gl)
            if total_n_core > HwConfig.N_CORE_OFFLINE:
                continue

            # If this is the first successful estimation or the total number of cores is less than
            # the previous one, update the result.
            if (
                est_result.status == _EstStatus.FAILED
                or total_n_core < est_result.n_core
            ):
                est_result.status = est.status
                est_result.n_core = total_n_core
                est_result.lcn = est.lcn
                best_n_tile_gl = n_tile_gl
                best_tile_size_gl = o_tile_size_gl

            # TODO After finding an available tile size, we should also consider the
            # #N of cores for copying the overlap between the adjacent input tiles.

    return est_result, best_tile_size_gl, best_n_tile_gl


def optimal_tiling_conv2d(
    in_shape: Size3Type,
    out_shape: Size3Type,
    ksize: Size2Type,
    stride: Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    *,
    traverse_order: Literal["all", "Lshape", "even"] = "all",
) -> tuple[_EstResult, TileSize3d, TileSize3d]:
    """Tile the 2d convolution & find the optimal block size of the output feature map.

    Args:
        in_shape (tuple[int, int, int]): the shape of the input feature map in C, H, W order.
        out_shape (tuple[int, int, int]): the shape of the output feature map in C, H, W order.
        ksize (Size2Type): the kernel size.
        stride (Size2Type): the stride.
        groups (int): the number of groups.
        core_n_fanin_base (int): the base fan-in of a core.
        core_n_fanout_base (int): the base fan-out of a core.
        traverse_order (Literal["all", "Lshape", "even"]): the order to traverse the possible optimal output \
            feature map. Defaults to "all".

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
        tile_size_ghw (tuple[int, int, int]): the block size on dimensions g, h, w.
        n_tile_ghw (tuple[int, int, int]): the number of blocks in each dimension of g, h, w.
    """
    ci, _, _ = in_shape
    co, ho, wo = out_shape

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    est_result = _EstResult(_EstStatus.FAILED, 0, LCN_EX.LCN_1X)

    # Tile the conv2d in g, h, w dimensions (channels will be followed by groups)
    best_n_tile_ghw = (0, 0, 0)
    best_tile_size_ghw = (0, 0, 0)

    for g_tl in range(1, groups + 1):
        n_tile_g = math.ceil(groups / g_tl)

        for ho_tl, wo_tl in _traverse_tile_size2d(ho, wo, traverse_order):
            o_tile_size_ghw = (g_tl, ho_tl, wo_tl)
            hi_tl, wi_tl = _i_tile_hw_conv2d((ho_tl, wo_tl), ksize, stride)

            i_tile_size = (g_tl, ci_in_grp, hi_tl, wi_tl)
            o_tile_size = (g_tl, co_in_grp, ho_tl, wo_tl)

            est = operator_core_cost_estimate(
                i_tile_size, o_tile_size, core_n_fanin_base, core_n_fanout_base
            )
            if est.status != _EstStatus.SUCCESS:
                continue

            # Even if there are incomplete tile, treat them as a `o_tile_size` block for calculating
            # resource costs.
            n_tile_h = math.ceil(ho / ho_tl)
            n_tile_w = math.ceil(wo / wo_tl)
            n_tile_ghw = (n_tile_g, n_tile_h, n_tile_w)

            total_n_core = est.n_core * shape2num(n_tile_ghw)
            if total_n_core > HwConfig.N_CORE_OFFLINE:
                continue

            # If this is the first successful estimation or the total number of cores is less than
            # the previous one, update the result.
            if (
                est_result.status == _EstStatus.FAILED
                or total_n_core < est_result.n_core
            ):
                est_result.status = est.status
                est_result.n_core = total_n_core
                est_result.lcn = est.lcn
                best_n_tile_ghw = n_tile_ghw
                best_tile_size_ghw = o_tile_size_ghw

            # TODO After finding an available tile size, we should also consider the
            # #N of cores for copying the overlap between the adjacent input tiles.

    return est_result, best_tile_size_ghw, best_n_tile_ghw


def _n_tile_ghw_by_tile_size(
    out_shape_hw: Size2Type, groups: int, tile_size: TileSize3d
) -> TileSize3d:
    """Compute the number of blocks in each dimension of g, h, w by given the tile size."""
    ho, wo = out_shape_hw
    g_tl, ho_tl, wo_tl = tile_size

    n_tile_g = math.ceil(groups / g_tl)
    n_tile_h = math.ceil(ho / ho_tl)
    n_tile_w = math.ceil(wo / wo_tl)

    return (n_tile_g, n_tile_h, n_tile_w)


def _tile_range2d_check(
    g: int, l: int, g_start: int, l_start: int, g_max: int, l_max: int
) -> None:
    if not (g_start < g_max and l_start < l_max):
        raise ValueError(
            f"Tile starting at (g, l)=({g_start}, {l_start}) "
            f"exceeds the input size ({g}, {l}).",
        )
    if not (g_start + g <= g_max and l_start + l <= l_max):
        raise ValueError(
            f"Tile with size (g, l)=({g}, {l}) starting at ({g_start}, {l_start}), "
            f"exceeds the input size ({g_max}, {l_max}).",
        )


def _tile_range3d_check(
    g: int,
    h: int,
    w: int,
    g_start: int,
    h_start: int,
    w_start: int,
    g_max: int,
    h_max: int,
    w_max: int,
) -> None:
    if not (g_start < g_max and h_start < h_max and w_start < w_max):
        raise ValueError(
            f"Tile starting at (g, h, w)=({g_start}, {h_start}, {w_start}) "
            f"exceeds the input size ({g}, {h}, {w}).",
        )
    if not (g_start + g <= g_max and h_start + h <= h_max and w_start + w <= w_max):
        raise ValueError(
            f"Tile with size (g, h, w)=({g}, {h}, {w}) starting at ({g_start}, {h_start}, {w_start}), "
            f"exceeds the input size ({g_max}, {h_max}, {w_max}).",
        )


def _conv1d_tile_unroll(
    kernel: WeightType,
    stride: Size1Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile_l: Size1Type,
    o_tile_l: Size1Type,
    i_tile_padding: Size2Type,
) -> np.ndarray:
    _, ci_in_grp, kl = kernel.shape
    kernel = kernel.reshape(groups, -1, ci_in_grp, kl)
    k_tile = kernel[g_start : g_start + g_tile]
    k_tile = k_tile.reshape(-1, ci_in_grp, kl)

    return _conv1d_unroll_asymmetric_padding(
        i_tile_l, o_tile_l, k_tile, stride, i_tile_padding, g_tile
    )


def _conv2d_tile_unroll(
    kernel: WeightType,
    stride: Size2Type,
    groups: int,
    g_start: int,
    g_tile: int,
    i_tile_hw: Size2Type,
    o_tile_hw: Size2Type,
    i_tile_padding: Size4Type,
) -> np.ndarray:
    _, ci_in_grp, kh, kw = kernel.shape
    kernel = kernel.reshape(groups, -1, ci_in_grp, kh, kw)
    k_tile = kernel[g_start : g_start + g_tile]
    k_tile = k_tile.reshape(-1, ci_in_grp, kh, kw)

    return _conv2d_unroll_asymmetric_padding(
        i_tile_hw, o_tile_hw, k_tile, stride, i_tile_padding, g_tile
    )


def _conv1d_tiling_visitor(
    n_tile: TileSize2d,
    tile_size2d: TileSize2d,
    groups: int,
    lo: int,
) -> Generator[tuple[TileSize2d, TileSize2d, TileSize2d], None, None]:
    """Visit all tiles in order of g, l & yield the starting indices & tile sizes of each tile."""
    n_g, n_l = n_tile

    g_tl = lo_tl = -1
    g_start = 0
    for g_idx in range(n_g):
        lo_start = 0
        for lo_idx in range(n_l):
            g_tl, lo_tl = _eff_tile_size2d(
                (g_start, lo_start), tile_size2d, (groups, lo)
            )
            yield (g_idx, lo_idx), (g_start, lo_start), (g_tl, lo_tl)
            lo_start += lo_tl
        g_start += g_tl


def _conv2d_tiling_visitor(
    n_tile: TileSize3d,
    tile_size3d: TileSize3d,
    groups: int,
    ho: int,
    wo: int,
) -> Generator[tuple[TileSize3d, TileSize3d, TileSize3d], None, None]:
    """Visit all tiles in order of g, h, w & yield the starting indices & tile sizes of each tile."""
    n_g, n_h, n_w = n_tile

    g_tl = ho_tl = wo_tl = -1
    g_start = 0
    for g_idx in range(n_g):
        ho_start = 0
        for ho_idx in range(n_h):
            wo_start = 0
            for wo_idx in range(n_w):
                g_tl, ho_tl, wo_tl = _eff_tile_size3d(
                    (g_start, ho_start, wo_start), tile_size3d, (groups, ho, wo)
                )
                yield (g_idx, ho_idx, wo_idx), (g_start, ho_start, wo_start), (
                    g_tl,
                    ho_tl,
                    wo_tl,
                )
                wo_start += wo_tl
            ho_start += ho_tl
        g_start += g_tl


def conv1d_tile_by_tile_size(
    in_shape: Size2Type,
    out_shape_l: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    groups: int,
    conv1d_tile_size: TileSize2d,
    n_tile: TileSize2d,
) -> tuple[np.ndarray, np.ndarray]:
    ci, li = in_shape
    (lo,) = out_shape_l
    _, _, kl = kernel.shape
    ksize = (kl,)

    ci_in_grp = ci // groups
    tiles = np.zeros(n_tile, dtype=object)
    copy_times = np.zeros((groups, ci_in_grp, li), dtype=np.uint16)

    for tile_idx, (g_start, lo_start), (g_tl, lo_tl) in _conv1d_tiling_visitor(
        n_tile, conv1d_tile_size, groups, lo
    ):
        li_tl, li_start, paddings = _eff_input_tile_size1d(
            (li,),
            ksize,
            stride,
            padding,
            lo_tl,
            lo_start,
        )
        _tile_range2d_check(g_tl, lo_tl, g_start, lo_start, groups, lo)
        _tile_range2d_check(g_tl, li_tl, g_start, li_start, groups, li)
        # Extract input & output tile indices
        i_tile_idx = TileSliceConv1d(
            g_start, g_start + g_tl, li_start, li_start + li_tl
        )
        o_tile_idx = TileSliceConv1d(
            g_start, g_start + g_tl, lo_start, lo_start + lo_tl
        )
        # Get unrolled kernel for each tile
        k_tile_unrolled = _conv1d_tile_unroll(
            kernel, stride, groups, g_start, g_tl, (li_tl,), (lo_tl,), paddings
        )
        tiles[tile_idx] = (i_tile_idx, o_tile_idx, k_tile_unrolled)
        copy_times[i_tile_idx.to_slice()] += 1

    return tiles, copy_times.reshape(in_shape)


def conv2d_tile_by_tile_size(
    in_shape: Size3Type,
    out_shape_hw: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int,
    conv2d_tile_size: TileSize3d,
    n_tile: TileSize3d,
) -> tuple[np.ndarray, np.ndarray]:
    ci, hi, wi = in_shape
    ho, wo = out_shape_hw
    _, _, kh, kw = kernel.shape
    ksize = (kh, kw)

    ci_in_grp = ci // groups
    tiles = np.zeros(n_tile, dtype=object)
    copy_times = np.zeros((groups, ci_in_grp, hi, wi), dtype=np.uint16)

    for (
        tile_idx,
        (g_start, ho_start, wo_start),
        (g_tl, ho_tl, wo_tl),
    ) in _conv2d_tiling_visitor(n_tile, conv2d_tile_size, groups, ho, wo):
        (hi_tl, wi_tl), (hi_start, wi_start), paddings = _eff_input_tile_size2d(
            (hi, wi),
            ksize,
            stride,
            padding,
            (ho_tl, wo_tl),
            (ho_start, wo_start),
        )
        _tile_range3d_check(
            g_tl, ho_tl, wo_tl, g_start, ho_start, wo_start, groups, ho, wo
        )
        _tile_range3d_check(
            g_tl, hi_tl, wi_tl, g_start, hi_start, wi_start, groups, hi, wi
        )
        # Extract input & output tile indices
        i_tile_idx = TileSliceConv2d(
            g_start,
            g_start + g_tl,
            hi_start,
            hi_start + hi_tl,
            wi_start,
            wi_start + wi_tl,
        )
        o_tile_idx = TileSliceConv2d(
            g_start,
            g_start + g_tl,
            ho_start,
            ho_start + ho_tl,
            wo_start,
            wo_start + wo_tl,
        )
        # Get unrolled kernel for each tile
        k_tile_unrolled = _conv2d_tile_unroll(
            kernel,
            stride,
            groups,
            g_start,
            g_tl,
            (hi_tl, wi_tl),
            (ho_tl, wo_tl),
            paddings,
        )
        tiles[tile_idx] = (i_tile_idx, o_tile_idx, k_tile_unrolled)
        copy_times[i_tile_idx.to_slice()] += 1

    return tiles, copy_times.reshape(in_shape)


def conv1d_tiling_optimize(
    in_shape: Size2Type,
    kernel: WeightType,
    stride: _Size1Type,
    padding: _Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Optional[Size2Type] = None,
) -> tuple[_EstResult, np.ndarray, np.ndarray]:
    """Tile the 1d convolution by the optimal tile size.

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
        tiles (np.ndarray): the tiled 1d convolution in tile format.
        copy_times (np.ndarray): the copy times for the input feature map due to the overlap between adjacent tiles.
    """
    ci, li = in_shape
    co, ci_in_grp, kl = kernel.shape
    ksize = (kl,)
    stride = _single(stride)
    padding = _single(padding)

    if out_shape is None:
        (lo,) = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (co, lo)
    else:
        _co, lo = out_shape
        assert _co == co

    _group_ch_check(ci, co, groups, ci_in_grp)

    est_result, conv1d_tile_size, n_tile_ghw = optimal_tiling_conv1d(
        in_shape,
        out_shape,
        ksize,
        stride,
        groups,
        core_n_fanin_base,
        core_n_fanout_base,
    )
    if est_result.status != _EstStatus.SUCCESS:
        tl_optim_log.debug(
            "Failed to find a valid tiling. Maybe the operator is too large to fit in."
        )
        return est_result, np.empty(n_tile_ghw, dtype=object), np.empty(in_shape)
    else:
        tl_optim_log.debug(
            f"Conv1d tiling success.\n"
            + f"\tEstimated optimal tile size: {conv1d_tile_size}, n_tile_ghw: {n_tile_ghw}.\n"
            + f"\tEstimated cost cores in total: {est_result.n_core}, lcn for each tile: {est_result.lcn.name}."
        )

    return (
        est_result,
        *conv1d_tile_by_tile_size(
            in_shape,
            (lo,),
            kernel,
            stride,
            padding,
            groups,
            conv1d_tile_size,
            n_tile_ghw,
        ),
    )


def conv2d_tiling_optimize(
    in_shape: Size3Type,
    kernel: WeightType,
    stride: _Size2Type,
    padding: _Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Optional[Size3Type] = None,
) -> tuple[_EstResult, np.ndarray, np.ndarray]:
    """Tile the 2d convolution by the optimal tile size.

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
        tiles (np.ndarray): the tiled 2d convolution in tile format.
        copy_times (np.ndarray): the copy times for the input feature map due to the overlap between adjacent tiles.
    """
    ci, hi, wi = in_shape
    co, ci_in_grp, kh, kw = kernel.shape
    ksize = (kh, kw)
    stride = _pair(stride)
    padding = _pair(padding)

    if out_shape is None:
        ho, wo = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (co, ho, wo)
    else:
        _co, ho, wo = out_shape
        assert _co == co

    _group_ch_check(ci, co, groups, ci_in_grp)

    est_result, conv2d_tile_size, n_tile_ghw = optimal_tiling_conv2d(
        in_shape,
        out_shape,
        ksize,
        stride,
        groups,
        core_n_fanin_base,
        core_n_fanout_base,
    )
    if est_result.status != _EstStatus.SUCCESS:
        tl_optim_log.debug(
            "Failed to find a valid tiling. Maybe the operator is too large to fit in."
        )
        return est_result, np.empty(n_tile_ghw, dtype=object), np.empty(in_shape)
    else:
        tl_optim_log.debug(
            f"Conv2d tiling success.\n"
            + f"\tEstimated optimal tile size: {conv2d_tile_size}, n_tile_ghw: {n_tile_ghw}.\n"
            + f"\tEstimated cost cores in total: {est_result.n_core}, lcn for each tile: {est_result.lcn.name}."
        )

    return (
        est_result,
        *conv2d_tile_by_tile_size(
            in_shape,
            (ho, wo),
            kernel,
            stride,
            padding,
            groups,
            conv2d_tile_size,
            n_tile_ghw,
        ),
    )


def count_copy_times_conv_by_tiles(
    conv_tiles: np.ndarray, in_shape: Union[Size2Type, Size3Type], groups: int
) -> np.ndarray:
    """Calculate the copy times for 1d/2d input feature map due to the overlap between adjacent tiles."""
    copt_times = np.zeros(in_shape, dtype=np.uint8).reshape(groups, -1, *in_shape[1:])

    for _, (i_blk_idx, _, _) in np.ndenumerate(conv_tiles):
        copt_times[i_blk_idx.to_slice()] += 1

    return copt_times


"""Used for testing only"""


def conv_in_tile_format(
    x: np.ndarray,
    out_shape: Union[Size2Type, Size3Type],
    groups: int,
    conv_tiles: np.ndarray,
) -> SynOutType:
    """Calculate the output of a 1d/2d convolution in tile format."""
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

    i_idx_n = np.prod(x.shape)
    o_idx_n = np.prod(out_shape)
    _assert_max_index(i_idx_n, o_idx_n)

    x_gchw = x.reshape(groups, ci_in_grp, *x.shape[1:])
    out = np.zeros((groups, co_in_grp, *out_shape[1:]), dtype=np.int64)

    for _, (i_blk_idx, o_blk_idx, k_tile_unrolled) in np.ndenumerate(conv_tiles):
        x = x_gchw[i_blk_idx.to_slice()]
        o = x.ravel().astype(np.int64) @ k_tile_unrolled

        out[o_blk_idx.to_slice()] = o.reshape(o_blk_idx.to_shape())

    return out.reshape(out_shape).astype(VOLTAGE_DTYPE)


def conv1d_unroll_tiled_by_tiles(
    in_shape: Size2Type,
    kernel: np.ndarray,
    stride: Size1Type,
    padding: Size1Type,
    groups: int,
    tile_size: Optional[TileSize2d] = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 1d convolution kernel by tiles. Return a generator of unrolled kernels."""
    ci, li = in_shape
    co, ci_in_grp, kl = kernel.shape
    out_shape = _conv1d_oshape((li,), (kl,), stride, padding)
    (lo,) = out_shape

    _group_ch_check(ci, co, groups, ci_in_grp)

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
            yield _conv1d_tile_unroll(
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
    tile_size: Optional[TileSize3d] = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 2d convolution kernel by tiles. Return a generator of unrolled kernels."""
    ci, hi, wi = in_shape
    co, ci_in_grp, kh, kw = kernel.shape
    out_shape = _conv2d_oshape((hi, wi), (kh, kw), stride, padding)
    ho, wo = out_shape

    _group_ch_check(ci, co, groups, ci_in_grp)

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
                yield _conv2d_tile_unroll(
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
    conv1d_tile_size: Optional[TileSize2d] = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 1d convolution kernel from the full unrolled kernel matrix. Return a generator  \
        of unrolled kernels.
    """
    ci, li = in_shape
    co, ci_in_grp, kl = kernel.shape
    out_shape = _conv1d_oshape((li,), (kl,), stride, padding)
    (lo,) = out_shape

    _group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if conv1d_tile_size is None:
        g_tl = groups
        lo_tl = lo
    else:
        g_tl, lo_tl = conv1d_tile_size

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

            li_start, li_end, _ = _eff_input_tile_size1d(
                (li,), (kl,), stride, padding, lo_tl_, lo_start
            )

            l = np.arange(li_start, li_end)
            input_idx = (ci[:, np.newaxis] * li + l[np.newaxis, :]).ravel()
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
    conv2d_tile_size: Optional[TileSize3d] = None,
) -> Generator[np.ndarray, None, None]:
    """Unroll the tiled 2d convolution kernel from the full unrolled kernel matrix. Return a generator  \
        of unrolled kernels.
    """
    ci, hi, wi = in_shape
    co, ci_in_grp, kh, kw = kernel.shape
    out_shape = _conv2d_oshape((hi, wi), (kh, kw), stride, padding)
    ho, wo = out_shape

    _group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if conv2d_tile_size is None:
        g_tl = groups
        ho_tl = ho
        wo_tl = wo
    else:
        g_tl, ho_tl, wo_tl = conv2d_tile_size

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

                (hi_start, hi_end), (wi_start, wi_end), _ = _eff_input_tile_size2d(
                    (hi, wi),
                    (kh, kw),
                    stride,
                    padding,
                    (ho_tl_, wo_tl_),
                    (ho_start, wo_start),
                )

                h = np.arange(hi_start, hi_end)
                w = np.arange(wi_start, wi_end)

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
