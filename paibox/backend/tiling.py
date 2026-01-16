import math
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import ClassVar, Literal, TypeVar, cast, overload

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from paicorelib import LCN_EX, HwConfig, OffCoreCfg, OnCoreCfg

from paibox import _logging
from paibox.components import Conv2d
from paibox.components.synapses.conv_types import (
    Size1Type,
    Size2Type,
    Size3Type,
    Size4Type,
    SizeAnyType,
    _Size1Type,
    _Size2Type,
    _Size4Type,
)
from paibox.components.synapses.conv_utils import (
    INDEX_DTYPE,
    _conv1d_oshape,
    _conv2d_oshape,
    _pair,
    _quadruple,
    _single,
    group_ch_check,
)
from paibox.types import Shape, WeightType
from paibox.utils import shape2num

from .kernel_unrolling import (
    conv1d_tiled_kernel_unroll,
    conv1d_tiled_kernel_unroll_no_pad,
    conv1d_tiled_kernel_unroll_no_pad_multi_grp,
    conv2d_tiled_kernel_unroll,
    conv2d_tiled_kernel_unroll_no_pad,
    conv2d_tiled_kernel_unroll_no_pad_multi_grp,
)

__all__ = [
    # Types
    "TileSize2d",
    "TileSize3d",
    "TileSlice2d",
    "TileSlice3d",
    "TileSliceConv1d",
    "TileSliceConv2d",
    "EstCoreCostStatus",
    "EstCoreCostResult",
    # Functions for estimating operator's cost
    "operator_core_cost_estimate",
    "optimal_lcn_matmul2d",
    "optimal_tiling_conv1d",
    "optimal_tiling_conv2d",
    # Functions for tiling
    "conv1d_tile_by_tile_size",
    "conv2d_tile_by_tile_size",
    "make_conv_tiled_idx_map",
    "make_output_conv_tiled_idx_map",
    "make_input_conv_tiled_idx_map",
    "make_conv1d_kernel_tiled_unrolled",
    "make_conv2d_kernel_tiled_unrolled",
    # Functions for optimizing tiling for conv
    "conv1d_tiling_optimize",
    "conv2d_tiling_optimize",
    "conv2d_optimize",
]

tl_optim_log = _logging.get_artifact_logger(__name__, "tiling_optim")

TileSize2d = Size2Type
TileSize3d = Size3Type


def _starts_ends_pair(starts: Sequence[int], lens: Sequence[int]) -> tuple[int, ...]:
    ends = tuple(s + l for s, l in zip(starts, lens))
    return tuple(x for pair in zip(starts, ends) for x in pair)


class TileSliceBase:
    pass


class TileSlice2d(TileSliceBase):
    ndim: ClassVar[int] = 2

    def __init__(self, x_start: int, x_end: int, y_start: int, y_end: int) -> None:
        assert x_start < x_end and y_start < y_end
        self.x = slice(x_start, x_end, 1)
        self.y = slice(y_start, y_end, 1)

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
    def size(self) -> int:
        return self.xlen * self.ylen

    @property
    def xlen(self) -> int:
        return self.x.stop - self.x.start

    @property
    def ylen(self) -> int:
        return self.y.stop - self.y.start

    @property
    def slice_range(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}"


class TileSlice3d(TileSliceBase):
    ndim: ClassVar[int] = 3

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
    def size(self) -> int:
        return self.xlen * self.ylen * self.zlen

    @property
    def xlen(self) -> int:
        return self.x.stop - self.x.start

    @property
    def ylen(self) -> int:
        return self.y.stop - self.y.start

    @property
    def zlen(self) -> int:
        return self.z.stop - self.z.start

    @property
    def slice_range(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}, z={self.z}"


SliceRangeNd = tuple[slice, ...]
SliceRange3d = tuple[slice, slice, slice]
SliceRange4d = tuple[slice, slice, slice, slice]


class TileSliceConv1d(TileSlice2d):
    def __init__(
        self,
        g_start: int,
        g_end: int,
        l_start: int,
        l_end: int,
        paddings: _Size2Type = 0,
    ) -> None:
        # Pad on the L dimension only
        pl, pr = _pair(paddings)
        self.paddings = ((0, 0), (pl, pr))

        super().__init__(g_start, g_end, l_start, l_end)

    @classmethod
    def make_tile_slice(
        cls, start_idx: Size2Type, side_lens: Size2Type, paddings: _Size2Type = 0
    ):
        return cls(*_starts_ends_pair(start_idx, side_lens), paddings=paddings)

    @property
    def shape_ext(self) -> Size3Type:
        """Extended shape g, c(=-1), l."""
        return (self.xlen, -1, self.ylen)

    @property
    def slice_range(self) -> SliceRange3d:
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
        paddings: _Size4Type = 0,
    ) -> None:
        pt, pb, pl, pr = _quadruple(paddings)
        self.paddings = ((0, 0), (pt, pb), (pl, pr))

        super().__init__(g_start, g_end, h_start, h_end, w_start, w_end)

    @classmethod
    def make_tile_slice(
        cls, start_idx: Size3Type, side_lens: Size3Type, paddings: _Size4Type = 0
    ):
        return cls(*_starts_ends_pair(start_idx, side_lens), paddings=paddings)

    @property
    def shape_ext(self) -> Size4Type:
        """Extended shape g, c(=-1), h, w."""
        return (self.xlen, -1, self.ylen, self.zlen)

    @property
    def slice_range(self) -> SliceRange4d:
        return (self.x, slice(None), self.y, self.z)


@unique
class EstCoreCostStatus(IntEnum):
    """Status of estimating core cost."""

    SUCCESS = 0
    FAN_IN_TOO_LARGE = -1
    """Too large fan-in for the computing cores."""
    CORES_MORE_THAN_ONE_CHIP = -2
    """Required cores more than one chip, impossible to be multicasted."""
    FAILED = -99


@dataclass
class EstCoreCostResult:
    status: EstCoreCostStatus
    n_core: int
    lcn: LCN_EX


EstStatus = EstCoreCostStatus
EstResult = EstCoreCostResult


def optimal_lcn_matmul2d(
    shape_a: Size2Type,
    shape_b: Size2Type,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
) -> EstResult:
    """Estimate the optimal LCN for 2d matmul tiling. A 2d matmul `Y = A(m*n) * B(n*k)` can be viewed as:
        [Y1, Y2, ..., Ym]^T = [A1, A2, ..., An]^T @ B

        where B is the patch matrix and `m` is the number of patches. `m` B matrix is arranged diagonally.

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores & LCN.
    """
    assert shape_a[1] == shape_b[0]

    est_result = EstResult(EstStatus.FAILED, 0, LCN_EX.LCN_1X)
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
        #     est_result.status = EstStatus.CORES_MORE_THAN_ONE_CHIP
        #     break

        # Get the #N of tiles to allocate for this matmul
        n_tl_ghw, n_patch_remain = divmod(n_patch, n_patch_in_tile)

        n_core = n_core_tile * n_tl_ghw
        if n_patch_remain > 0:
            n_core += math.ceil(n_patch_remain * n_fanout_patch / fout_capacity)

        tl_optim_log.debug(f"Current lcn: {lcn}, estimated cores: {n_core}")

        if n_core > HwConfig.N_CORE_OFFLINE:
            # Exceeds the maximum number of cores in a single chip.
            continue

        # If this is the first successful estimation or the total number of cores is less than
        # the previous one, update the result.
        if est_result.status == EstStatus.FAILED or n_core < est_result.n_core:
            est_result.status = EstStatus.SUCCESS
            est_result.n_core = n_core
            est_result.lcn = lcn

    return est_result


def operator_core_cost_estimate(
    in_shape: Shape,
    out_shape: Shape,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
) -> EstResult:
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
        return EstResult(EstStatus.FAN_IN_TOO_LARGE, 0, LCN_EX.LCN_64X)

    est_n_core = math.ceil(n_fanout / (core_n_fanout_base >> _lcn))
    if est_n_core > HwConfig.N_CORE_OFFLINE:
        return EstResult(EstStatus.CORES_MORE_THAN_ONE_CHIP, est_n_core, LCN_EX(_lcn))

    return EstResult(EstStatus.SUCCESS, est_n_core, LCN_EX(_lcn))


def _eff_step(start: int, step: int, len: int) -> int:
    """Return the effective step."""
    if start + step <= len:
        return step
    else:
        return len - start


def _eff_tile_size2d(
    tile_start_idx: Size2Type, o_tile_size: TileSize2d, range: Size2Type
) -> TileSize2d:
    g, l = map(lambda args: _eff_step(*args), zip(tile_start_idx, o_tile_size, range))
    return g, l


def _eff_tile_size3d(
    tile_start_idx: Size3Type, o_tile_size: TileSize3d, range: Size3Type
) -> TileSize3d:
    g, h, w = map(
        lambda args: _eff_step(*args), zip(tile_start_idx, o_tile_size, range)
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


def _i_tile_hw_conv1d(o_tl_size_gl: int, ksize: Size1Type, stride: Size1Type) -> int:
    """Compute the input tile size by given the output tile size, kernel size & stride."""
    return _conv_len_o2i(o_tl_size_gl, ksize[0], stride[0])


def _i_tile_hw_conv2d(
    o_tl_size_ghw: Size2Type, ksize: Size2Type, stride: Size2Type
) -> Size2Type:
    """Compute the input tile size by given the output tile size, kernel size & stride."""
    hi_tl, wi_tl = map(
        lambda args: _conv_len_o2i(*args), zip(o_tl_size_ghw, ksize, stride)
    )
    return hi_tl, wi_tl


def _i_tile_hw_conv(
    o_tile_size: SizeAnyType, ksize: SizeAnyType, stride: SizeAnyType
) -> SizeAnyType:
    """Compute the input tile size by given the output tile size, kernel size & stride."""
    assert len(o_tile_size) == len(ksize) == len(stride)
    return tuple(
        map(lambda args: _conv_len_o2i(*args), zip(o_tile_size, ksize, stride))
    )


def _eff_input_tile_size1d(
    in_shape_l: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
    o_tile_l: int,
    o_tile_l_start: int,
) -> tuple[int, int, Size2Type]:
    li = in_shape_l[0]

    _li_start = _input_pos_conv1d(o_tile_l_start, stride, padding)
    _li_tl = _i_tile_hw_conv1d(o_tile_l, ksize, stride)
    _li_end = _li_start + _li_tl

    pl = max(-_li_start, 0)
    pr = max(_li_end - li, 0)

    li_start = max(_li_start, 0)

    # Actual block size without padding
    li_tl = _li_tl - pl - pr

    return li_tl, li_start, (pl, pr)


def _eff_input_tile_size2d(
    in_shape_hw: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    o_tile_hw: Size2Type,
    o_tile_start_hw: Size2Type,
) -> tuple[Size2Type, Size2Type, Size4Type]:
    hi, wi = in_shape_hw

    _hi_start, _wi_start = _input_pos_conv2d(o_tile_start_hw, stride, padding)
    _hi_tl, _wi_tl = _i_tile_hw_conv2d(o_tile_hw, ksize, stride)
    _hi_end = _hi_start + _hi_tl
    _wi_end = _wi_start + _wi_tl

    pt = max(-_hi_start, 0)
    pb = max(_hi_end - hi, 0)
    pl = max(-_wi_start, 0)
    pr = max(_wi_end - wi, 0)

    hi_start = max(_hi_start, 0)
    wi_start = max(_wi_start, 0)

    # Actual block size without padding
    hi_tl = _hi_tl - pt - pb
    wi_tl = _wi_tl - pl - pr

    return (hi_tl, wi_tl), (hi_start, wi_start), (pt, pb, pl, pr)


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


def get_factors(n: int) -> list[int]:
    factors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)

    return sorted(factors)


def tile_size_ghw_o2i(
    o_tl_size_ghw: TileSize3d, ksize: Size2Type, stride: Size2Type
) -> TileSize3d:
    return (o_tl_size_ghw[0],) + _i_tile_hw_conv2d(o_tl_size_ghw[1:], ksize, stride)


def tile_size_o_gl2i_cl(
    o_tl_size_gl: TileSize2d, ci_in_grp: int, ksize: Size1Type, stride: Size1Type
) -> TileSize2d:
    g_tl = o_tl_size_gl[0]
    li_tl = _i_tile_hw_conv1d(o_tl_size_gl[-1], ksize, stride)
    return (g_tl * ci_in_grp, li_tl)


def tile_size_o_ghw2i_chw(
    o_tl_size_ghw: TileSize3d, ci_in_grp: int, ksize: Size2Type, stride: Size2Type
) -> TileSize3d:
    g_tl = o_tl_size_ghw[0]
    hi_tl, wi_tl = _i_tile_hw_conv2d(o_tl_size_ghw[1:], ksize, stride)
    return (g_tl * ci_in_grp, hi_tl, wi_tl)


TST = TypeVar("TST", TileSize2d, TileSize3d)


def tile_size_gl2cl(tl_slice: TST, c_in_grp: int) -> TST:
    return (tl_slice[0] * c_in_grp,) + tl_slice[1:]


def tile_size_ghw2chw(tl_slice: TST, c_in_grp: int) -> TST:
    return tile_size_gl2cl(tl_slice, c_in_grp)


def optimal_tiling_conv1d(
    in_shape: Size2Type,
    out_shape: Size2Type,
    ksize: Size1Type,
    stride: Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    traverse_order: Literal["all", "even"] = "all",
) -> tuple[EstResult, TileSize2d, TileSize2d]:
    """Tile the 1d convolution & find the optimal block size of the output feature map.

    Args:
        core_n_fanin_base (int): the base fan-in of a core.
        core_n_fanout_base (int): the base fan-out of a core.
        traverse_order ("all" or "even"): the order to traverse the possible optimal output feature map.\
            Default is "all".

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores \
            & LCN.
        best_tl_size_gl (Size2Type): the optimal block size on dimensions g, l.
        best_n_tl_gl (Size2Type): the number of blocks in each dimension of g, l.
    """
    ci, _ = in_shape
    co, lo = out_shape

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    est_result = EstResult(EstStatus.FAILED, 0, LCN_EX.LCN_1X)

    # Tile the conv1d in g, l dimensions (channels will be followed by groups)
    best_n_tl_gl = (0, 0)
    best_tl_size_gl = (0, 0)

    # NOTE: Make sure candidate `g_tl` is a factor of `groups`
    for g_tl in get_factors(groups):
        n_tl_g = math.ceil(groups / g_tl)

        for lo_tl in _traverse_tile_size1d(lo, traverse_order):
            o_tl_size_gl = (g_tl, lo_tl)
            o_tl_size_gcl = tile_size_gl2cl(o_tl_size_gl, co_in_grp)
            i_tl_size_gcl = tile_size_o_gl2i_cl(o_tl_size_gcl, ci_in_grp, ksize, stride)

            est = operator_core_cost_estimate(
                i_tl_size_gcl, o_tl_size_gcl, core_n_fanin_base, core_n_fanout_base
            )
            if est.status != EstStatus.SUCCESS:
                continue

            # Even if there are incomplete tile, treat them as a `o_tl_size_ghw` block for calculating
            # resource costs.
            n_tl_l = math.ceil(lo / lo_tl)
            n_tl_gl = (n_tl_g, n_tl_l)

            total_n_core = est.n_core * shape2num(n_tl_gl)
            if total_n_core > HwConfig.N_CORE_OFFLINE:
                continue

            # If this is the first successful estimation or the total number of cores is less than
            # the previous one, update the result.
            if (
                est_result.status == EstStatus.FAILED
                or total_n_core < est_result.n_core
            ):
                est_result.status = est.status
                est_result.n_core = total_n_core
                est_result.lcn = est.lcn
                best_n_tl_gl = n_tl_gl
                best_tl_size_gl = o_tl_size_gl

            # TODO After finding an available tile size, we should also consider the
            # #N of cores for copying the overlap between the adjacent input tiles.

    return est_result, best_tl_size_gl, best_n_tl_gl


def optimal_tiling_conv2d(
    in_shape: Size3Type,
    out_shape: Size3Type,
    ksize: Size2Type,
    stride: Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    traverse_order: Literal["all", "Lshape", "even"] = "all",
) -> tuple[EstResult, TileSize3d, TileSize3d]:
    """Tile the 2d convolution & find the optimal block size of the output feature map.

    Args:
        core_n_fanin_base (int): the base fan-in of a core.
        core_n_fanout_base (int): the base fan-out of a core.
        traverse_order ("all", "Lshape" or "even"): the order to traverse the possible optimal output   \
            feature map. Default is "all".

    Returns:
        est_result (EstCoreCostResult): the result of the estimation, including the status, #N of cores \
            & LCN.
        tile_size_ghw (Size3Type): the block size on dimensions g, h, w.
        n_tl_ghw (Size3Type): the number of blocks in each dimension of g, h, w.
    """
    ci, _, _ = in_shape
    co, ho, wo = out_shape

    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    est_result = EstResult(EstStatus.FAILED, 0, LCN_EX.LCN_1X)

    # Tile the conv2d in g, h, w dimensions (channels will be followed by groups)
    best_n_tl_ghw = (0, 0, 0)
    best_tl_size_ghw = (0, 0, 0)

    # NOTE: Make sure candidate `g_tl` is a factor of `groups`
    for g_tl in get_factors(groups):
        n_tl_g = math.ceil(groups / g_tl)

        for ho_tl, wo_tl in _traverse_tile_size2d(ho, wo, traverse_order):
            o_tl_size_ghw = (g_tl, ho_tl, wo_tl)
            o_tl_size_chw = tile_size_ghw2chw(o_tl_size_ghw, co_in_grp)
            i_tl_size_chw = tile_size_o_ghw2i_chw(
                o_tl_size_ghw, ci_in_grp, ksize, stride
            )

            est = operator_core_cost_estimate(
                i_tl_size_chw, o_tl_size_chw, core_n_fanin_base, core_n_fanout_base
            )
            if est.status != EstStatus.SUCCESS:
                continue

            # Even if there are incomplete tile, treat them as a `o_tl_size_ghw` block for calculating
            # resource costs.
            n_tl_h = math.ceil(ho / ho_tl)
            n_tl_w = math.ceil(wo / wo_tl)
            n_tl_ghw = (n_tl_g, n_tl_h, n_tl_w)

            total_n_core = int(est.n_core * np.prod(n_tl_ghw))
            if total_n_core > HwConfig.N_CORE_OFFLINE:
                continue

            # If this is the first successful estimation or the total number of cores is less than
            # the previous one, update the result.
            if (
                est_result.status == EstStatus.FAILED
                or total_n_core < est_result.n_core
            ):
                est_result.status = est.status
                est_result.n_core = total_n_core
                est_result.lcn = est.lcn
                best_n_tl_ghw = n_tl_ghw
                best_tl_size_ghw = o_tl_size_ghw

            # TODO After finding an available tile size, we should also consider the
            # #N of cores for copying the overlap between the adjacent input tiles.

    return est_result, best_tl_size_ghw, best_n_tl_ghw


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


def _conv1d_o_tile_visitor(
    n_tl_ghw: TileSize2d, tile_size2d: TileSize2d, groups: int, lo: int
) -> Generator[tuple[TileSize2d, TileSize2d, TileSize2d], None, None]:
    """Visit all tiles in order of g, l & yield the starting indices & tile sizes of each tile."""
    n_g, n_l = n_tl_ghw

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


def _conv2d_o_tile_visitor(
    n_tile: TileSize3d, tile_size3d: TileSize3d, groups: int, ho: int, wo: int
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
                yield (
                    (g_idx, ho_idx, wo_idx),
                    (g_start, ho_start, wo_start),
                    (
                        g_tl,
                        ho_tl,
                        wo_tl,
                    ),
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
    o_tile_size: TileSize2d,
    n_tl_ghw: TileSize2d,
) -> tuple[np.ndarray, np.ndarray]:
    ci, li = in_shape
    (lo,) = out_shape_l
    _, _, kl = kernel.shape
    ksize = (kl,)

    ci_in_grp = ci // groups
    tiles = np.zeros(n_tl_ghw, dtype=object)
    copy_times = np.zeros((groups, ci_in_grp, li), dtype=np.uint8)

    for tl_idx, (g_start, lo_start), (g_tl, lo_tl) in _conv1d_o_tile_visitor(
        n_tl_ghw, o_tile_size, groups, lo
    ):
        li_tl, li_start, paddings = _eff_input_tile_size1d(
            (li,), ksize, stride, padding, lo_tl, lo_start
        )
        _tile_range2d_check(g_tl, lo_tl, g_start, lo_start, groups, lo)
        _tile_range2d_check(g_tl, li_tl, g_start, li_start, groups, li)
        # Extract input & output tile indices
        i_tl_slice = TileSliceConv1d(
            g_start, g_start + g_tl, li_start, li_start + li_tl
        )
        o_tl_slice = TileSliceConv1d(
            g_start, g_start + g_tl, lo_start, lo_start + lo_tl
        )
        # Get unrolled kernel for each tile
        k_tl_unrolled = conv1d_tiled_kernel_unroll(
            kernel, stride, groups, g_start, g_tl, (li_tl,), (lo_tl,), paddings
        )
        tiles[tl_idx] = (i_tl_slice, o_tl_slice, k_tl_unrolled)
        copy_times[i_tl_slice.slice_range] += 1

    return tiles, copy_times.reshape(in_shape)


def conv2d_tile_by_tile_size(
    in_shape: Size3Type,
    out_shape_hw: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int,
    o_tile_size: TileSize3d,
    n_tl_ghw: TileSize3d,
) -> tuple[np.ndarray, np.ndarray]:
    _, hi, wi = in_shape
    ho, wo = out_shape_hw
    _, ci_in_grp, kh, kw = kernel.shape
    ksize = (kh, kw)

    tiles = np.zeros(n_tl_ghw, dtype=object)
    copy_times = np.zeros((groups, ci_in_grp, hi, wi), dtype=np.uint8)

    for (
        tl_idx,
        (g_start, ho_start, wo_start),
        (g_tl, ho_tl, wo_tl),
    ) in _conv2d_o_tile_visitor(n_tl_ghw, o_tile_size, groups, wo, wo):
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
        i_tl_slice = TileSliceConv2d(
            g_start,
            g_start + g_tl,
            hi_start,
            hi_start + hi_tl,
            wi_start,
            wi_start + wi_tl,
        )
        o_tl_slice = TileSliceConv2d(
            g_start,
            g_start + g_tl,
            ho_start,
            ho_start + ho_tl,
            wo_start,
            wo_start + wo_tl,
        )
        # Get unrolled kernel for each tile
        k_tl_unrolled = conv2d_tiled_kernel_unroll(
            kernel,
            stride,
            groups,
            g_start,
            g_tl,
            (hi_tl, wi_tl),
            (ho_tl, wo_tl),
            paddings,
        )
        tiles[tl_idx] = (i_tl_slice, o_tl_slice, k_tl_unrolled)
        copy_times[i_tl_slice.slice_range] += 1

    return tiles, copy_times.reshape(in_shape)


INVALID_ADDR_IDX = -1


def _invalid_addr_idx_value(zero_as_invalid_addr: bool):
    return 0 if zero_as_invalid_addr else INVALID_ADDR_IDX


INDEX_DTYPE_WITH_INVALID = np.int32  # -1 as invalid idx
INDEX_DTYPE_ZERO_AS_INVALID = INDEX_DTYPE

# For conv1d, index map array is in shape:
#   (#N of tiles in C-dim, #N of tiles in L-dim, C in tile, L in tile)
# For conv2d, index map array is in shape:
#   (#N of tiles in C-dim, #N of tiles in H-dim, #N of tiles in W-dim, C in tile, H in tile, W in tile)
IndexMapArrayType = NDArray[INDEX_DTYPE_WITH_INVALID | INDEX_DTYPE_ZERO_AS_INVALID]


def _cast_size2type(shape: SizeAnyType) -> Size2Type:
    return cast(Size2Type, shape)


def _cast_size3type(shape: SizeAnyType) -> Size3Type:
    return cast(Size3Type, shape)


def creat_index_map(
    shape: SizeAnyType, zero_as_invalid_addr: bool = False
) -> IndexMapArrayType:
    if zero_as_invalid_addr:
        return np.arange(
            1, np.prod(shape) + 1, dtype=INDEX_DTYPE_ZERO_AS_INVALID
        ).reshape(shape)
    else:
        return np.arange(np.prod(shape), dtype=INDEX_DTYPE_WITH_INVALID).reshape(shape)


def create_idx_map_with_pad(
    shape: SizeAnyType,
    tl_padding: SizeAnyType,
    conv_padding: SizeAnyType | None = None,
    zero_as_invalid_addr: bool = False,
) -> IndexMapArrayType:
    """Create an index map with tiling padding & conv padding."""
    assert len(shape) == len(tl_padding)
    if conv_padding is None:
        conv_p = (0,) * len(shape)
    else:
        conv_p = (0,) + conv_padding
        assert len(shape) == len(conv_p)

    idx_map = creat_index_map(shape, zero_as_invalid_addr)
    padding = tuple((p1, p1 + p2) for p1, p2 in zip(conv_p, tl_padding))

    mask_value = _invalid_addr_idx_value(zero_as_invalid_addr)
    return np.pad(idx_map, padding, constant_values=mask_value)


def _o_divisible_tiling_pad(
    shape: SizeAnyType, inner_shape: SizeAnyType
) -> SizeAnyType:
    assert len(shape) == len(inner_shape)
    # Pad in +C, +H, +W directions to make the shape divisible by the tile shape
    return tuple(math.ceil(s / t) * t - s for s, t in zip(shape, inner_shape))


def _i_divisible_tiling_pad(
    shape: SizeAnyType,
    inner_shape: SizeAnyType,
    tile_shape: SizeAnyType,
    ksize: SizeAnyType,
    stride: SizeAnyType,
    padding: SizeAnyType,
) -> SizeAnyType:
    assert len(shape) == len(inner_shape)
    assert len(shape) - 1 == len(ksize) == len(stride) == len(padding)

    tl_pad_c = 0  # XXX Now, assume that no pad on C dimension

    return (tl_pad_c,) + tuple(
        max((tl * i - (tl - 1) * (k - s) - (l + 2 * p)), 0)
        for tl, i, k, s, l, p in zip(
            tile_shape[1:], inner_shape[1:], ksize, stride, shape[1:], padding
        )
    )


def _tile_range_check(shape: SizeAnyType, inner_shape: SizeAnyType) -> None:
    assert len(shape) == len(inner_shape)
    assert all(s >= t for s, t in zip(shape, inner_shape))


def get_tile_shape(shape: SizeAnyType, inner_shape: SizeAnyType) -> SizeAnyType:
    """Return the number of tiles for each dimension."""
    assert len(shape) == len(inner_shape)
    # Pad in +X, +Y, +Z directions to make the shape divisible by the tile shape
    return tuple(math.ceil(s / t) for s, t in zip(shape, inner_shape))


def make_output_conv_tiled_idx_map(
    shape: SizeAnyType, inner_shape: SizeAnyType, zero_as_invalid_addr: bool = False
) -> IndexMapArrayType:
    """Tile the output index map for a convolution.

    Return:
        The tiled output index map is in shape:
        - conv1d: (#N of tiles in C-dim, #N of tiles in L-dim, C in tile, L in tile).
        - conv2d: (#N of tiles in C-dim, #N of tiles in H-dim, #N of tiles in W-dim, C in tile, H in    \
            tile, W in tile).

    Example:
        A feature map of shape (8, 8) is tiled with an inner shape(or tile size) of (4, 4). (2, 2) is   \
            the tile shape.
    """
    _tile_range_check(shape, inner_shape)
    tl_padding = _o_divisible_tiling_pad(shape, inner_shape)

    tl_optim_log.debug(
        f"Output feature map shape: {shape}\n"
        + f"Inner shape: {inner_shape}\n"
        + f"Tiling padding: {tl_padding}\n"
        + f"Index map windows stride: {inner_shape}"
    )

    idx_map = create_idx_map_with_pad(shape, tl_padding, None, zero_as_invalid_addr)
    strides = tuple(slice(None, None, st) for st in inner_shape)
    return sliding_window_view(idx_map, inner_shape)[strides]


def make_input_conv_tiled_idx_map(
    shape: SizeAnyType,
    o_inner_shape: SizeAnyType,
    tile_shape: SizeAnyType,
    i_inner_ch: int,
    ksize: SizeAnyType,
    stride: SizeAnyType,
    padding: SizeAnyType,
    zero_as_invalid_addr: bool = False,
) -> IndexMapArrayType:
    """Tile the input index map for a convolution.

    Return:
        The tiled input index map is in shape:
        - conv1d: (#N of tiles in C-dim, #N of tiles in L-dim, C in tile, L in tile).
        - conv2d: (#N of tiles in C-dim, #N of tiles in H-dim, #N of tiles in W-dim, C in tile, H in tile, W in tile).
    """
    # output inner shape hw -> input inner shape hw
    inner_shape = (i_inner_ch,) + _i_tile_hw_conv(o_inner_shape[1:], ksize, stride)
    _tile_range_check(shape, inner_shape)

    tl_padding = _i_divisible_tiling_pad(
        shape, inner_shape, tile_shape, ksize, stride, padding
    )

    idx_map_windows_stride = (i_inner_ch,) + tuple(
        o * s for o, s in zip(o_inner_shape[1:], stride)
    )

    tl_optim_log.debug(
        f"Input feature map shape: {shape}\n"
        + f"Inner shape: {inner_shape}\n"
        + f"Conv padding: {padding}\n"
        + f"Tiling padding: {tl_padding}\n"
        + f"Index map windows stride: {idx_map_windows_stride}"
    )

    idx_map = create_idx_map_with_pad(shape, tl_padding, padding, zero_as_invalid_addr)
    strides = tuple(slice(None, None, st) for st in idx_map_windows_stride)
    return sliding_window_view(idx_map, inner_shape)[strides]


@overload
def make_conv_tiled_idx_map(
    in_shape: Size2Type,
    out_shape: Size2Type,
    o_inner_shape: Size2Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
    groups: int = 1,
    zero_as_invalid_addr: bool = False,
) -> tuple[IndexMapArrayType, IndexMapArrayType, NDArray[np.intp]]: ...


@overload
def make_conv_tiled_idx_map(
    in_shape: Size3Type,
    out_shape: Size3Type,
    o_inner_shape: Size3Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
    zero_as_invalid_addr: bool = False,
) -> tuple[IndexMapArrayType, IndexMapArrayType, NDArray[np.intp]]: ...


def make_conv_tiled_idx_map(
    in_shape: Size2Type | Size3Type,
    out_shape: Size2Type | Size3Type,
    o_inner_shape: Size2Type | Size3Type,
    ksize: Size1Type | Size2Type,
    stride: Size1Type | Size2Type,
    padding: Size1Type | Size2Type,
    groups: int = 1,
    zero_as_invalid_addr: bool = False,
) -> tuple[IndexMapArrayType, IndexMapArrayType, NDArray[np.intp]]:
    assert len(in_shape) == len(out_shape)
    assert len(ksize) == len(in_shape) - 1

    ci = in_shape[0]
    co = out_shape[0]
    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci}, {co} must be divisible by groups {groups}."
    ci_in_grp = ci // groups
    co_in_grp = co // groups

    # Calculate the #N of groups inside the tile
    g_tl = o_inner_shape[0]
    assert (
        groups % g_tl == 0
    ), f"The #N of groups inside the tile {g_tl} must be divisible by the #N of groups {groups}."
    i_inner_ch = ci_in_grp * g_tl

    # Output inner shape: ghw -> chw or gl -> cl
    o_inner_shape = (co_in_grp * g_tl,) + o_inner_shape[1:]
    tl_shape = get_tile_shape(out_shape, o_inner_shape)

    o_tiled_idx_map = make_output_conv_tiled_idx_map(
        out_shape, o_inner_shape, zero_as_invalid_addr
    )
    i_tiled_idx_map = make_input_conv_tiled_idx_map(
        in_shape,
        o_inner_shape,
        tl_shape,
        i_inner_ch,
        ksize,
        stride,
        padding,
        zero_as_invalid_addr,
    )

    # Record the #N of copies for each pixel in the input feature map
    copy_times = get_tiled_conv_copy_times(
        i_tiled_idx_map, in_shape, zero_as_invalid_addr
    )

    return i_tiled_idx_map, o_tiled_idx_map, copy_times


def make_conv1d_kernel_tiled_unrolled(
    kernel: WeightType,
    stride: Size1Type,
    groups: int,
    g_tile: int,
    tile_shape: Size2Type,
    i_tile_size: Size2Type,
    o_tile_size: Size2Type,
) -> WeightType:
    """Unroll the 1d convolution kernel in tiles."""
    if groups == 1:
        return conv1d_tiled_kernel_unroll_no_pad(
            kernel, stride, i_tile_size[1:], o_tile_size[1:]
        )

    tl_unrolled_shape = (np.prod(i_tile_size), np.prod(o_tile_size))
    k_tiles = np.zeros(tile_shape + tl_unrolled_shape, dtype=kernel.dtype)

    for tl_idx in np.ndindex(tile_shape):
        g_start = tl_idx[0] * g_tile
        k_tiles[tl_idx] = conv1d_tiled_kernel_unroll_no_pad_multi_grp(
            kernel, stride, groups, g_start, g_tile, i_tile_size[1:], o_tile_size[1:]
        )

    return k_tiles


def make_conv2d_kernel_tiled_unrolled(
    kernel: WeightType,
    stride: Size2Type,
    groups: int,
    g_tile: int,
    tile_shape: Size3Type,
    i_tile_size: Size3Type,
    o_tile_size: Size3Type,
) -> WeightType:
    """Unroll the 2d convolution kernel in tiles."""
    if groups == 1:
        return conv2d_tiled_kernel_unroll_no_pad(
            kernel, stride, i_tile_size[1:], o_tile_size[1:]
        )

    tl_unrolled_shape = (np.prod(i_tile_size), np.prod(o_tile_size))
    k_tiles = np.zeros(tile_shape + tl_unrolled_shape, dtype=kernel.dtype)

    for tl_idx in np.ndindex(tile_shape):
        g_start = tl_idx[0] * g_tile
        k_tiles[tl_idx] = conv2d_tiled_kernel_unroll_no_pad_multi_grp(
            kernel, stride, groups, g_start, g_tile, i_tile_size[1:], o_tile_size[1:]
        )

    return k_tiles


def compact_and_flatten_idx_map(
    idx_map: IndexMapArrayType, zero_as_invalid_addr: bool = False
):
    """Remove the invalid values from the index map & return a flattened array."""
    if zero_as_invalid_addr:
        return idx_map[idx_map > 0] - 1
    else:
        return idx_map[idx_map > INVALID_ADDR_IDX].astype(INDEX_DTYPE_ZERO_AS_INVALID)


def get_tiled_conv_copy_times(
    idx_map: IndexMapArrayType,
    in_shape: Size2Type | Size3Type,
    zero_as_invalid_addr: bool = False,
) -> NDArray[np.intp]:
    valid_addr = compact_and_flatten_idx_map(idx_map, zero_as_invalid_addr)
    return np.bincount(valid_addr, minlength=np.prod(in_shape)).reshape(in_shape)


@overload
def conv1d_tiling_optimize(
    in_shape: Size2Type,
    kernel: WeightType,
    stride: _Size1Type,
    padding: _Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size2Type | None = None,
    traverse_order: Literal["all", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: Literal[True] = True,
) -> tuple[EstResult, np.ndarray, np.ndarray]: ...


@overload
def conv1d_tiling_optimize(
    in_shape: Size2Type,
    kernel: WeightType,
    stride: _Size1Type,
    padding: _Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size2Type | None = None,
    traverse_order: Literal["all", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: Literal[False] = False,
) -> tuple[
    EstResult, IndexMapArrayType, IndexMapArrayType, WeightType, NDArray[np.intp]
]: ...


def conv1d_tiling_optimize(
    in_shape: Size2Type,
    kernel: WeightType,
    stride: _Size1Type,
    padding: _Size1Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size2Type | None = None,
    traverse_order: Literal["all", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: bool = False,
):
    """Tile the 1d convolution by the optimal tile size.

    Args:
        core_n_fanin_base (int): the base #N of cores for the input feature map.
        core_n_fanout_base (int): the base #N of cores for the output feature map.
        out_shape (Size2Type, optional): the shape of output feature map.
        traverse_order ("all" or "even"): the order to traverse the possible optimal output \
            feature map. Default is "all".
        zero_as_invalid_addr (bool): whether to use 0 or `INVALID_ADDR_IDX` to represent    \
            invalid addresses.
        compact_idx_map (bool): whether to return the index maps in compact format or not.  \
            Default is false.
    """
    _, li = in_shape
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

    est_result, o_tl_size, n_tl_gl = optimal_tiling_conv1d(
        in_shape,
        out_shape,
        ksize,
        stride,
        groups,
        core_n_fanin_base,
        core_n_fanout_base,
        traverse_order,
    )
    if est_result.status != EstStatus.SUCCESS:
        tl_optim_log.debug(
            "Failed to find a valid tiling. Maybe the operator is too large to fit in."
        )
        if compact_idx_map:
            return est_result, np.empty(n_tl_gl, dtype=object), np.empty(in_shape)
        else:
            return (
                est_result,
                np.empty(0, dtype=INDEX_DTYPE_ZERO_AS_INVALID),
                np.empty(0, dtype=INDEX_DTYPE_ZERO_AS_INVALID),
                np.empty(0, dtype=kernel.dtype),
                np.empty(0, dtype=np.intp),
            )
    else:
        tl_optim_log.debug(
            "Conv1d tiling success.\n"
            + f"\tEstimated optimal tile size: {o_tl_size}, n_tl_gl: {n_tl_gl}.\n"
            + f"\tEstimated cost cores in total: {est_result.n_core}, "
            + f"lcn for each tile: {est_result.lcn.name}."
        )

    if compact_idx_map:
        return (
            est_result,
            *conv1d_tile_by_tile_size(
                in_shape, (lo,), kernel, stride, padding, groups, o_tl_size, n_tl_gl
            ),
        )
    else:
        i_tiled_idx_map, o_tiled_idx_map, copy_times = make_conv_tiled_idx_map(
            in_shape,
            out_shape,
            o_tl_size,
            ksize,
            stride,
            padding,
            groups,
            zero_as_invalid_addr,
        )
        assert n_tl_gl == i_tiled_idx_map.shape[:2]

        tl_shape = _cast_size2type(i_tiled_idx_map.shape[:2])
        # #N of channels in a tile / #N of channels in a group -> #N of groups in a tile
        g_tl = i_tiled_idx_map.shape[2] // ci_in_grp
        i_tl_size = _cast_size2type(i_tiled_idx_map.shape[-2:])
        o_tl_size = _cast_size2type(o_tiled_idx_map.shape[-2:])

        k_tiles = make_conv1d_kernel_tiled_unrolled(
            kernel, stride, groups, g_tl, tl_shape, i_tl_size, o_tl_size
        )

        return (est_result, i_tiled_idx_map, o_tiled_idx_map, k_tiles, copy_times)


@overload
def conv2d_tiling_optimize(
    in_shape: Size3Type,
    kernel: WeightType,
    stride: _Size2Type,
    padding: _Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size3Type | None = None,
    traverse_order: Literal["all", "Lshape", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: Literal[True] = True,
) -> tuple[EstResult, np.ndarray, np.ndarray]: ...


@overload
def conv2d_tiling_optimize(
    in_shape: Size3Type,
    kernel: WeightType,
    stride: _Size2Type,
    padding: _Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size3Type | None = None,
    traverse_order: Literal["all", "Lshape", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: Literal[False] = False,
) -> tuple[
    EstResult, IndexMapArrayType, IndexMapArrayType, WeightType, NDArray[np.intp]
]: ...


def conv2d_tiling_optimize(
    in_shape: Size3Type,
    kernel: WeightType,
    stride: _Size2Type,
    padding: _Size2Type,
    groups: int,
    core_n_fanin_base: int,
    core_n_fanout_base: int,
    out_shape: Size3Type | None = None,
    traverse_order: Literal["all", "Lshape", "even"] = "all",
    zero_as_invalid_addr: bool = False,
    compact_idx_map: bool = False,
):
    """Tile the 2d convolution by the optimal tile size.

    Args:
        core_n_fanin_base (int): the base #N of cores for the input feature map.
        core_n_fanout_base (int): the base #N of cores for the output feature map.
        out_shape (Size3Type, optional): the shape of output feature map.
        traverse_order ("all", "Lshape" or "even"): the order to traverse the possible      \
            optimal output feature map. Default is "all".
        zero_as_invalid_addr (bool): whether to use 0 or `INVALID_ADDR_IDX` to represent    \
            invalid addresses.
        compact_idx_map (bool): whether to return the index maps in compact format or not.  \
            Default is false.
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

    group_ch_check(ci, co, groups, ci_in_grp)

    est_result, o_tl_size, n_tl_ghw = optimal_tiling_conv2d(
        in_shape,
        out_shape,
        ksize,
        stride,
        groups,
        core_n_fanin_base,
        core_n_fanout_base,
        traverse_order,
    )
    if est_result.status != EstStatus.SUCCESS:
        tl_optim_log.debug(
            "Failed to find a valid tiling. Maybe the operator is too large to fit in."
        )
        if compact_idx_map:
            return est_result, np.empty(n_tl_ghw, dtype=object), np.empty(in_shape)
        else:
            return (
                est_result,
                np.empty(0, dtype=INDEX_DTYPE_ZERO_AS_INVALID),
                np.empty(0, dtype=INDEX_DTYPE_ZERO_AS_INVALID),
                np.empty(0, dtype=kernel.dtype),
                np.empty(0, dtype=np.intp),
            )
    else:
        tl_optim_log.debug(
            "Conv2d tiling success.\n"
            + f"\tEstimated optimal tile size: {o_tl_size}, n_tl_ghw: {n_tl_ghw}.\n"
            + f"\tEstimated cost cores in total: {est_result.n_core}, lcn for each tile: {est_result.lcn.name}."
        )

    if compact_idx_map:
        return (
            est_result,
            *conv2d_tile_by_tile_size(
                in_shape, (ho, wo), kernel, stride, padding, groups, o_tl_size, n_tl_ghw
            ),
        )
    else:
        i_tiled_idx_map, o_tiled_idx_map, copy_times = make_conv_tiled_idx_map(
            in_shape,
            out_shape,
            o_tl_size,
            ksize,
            stride,
            padding,
            groups,
            zero_as_invalid_addr,
        )
        assert n_tl_ghw == i_tiled_idx_map.shape[:3]

        tl_shape = _cast_size3type(i_tiled_idx_map.shape[:3])
        # #N of channels in a tile / #N of channels in a group -> #N of groups in a tile
        g_tl = i_tiled_idx_map.shape[3] // ci_in_grp
        i_tl_size = _cast_size3type(i_tiled_idx_map.shape[-3:])
        o_tl_size = _cast_size3type(o_tiled_idx_map.shape[-3:])

        k_tiles = make_conv2d_kernel_tiled_unrolled(
            kernel, stride, groups, g_tl, tl_shape, i_tl_size, o_tl_size
        )

        return (est_result, i_tiled_idx_map, o_tiled_idx_map, k_tiles, copy_times)


def conv2d_optimize(
    conv2d_edge: Conv2d,
    online: bool = False,
):
    in_shape = conv2d_edge.shape_in
    kernel = conv2d_edge.weights
    stride = conv2d_edge.comm.stride
    padding = conv2d_edge.comm.padding
    groups = conv2d_edge.comm.groups
    out_shape = conv2d_edge.shape_out
    if online:
        core_base_fanin = OnCoreCfg.ADDR_AXON_MAX
        core_base_fanout = OnCoreCfg.N_DENDRITE_MAX
    else:
        core_base_fanin = OffCoreCfg.ADDR_AXON_MAX
        core_base_fanout = OffCoreCfg.N_DENDRITE_MAX_SNN

    if len(in_shape) == 3:
        in_shape = _cast_size3type(in_shape)
    else:
        raise ValueError("Only 2D convolution is supported in conv2d_optimize.")

    if len(out_shape) == 3:
        out_shape = _cast_size3type(out_shape)
    else:
        raise ValueError("Only 2D convolution is supported in conv2d_optimize.")

    return conv2d_tiling_optimize(
        in_shape,
        kernel,
        stride,
        padding,
        groups,
        core_base_fanin,
        core_base_fanout,
        out_shape,
        traverse_order="all",
        zero_as_invalid_addr=False,
        compact_idx_map=False,
    )
