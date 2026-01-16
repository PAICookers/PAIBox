import itertools
import timeit
from collections.abc import Sequence

import numpy as np
import pytest
from paicorelib import CoreMode, OffCoreCfg

from paibox._logging import set_logs
from paibox.backend.tiling import (
    EstCoreCostStatus,
    TileSliceConv2d,
    conv1d_tile_by_tile_size,
    conv1d_tiling_optimize,
    conv2d_tile_by_tile_size,
    conv2d_tiling_optimize,
    get_tile_shape,
    make_conv1d_kernel_tiled_unrolled,
    make_conv2d_kernel_tiled_unrolled,
    make_conv_tiled_idx_map,
    make_input_conv_tiled_idx_map,
    make_output_conv_tiled_idx_map,
    operator_core_cost_estimate,
    optimal_lcn_matmul2d,
    optimal_tiling_conv2d,
)
from paibox.components.synapses.conv_utils import (
    SizeAnyType,
    _conv1d_oshape,
    _conv2d_oshape,
    _pair,
    _single,
    conv1d_faster,
    conv2d_faster,
)
from tests.utils import gen_random_array, is_ci_env

from .tiling_test_utils import (
    conv1d_unroll_tiled_by_tiles,
    conv1d_unroll_tiled_from_full_kernel,
    conv2d_unroll_tiled_by_tiles,
    conv2d_unroll_tiled_from_full_kernel,
    tiled_vmm_conv1d,
    tiled_vmm_conv2d,
    tiled_vmm_conv_compact,
)

set_logs(tiling_optim=True)


def _fan_attrs_at_xbit(cm: CoreMode, xbit: int = 8):
    assert xbit in (1, 2, 4, 8)
    bw = xbit.bit_length() - 1
    if cm.is_snn:
        return (OffCoreCfg.N_FANIN_PER_DENDRITE_SNN, OffCoreCfg.N_NEURON_MAX_SNN >> bw)
    else:
        return OffCoreCfg.N_FANIN_PER_DENDRITE_ANN, OffCoreCfg.FANOUT_IW8[bw]


class TestTileSlice:
    def test_make_tile_slice(self):
        starts = (2, 3, 4)
        lens = (4, 5, 6)
        tl_slice = TileSliceConv2d.make_tile_slice(starts, lens)

        assert tl_slice.start == (2, 3, 4)
        assert tl_slice.end == (6, 8, 10)


class TestTilingOptimMatmul:
    @pytest.mark.parametrize(
        "shape_a, shape_b, core_mode",
        [
            ((64, 256), (256, 128), CoreMode.MODE_SNN),
            ((120, 240), (240, 240), CoreMode.MODE_ANN),
            ((36, 256), (256, 36), CoreMode.MODE_ANN),
        ],
    )
    def test_optimal_lcn_matmul2d(self, shape_a, shape_b, core_mode):
        est_result = optimal_lcn_matmul2d(
            shape_a, shape_b, *_fan_attrs_at_xbit(core_mode)
        )

        if est_result.status == EstCoreCostStatus.SUCCESS:
            print(
                f"Estimated status {est_result.status.name}, "
                f"Estimated cost cores in total: {est_result.n_core}, "
                f"lcn for each tile: {est_result.lcn.name}."
            )
        else:
            print("No optimization possible")


def prepare_tiled_vmm_conv_perf_data(
    in_shape: SizeAnyType,
    co: int,
    k_opt: int | Sequence[int],
    s_opt: int | Sequence[int],
    p_opt: int | Sequence[int],
    g_opt: int | Sequence[int],
    tile_size: SizeAnyType,
):
    cfg = []

    if isinstance(k_opt, int):
        k_opt = (k_opt,)
    if isinstance(s_opt, int):
        s_opt = (s_opt,)
    if isinstance(p_opt, int):
        p_opt = (p_opt,)
    if isinstance(g_opt, int):
        g_opt = (g_opt,)

    for k, s, p, g in itertools.product(k_opt, s_opt, p_opt, g_opt):
        assert tile_size[0] <= g
        cfg.append((in_shape, co, k, s, p, g, tile_size))

    return cfg


class TestConvKernelUnrollingPerf:
    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, l dimensions
            ((4, 16), 8, (3,), (1,), (0,), 1, (1, 2)),
            ((64, 128), 16, (32,), (1,), (2,), 2, (1, 4)),
            ((80, 512), 32, (16,), (4,), (8,), 4, (4, 4)),
            ((128, 1024), 16, (32,), (8,), (4,), 16, (2, 8)),
        ],
    )
    def test_conv1d_unroll_tiled_perf(
        self, in_shape, co, ksize, stride, padding, groups, tile_size
    ):
        ci = in_shape[0]
        g_tl, lo_tl = tile_size

        assert ci % groups == 0 and co % groups == 0
        ci_in_grp = ci // groups
        k_shape = (co, ci_in_grp) + ksize

        kernel = np.arange(np.prod(k_shape)).reshape(k_shape)

        tl_unrolled1 = conv1d_unroll_tiled_by_tiles(
            in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
        )
        tl_unrolled2 = conv1d_unroll_tiled_from_full_kernel(
            in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
        )

        def run_conv1d_unroll_tiled_by_tiles():
            g1 = conv1d_unroll_tiled_by_tiles(
                in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
            )
            for _ in g1:
                pass

        def run_conv1d_unroll_tiled_from_full_kernel():
            g2 = conv1d_unroll_tiled_from_full_kernel(
                in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
            )
            for _ in g2:
                pass

        if not is_ci_env():
            n = 5
            t1 = timeit.timeit(run_conv1d_unroll_tiled_by_tiles, number=n)
            t2 = timeit.timeit(run_conv1d_unroll_tiled_from_full_kernel, number=n)
            print(
                f"conv1d_unroll_tiled_by_tiles: {t1 / n}, "
                + f"conv1d_unroll_tiled_from_full_kernel: {t2 / n}"
            )

        for t1, t2 in zip(tl_unrolled1, tl_unrolled2):
            assert t1.size > 0 and t2.size > 0
            assert np.array_equal(t1, t2)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, h, w dimensions
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, (1, 2, 4)),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 2, (1, 4, 4)),
            ((16, 32, 32), 32, (3, 3), (1, 1), (1, 1), 2, (16, 2, 8)),
            ((16, 64, 64), 16, (4, 4), (2, 2), (0, 0), 16, (2, 8, 8)),
        ],
    )
    def test_conv2d_unroll_tiled_perf(
        self, in_shape, co, ksize, stride, padding, groups, tile_size
    ):
        ksize = _single(ksize)
        stride = _single(stride)
        padding = _single(padding)
        ci = in_shape[0]
        g_tl, ho_tl, wo_tl = tile_size

        assert ci % groups == 0 and co % groups == 0
        ci_in_grp = ci // groups
        k_shape = (co, ci_in_grp) + ksize

        kernel = np.arange(np.prod(k_shape)).reshape(k_shape)

        tl_unrolled1 = conv2d_unroll_tiled_by_tiles(
            in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
        )
        tl_unrolled2 = conv2d_unroll_tiled_from_full_kernel(
            in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
        )

        def run_conv2d_unroll_tiled_by_tiles():
            g1 = conv2d_unroll_tiled_by_tiles(
                in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
            )
            for _ in g1:
                pass

        def run_conv2d_unroll_tiled_from_full_kernel():
            g2 = conv2d_unroll_tiled_from_full_kernel(
                in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
            )
            for _ in g2:
                pass

        if not is_ci_env():
            n = 5
            t1 = timeit.timeit(run_conv2d_unroll_tiled_by_tiles, number=n)
            t2 = timeit.timeit(run_conv2d_unroll_tiled_from_full_kernel, number=n)
            print(
                f"conv2d_unroll_tiled_by_tiles: {t1 / n}, "
                + f"conv2d_unroll_tiled_from_full_kernel: {t2 / n}"
            )

        for t1, t2 in zip(tl_unrolled1, tl_unrolled2):
            assert t1.size > 0 and t2.size > 0
            assert np.array_equal(t1, t2)


def idx_map_dtype_check(arr: np.ndarray, zero_as_invalid_addr: bool) -> None:
    if zero_as_invalid_addr:
        assert np.issubdtype(arr.dtype, np.unsignedinteger)
    else:
        assert np.issubdtype(arr.dtype, np.signedinteger)


class TestConvTiledVMM:
    @pytest.mark.parametrize(
        "shape, tile_shape, idx_map_shape",
        [((2, 128), (1, 32), (2, 4)), ((16, 1024), (5, 500), (4, 3))],
    )
    def test_output_conv1d_tiled_idx_map(self, shape, tile_shape, idx_map_shape):
        tile1 = make_output_conv_tiled_idx_map(shape, tile_shape, True)
        idx_map_dtype_check(tile1, True)

        tile2 = make_output_conv_tiled_idx_map(shape, tile_shape, False)
        idx_map_dtype_check(tile2, False)

        assert tile1.shape[:2] == idx_map_shape

    @pytest.mark.parametrize(
        "shape, tile_shape, idx_map_shape",
        [
            ((2, 8, 8), (1, 4, 4), (2, 2, 2)),
            ((4, 8, 8), (4, 3, 3), (1, 3, 3)),
            ((16, 64, 64), (5, 4, 8), (4, 16, 8)),
        ],
    )
    def test_output_conv2d_tiled_idx_map(self, shape, tile_shape, idx_map_shape):
        tile1 = make_output_conv_tiled_idx_map(shape, tile_shape, True)
        idx_map_dtype_check(tile1, True)

        tile2 = make_output_conv_tiled_idx_map(shape, tile_shape, False)
        idx_map_dtype_check(tile2, False)

        assert tile1.shape[:3] == idx_map_shape

    @pytest.mark.parametrize(
        "ishape, oc, ksize, stride, padding, groups, tile_shape",
        [
            ((4, 16), 8, (3,), (1,), (0,), 1, (1, 4)),
            ((4, 32), 8, (4,), (1,), (0,), 1, (1, 6)),
            ((8, 1024), 4, (32,), (4,), (8,), 2, (2, 8)),
            ((16, 2048), 8, (64,), (8,), (8,), 8, (5, 16)),
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    def test_input_conv1d_tiled_idx_map(
        self,
        ishape,
        oc,
        ksize,
        stride,
        padding,
        groups,
        tile_shape,
        zero_as_invalid_addr,
    ):
        assert ishape[0] % groups == 0 and oc % groups == 0
        oshape = (oc,) + _conv1d_oshape(ishape[1:], ksize, stride, padding)

        assert all(o >= t for o, t in zip(oshape, tile_shape))
        o_inner_shape = get_tile_shape(oshape, tile_shape)

        # Shape of the output feature map after padding for complete tiling
        ref_oshape_aft_tiling_pad = tuple(
            l * n for l, n in zip(o_inner_shape, tile_shape)
        )

        assert tile_shape[0] <= groups
        i_inner_ch = ishape[0] // tile_shape[0]

        tile = make_input_conv_tiled_idx_map(
            ishape,
            o_inner_shape,
            tile_shape,
            i_inner_ch,
            ksize,
            stride,
            padding,
            zero_as_invalid_addr,
        )
        idx_map_dtype_check(tile, zero_as_invalid_addr)

        tile_l = tile.shape[-1:]
        n_tile_l = tile.shape[1:2]

        # Already padded for complete tiling in `tile`
        oshape_tile_l = _conv1d_oshape(tile_l, ksize, stride, 0)

        # Compare (#N of tiles * length of each tile) in each dimension with the reference
        assert oshape_tile_l == o_inner_shape[1:]
        assert all(
            n_tile * oshape_tile == ref
            for n_tile, oshape_tile, ref in zip(
                n_tile_l, oshape_tile_l, ref_oshape_aft_tiling_pad[1:]
            )
        )

    @pytest.mark.parametrize(
        "ishape, oc, tile_shape, ksize, stride, padding, groups",
        [
            ((4, 16, 16), 8, (1, 3, 3), (3, 3), (1, 1), (0, 0), 1),
            ((4, 16, 16), 8, (1, 3, 3), (3, 3), (1, 1), (0, 0), 1),
            ((8, 16, 16), 4, (2, 2, 2), (3, 3), (2, 2), (1, 1), 2),
            ((16, 32, 32), 8, (5, 4, 6), (3, 3), (2, 2), (1, 1), 8),
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    def test_input_conv2d_tiled_idx_map(
        self,
        ishape,
        oc,
        tile_shape,
        ksize,
        stride,
        padding,
        groups,
        zero_as_invalid_addr,
    ):
        assert ishape[0] % groups == 0 and oc % groups == 0
        oshape = (oc,) + _conv2d_oshape(ishape[1:], ksize, stride, padding)

        assert all(o >= t for o, t in zip(oshape, tile_shape))
        o_inner_shape = get_tile_shape(oshape, tile_shape)

        # Shape of the output feature map after padding for complete tiling
        ref_oshape_aft_tiling_pad = tuple(
            tl_size * tl_n for tl_size, tl_n in zip(o_inner_shape, tile_shape)
        )

        assert tile_shape[0] <= groups
        i_inner_ch = ishape[0] // tile_shape[0]

        tile = make_input_conv_tiled_idx_map(
            ishape,
            o_inner_shape,
            tile_shape,
            i_inner_ch,
            ksize,
            stride,
            padding,
            zero_as_invalid_addr,
        )
        idx_map_dtype_check(tile, zero_as_invalid_addr)

        tile_hw = tile.shape[-2:]
        n_tile_hw = tile.shape[1:3]

        # Already padded for complete tiling in `tile`
        oshape_tile_hw = _conv2d_oshape(tile_hw, ksize, stride, 0)

        # Compare (#N of tiles * length of each tile) in each dimension with the reference
        assert oshape_tile_hw == o_inner_shape[1:]
        assert all(
            n_tile * oshape_tile == ref
            for n_tile, oshape_tile, ref in zip(
                n_tile_hw, oshape_tile_hw, ref_oshape_aft_tiling_pad[1:]
            )
        )

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, l dimensions
            ((4, 16), 8, (3,), (1,), (0,), 1, (1, 2)),
            ((64, 128), 16, (32,), (1,), (2,), 2, (1, 4)),
            ((80, 512), 32, (16,), (4,), (8,), 4, (4, 4)),
            ((128, 1024), 16, (32,), (8,), (4,), 16, (2, 8)),
        ],
    )
    def test_tiled_vmm_conv1d_compact(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        tile_size,
        fixed_rng,
    ):
        ci, li = in_shape
        lo = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (co,) + lo

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        n_tile_gl = get_tile_shape((groups,) + lo, tile_size)

        tiles, _ = conv1d_tile_by_tile_size(
            in_shape, lo, kernel, stride, padding, groups, tile_size, n_tile_gl
        )

        result = tiled_vmm_conv_compact(x, out_shape, groups, tiles)
        expected = conv1d_faster(x, lo, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, h, w dimensions
            ((1, 8, 8), 2, (3, 3), (1, 1), (0, 0), 1, (1, 3, 3)),
            ((1, 8, 8), 2, (3, 3), (1, 1), (1, 1), 1, (1, 3, 3)),
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, (1, 2, 4)),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 2, (1, 4, 4)),
            ((16, 32, 32), 32, (1, 1), (1, 1), (1, 1), 4, (4, 4, 12)),
            ((16, 64, 64), 16, (4, 4), (2, 2), (0, 0), 16, (2, 8, 8)),
        ],
    )
    def test_tiled_vmm_conv2d_compact(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        tile_size,
        fixed_rng,
    ):
        ci, hi, wi = in_shape
        osize = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (co,) + osize

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        n_tile_ghw = get_tile_shape((groups,) + osize, tile_size)

        tiles, _ = conv2d_tile_by_tile_size(
            in_shape, osize, kernel, stride, padding, groups, tile_size, n_tile_ghw
        )

        result = tiled_vmm_conv_compact(x, out_shape, groups, tiles)
        expected = conv2d_faster(x, osize, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, o_inner_shape_gl",
        [
            # tile_size/tile_start_idx in g, l dimensions
            ((4, 16), 8, 3, 1, 0, 1, (1, 3)),  # k,s,p,g = 3,1,0,1
            ((8, 64), 16, 3, 1, 1, 1, (1, 4)),  # 3,1,1,1
            ((16, 512), 32, 16, 4, 8, 4, (4, 32)),  # 16,4,8,4
            ((16, 1024), 16, 64, 8, 0, 16, (4, 32)),  # 64,8,0,16
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    def test_tiled_vmm_conv1d(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        o_inner_shape_gl,
        zero_as_invalid_addr,
        fixed_rng,
    ):
        ci, li = in_shape
        ksize = _single(ksize)
        stride = _single(stride)
        padding = _single(padding)

        osize = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (co,) + osize

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups
        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        i_tiled_idx_map, o_tiled_idx_map, copy_times = make_conv_tiled_idx_map(
            in_shape,
            out_shape,
            o_inner_shape_gl,
            ksize,
            stride,
            padding,
            groups,
            zero_as_invalid_addr,
        )

        tl_shape = i_tiled_idx_map.shape[:2]
        g_tl = i_tiled_idx_map.shape[2] // ci_in_grp
        i_tl_size = i_tiled_idx_map.shape[-2:]
        o_tl_size = o_tiled_idx_map.shape[-2:]

        k_tiles_unrolled = make_conv1d_kernel_tiled_unrolled(
            kernel, stride, groups, g_tl, tl_shape, i_tl_size, o_tl_size
        )

        result1 = tiled_vmm_conv1d(
            x,
            out_shape,
            kernel,
            stride,
            groups,
            i_tiled_idx_map,
            o_tiled_idx_map,
            zero_as_invalid_addr,
            k_tiles_unrolled,
        )
        result2 = tiled_vmm_conv1d(
            x,
            out_shape,
            kernel,
            stride,
            groups,
            i_tiled_idx_map,
            o_tiled_idx_map,
            zero_as_invalid_addr,
        )

        expected = conv1d_faster(x, osize, kernel, stride, padding, groups=groups)
        assert np.array_equal(result1, expected)
        assert np.array_equal(result2, expected)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, o_inner_shape_ghw",
        [
            # tile_size/tile_start_idx in g, h, w dimensions
            ((4, 16, 16), 8, 3, 1, 0, 1, (1, 2, 4)),  # k,s,p,g = 3,1,0,1
            ((8, 64, 64), 16, 3, 1, 1, 1, (1, 4, 4)),  # 3,1,1,1
            ((16, 32, 32), 32, 3, 1, 1, 4, (4, 4, 12)),  # 3,1,1,4
            ((16, 64, 64), 16, 4, 2, 0, 16, (4, 12, 12)),  # 4,2,0,16
            ((16, 64, 64), 16, 4, 2, 0, 1, (1, 8, 8)),  # 4,2,0,1
            ((8, 64, 64), 16, 4, 1, 1, 8, (2, 18, 12)),  # 4,1,1,8
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    def test_tiled_vmm_conv2d(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        o_inner_shape_ghw,
        zero_as_invalid_addr,
        fixed_rng,
    ):
        ci, hi, wi = in_shape
        ksize = _pair(ksize)
        stride = _pair(stride)
        padding = _pair(padding)

        osize = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (co,) + osize

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups
        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        i_tiled_idx_map, o_tiled_idx_map, copy_times = make_conv_tiled_idx_map(
            in_shape,
            out_shape,
            o_inner_shape_ghw,
            ksize,
            stride,
            padding,
            groups,
            zero_as_invalid_addr,
        )

        tl_shape = i_tiled_idx_map.shape[:3]
        # #N of channels in a tile / #N of channels in a group -> #N of groups in a tile
        g_tl = i_tiled_idx_map.shape[3] // ci_in_grp
        i_tl_size = i_tiled_idx_map.shape[-3:]
        o_tl_size = o_tiled_idx_map.shape[-3:]

        k_tiles_unrolled = make_conv2d_kernel_tiled_unrolled(
            kernel, stride, groups, g_tl, tl_shape, i_tl_size, o_tl_size
        )

        result1 = tiled_vmm_conv2d(
            x,
            out_shape,
            kernel,
            stride,
            groups,
            i_tiled_idx_map,
            o_tiled_idx_map,
            zero_as_invalid_addr,
            k_tiles_unrolled,
        )
        result2 = tiled_vmm_conv2d(
            x,
            out_shape,
            kernel,
            stride,
            groups,
            i_tiled_idx_map,
            o_tiled_idx_map,
            zero_as_invalid_addr,
        )

        expected = conv2d_faster(x, osize, kernel, stride, padding, groups=groups)
        assert np.array_equal(result1, expected)
        assert np.array_equal(result2, expected)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, tile_size",
        prepare_tiled_vmm_conv_perf_data(
            (4, 32, 32),
            8,
            (3, 4),
            (1, 2),
            (1, 2),
            (1, 4),
            (1, 8, 12),
        )
        + prepare_tiled_vmm_conv_perf_data(
            (8, 32, 32),
            16,
            (3, 4),
            (1, 2),
            (1, 2),
            (4, 8),
            (4, 10, 10),
        ),
    )
    def test_tiled_vmm_conv2d_perf(
        self, in_shape, co, ksize, stride, padding, groups, tile_size, fixed_rng
    ):
        ci, hi, wi = in_shape
        ksize = _pair(ksize)
        stride = _pair(stride)
        padding = _pair(padding)

        osize = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (co,) + osize

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        n_tile_ghw = get_tile_shape((groups,) + osize, tile_size)

        def run1():
            tiles, _ = conv2d_tile_by_tile_size(
                in_shape, osize, kernel, stride, padding, groups, tile_size, n_tile_ghw
            )

            return tiled_vmm_conv_compact(x, out_shape, groups, tiles)

        def run2():
            zero_as_invalid_addr = True
            i_tiled_idx_map, o_tiled_idx_map, copy_times = make_conv_tiled_idx_map(
                in_shape,
                out_shape,
                tile_size,
                ksize,
                stride,
                padding,
                groups,
                zero_as_invalid_addr,
            )

            return tiled_vmm_conv2d(
                x,
                out_shape,
                kernel,
                stride,
                groups,
                i_tiled_idx_map,
                o_tiled_idx_map,
                zero_as_invalid_addr,
            )

        if not is_ci_env():
            n = 10
            t1 = timeit.timeit(run1, number=n)
            t2 = timeit.timeit(run2, number=n)
            print(f"run1: {t1 / n}, run2: {t2 / n}")

        result1 = run1()
        result2 = run2()

        expected = conv2d_faster(x, osize, kernel, stride, padding, groups=groups)
        assert np.array_equal(result1, expected)
        assert np.array_equal(result2, expected)


class TestConvTilingOptim:
    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, core_mode",
        [
            ((3, 32, 32), 8, (3, 3), 1, 0, 1, CoreMode.MODE_SNN),
            ((16, 64, 64), 32, (3, 3), 1, 0, 1, CoreMode.MODE_SNN),
            ((16, 32, 32), 32, (3, 3), 1, 1, 2, CoreMode.MODE_ANN),
            ((16, 80, 120), 16, (3, 3), 2, 0, 16, CoreMode.MODE_ANN),  # depthwise conv
            ((32, 64, 64), 32, (1, 1), 1, 0, 1, CoreMode.MODE_ANN),  # pointwise conv
            ((16, 64, 64), 16, (3, 3), 1, 0, 1, CoreMode.MODE_ANN),
        ],
        ids=["snn1", "snn2", "ann1", "ann2", "ann3", "ann4"],
    )
    def test_optimal_tiling_conv2d(
        self, in_shape, co, ksize, stride, padding, groups, core_mode
    ):
        stride = _pair(stride)
        padding = _pair(padding)

        ho, wo = _conv2d_oshape(in_shape[1:], ksize, stride, padding)
        out_shape = (co,) + (ho, wo)
        n_fanin_base, n_fanout_base = _fan_attrs_at_xbit(core_mode)

        # Result for non-optimized conv if the fan-in is not too large to fit in.
        est_result = operator_core_cost_estimate(
            in_shape, out_shape, n_fanin_base, n_fanout_base
        )

        print("Before optimization:")
        if est_result.status == EstCoreCostStatus.SUCCESS:
            print(f"\tNumber of cores: {est_result.n_core}, lcn: {est_result.lcn}")
        else:
            print(f"\t{est_result.status.name}")

        est_result, best_tile_size_ghw, best_n_tile_ghw = optimal_tiling_conv2d(
            in_shape, out_shape, ksize, stride, groups, n_fanin_base, n_fanout_base
        )

        if est_result.status == EstCoreCostStatus.SUCCESS:
            print("After optimization:")
            print(f"\tNumber of cores: {est_result.n_core}, lcn: {est_result.lcn}")
            print(
                f"\tBest block size: {best_tile_size_ghw}, #blocks: {best_n_tile_ghw}"
            )
        else:
            print("No optimization possible")

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, core_mode",
        [
            ((4, 32), 8, (3,), (1,), (0,), 1, CoreMode.MODE_SNN),
            ((16, 64), 16, (8,), (2,), (1,), 1, CoreMode.MODE_ANN),
            ((16, 256), 32, (16,), (8,), (3,), 4, CoreMode.MODE_SNN),
            ((16, 1024), 16, (64,), (24,), (16,), 8, CoreMode.MODE_ANN),
            ((32, 1024), 32, (64,), (16,), (8,), 4, CoreMode.MODE_SNN),
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    @pytest.mark.parametrize("compact_idx_map", [True, False])
    def test_conv1d_tiling_optimize(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        core_mode,
        zero_as_invalid_addr,
        compact_idx_map,
        fixed_rng,
    ):
        ci, li = in_shape
        lo = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (co,) + lo
        x = gen_random_array(in_shape, np.uint8, fixed_rng)
        ci_in_grp = ci // groups
        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        if compact_idx_map:
            est_result, conv1d_tiles, copy_times = conv1d_tiling_optimize(
                in_shape,
                kernel,
                stride,
                padding,
                groups,
                *_fan_attrs_at_xbit(core_mode),
                zero_as_invalid_addr=zero_as_invalid_addr,
                compact_idx_map=True,
            )
        else:
            (
                est_result,
                i_tiled_idx_map,
                o_tiled_idx_map,
                k_tiles_unrolled,
                copy_times,
            ) = conv1d_tiling_optimize(
                in_shape,
                kernel,
                stride,
                padding,
                groups,
                *_fan_attrs_at_xbit(core_mode),
                zero_as_invalid_addr=zero_as_invalid_addr,
                compact_idx_map=False,
            )

        if est_result.status != EstCoreCostStatus.SUCCESS:
            pytest.skip("No optimization possible")

        if compact_idx_map:
            result = tiled_vmm_conv_compact(x, out_shape, groups, conv1d_tiles)
        else:
            result = tiled_vmm_conv1d(
                x,
                out_shape,
                kernel,
                stride,
                groups,
                i_tiled_idx_map,
                o_tiled_idx_map,
                zero_as_invalid_addr,
                k_tiles_unrolled,
            )

        expected = conv1d_faster(x, lo, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, co, ksize, stride, padding, groups, core_mode",
        [
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, CoreMode.MODE_SNN),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 1, CoreMode.MODE_ANN),
            ((16, 32, 32), 32, (1, 1), (1, 1), (1, 1), 4, CoreMode.MODE_SNN),
            ((16, 54, 54), 16, (4, 4), (2, 2), (0, 0), 16, CoreMode.MODE_ANN),
            ((3, 64, 64), 16, (5, 5), (1, 1), (0, 0), 1, CoreMode.MODE_SNN),
        ],
    )
    @pytest.mark.parametrize("zero_as_invalid_addr", [True, False])
    @pytest.mark.parametrize("compact_idx_map", [True, False])
    def test_conv2d_tiling_optimize(
        self,
        in_shape,
        co,
        ksize,
        stride,
        padding,
        groups,
        core_mode,
        zero_as_invalid_addr,
        compact_idx_map,
        fixed_rng,
    ):
        ci, hi, wi = in_shape
        oshape = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (co,) + oshape
        x = gen_random_array(in_shape, np.uint8, fixed_rng)
        ci_in_grp = ci // groups
        kshape = (co, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        if compact_idx_map:
            est_result, conv2d_tiles, copy_times = conv2d_tiling_optimize(
                in_shape,
                kernel,
                stride,
                padding,
                groups,
                *_fan_attrs_at_xbit(core_mode),
                zero_as_invalid_addr=zero_as_invalid_addr,
                compact_idx_map=True,
            )
        else:
            (
                est_result,
                i_tiled_idx_map,
                o_tiled_idx_map,
                k_tiles_unrolled,
                copy_times,
            ) = conv2d_tiling_optimize(
                in_shape,
                kernel,
                stride,
                padding,
                groups,
                *_fan_attrs_at_xbit(core_mode),
                zero_as_invalid_addr=zero_as_invalid_addr,
                compact_idx_map=False,
            )

        if est_result.status != EstCoreCostStatus.SUCCESS:
            pytest.skip("No optimization possible")

        if compact_idx_map:
            result = tiled_vmm_conv_compact(x, out_shape, groups, conv2d_tiles)
        else:
            result = tiled_vmm_conv2d(
                x,
                out_shape,
                kernel,
                stride,
                groups,
                i_tiled_idx_map,
                o_tiled_idx_map,
                zero_as_invalid_addr,
                k_tiles_unrolled,
            )

        expected = conv2d_faster(x, oshape, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)
