import math
import pytest
import timeit
import numpy as np

from paicorelib import OffCoreCfg, CoreMode
from paibox.backend.tiling import *
from paibox.components.synapses.conv_types import Size1Type, Size2Type
from paibox.components.synapses.conv_utils import (
    _conv1d_oshape,
    _conv2d_oshape,
    _pair,
    conv1d_faster,
    conv2d_faster,
)
from tests.utils import gen_random_array, is_ci_env

from paibox._logging import set_logs

set_logs(tiling_optim=True)


def _fan_attrs_at_xbit(cm: CoreMode, xbit: int = 8):
    assert xbit in (1, 2, 4, 8)
    bw = xbit.bit_length() - 1
    if cm.is_snn:
        return (OffCoreCfg.N_FANIN_PER_DENDRITE_SNN, OffCoreCfg.N_NEURON_MAX_SNN >> bw)
    else:
        return OffCoreCfg.N_FANIN_PER_DENDRITE_ANN, OffCoreCfg.FANOUT_IW8[bw]


def _n_tile_ghw_by_tile_size2d(
    out_shape_l: Size1Type, groups: int, tile_size: TileSize2d
) -> TileSize2d:
    """Compute the number of blocks in each dimension of g, l by given the tile size."""
    (lo,) = out_shape_l
    g_tl, lo_tl = tile_size

    n_tile_g = math.ceil(groups / g_tl)
    n_tile_l = math.ceil(lo / lo_tl)

    return (n_tile_g, n_tile_l)


def _n_tile_ghw_by_tile_size3d(
    out_shape_hw: Size2Type, groups: int, tile_size: TileSize3d
) -> TileSize3d:
    """Compute the number of blocks in each dimension of g, h, w by given the tile size."""
    ho, wo = out_shape_hw
    g_tl, ho_tl, wo_tl = tile_size

    n_tile_g = math.ceil(groups / g_tl)
    n_tile_h = math.ceil(ho / ho_tl)
    n_tile_w = math.ceil(wo / wo_tl)

    return (n_tile_g, n_tile_h, n_tile_w)


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


class TestTilingOptimConv:
    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, core_mode",
        [
            ((3, 32, 32), 8, (3, 3), 1, 0, 1, CoreMode.MODE_SNN),
            ((16, 64, 64), 32, (3, 3), 1, 0, 1, CoreMode.MODE_SNN),
            ((16, 32, 32), 32, (3, 3), 1, 1, 2, CoreMode.MODE_ANN),
            ((16, 80, 120), 16, (3, 3), 2, 0, 16, CoreMode.MODE_ANN),  # depthwise conv
            ((32, 64, 64), 32, (1, 1), 1, 0, 1, CoreMode.MODE_ANN),  # pointwise conv
            ((16, 64, 64), 16, (3, 3), 1, 0, 1, CoreMode.MODE_ANN),
        ],
    )
    def test_optimal_tiling_conv2d(
        self, in_shape, out_channels, ksize, stride, padding, groups, core_mode
    ):
        stride = _pair(stride)
        padding = _pair(padding)

        ho, wo = _conv2d_oshape(in_shape[1:], ksize, stride, padding)
        out_shape = (out_channels,) + (ho, wo)
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
        "in_shape, out_channels, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, l dimensions
            ((4, 16), 8, (3,), (1,), (0,), 1, (1, 2)),
            ((64, 128), 16, (32,), (1,), (2,), 1, (2, 4)),
            ((80, 512), 32, (16,), (4,), (8,), 4, (4, 4)),
            ((128, 1024), 16, (32,), (8,), (4,), 16, (2, 8)),
        ],
    )
    def test_conv1d_tile_by_tile_size(
        self,
        in_shape,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        tile_size,
        fixed_rng,
    ):
        ci, li = in_shape
        lo = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (out_channels,) + lo

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (out_channels, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        n_tile_ghw = _n_tile_ghw_by_tile_size2d(lo, groups, tile_size)

        tiles, _ = conv1d_tile_by_tile_size(
            in_shape, lo, kernel, stride, padding, groups, tile_size, n_tile_ghw
        )

        result = conv_in_tile_format(x, out_shape, groups, tiles)
        expected = conv1d_faster(x, lo, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, h, w dimensions
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, (1, 2, 4)),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 1, (2, 4, 4)),
            ((16, 32, 32), 32, (1, 1), (1, 1), (1, 1), 4, (4, 4, 12)),
            ((16, 64, 64), 16, (4, 4), (2, 2), (0, 0), 16, (2, 8, 8)),
        ],
    )
    def test_conv2d_tile_by_tile_size(
        self,
        in_shape,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        tile_size,
        fixed_rng,
    ):
        ci, hi, wi = in_shape
        osize = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (out_channels,) + osize

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (out_channels, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        n_tile_ghw = _n_tile_ghw_by_tile_size3d(osize, groups, tile_size)

        tiles, _ = conv2d_tile_by_tile_size(
            in_shape, osize, kernel, stride, padding, groups, tile_size, n_tile_ghw
        )

        result = conv_in_tile_format(x, out_shape, groups, tiles)
        expected = conv2d_faster(x, osize, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, core_mode",
        [
            ((4, 16), 8, (3,), (1,), (0,), 1, CoreMode.MODE_SNN),
            ((16, 32), 16, (8,), (2,), (1,), 1, CoreMode.MODE_ANN),
            ((16, 128), 32, (16,), (8,), (2,), 4, CoreMode.MODE_SNN),
            ((16, 1024), 16, (40,), (20,), (10,), 16, CoreMode.MODE_ANN),
            ((32, 1024 * 8), 32, (64,), (16,), (8,), 1, CoreMode.MODE_SNN),
        ],
    )
    def test_conv1d_tiling_optimize(
        self,
        in_shape,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        core_mode,
        fixed_rng,
    ):
        ci, li = in_shape
        lo = _conv1d_oshape((li,), ksize, stride, padding)
        out_shape = (out_channels,) + lo

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (out_channels, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        est_result, conv2d_tiles, _ = conv1d_tiling_optimize(
            in_shape,
            kernel,
            stride,
            padding,
            groups,
            *_fan_attrs_at_xbit(core_mode),
        )

        if est_result.status != EstCoreCostStatus.SUCCESS:
            pytest.skip("No optimization possible")

        result = conv_in_tile_format(x, out_shape, groups, conv2d_tiles)
        expected = conv1d_faster(x, lo, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, core_mode",
        [
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, CoreMode.MODE_SNN),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 1, CoreMode.MODE_ANN),
            ((16, 32, 32), 32, (1, 1), (1, 1), (1, 1), 4, CoreMode.MODE_SNN),
            ((16, 54, 54), 16, (4, 4), (2, 2), (0, 0), 16, CoreMode.MODE_ANN),
            ((3, 224, 224), 16, (4, 4), (1, 1), (0, 0), 1, CoreMode.MODE_SNN),
        ],
    )
    def test_conv2d_tiling_optimize(
        self,
        in_shape,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        core_mode,
        fixed_rng,
    ):
        ci, hi, wi = in_shape
        oshape = _conv2d_oshape((hi, wi), ksize, stride, padding)
        out_shape = (out_channels,) + oshape

        x = gen_random_array(in_shape, np.uint8, fixed_rng)

        ci_in_grp = ci // groups

        kshape = (out_channels, ci_in_grp) + ksize
        kernel = gen_random_array(kshape, np.int8, fixed_rng)

        est_result, conv2d_tiles, _ = conv2d_tiling_optimize(
            in_shape,
            kernel,
            stride,
            padding,
            groups,
            *_fan_attrs_at_xbit(core_mode),
        )

        if est_result.status != EstCoreCostStatus.SUCCESS:
            pytest.skip("No optimization possible")

        result = conv_in_tile_format(x, out_shape, groups, conv2d_tiles)
        expected = conv2d_faster(x, oshape, kernel, stride, padding, groups=groups)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, l dimensions
            ((4, 16), 8, (3,), (1,), (0,), 1, (1, 2)),
            ((64, 128), 16, (32,), (1,), (2,), 1, (2, 4)),
            ((80, 512), 32, (16,), (4,), (8,), 4, (4, 4)),
            ((128, 1024), 16, (32,), (8,), (4,), 16, (2, 8)),
        ],
    )
    def test_conv1d_unroll_tiled_perf(
        self, in_shape, out_channels, ksize, stride, padding, groups, tile_size
    ):
        ci = in_shape[0]
        g_tl, lo_tl = tile_size

        assert ci % groups == 0 and out_channels % groups == 0
        ci_in_grp = ci // groups
        k_shape = (out_channels, ci_in_grp) + ksize

        kernel = np.arange(np.prod(k_shape)).reshape(k_shape)

        unroll_tiles1 = conv1d_unroll_tiled_by_tiles(
            in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
        )
        unroll_tiles2 = conv1d_unroll_tiled_from_full_kernel(
            in_shape, kernel, stride, padding, groups, (g_tl, lo_tl)
        )

        def run_conv1d_unroll_tiled_by_tiles():
            for _ in unroll_tiles1:
                pass

        def run_conv1d_unroll_tiled_from_full_kernel():
            for _ in unroll_tiles2:
                pass

        if not is_ci_env():
            n = 5
            t1 = timeit.timeit(run_conv1d_unroll_tiled_by_tiles, number=n)
            t2 = timeit.timeit(run_conv1d_unroll_tiled_from_full_kernel, number=n)
            print(
                f"conv1d_unroll_tiled_by_tiles: {t1 / n}, "
                + f"conv1d_unroll_tiled_from_full_kernel: {t2 / n}"
            )

        for k_tile_ur1, k_tile_ur2 in zip(unroll_tiles1, unroll_tiles2):
            assert np.array_equal(k_tile_ur1, k_tile_ur2)

    @pytest.mark.parametrize(
        "in_shape, out_channels, ksize, stride, padding, groups, tile_size",
        [
            # tile_size/tile_start_idx in g, h, w dimensions
            ((4, 16, 16), 8, (3, 3), (1, 1), (0, 0), 1, (1, 2, 4)),
            ((8, 32, 32), 16, (3, 3), (2, 2), (1, 1), 1, (2, 4, 4)),
            ((16, 32, 32), 32, (3, 3), (1, 1), (1, 1), 2, (16, 2, 8)),
            ((16, 64, 64), 16, (4, 4), (2, 2), (0, 0), 16, (2, 8, 8)),
        ],
    )
    def test_conv2d_unroll_tiled_perf(
        self, in_shape, out_channels, ksize, stride, padding, groups, tile_size
    ):
        ci = in_shape[0]
        g_tl, ho_tl, wo_tl = tile_size

        assert ci % groups == 0 and out_channels % groups == 0
        ci_in_grp = ci // groups
        k_shape = (out_channels, ci_in_grp) + ksize

        kernel = np.arange(np.prod(k_shape)).reshape(k_shape)

        unroll_tiles1 = conv2d_unroll_tiled_by_tiles(
            in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
        )
        unroll_tiles2 = conv2d_unroll_tiled_from_full_kernel(
            in_shape, kernel, stride, padding, groups, (g_tl, ho_tl, wo_tl)
        )

        def run_conv2d_unroll_tiled_by_tiles():
            for _ in unroll_tiles1:
                pass

        def run_conv2d_unroll_tiled_from_full_kernel():
            for _ in unroll_tiles2:
                pass

        if not is_ci_env():
            n = 5
            t1 = timeit.timeit(run_conv2d_unroll_tiled_by_tiles, number=n)
            t2 = timeit.timeit(run_conv2d_unroll_tiled_from_full_kernel, number=n)
            print(
                f"conv2d_unroll_tiled_by_tiles: {t1 / n}, "
                + f"conv2d_unroll_tiled_from_full_kernel: {t2 / n}"
            )

        for k_tile_ur1, k_tile_ur2 in zip(unroll_tiles1, unroll_tiles2):
            assert np.array_equal(k_tile_ur1, k_tile_ur2)
