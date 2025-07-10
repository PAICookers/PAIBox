import numpy as np
import pytest
import timeit

from paibox.components.synapses.conv_utils import (
    _conv1d_unroll,
    _conv2d_unroll,
    conv1d_faster,
    conv2d_faster,
)
from tests.conftest import ParametrizedTestData
from tests.utils import gen_random_array


try:
    from paibox.components.synapses.conv_utils import conv1d_faster_legacy

    skip_conv1d_faster_test = False
except ImportError:
    skip_conv1d_faster_test = True


test_conv1d_faster_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups",
    data=[
        ((2048,), 16, 32, (128,), (20,), (50,), (8,), 1),
        ((512,), 64, 32, (16,), (8,), (0,), (1,), 4),
    ],
)


@pytest.mark.skipif(
    skip_conv1d_faster_test,
    reason="Legacy function 'conv1d_faster_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv1d_faster_data["args"], test_conv1d_faster_data["data"]
)
def test_conv1d_faster_perf(
    in_shape,
    in_channels,
    out_channels,
    ksize,
    stride,
    padding,
    dilation,
    groups,
    fixed_rng,
    request,
):
    cin_per_grp = in_channels // groups

    x = gen_random_array((in_channels,) + in_shape, np.uint8, fixed_rng)
    kernel = gen_random_array((out_channels, cin_per_grp) + ksize, np.int8, fixed_rng)

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
    )

    def run_conv1d_faster():
        return conv1d_faster(x, out_shape, kernel, stride, padding, dilation, groups)

    def run_conv1d_faster_legacy():
        return conv1d_faster_legacy(
            x, out_shape, kernel, stride, padding, dilation, groups
        )

    if not request.node.get_closest_marker("perf"):
        # ~10x faster
        t1 = timeit.timeit(lambda: run_conv1d_faster(), number=10)
        t2 = timeit.timeit(lambda: run_conv1d_faster_legacy(), number=10)
        print("Use im2col: ", t1, "Use legacy:", t2)

    r_opt = run_conv1d_faster()
    r_legacy = run_conv1d_faster_legacy()
    assert np.array_equal(r_opt, r_legacy)


try:
    from paibox.components.synapses.conv_utils import _conv1d_unroll_legacy

    skip_conv1d_unroll_test = False
except ImportError:
    skip_conv1d_unroll_test = True


test_conv1d_unroll_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, groups",
    data=[
        ((32,), 8, 16, (3,), (1,), (2,), 1),
        ((240,), 16, 32, (40,), (5,), (0,), 2),
        ((1000,), 32, 64, (100,), (20,), (20,), 1),
    ],
)


@pytest.mark.skipif(
    skip_conv1d_unroll_test,
    reason="Legacy function '_conv1d_unroll_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv1d_unroll_data["args"], test_conv1d_unroll_data["data"]
)
def test_conv1d_unroll_perf(
    in_shape,
    in_channels,
    out_channels,
    ksize,
    stride,
    padding,
    groups,
    fixed_rng,
    request,
):
    out_shape = ((in_shape[0] + 2 * padding[0] - ksize[0]) // stride[0] + 1,)

    k_shape = (out_channels, in_channels) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    def run_conv1d_unroll():
        return _conv1d_unroll(in_shape, out_shape, kernel, stride, padding, groups)

    def run_conv1d_unroll_legacy():
        return _conv1d_unroll_legacy(
            in_shape, out_shape, kernel, stride, padding, groups
        )

    if not request.node.get_closest_marker("perf"):
        # 10~40x faster
        t1 = timeit.timeit(lambda: run_conv1d_unroll(), number=5)
        t2 = timeit.timeit(lambda: run_conv1d_unroll_legacy(), number=5)
        print("Optimized: ", t1, "Legacy: ", t2)

    r_opt = run_conv1d_unroll()
    r_legacy = run_conv1d_unroll_legacy()
    assert np.array_equal(r_opt, r_legacy)


try:
    from paibox.components.synapses.conv_utils import conv2d_faster_legacy

    skip_conv2d_faster_test = False
except ImportError:
    skip_conv2d_faster_test = True


test_conv2d_faster_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups",
    data=[
        ((64, 64), 16, 32, (3, 3), (1, 1), (1, 1), (2, 2), 4),
        ((32, 32), 24, 24, (4, 4), (2, 2), (2, 1), (1, 1), 2),
        ((24, 24), 16, 8, (3, 3), (1, 2), (0, 0), (1, 1), 1),
    ],
)


@pytest.mark.skipif(
    skip_conv2d_faster_test, reason="Legacy function 'conv2d_faster_legacy' is removed"
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv2d_faster_data["args"], test_conv2d_faster_data["data"]
)
def test_conv2d_faster_perf(
    in_shape,
    in_channels,
    out_channels,
    ksize,
    stride,
    padding,
    dilation,
    groups,
    fixed_rng,
    request,
):
    cin_per_grp = in_channels // groups

    x = gen_random_array((in_channels,) + in_shape, np.uint8, fixed_rng)
    kernel = gen_random_array((out_channels, cin_per_grp) + ksize, np.int8, fixed_rng)

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
        (in_shape[1] + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1) // stride[1]
        + 1,
    )

    def run_conv2d_faster():
        return conv2d_faster(x, out_shape, kernel, stride, padding, dilation, groups)

    def run_conv2d_faster_legacy():
        return conv2d_faster_legacy(
            x, out_shape, kernel, stride, padding, dilation, groups
        )

    if not request.node.get_closest_marker("perf"):
        # ~3x faster
        t1 = timeit.timeit(lambda: run_conv2d_faster(), number=10)
        t2 = timeit.timeit(lambda: run_conv2d_faster_legacy(), number=10)
        print("Use im2col: ", t1, "Use legacy:", t2)

    r_opt = run_conv2d_faster()
    r_legacy = run_conv2d_faster_legacy()
    assert np.array_equal(r_opt, r_legacy)


try:
    from paibox.components.synapses.conv_utils import _conv2d_unroll_legacy

    skip_conv2d_unroll_test = False
except ImportError:
    skip_conv2d_unroll_test = True


test_conv2d_unroll_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, groups",
    data=[
        ((16, 16), 4, 16, (3, 3), (1, 1), (0, 0), 1),
        ((32, 32), 8, 16, (3, 3), (1, 1), (1, 1), 1),
        ((32, 32), 32, 8, (4, 4), (2, 2), (0, 0), 2),
    ],
)


@pytest.mark.skipif(
    skip_conv2d_unroll_test,
    reason="Legacy function '_conv2d_unroll_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv2d_unroll_data["args"], test_conv2d_unroll_data["data"]
)
def test_conv2d_unroll_perf(
    in_shape,
    in_channels,
    out_channels,
    ksize,
    stride,
    padding,
    groups,
    fixed_rng,
    request,
):
    out_shape = (
        (in_shape[0] + 2 * padding[0] - ksize[0]) // stride[0] + 1,
        (in_shape[1] + 2 * padding[1] - ksize[1]) // stride[1] + 1,
    )

    k_shape = (out_channels, in_channels) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    def run_conv2d_unroll():
        return _conv2d_unroll(in_shape, out_shape, kernel, stride, padding, groups)

    def run_conv2d_unroll_legacy():
        return _conv2d_unroll_legacy(
            in_shape, out_shape, kernel, stride, padding, groups
        )

    if not request.node.get_closest_marker("perf"):
        # 50~100x faster
        t1 = timeit.timeit(lambda: run_conv2d_unroll(), number=5)
        t2 = timeit.timeit(lambda: run_conv2d_unroll_legacy(), number=5)
        print("Optimized: ", t1, "Legacy: ", t2)

    r_opt = run_conv2d_unroll()
    r_legacy = run_conv2d_unroll_legacy()
    assert np.array_equal(r_opt, r_legacy)
