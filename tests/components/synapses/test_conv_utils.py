import numpy as np
import pytest
import timeit

from paibox.components.synapses.conv_utils import (
    _conv2d_unroll,
    conv1d_faster,
    conv2d_faster,
)
from paibox.components.synapses import conv_utils
from tests.conftest import ParametrizedTestData

RNG = np.random.default_rng()


test_conv1d_faster_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups",
    data=[
        ((2048,), 16, 32, (32,), (1,), (1,), (8,), 2),
        ((512,), 64, 32, (16,), (8,), (0,), (4,), 8),
    ],
)


@pytest.mark.skipif(
    not hasattr(conv_utils, "conv1d_faster_legacy"),
    reason="Legacy function 'conv1d_faster_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv1d_faster_data["args"], test_conv1d_faster_data["data"]
)
def test_conv1d_faster_perf(
    in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups
):
    cin_per_grp = in_channels // groups

    x = RNG.integers(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max + 1,
        size=(in_channels,) + in_shape,
        dtype=np.uint8,
    )
    kernel = RNG.integers(
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max + 1,
        size=(out_channels, cin_per_grp) + ksize,
        dtype=np.int8,
    )

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
    )

    def run_conv1d_faster():
        return conv1d_faster(x, out_shape, kernel, stride, padding, dilation, groups)

    def run_conv1d_faster_legacy():
        return conv_utils.conv1d_faster_legacy(
            x, out_shape, kernel, stride, padding, dilation, groups
        )

    t1 = timeit.timeit(lambda: run_conv1d_faster(), number=100)
    t2 = timeit.timeit(lambda: run_conv1d_faster_legacy(), number=100)
    print("Use im2col: ", t1, "Use legacy:", t2)


@pytest.mark.skipif(
    not hasattr(conv_utils, "conv1d_faster_legacy"),
    reason="Legacy function 'conv1d_faster_legacy' is removed",
)
@pytest.mark.parametrize(
    test_conv1d_faster_data["args"], test_conv1d_faster_data["data"]
)
def test_conv1d_faster(
    in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups
):
    cin_per_grp = in_channels // groups

    x = RNG.integers(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max + 1,
        size=(in_channels,) + in_shape,
        dtype=np.uint8,
    )
    kernel = RNG.integers(
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max + 1,
        size=(out_channels, cin_per_grp) + ksize,
        dtype=np.int8,
    )

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
    )

    out1 = conv1d_faster(x, out_shape, kernel, stride, padding, dilation, groups)
    out2 = conv_utils.conv1d_faster_legacy(
        x, out_shape, kernel, stride, padding, dilation, groups
    )
    assert np.array_equal(out1, out2)


test_conv2d_faster_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups",
    data=[
        ((64, 64), 16, 32, (3, 3), (1, 1), (1, 1), (2, 2), 4),
        ((32, 32), 24, 12, (3, 3), (2, 2), (2, 1), (1, 1), 3),
        ((28, 28), 16, 8, (3, 3), (1, 2), (0, 0), (1, 1), 8),
    ],
)


@pytest.mark.skipif(
    not hasattr(conv_utils, "conv2d_faster_legacy"),
    reason="Legacy function 'conv2d_faster_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv2d_faster_data["args"], test_conv2d_faster_data["data"]
)
def test_conv2d_faster_perf(
    in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups
):
    cin_per_grp = in_channels // groups

    x = RNG.integers(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max + 1,
        size=(in_channels,) + in_shape,
        dtype=np.uint8,
    )
    kernel = RNG.integers(
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max + 1,
        size=(out_channels, cin_per_grp) + ksize,
        dtype=np.int8,
    )

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
        (in_shape[1] + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1) // stride[1]
        + 1,
    )

    def run_conv2d_faster():
        return conv2d_faster(x, out_shape, kernel, stride, padding, dilation, groups)

    def run_conv2d_faster_legacy():
        return conv_utils.conv2d_faster_legacy(
            x, out_shape, kernel, stride, padding, dilation, groups
        )

    t1 = timeit.timeit(lambda: run_conv2d_faster(), number=100)
    t2 = timeit.timeit(lambda: run_conv2d_faster_legacy(), number=100)
    print("Use im2col: ", t1, "Use legacy:", t2)


@pytest.mark.skipif(
    not hasattr(conv_utils, "conv2d_faster_legacy"),
    reason="Legacy function 'conv2d_faster_legacy' is removed",
)
@pytest.mark.parametrize(
    test_conv2d_faster_data["args"], test_conv2d_faster_data["data"]
)
def test_conv2d_faster(
    in_shape, in_channels, out_channels, ksize, stride, padding, dilation, groups
):
    cin_per_grp = in_channels // groups

    x = RNG.integers(
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max + 1,
        size=(in_channels,) + in_shape,
        dtype=np.uint8,
    )
    kernel = RNG.integers(
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max + 1,
        size=(out_channels, cin_per_grp) + ksize,
        dtype=np.int8,
    )

    out_shape = (
        (in_shape[0] + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0]
        + 1,
        (in_shape[1] + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1) // stride[1]
        + 1,
    )

    out1 = conv2d_faster(x, out_shape, kernel, stride, padding, dilation, groups)
    out2 = conv_utils.conv2d_faster_legacy(
        x, out_shape, kernel, stride, padding, dilation, groups
    )
    assert np.array_equal(out1, out2)


test_conv2d_unroll_data = ParametrizedTestData(
    args="in_shape, in_channels, out_channels, ksize, stride, padding, groups",
    data=[
        ((32, 32), 8, 16, (3, 3), (1, 1), (1, 1), 1),
        ((32, 32), 8, 16, (4, 4), (2, 2), (0, 0), 4),
    ],
)


@pytest.mark.skipif(
    not hasattr(conv_utils, "_conv2d_unroll_legacy"),
    reason="Legacy function '_conv2d_unroll_legacy' is removed",
)
@pytest.mark.perf
@pytest.mark.parametrize(
    test_conv2d_unroll_data["args"], test_conv2d_unroll_data["data"]
)
def test_conv2d_unroll_perf(
    in_shape, in_channels, out_channels, ksize, stride, padding, groups
):
    out_shape = (
        (in_shape[0] + 2 * padding[0] - ksize[0]) // stride[0] + 1,
        (in_shape[1] + 2 * padding[1] - ksize[1]) // stride[1] + 1,
    )

    k_shape = (out_channels, in_channels) + ksize
    kernel = RNG.integers(
        np.iinfo(np.int8).min, np.iinfo(np.int8).max + 1, size=k_shape, dtype=np.int8
    )

    def run_conv2d_unroll():
        return _conv2d_unroll(in_shape, out_shape, kernel, stride, padding, groups)

    def run_conv2d_unroll_legacy():
        return conv_utils._conv2d_unroll_legacy(
            in_shape, out_shape, kernel, stride, padding, groups
        )

    t1 = timeit.timeit(lambda: run_conv2d_unroll(), number=5)
    t2 = timeit.timeit(lambda: run_conv2d_unroll_legacy(), number=5)
    print("Optimized: ", t1, "Legacy: ", t2)


@pytest.mark.skipif(
    not hasattr(conv_utils, "_conv2d_unroll_legacy"),
    reason="Legacy function '_conv2d_unroll_legacy' is removed",
)
@pytest.mark.parametrize(
    test_conv2d_unroll_data["args"], test_conv2d_unroll_data["data"]
)
def test_conv2d_unroll(
    in_shape, in_channels, out_channels, ksize, stride, padding, groups
):
    out_shape = (
        (in_shape[0] + 2 * padding[0] - ksize[0]) // stride[0] + 1,
        (in_shape[1] + 2 * padding[1] - ksize[1]) // stride[1] + 1,
    )

    k_shape = (out_channels, in_channels) + ksize
    kernel = RNG.integers(
        np.iinfo(np.int8).min, np.iinfo(np.int8).max + 1, size=k_shape, dtype=np.int8
    )

    out1 = _conv2d_unroll(in_shape, out_shape, kernel, stride, padding, groups)
    out2 = conv_utils._conv2d_unroll_legacy(
        in_shape, out_shape, kernel, stride, padding, groups
    )
    assert np.array_equal(out1, out2)
