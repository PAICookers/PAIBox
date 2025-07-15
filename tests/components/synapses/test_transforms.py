import numpy as np
import pytest

from paibox.components.synapses import transforms as tfm
from paibox.components.synapses.conv_utils import _conv1d_oshape, _conv2d_oshape
from paibox.exceptions import AutoOptimizationWarning
from paibox.types import VOLTAGE_DTYPE, WEIGHT_DTYPE
from paibox.utils import shape2num

from tests.utils import gen_random_array
from tests.components.utils import *


class TestTransforms:
    @pytest.mark.parametrize(
        "weight",
        [
            np.array([1, 2, 3], dtype=np.int8),
            np.array([1, 0, 1], dtype=np.bool_),
            np.array([True, False]),
            np.array([True, False], dtype=np.int8),
            10,
            1,
            True,
            np.int8(1),  # automatically optimizated
            np.uint8(99),
            np.array([-128, 1, 127], dtype=np.int8),
            [1, 2, 3],
            (0, 1, 0, 1),
        ],
    )
    def test_weight_dtype_convert(self, weight):
        t = tfm.Transform(weight)
        assert t.weights.dtype == WEIGHT_DTYPE

    @pytest.mark.parametrize(
        "weight",
        [
            np.array([1, 2, 3]),
            # Only automatically optimized to int8 unless specified as bool
            np.array([True, False], dtype=np.int16),
            np.array([1, 0, 1], dtype=np.int16),  # Same as above
            np.array([-128, 1, 127], dtype=np.int32),
            np.array([-8, 4, 7]),
            [-100, 0, 100],
        ],
    )
    def test_weight_dtype_convert_warning(self, weight):
        with pytest.warns(AutoOptimizationWarning):
            t = tfm.Transform(weight)

        assert t.weights.dtype == WEIGHT_DTYPE

    @pytest.mark.parametrize(
        "weight",
        [
            (np.array([1.0, 2.1, 3.2])),  # float is forbidden
            (np.array([1, 2, 3], dtype=np.float32)),
            (np.array([111, 222, -333], dtype=np.int16)),  # out of range int8
            (999),
            (3.14),
            ([-100, 200, 0]),
            ((1.1, 0.5)),
        ],
    )
    def test_weight_dtype_convert_illegal(self, weight):
        with pytest.raises((TypeError, ValueError)):
            t = tfm.Transform(weight)

    @pytest.mark.parametrize(
        "weight",
        [
            (np.array([1, 2, 3], dtype=np.int8)),
            (np.array([1, 0, 1], dtype=np.bool_)),
            (np.array([1, 0, 1], dtype=np.int8)),
            (10),
            (np.int8(-1)),
            (np.array([127, 0, 1], dtype=np.int8)),
            (np.array([-128, 1, 127], dtype=np.int8)),
        ],
        ids=[
            "array_1",
            "array_2",
            "array_3",
            "scalar_pos",
            "scalar_neg",
            "array_int8_1",
            "array_int8_2",
        ],
    )
    def test_OneToOne_dtype(self, weight):
        num = 3
        f = tfm.OneToOne(num, weight)
        x = np.array([1, 0, 1], dtype=np.uint8)
        y = f(x)
        expected = x * weight

        assert y.dtype == np.int32
        assert y.shape == (num,)
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (num, num)

    def test_OneToOne(self):
        weight = np.array([1, 2, 3, 4], dtype=np.int8)
        f = tfm.OneToOne(4, weight)
        assert f.connectivity.shape == (4, 4)

        # The last spike is an array.
        x1 = np.array([0, 1, 1, 0], dtype=np.uint8)
        y = f(x1)
        assert y.shape == (4,)

        # The last spike is a scalar.
        x2 = np.array(1, dtype=np.uint8)
        y = f(x2)
        assert y.shape == (4,)

    @pytest.mark.parametrize(
        "weight",
        [1, -1, 10, -100, -128, 127],
        ids=[
            "scalar_1",
            "scalar_-1",
            "scalar_10",
            "scalar_-100",
            "scalar_-128",
            "scalar_-127",
        ],
    )
    def test_AllToAll_weight_scalar(self, weight):
        """Test `AllToAll` when weight is a scalar"""

        num_in, num_out = 10, 20
        x = gen_random_array((num_in,), np.bool_)
        f = tfm.AllToAll((num_in, num_out), weight)
        y = f(x)
        expected = np.full((num_out,), np.sum(x, axis=None), dtype=np.int32) * weight

        assert f.connectivity.dtype == WEIGHT_DTYPE
        assert y.dtype == np.int32
        assert y.shape == (num_out,)
        assert y.ndim == 1
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (num_in, num_out)

    @pytest.mark.parametrize(
        "shape, x, weights",
        [
            (
                (3, 4),
                gen_random_array((3,), np.bool_),
                gen_random_array((3, 4), np.bool_),
            ),
            (
                (10, 20),
                gen_random_array((10,), np.bool_),
                gen_random_array((10, 20), np.int8),
            ),
            (
                (20, 10),
                gen_random_array((20,), np.bool_),
                gen_random_array((20, 10), np.bool_),
            ),
            (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[1, 2], [3, 4]], dtype=np.int8),
            ),
            (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[127, 0], [3, -128]], dtype=np.int8),
            ),
        ],
        ids=["bool_1", "int8_1", "int8_2", "int8_3", "int8_4"],
    )
    def test_AllToAll_array(self, shape, x, weights):
        """Test `AllToAll` when weights is an array"""

        f = tfm.AllToAll(shape, weights)
        y = f(x)
        expected = x @ weights.astype(np.int32)

        assert f.connectivity.dtype == WEIGHT_DTYPE
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == shape

    @pytest.mark.parametrize(
        "x, weights",
        [
            (
                np.arange(12, dtype=np.int8).reshape(3, 4),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int8),
            ),
            (gen_random_array((10,), np.bool_), gen_random_array((10, 20), np.int8)),
            (np.ones((20, 10), dtype=np.bool_), gen_random_array((20, 10), np.bool_)),
            (
                np.array((1, 1), dtype=np.bool_),
                np.array([[127, 0], [3, -128]], dtype=np.int8),
            ),
        ],
    )
    def test_MaskedLinear(
        self,
        x,
        weights,
    ):
        if x.ndim == 1:
            in_shape = (1, x.shape[0])
        else:
            in_shape = x.shape

        if in_shape[0] == weights.shape[0]:
            axes = (1, 0)
        else:
            axes = (0, 1)

        _in_shape = tuple(in_shape[i] for i in axes)
        oshape = _in_shape[:-1] + weights.shape[1:]

        f = tfm.MaskedLinear(x.shape, oshape, weights)
        y = f(x)
        y2 = x.ravel() @ f.connectivity.astype(np.int32)
        expected = x.reshape(in_shape).transpose(axes) @ weights.astype(np.int32)

        assert f.connectivity.dtype == WEIGHT_DTYPE
        assert y.shape == oshape
        assert y2.dtype == np.int32
        assert np.array_equal(y, expected)
        assert np.array_equal(y2, y.ravel())
        assert f.connectivity.shape == (x.size, y.size)

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, ksize, stride, padding, groups, kdtype",
        [
            (np.bool_, (8,), 16, 8, (3,), (1,), (1,), 2, np.int8),
            (np.bool_, (28,), 16, 8, (3,), (1,), (1,), 4, np.bool_),
            (np.bool_, (28,), 24, 12, (3,), (2,), (2,), 3, np.bool_),
            (np.bool_, (28,), 24, 12, (5,), (2,), (2,), 6, np.bool_),
            (np.bool_, (16,), 8, 16, (3,), (2,), (0,), 8, np.bool_),
            (np.bool_, (28,), 16, 8, (3,), (1,), (0,), 1, np.int8),
            (np.bool_, (28,), 24, 12, (3,), (2,), (0,), 1, np.int8),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), 4, np.int8),
            (np.bool_, (16,), 8, 16, (3,), (2,), (0,), 2, np.int8),
            (np.uint8, (8,), 16, 8, (3,), (1,), (1,), 2, np.int8),
            (np.uint8, (28,), 16, 8, (3,), (1,), (1,), 4, np.bool_),
            (np.uint8, (28,), 24, 12, (3,), (2,), (2,), 3, np.bool_),
            (np.int8, (28,), 24, 12, (5,), (2,), (2,), 3, np.bool_),
            (np.int8, (16,), 8, 16, (3,), (2,), (0,), 8, np.bool_),
            (np.int8, (28,), 16, 8, (3,), (1,), (0,), 1, np.int8),
            (np.int8, (28,), 24, 12, (3,), (2,), (0,), 1, np.int8),
            (np.int8, (28,), 24, 12, (5,), (2,), (0,), 4, np.int8),
            (np.int8, (16,), 8, 16, (3,), (2,), (0,), 8, np.int8),
        ],
    )
    def test_Conv1dForward(
        self,
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        kdtype,
    ):
        ci_in_grp = in_channels // groups
        x_shape = (in_channels,) + in_shape

        kernel = gen_random_array((out_channels, ci_in_grp) + ksize, kdtype)
        x = gen_random_array(x_shape, xdtype)

        out_shape = _conv1d_oshape(in_shape, ksize, stride, padding)
        f = tfm.Conv1dForward(
            in_shape, out_shape, kernel, stride, padding, groups=groups
        )

        xf = x.ravel()

        # The result of traditional conv
        ygolden = conv1d_golden(
            x, out_shape, kernel, stride, padding, groups=groups
        ).astype(VOLTAGE_DTYPE)

        # The result of __call__ using faster conv
        y1 = f(xf)

        # The result of matmul using the unrolled matrix
        w_unrolled = f.connectivity
        y2 = (xf.astype(np.int32) @ w_unrolled).astype(VOLTAGE_DTYPE)

        assert np.array_equal(ygolden, y1)
        assert np.array_equal(y2, y1.ravel())
        assert w_unrolled.shape == (
            shape2num(x_shape),
            shape2num((out_channels,) + out_shape),
        )

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, ksize, stride, padding, groups, kdtype",
        [
            (np.bool_, (28, 28), 16, 8, (3, 3), (1, 1), (1, 1), 2, np.bool_),
            (np.bool_, (28, 28), 24, 12, (3, 3), (2, 2), (2, 1), 3, np.bool_),
            (np.bool_, (28, 28), 16, 8, (3, 3), (1, 1), (2, 3), 8, np.bool_),
            (np.bool_, (28, 28), 24, 12, (3, 3), (2, 2), (0, 0), 1, np.int8),
            (np.bool_, (28, 28), 24, 12, (5, 5), (2, 1), (0, 0), 4, np.int8),
            (np.bool_, (8, 8), 8, 16, (3, 3), (2, 2), (1, 1), 1, np.int8),
            (np.uint8, (28, 28), 16, 8, (3, 3), (1, 1), (1, 1), 8, np.bool_),
            (np.uint8, (28, 28), 24, 12, (3, 3), (2, 2), (2, 1), 1, np.bool_),
            (np.int8, (28, 28), 16, 8, (3, 3), (1, 1), (2, 3), 4, np.bool_),
            (np.int8, (28, 28), 24, 12, (3, 3), (2, 2), (0, 0), 12, np.int8),
            (np.int8, (28, 28), 24, 12, (5, 5), (2, 1), (0, 0), 3, np.int8),
            (np.int8, (8, 8), 8, 16, (3, 3), (2, 2), (1, 1), 2, np.int8),
        ],
    )
    def test_Conv2dForward(
        self,
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        ksize,
        stride,
        padding,
        groups,
        kdtype,
    ):
        ci_in_grp = in_channels // groups
        x_shape = (in_channels,) + in_shape

        kernel = gen_random_array((out_channels, ci_in_grp) + ksize, kdtype)
        x = gen_random_array(x_shape, xdtype)

        out_shape = _conv2d_oshape(in_shape, ksize, stride, padding)

        f = tfm.Conv2dForward(
            in_shape, out_shape, kernel, stride, padding, groups=groups
        )

        xf = x.ravel()

        # The result of traditional conv
        ygolden = conv2d_golden(
            x, out_shape, kernel, stride, padding, groups=groups
        ).astype(VOLTAGE_DTYPE)

        # The result of __call__ using faster conv
        y1 = f(xf)

        # The result of matmul using the unrolled matrix
        w_unrolled = f.connectivity
        y2 = (xf.astype(np.int32) @ w_unrolled).astype(VOLTAGE_DTYPE)

        assert np.array_equal(ygolden, y1)
        assert np.array_equal(y2, y1.ravel())
        assert w_unrolled.shape == (
            shape2num(x_shape),
            shape2num((out_channels,) + out_shape),
        )

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, ksize, stride, padding, output_padding, kdtype",
        [
            (np.bool_, (28,), 16, 8, (3,), (1,), (0,), (0,), np.bool_),
            (np.bool_, (28,), 24, 12, (3,), (2,), (2,), (2,), np.bool_),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), (1,), np.bool_),
            (np.bool_, (16,), 8, 16, (3,), (2,), (1,), (0,), np.bool_),
            (np.bool_, (28,), 16, 8, (3,), (3,), (0,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (3,), (2,), (3,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), (0,), np.int8),
            (np.bool_, (16,), 8, 16, (3,), (2,), (1,), (1,), np.int8),
            (np.uint8, (28,), 16, 8, (3,), (1,), (0,), (0,), np.bool_),
            (np.uint8, (28,), 24, 12, (3,), (2,), (2,), (2,), np.bool_),
            (np.uint8, (28,), 24, 12, (5,), (2,), (0,), (1,), np.bool_),
            (np.uint8, (16,), 8, 16, (3,), (2,), (1,), (0,), np.bool_),
            (np.int8, (28,), 16, 8, (3,), (3,), (0,), (0,), np.int8),
            (np.int8, (28,), 24, 12, (3,), (2,), (3,), (0,), np.int8),
            (np.int8, (28,), 24, 12, (5,), (2,), (0,), (0,), np.int8),
            (np.int8, (16,), 8, 16, (3,), (2,), (1,), (1,), np.int8),
            # ((28,), 16, 8, (3,), (1,), (0,), "LC"),
            # ((24,), 8, 8, (3,), (2,), (0,), "LC"),
            # ((24,), 8, 16, (7,), (2,), (0,), "LC"),
            # ((32,), 4, 12, (5,), (1,), (0,), "LC"),
        ],
    )
    def test_ConvTranspose1dForward(
        self,
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        ksize,
        stride,
        padding,
        output_padding,
        kdtype,
    ):
        x_shape = (in_channels,) + in_shape
        kernel = gen_random_array((out_channels, in_channels) + ksize, kdtype)
        x = gen_random_array(x_shape, xdtype)

        out_shape = (
            (in_shape[0] - 1) * stride[0]
            - 2 * padding[0]
            + ksize[0]
            + output_padding[0],
        )
        f = tfm.ConvTranspose1dForward(
            in_shape, out_shape, kernel, stride, padding, output_padding=output_padding
        )

        xf = x.ravel()

        # The result of traditional conv
        ygolden = convtranspose1d_golden(
            x, out_shape, kernel, stride, padding, output_padding
        )

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        assert np.array_equal(y1, ygolden)
        assert np.array_equal(y2, y1.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, ksize, stride, padding, output_padding, kdtype",
        [
            (np.bool_, (12, 12), 16, 8, (3, 3), (1, 1), (1, 1), (1, 1), np.bool_),
            (np.bool_, (12, 12), 24, 12, (3, 3), (2, 2), (2, 2), (1, 0), np.bool_),
            (np.bool_, (12, 12), 16, 8, (3, 3), (1, 1), (0, 0), (0, 0), np.bool_),
            (np.bool_, (12, 12), 24, 12, (3, 3), (2, 2), (1, 2), (0, 1), np.int8),
            (np.bool_, (10, 10), 24, 12, (5, 5), (2, 1), (1, 1), (2, 2), np.int8),
            (np.bool_, (16, 16), 8, 16, (3, 3), (2, 2), (1, 3), (2, 0), np.int8),
            (np.uint8, (12, 12), 16, 8, (3, 3), (1, 1), (1, 1), (1, 1), np.bool_),
            (np.uint8, (12, 12), 24, 12, (3, 3), (2, 2), (2, 2), (1, 0), np.bool_),
            (np.int8, (12, 12), 16, 8, (3, 3), (1, 1), (0, 0), (0, 0), np.bool_),
            (np.int8, (12, 12), 24, 12, (3, 3), (2, 2), (1, 2), (0, 1), np.int8),
            (np.int8, (10, 10), 24, 12, (5, 5), (2, 1), (1, 1), (2, 2), np.int8),
            (np.int8, (16, 16), 8, 16, (3, 3), (2, 2), (1, 3), (2, 0), np.int8),
            # ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0), "HWC", np.bool_),
            # ((24, 32), 8, 8, (3, 4), (2, 1), (0, 0), "HWC", np.bool_),
            # ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0), "HWC", np.bool_),
            # ((32, 16), 4, 12, (5, 7), (1, 2), (0, 0), "HWC", np.int8),
            # ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0), "HWC", np.int8),
            # ((32, 16), 4, 12, (5, 7), (1, 2), (0, 0), "HWC", np.int8),
        ],
    )
    def test_ConvTranspose2dForward(
        self,
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        ksize,
        stride,
        padding,
        output_padding,
        kdtype,
    ):
        x_shape = (in_channels,) + in_shape
        kernel = gen_random_array((out_channels, in_channels) + ksize, kdtype)
        x = gen_random_array(x_shape, xdtype)

        out_shape = (
            (in_shape[0] - 1) * stride[0]
            - 2 * padding[0]
            + ksize[0]
            + output_padding[0],
            (in_shape[1] - 1) * stride[1]
            - 2 * padding[1]
            + ksize[1]
            + output_padding[1],
        )

        f = tfm.ConvTranspose2dForward(
            in_shape, out_shape, kernel, stride, padding, output_padding=output_padding
        )

        xf = x.ravel()

        # The result of traditional conv
        ygolden = convtranspose2d_golden(
            x, out_shape, kernel, stride, padding, output_padding
        )

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        assert np.array_equal(y1, ygolden)
        assert np.array_equal(y2, y1.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @pytest.mark.parametrize("n_compare, n_group", [(4, 8), (9, 12), (25, 1)])
    def test_CompareMax(self, n_compare, n_group):
        n = n_compare * n_group
        w = np.zeros((n, n_group), dtype=np.int8)
        for i in range(n_group):
            w[n_compare * i : n_compare * (i + 1), i] = 1

        f = tfm.CompareMax((n, n_group), w)

        x = gen_random_array((n_compare, n_group), np.uint8)
        x_cm_order = x.ravel(order="F")
        y1 = f(x_cm_order)  # flatten in column-major order
        expected = np.zeros((n_group,), dtype=np.uint8)

        for i in range(n_group):
            expected[i] = np.max(x[:, i])

        assert np.array_equal(y1, expected)
