from typing import Tuple

import numpy as np
import pytest

from paibox.synapses.transforms import *
from paibox.utils import shape2num


class TestTransforms:
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
        f = OneToOne(num, weight)
        x = np.array([1, 0, 1], dtype=np.bool_)
        y = f(x)
        expected = x * weight

        assert y.dtype == np.int32
        assert y.shape == (num,)
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (num, num)

    def test_OneToOne(self):
        weight = np.array([1, 2, 3, 4], dtype=np.int8)
        f = OneToOne(4, weight)
        assert f.connectivity.shape == (4, 4)

        # The last spike is an array.
        x1 = np.array([1, 2, 3, 4], dtype=np.int8)
        y = f(x1)
        assert y.shape == (4,)

        # The last spike is a scalar.
        x2 = np.array(2, dtype=np.int8)
        y = f(x2)
        assert y.shape == (4,)

    @pytest.mark.parametrize(
        "weight, expected_dtype",
        [
            (1, np.bool_),
            (-1, np.int8),
            (10, np.int8),
            (-100, np.int8),
            (-128, np.int8),
            (127, np.int8),
        ],
        ids=[
            "scalar_1",
            "scalar_-1",
            "scalar_10",
            "scalar_-100",
            "scalar_-128",
            "scalar_-127",
        ],
    )
    def test_AllToAll_weight_scalar(self, weight, expected_dtype):
        """Test `AllToAll` when weight is a scalar"""

        num_in, num_out = 10, 20
        x = np.random.randint(2, size=(10,))
        f = AllToAll((num_in, num_out), weight)
        y = f(x)
        expected = np.full((num_out,), np.sum(x, axis=None), dtype=np.int32) * weight

        assert f.conn_dtype == expected_dtype
        assert y.dtype == np.int32
        assert y.shape == (num_out,)
        assert y.ndim == 1
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (num_in, num_out)

    @pytest.mark.parametrize(
        "shape, x, weights, expected_dtype",
        [
            (
                (3, 4),
                np.random.randint(2, size=(3,), dtype=np.bool_),
                np.random.randint(2, size=(3, 4), dtype=np.bool_),
                np.bool_,
            ),
            (
                (10, 20),
                np.random.randint(2, size=(10,), dtype=np.bool_),
                np.random.randint(127, size=(10, 20), dtype=np.int8),
                np.int8,
            ),
            (
                (20, 10),
                np.random.randint(2, size=(20,), dtype=np.bool_),
                np.random.randint(2, size=(20, 10), dtype=np.int8),
                np.bool_,
            ),
            (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[1, 2], [3, 4]], dtype=np.int8),
                np.int8,
            ),
            (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[127, 0], [3, -128]], dtype=np.int8),
                np.int8,
            ),
        ],
        ids=[
            "weights_bool_1",
            "weights_int8_1",
            "weights_int8_2",
            "weights_int8_3",
            "weights_int8_4",
        ],
    )
    def test_AllToAll_array(self, shape, x, weights, expected_dtype):
        """Test `AllToAll` when weights is an array"""

        f = AllToAll(shape, weights)
        y = f(x)
        expected = x @ weights.copy().astype(np.int32)

        assert f.conn_dtype == expected_dtype
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == shape

    @pytest.mark.parametrize(
        "shape, x, weights, expected_dtype",
        [
            (
                (3, 4),
                np.array([1, 1, 1], dtype=np.bool_),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int8),
                np.int8,
            ),
            (
                (10, 20),
                np.random.randint(2, size=(10,), dtype=np.bool_),
                np.random.randint(-10, 10, size=(10, 20), dtype=np.int8),
                np.int8,
            ),
            (
                (20, 10),
                np.ones((20,), dtype=np.bool_),
                np.random.randint(2, size=(20, 10), dtype=np.int8),
                np.bool_,
            ),
            (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[127, 0], [3, -128]], dtype=np.int8),
                np.int8,
            ),
        ],
        ids=["weights_int8_1", "weights_int8_2", "weights_bool", "weights_int8_3"],
    )
    def test_MaskedLinear_conn(self, shape, x, weights, expected_dtype):
        f = MaskedLinear(shape, weights)
        y = f(x)
        expected = x @ weights.copy().astype(np.int32)

        assert f.conn_dtype == expected_dtype
        assert f.connectivity.dtype == expected_dtype
        assert y.shape == (shape[1],)
        assert y.dtype == np.int32
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == shape

    @staticmethod
    def _conv1d_golden(
        x: np.ndarray,
        out_shape: Tuple[int],
        kernel: np.ndarray,
        stride: Tuple[int],
        padding: Tuple[int],
        fm_order: str,
    ):
        cout, cin, kl = kernel.shape

        if fm_order == "LC":
            _x = x.T
        else:
            _x = x.copy()

        xcin, il = _x.shape

        assert cin == xcin

        ol = (il - kl + 2 * padding[0]) // stride[0] + 1

        assert ol == out_shape[0]

        out = np.zeros((cout,) + out_shape, dtype=np.int64)

        x_padded = np.pad(_x, (0, padding[0]), mode="constant")

        for o in range(cout):
            for i in range(cin):
                conv_result = np.zeros((ol,), dtype=np.int64)
                for l in range(ol):
                    window = x_padded[i, l * stride[0] : l * stride[0] + kl]
                    conv_result[l] = np.sum(window * kernel[o, i, :])

                out[o] += conv_result

        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, fm_order",
        # Padding is fixed at (0, 0)
        [
            ((28,), 16, 8, (3,), (1,), (0,), "CL"),
            ((28,), 24, 12, (3,), (2,), (0,), "CL"),
            ((28,), 24, 12, (5,), (2,), (0,), "CL"),
            ((16,), 8, 16, (3,), (2,), (0,), "CL"),
            ((28,), 16, 8, (3,), (1,), (0,), "LC"),
            ((24,), 8, 8, (3,), (2,), (0,), "LC"),
            ((24,), 8, 16, (7,), (2,), (0,), "LC"),
            ((32,), 4, 12, (5,), (1,), (0,), "LC"),
        ],
    )
    def test_Conv1dForward(
        self,
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        fm_order,
    ):
        kernel = np.random.randint(
            -128, 127, size=(out_channels, in_channels) + kernel_size, dtype=np.int8
        )

        out_shape = ((in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,)

        f = Conv1dForward(in_shape, out_shape, kernel, stride, padding, fm_order)

        if fm_order == "CL":
            fm_shape = (in_channels,) + in_shape
        else:
            fm_shape = in_shape + (in_channels,)

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.copy().flatten()

        y = f(xf)
        expected = self._conv1d_golden(x, out_shape, kernel, stride, padding, fm_order)

        # Flattened output
        y = y.reshape((out_channels,) + out_shape)

        assert y.shape == expected.shape
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @staticmethod
    def _conv2d_golden(
        x: np.ndarray,
        out_shape: Tuple[int, int],
        kernel: np.ndarray,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        fm_order: str,
    ):
        cout, cin, kh, kw = kernel.shape

        if fm_order == "HWC":
            _x = x.transpose(2, 0, 1)
        else:
            _x = x.copy()

        xcin, ih, iw = _x.shape

        assert cin == xcin

        oh = (ih - kh + 2 * padding[0]) // stride[0] + 1
        ow = (iw - kw + 2 * padding[1]) // stride[1] + 1

        assert oh, ow == out_shape

        out = np.zeros((cout,) + out_shape, dtype=np.int64)

        x_padded = np.pad(
            _x,
            ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            mode="constant",
        )

        for o in range(cout):
            for i in range(cin):
                conv_result = np.zeros((oh, ow), dtype=np.int64)
                for h in range(oh):
                    for w in range(ow):
                        window = x_padded[
                            i,
                            h * stride[0] : h * stride[0] + kh,
                            w * stride[1] : w * stride[1] + kw,
                        ]
                        conv_result[h, w] = np.sum(window * kernel[o, i, :, :])

                out[o] += conv_result

        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, fm_order",
        # Padding is fixed at (0, 0)
        [
            ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0), "CHW"),
            ((28, 28), 24, 12, (3, 3), (2, 2), (0, 0), "CHW"),
            ((28, 28), 24, 12, (5, 5), (2, 1), (0, 0), "CHW"),
            ((16, 16), 8, 16, (3, 3), (2, 2), (0, 0), "CHW"),
            ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0), "HWC"),
            ((24, 32), 8, 8, (3, 4), (2, 1), (0, 0), "HWC"),
            ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0), "HWC"),
            ((32, 16), 4, 12, (5, 7), (1, 2), (0, 0), "HWC"),
        ],
    )
    def test_Conv2dForward(
        self,
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        fm_order,
    ):
        kernel = np.random.randint(
            -128, 127, size=(out_channels, in_channels) + kernel_size, dtype=np.int8
        )

        out_shape = (
            (in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
            (in_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
        )

        f = Conv2dForward(in_shape, out_shape, kernel, stride, padding, fm_order)

        if fm_order == "CHW":
            fm_shape = (in_channels,) + in_shape
        else:
            fm_shape = in_shape + (in_channels,)

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.copy().flatten()

        y = f(xf)
        expected = self._conv2d_golden(x, out_shape, kernel, stride, padding, fm_order)

        # Flattened output
        y = y.reshape((out_channels,) + out_shape)

        assert y.shape == expected.shape
        assert np.array_equal(y, expected)
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )
