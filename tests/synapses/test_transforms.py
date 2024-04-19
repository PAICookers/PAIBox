from typing import Tuple

import numpy as np
import pytest

from paibox.exceptions import AutoOptimizationWarning
from paibox.synapses.transforms import *
from paibox.synapses.transforms import Transform
from paibox.utils import shape2num


class TestTransforms:
    @pytest.mark.parametrize(
        "weight, expected_dtype",
        [
            (np.array([1, 2, 3], dtype=np.int8), np.int8),
            (np.array([1, 0, 1], dtype=np.bool_), np.bool_),
            (np.array([True, False]), np.bool_),
            (np.array([True, False], dtype=np.int8), np.int8),
            (10, np.int8),
            (1, np.bool_),
            (True, np.bool_),
            (np.int8(1), np.bool_),  # automatically optimizated
            (np.uint8(99), np.int8),
            (np.array([-128, 1, 127], dtype=np.int8), np.int8),
            ([1, 2, 3], np.int8),
            ((0, 1, 0, 1), np.int8),
        ],
    )
    def test_weight_dtype_convert(self, weight, expected_dtype):
        tfm = Transform(weight)
        assert tfm.weights.dtype == expected_dtype

    @pytest.mark.parametrize(
        "weight, expected_dtype",
        [
            (np.array([1, 2, 3]), np.int8),
            # Only automatically optimized to int8 unless specified as bool
            (np.array([True, False], dtype=np.int16), np.int8),
            (np.array([1, 0, 1], dtype=np.int16), np.int8),  # Same as above
            (np.array([-128, 1, 127], dtype=np.int32), np.int8),
            (np.array([-8, 4, 7]), np.int8),
            ([-100, 0, 100], np.int8),
        ],
    )
    def test_weight_dtype_convert_warning(self, weight, expected_dtype):
        with pytest.warns(AutoOptimizationWarning):
            tfm = Transform(weight)

        assert tfm.weights.dtype == expected_dtype

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
            tfm = Transform(weight)

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

        assert f.connectivity.dtype == expected_dtype
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
                np.random.randint(2, size=(20, 10), dtype=np.bool_),
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

        assert f.connectivity.dtype == expected_dtype
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
                np.random.randint(2, size=(20, 10), dtype=np.bool_),
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

        # if fm_order == "LC":
        #     _x = x.T
        # else:
        #     _x = x.copy()
        _x = x.copy()

        xcin, il = _x.shape

        assert cin == xcin

        ol = (il - kl + 2 * padding[0]) // stride[0] + 1

        assert ol == out_shape[0]

        out = np.zeros((cout,) + out_shape, dtype=np.int64)

        x_padded = np.pad(_x, ((0, 0), (padding[0], padding[0])), mode="constant")

        for o in range(cout):
            for i in range(cin):
                conv_result = np.zeros((ol,), dtype=np.int64)
                for l in range(ol):
                    window = x_padded[i, l * stride[0] : l * stride[0] + kl]
                    conv_result[l] = np.sum(window * kernel[o, i, :])

                out[o] += conv_result

        # if fm_order == "LC":
        #     return out.T
        # else:
        #     return out
        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, fm_order, kdtype",
        # Padding is fixed at (0, 0)
        [
            ((8,), 16, 8, (3,), (1,), (1,), "CL", np.int8),
            ((28,), 16, 8, (3,), (1,), (1,), "CL", np.bool_),
            ((28,), 24, 12, (3,), (2,), (2,), "CL", np.bool_),
            ((28,), 24, 12, (5,), (2,), (2,), "CL", np.bool_),
            ((16,), 8, 16, (3,), (2,), (0,), "CL", np.bool_),
            ((28,), 16, 8, (3,), (1,), (0,), "CL", np.int8),
            ((28,), 24, 12, (3,), (2,), (0,), "CL", np.int8),
            ((28,), 24, 12, (5,), (2,), (0,), "CL", np.int8),
            ((16,), 8, 16, (3,), (2,), (0,), "CL", np.int8),
            # ((28,), 16, 8, (3,), (1,), (0,), "LC"),
            # ((24,), 8, 8, (3,), (2,), (0,), "LC"),
            # ((24,), 8, 16, (7,), (2,), (0,), "LC"),
            # ((32,), 4, 12, (5,), (1,), (0,), "LC"),
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
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        out_shape = ((in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,)

        f = Conv1dForward(in_shape, out_shape, kernel, stride, padding)

        # if fm_order == "CL":
        #     fm_shape = (in_channels,) + in_shape
        # else:
        #     fm_shape = in_shape + (in_channels,)
        fm_shape = (in_channels,) + in_shape

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)
        y3 = f.connectivity.astype(np.int32)

        expected = self._conv1d_golden(x, out_shape, kernel, stride, padding, fm_order)

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
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

        # if fm_order == "HWC":
        #     _x = x.transpose(2, 0, 1)
        # else:
        #     _x = x
        _x = x

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

        # if fm_order == "HWC":
        #     return out.transpose(1, 2, 0)
        # else:
        #     return out
        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, fm_order, kdtype",
        # Padding is fixed at (0, 0)
        [
            ((28, 28), 16, 8, (3, 3), (1, 1), (1, 1), "CHW", np.bool_),
            ((28, 28), 24, 12, (3, 3), (2, 2), (2, 1), "CHW", np.bool_),
            ((28, 28), 16, 8, (3, 3), (1, 1), (2, 3), "CHW", np.bool_),
            ((28, 28), 24, 12, (3, 3), (2, 2), (0, 0), "CHW", np.int8),
            ((28, 28), 24, 12, (5, 5), (2, 1), (0, 0), "CHW", np.int8),
            ((8, 8), 8, 16, (3, 3), (2, 2), (1, 1), "CHW", np.int8),
            # ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0), "HWC", np.bool_),
            # ((24, 32), 8, 8, (3, 4), (2, 1), (0, 0), "HWC", np.bool_),
            # ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0), "HWC", np.bool_),
            # ((32, 16), 4, 12, (5, 7), (1, 2), (0, 0), "HWC", np.int8),
            # ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0), "HWC", np.int8),
            # ((32, 16), 4, 12, (5, 7), (1, 2), (0, 0), "HWC", np.int8),
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
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        out_shape = (
            (in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
            (in_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
        )

        f = Conv2dForward(in_shape, out_shape, kernel, stride, padding)

        # if fm_order == "CHW":
        #     fm_shape = (in_channels,) + in_shape
        # else:
        #     fm_shape = in_shape + (in_channels,)
        fm_shape = (in_channels,) + in_shape

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        # y3 = f.connectivity.astype(np.int32)
        y2 = xf @ f.connectivity.astype(np.int32)


        expected = self._conv2d_golden(x, out_shape, kernel, stride, padding, fm_order)

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @staticmethod
    def _convtranspose1d_golden(
        x: np.ndarray,
        out_shape: Tuple[int],
        kernel: np.ndarray,
        stride: Tuple[int],
        padding: Tuple[int],
        output_padding: Tuple[int],
        # fm_order: str,
    ):
        cout, cin, kl = kernel.shape

        # if fm_order == "LC":
        #     _x = x.T
        # else:
        #     _x = x.copy()

        xcin, il = x.shape

        assert cin == xcin

        ol = (il - 1) * stride[0] - 2 * padding[0] + kl + output_padding[0]

        assert ol == out_shape[0]

        nol = ol - output_padding[0] + 2 * padding[0]

        out = np.zeros((cout,) + (nol,), dtype=np.int64)

        # generate new input array : transpose padding 0 & stride 0
        # Insert 0 between rows and columns (for stride)
        xc_t = xcin
        xl_t = il + (il - 1) * (stride[0] - 1)
        x_transpose = np.zeros((xc_t, xl_t), dtype=x.dtype)
        x_transpose[::1, :: stride[0]] = x
        # padding 0 for transpose not for parameter padding, get new input array x_transpose
        x_transpose = np.pad(x_transpose, ((0, 0), (kl - 1, kl - 1)), mode="constant")

        kernel_flip = np.flip(kernel, axis=2)
        stride_transpose = 1
        for o in range(cout):
            for i in range(cin):
                conv_result = np.zeros((nol,), dtype=np.int64)
                for l in range(nol):
                    window = x_transpose[
                        i, l * stride_transpose : l * stride_transpose + kl
                    ]
                    conv_result[l] = np.sum(window * kernel_flip[o, i, :])

                out[o] += conv_result

        # inverse padding : (cout, (xl-1)*stride+kernel) -> (cout, (xl-1)*stride+kernel-2*padding)
        out = (
            out[:, padding[0]: (-1 * padding[0])]
            if padding[0] > 0
            else out
        )

        # output_padding
        out = np.pad(out, ((0, 0), (0, output_padding[0])), mode="constant")

        # if fm_order == "LC":
        #     return out.T
        # else:
        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, output_padding, fm_order, kdtype",
        # Padding is fixed at (0, 0)
        [
            ((28,), 16, 8, (3,), (1,), (0,), (0,), "CL", np.bool_),
            ((28,), 24, 12, (3,), (2,), (2,), (2,), "CL", np.bool_),
            ((28,), 24, 12, (5,), (2,), (0,), (1,), "CL", np.bool_),
            ((16,), 8, 16, (3,), (2,), (1,), (0,), "CL", np.bool_),
            ((28,), 16, 8, (3,), (3,), (0,), (0,), "CL", np.int8),
            ((28,), 24, 12, (3,), (2,), (3,), (0,), "CL", np.int8),
            ((28,), 24, 12, (5,), (2,), (0,), (0,), "CL", np.int8),
            ((16,), 8, 16, (3,), (2,), (1,), (1,), "CL", np.int8),
            # ((28,), 16, 8, (3,), (1,), (0,), "LC"),
            # ((24,), 8, 8, (3,), (2,), (0,), "LC"),
            # ((24,), 8, 16, (7,), (2,), (0,), "LC"),
            # ((32,), 4, 12, (5,), (1,), (0,), "LC"),
        ],
    )
    def test_ConvTranspose1dForward(
        self,
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        fm_order,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        # out_shape = ((in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,)
        out_shape = ((in_shape[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0],)
        f = ConvTranspose1dForward(in_shape, out_shape, kernel, stride, padding, output_padding)

        # if fm_order == "CL":
        #     fm_shape = (in_channels,) + in_shape
        # else:
        #     fm_shape = in_shape + (in_channels,)
        fm_shape = (in_channels,) + in_shape

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        expected = self._convtranspose1d_golden(x, out_shape, kernel, stride, padding, output_padding)

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @staticmethod
    def _convtranspose2d_golden(
        x: np.ndarray,
        out_shape: Tuple[int, int],
        kernel: np.ndarray,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        output_padding: Tuple[int, int]
        # fm_order: str,
    ):
        cout, cin, kh, kw = kernel.shape

        # if fm_order == "HWC":
        #     _x = x.transpose(2, 0, 1)
        # else:
        #     _x = x

        xcin, ih, iw = x.shape

        assert cin == xcin

        oh = (ih - 1) * stride[0] - 2 * padding[0] + kh + output_padding[0]
        ow = (iw - 1) * stride[1] - 2 * padding[1] + kw + output_padding[1]

        assert oh, ow == out_shape

        noh = oh - output_padding[0] + 2 * padding[0]
        now = ow - output_padding[1] + 2 * padding[1]

        out = np.zeros((cout,) + (noh, now), dtype=np.int64)

        # Generate the transpose input arrary : transpose padding 0 & stride 0
        xc_t = xcin
        xh_t = ih + (ih - 1) * (stride[0] - 1)
        xw_t = iw + (iw - 1) * (stride[1] - 1)
        x_transpose = np.zeros((xc_t, xh_t, xw_t), dtype=x.dtype)
        x_transpose[::1, :: stride[0], :: stride[1]] = x
        # padding 0 for transpose not for parameter padding, get new input array x_transpose
        x_transpose = np.pad(
            x_transpose, ((0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)), mode="constant"
        )

        kernel_flip = np.flip(kernel, axis=(2, 3))

        stride_transpose = (1, 1)

        for o in range(cout):
            for i in range(cin):
                conv_result = np.zeros((noh, now), dtype=np.int64)
                for h in range(noh):
                    for w in range(now):
                        window = x_transpose[
                            i,
                            h * stride_transpose[0] : h * stride_transpose[0] + kh,
                            w * stride_transpose[1] : w * stride_transpose[1] + kw,
                        ]
                        conv_result[h, w] = np.sum(window * kernel_flip[o, i, :, :])

                out[o] += conv_result

        # inverse padding
        ph_start = padding[0] if padding[0] > 0 else None
        ph_end = (-1 * padding[0]) if padding[0] > 0 else None
        pw_start = padding[1] if padding[1] > 0 else None
        pw_end = (-1 * padding[1]) if padding[1] > 0 else None
        out = out[:, ph_start:ph_end, pw_start:pw_end]

        # output_padding
        out = np.pad(out, ((0, 0), (0, output_padding[0]), (0, output_padding[1])), mode="constant")

        # if fm_order == "HWC":
        #     return out.transpose(1, 2, 0)
        # else:
        #     return out
        return out

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding, output_padding, fm_order, kdtype",
        # Padding is fixed at (0, 0)
        [
            ((12, 12), 16, 8, (3, 3), (1, 1), (1, 1), (1, 1), "CHW", np.bool_),
            ((12, 12), 24, 12, (3, 3), (2, 2), (2, 2), (1, 0), "CHW", np.bool_),
            ((12, 12), 16, 8, (3, 3), (1, 1), (0, 0), (0, 0), "CHW", np.bool_),
            ((12, 12), 24, 12, (3, 3), (2, 2), (1, 2), (0, 1), "CHW", np.int8),
            ((10, 10), 24, 12, (5, 5), (2, 1), (1, 1), (2, 2), "CHW", np.int8),
            ((16, 16), 8, 16, (3, 3), (2, 2), (1, 3), (2, 0), "CHW", np.int8),
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
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        fm_order,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        out_shape = (
            (in_shape[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0],
            (in_shape[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1],
        )

        f = ConvTranspose2dForward(in_shape, out_shape, kernel, stride, padding, output_padding)

        fm_shape = (in_channels,) + in_shape

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)
        y3 = f.connectivity.astype(np.int32)
        expected = self._convtranspose2d_golden(x, out_shape, kernel, stride, padding, output_padding)
        y4 = expected.ravel()

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )
