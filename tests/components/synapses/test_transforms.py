import numpy as np
import pytest

from paibox.components.synapses import transforms as tfm
from paibox.components.synapses.conv_utils import _conv1d_faster, _conv2d_faster
from paibox.exceptions import AutoOptimizationWarning
from paibox.types import WEIGHT_DTYPE
from paibox.utils import shape2num


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
        x = np.array([1, 0, 1], dtype=np.bool_)
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
        x1 = np.array([0, 1, 1, 0], dtype=np.bool_)
        y = f(x1)
        assert y.shape == (4,)

        # The last spike is a scalar.
        x2 = np.array(1, dtype=np.bool_)
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
        x = np.random.randint(2, size=(10,), dtype=np.bool_)
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
                np.random.randint(2, size=(3,), dtype=np.bool_),
                np.random.randint(2, size=(3, 4), dtype=np.bool_),
            ),
            (
                (10, 20),
                np.random.randint(2, size=(10,), dtype=np.bool_),
                np.random.randint(127, size=(10, 20), dtype=np.int8),
            ),
            (
                (20, 10),
                np.random.randint(2, size=(20,), dtype=np.bool_),
                np.random.randint(2, size=(20, 10), dtype=np.bool_),
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
        ids=[
            "weights_bool_1",
            "weights_int8_1",
            "weights_int8_2",
            "weights_int8_3",
            "weights_int8_4",
        ],
    )
    def test_AllToAll_array(self, shape, x, weights):
        """Test `AllToAll` when weights is an array"""

        f = tfm.AllToAll(shape, weights)
        y = f(x)
        expected = x @ weights.copy().astype(np.int32)

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
            (
                np.random.randint(2, size=(10,), dtype=np.bool_),
                np.random.randint(-10, 10, size=(10, 20), dtype=np.int8),
            ),
            (
                np.ones((20, 10), dtype=np.bool_),
                np.random.randint(2, size=(20, 10), dtype=np.bool_),
            ),
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
        y2 = x.flatten() @ f.connectivity.astype(np.int32)
        expected = x.reshape(in_shape).transpose(axes) @ weights.copy().astype(np.int32)

        assert f.connectivity.dtype == WEIGHT_DTYPE
        assert y.shape == oshape
        assert y2.dtype == np.int32
        assert np.array_equal(y, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (x.size, y.size)

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, kernel_size, stride, padding, kdtype",
        [
            (np.bool_, (8,), 16, 8, (3,), (1,), (1,), np.int8),
            (np.bool_, (28,), 16, 8, (3,), (1,), (1,), np.bool_),
            (np.bool_, (28,), 24, 12, (3,), (2,), (2,), np.bool_),
            (np.bool_, (28,), 24, 12, (5,), (2,), (2,), np.bool_),
            (np.bool_, (16,), 8, 16, (3,), (2,), (0,), np.bool_),
            (np.bool_, (28,), 16, 8, (3,), (1,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (3,), (2,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), np.int8),
            (np.bool_, (16,), 8, 16, (3,), (2,), (0,), np.int8),
            (np.int8, (8,), 16, 8, (3,), (1,), (1,), np.int8),
            (np.int8, (28,), 16, 8, (3,), (1,), (1,), np.bool_),
            (np.int8, (28,), 24, 12, (3,), (2,), (2,), np.bool_),
            (np.int8, (28,), 24, 12, (5,), (2,), (2,), np.bool_),
            (np.int8, (16,), 8, 16, (3,), (2,), (0,), np.bool_),
            (np.int8, (28,), 16, 8, (3,), (1,), (0,), np.int8),
            (np.int8, (28,), 24, 12, (3,), (2,), (0,), np.int8),
            (np.int8, (28,), 24, 12, (5,), (2,), (0,), np.int8),
            (np.int8, (16,), 8, 16, (3,), (2,), (0,), np.int8),
            # ((28,), 16, 8, (3,), (1,), (0,), "LC"),
            # ((24,), 8, 8, (3,), (2,), (0,), "LC"),
            # ((24,), 8, 16, (7,), (2,), (0,), "LC"),
            # ((32,), 4, 12, (5,), (1,), (0,), "LC"),
        ],
    )
    def test_Conv1dForward(
        self,
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max + 1,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        fm_shape = (in_channels,) + in_shape
        if xdtype == np.bool_:
            x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        else:
            x = np.random.randint(
                np.iinfo(xdtype).min,
                np.iinfo(xdtype).max + 1,
                size=fm_shape,
                dtype=xdtype,
            )

        out_shape = ((in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,)

        f = tfm.Conv1dForward(in_shape, out_shape, kernel, stride, padding)

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        expected = _conv1d_faster(x, out_shape, kernel, stride, padding)

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, kernel_size, stride, padding, kdtype",
        [
            (np.bool_, (28, 28), 16, 8, (3, 3), (1, 1), (1, 1), np.bool_),
            (np.bool_, (28, 28), 24, 12, (3, 3), (2, 2), (2, 1), np.bool_),
            (np.bool_, (28, 28), 16, 8, (3, 3), (1, 1), (2, 3), np.bool_),
            (np.bool_, (28, 28), 24, 12, (3, 3), (2, 2), (0, 0), np.int8),
            (np.bool_, (28, 28), 24, 12, (5, 5), (2, 1), (0, 0), np.int8),
            (np.bool_, (8, 8), 8, 16, (3, 3), (2, 2), (1, 1), np.int8),
            (np.int8, (28, 28), 16, 8, (3, 3), (1, 1), (1, 1), np.bool_),
            (np.int8, (28, 28), 24, 12, (3, 3), (2, 2), (2, 1), np.bool_),
            (np.int8, (28, 28), 16, 8, (3, 3), (1, 1), (2, 3), np.bool_),
            (np.int8, (28, 28), 24, 12, (3, 3), (2, 2), (0, 0), np.int8),
            (np.int8, (28, 28), 24, 12, (5, 5), (2, 1), (0, 0), np.int8),
            (np.int8, (8, 8), 8, 16, (3, 3), (2, 2), (1, 1), np.int8),
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
        xdtype,
        in_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max + 1,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        fm_shape = (in_channels,) + in_shape
        if xdtype == np.bool_:
            x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        else:
            x = np.random.randint(
                np.iinfo(xdtype).min,
                np.iinfo(xdtype).max + 1,
                size=fm_shape,
                dtype=xdtype,
            )

        out_shape = (
            (in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
            (in_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
        )

        f = tfm.Conv2dForward(in_shape, out_shape, kernel, stride, padding)

        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        expected = _conv2d_faster(x, out_shape, kernel, stride, padding)

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @staticmethod
    def _convtranspose1d_golden(
        x: np.ndarray,
        out_shape: tuple[int],
        kernel: np.ndarray,
        stride: tuple[int],
        padding: tuple[int],
        output_padding: tuple[int],
    ):
        cout, cin, kl = kernel.shape
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
        conv_result = np.zeros((nol,), dtype=np.int64)

        for o in range(cout):
            for i in range(cin):
                conv_result.fill(0)
                for l in range(nol):
                    window = x_transpose[
                        i, l * stride_transpose : l * stride_transpose + kl
                    ].astype(np.int64)
                    conv_result[l] = np.sum(window * kernel_flip[o, i, :])

                out[o] += conv_result

        # inverse padding : (cout, (xl-1)*stride+kernel) -> (cout, (xl-1)*stride+kernel-2*padding)
        out = out[:, padding[0] : (-1 * padding[0])] if padding[0] > 0 else out

        # output_padding
        out = np.pad(out, ((0, 0), (0, output_padding[0])), mode="constant")

        return out

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, kernel_size, stride, padding, output_padding, kdtype",
        [
            (np.bool_, (28,), 16, 8, (3,), (1,), (0,), (0,), np.bool_),
            (np.bool_, (28,), 24, 12, (3,), (2,), (2,), (2,), np.bool_),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), (1,), np.bool_),
            (np.bool_, (16,), 8, 16, (3,), (2,), (1,), (0,), np.bool_),
            (np.bool_, (28,), 16, 8, (3,), (3,), (0,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (3,), (2,), (3,), (0,), np.int8),
            (np.bool_, (28,), 24, 12, (5,), (2,), (0,), (0,), np.int8),
            (np.bool_, (16,), 8, 16, (3,), (2,), (1,), (1,), np.int8),
            (np.int8, (28,), 16, 8, (3,), (1,), (0,), (0,), np.bool_),
            (np.int8, (28,), 24, 12, (3,), (2,), (2,), (2,), np.bool_),
            (np.int8, (28,), 24, 12, (5,), (2,), (0,), (1,), np.bool_),
            (np.int8, (16,), 8, 16, (3,), (2,), (1,), (0,), np.bool_),
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
        kernel_size,
        stride,
        padding,
        output_padding,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max + 1,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        fm_shape = (in_channels,) + in_shape
        if xdtype == np.bool_:
            x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        else:
            x = np.random.randint(
                np.iinfo(xdtype).min,
                np.iinfo(xdtype).max + 1,
                size=fm_shape,
                dtype=xdtype,
            )

        # out_shape = ((in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,)
        out_shape = (
            (in_shape[0] - 1) * stride[0]
            - 2 * padding[0]
            + kernel_size[0]
            + output_padding[0],
        )
        f = tfm.ConvTranspose1dForward(
            in_shape, out_shape, kernel, stride, padding, output_padding
        )

        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        expected = self._convtranspose1d_golden(
            x, out_shape, kernel, stride, padding, output_padding
        )

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @staticmethod
    def _convtranspose2d_golden(
        x: np.ndarray,
        out_shape: tuple[int, int],
        kernel: np.ndarray,
        stride: tuple[int, int],
        padding: tuple[int, int],
        output_padding: tuple[int, int],
    ):
        cout, cin, kh, kw = kernel.shape
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
        conv_result = np.zeros((noh, now), dtype=np.int64)

        for o in range(cout):
            for i in range(cin):
                conv_result.fill(0)
                for h in range(noh):
                    for w in range(now):
                        window = x_transpose[
                            i,
                            h * stride_transpose[0] : h * stride_transpose[0] + kh,
                            w * stride_transpose[1] : w * stride_transpose[1] + kw,
                        ].astype(np.int64)
                        conv_result[h, w] = np.sum(window * kernel_flip[o, i, :, :])

                out[o] += conv_result

        # inverse padding
        ph_start = padding[0] if padding[0] > 0 else None
        ph_end = (-1 * padding[0]) if padding[0] > 0 else None
        pw_start = padding[1] if padding[1] > 0 else None
        pw_end = (-1 * padding[1]) if padding[1] > 0 else None
        out = out[:, ph_start:ph_end, pw_start:pw_end]

        # output_padding
        out = np.pad(
            out,
            ((0, 0), (0, output_padding[0]), (0, output_padding[1])),
            mode="constant",
        )

        return out

    @pytest.mark.parametrize(
        "xdtype, in_shape, in_channels, out_channels, kernel_size, stride, padding, output_padding, kdtype",
        [
            (np.bool_, (12, 12), 16, 8, (3, 3), (1, 1), (1, 1), (1, 1), np.bool_),
            (np.bool_, (12, 12), 24, 12, (3, 3), (2, 2), (2, 2), (1, 0), np.bool_),
            (np.bool_, (12, 12), 16, 8, (3, 3), (1, 1), (0, 0), (0, 0), np.bool_),
            (np.bool_, (12, 12), 24, 12, (3, 3), (2, 2), (1, 2), (0, 1), np.int8),
            (np.bool_, (10, 10), 24, 12, (5, 5), (2, 1), (1, 1), (2, 2), np.int8),
            (np.bool_, (16, 16), 8, 16, (3, 3), (2, 2), (1, 3), (2, 0), np.int8),
            (np.int8, (12, 12), 16, 8, (3, 3), (1, 1), (1, 1), (1, 1), np.bool_),
            (np.int8, (12, 12), 24, 12, (3, 3), (2, 2), (2, 2), (1, 0), np.bool_),
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
        kernel_size,
        stride,
        padding,
        output_padding,
        kdtype,
    ):
        if kdtype == np.bool_:
            kernel = np.random.randint(
                0, 2, size=(out_channels, in_channels) + kernel_size, dtype=np.bool_
            )
        else:
            kernel = np.random.randint(
                np.iinfo(kdtype).min,
                np.iinfo(kdtype).max + 1,
                size=(out_channels, in_channels) + kernel_size,
                dtype=kdtype,
            )

        fm_shape = (in_channels,) + in_shape
        if xdtype == np.bool_:
            x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        else:
            x = np.random.randint(
                np.iinfo(xdtype).min,
                np.iinfo(xdtype).max + 1,
                size=fm_shape,
                dtype=xdtype,
            )

        out_shape = (
            (in_shape[0] - 1) * stride[0]
            - 2 * padding[0]
            + kernel_size[0]
            + output_padding[0],
            (in_shape[1] - 1) * stride[1]
            - 2 * padding[1]
            + kernel_size[1]
            + output_padding[1],
        )

        f = tfm.ConvTranspose2dForward(
            in_shape, out_shape, kernel, stride, padding, output_padding
        )

        x = np.random.randint(0, 2, size=fm_shape, dtype=np.bool_)
        xf = x.ravel()

        # The result of __call__ using traditional conv
        y1 = f(xf)
        # The result of matmul using the unrolled matrix
        y2 = xf @ f.connectivity.astype(np.int32)

        expected = self._convtranspose2d_golden(
            x, out_shape, kernel, stride, padding, output_padding
        )

        assert np.array_equal(y1, expected)
        assert np.array_equal(y2, expected.ravel())
        assert f.connectivity.shape == (
            shape2num((kernel.shape[1],) + in_shape),
            shape2num((kernel.shape[0],) + out_shape),
        )

    @pytest.mark.parametrize("n_compare, n_group", [(4, 8), (9, 12), (25, 1)])
    def test_CompareMax(self, n_compare, n_group):
        from paibox.components.synapses.transforms import _CompareMax

        n = n_compare * n_group
        w = np.zeros((n, n_group), dtype=np.int8)
        for i in range(n_group):
            w[n_compare * i : n_compare * (i + 1), i] = 1

        f = _CompareMax((n, n_group), w)

        x = np.random.randint(0, 256, size=(n_compare, n_group), dtype=np.uint8)
        y1 = f(x.ravel(order="F"))  # flatten in column-major order
        expected = np.zeros((n_group,), dtype=np.int32)

        for i in range(n_group):
            expected[i] = np.max(x[:, i])

        assert np.array_equal(y1, expected)
