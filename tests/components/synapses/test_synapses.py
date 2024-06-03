from contextlib import nullcontext

import numpy as np
import pytest
from paicorelib import WeightPrecision as WP

import paibox as pb
from paibox.components import FullConnectedSyn
from paibox.exceptions import RegisterError, ShapeError
from paibox.utils import shape2num


class TestFullConnectedSyn:
    def test_FullConnectedSyn_properties(self):
        n1 = pb.IF((10, 10), 10)
        n2 = pb.IF((20, 10), 10)
        s1 = pb.FullConn(
            n1, n2, np.random.randint(-128, 128, (100, 200), dtype=np.int8)
        )

        new_source1 = pb.LIF((100,), 3)
        new_source2 = pb.LIF((10,), 5)
        new_target1 = pb.LIF((10, 20), 7)
        new_target2 = pb.LIF(100, 9)

        s1.source = new_source1
        with pytest.raises(RegisterError):
            s1.source = new_source2

        s1.target = new_target1
        with pytest.raises(RegisterError):
            s1.target = new_target2

    def test_FullConn_copy(self):
        n1 = pb.IF((10, 10), 10)
        n2 = pb.IF((20, 10), 10)
        s1 = pb.FullConn(
            n1, n2, np.random.randint(-128, 128, (100, 200), dtype=np.int8)
        )
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        # Check the size
        s2.source = n1
        s2.target = n2

        new_target1 = pb.LIF((10, 20), 7)
        s2.target = new_target1

        assert s1.target != s2.target

    def test_MatMul2d_copy(self):
        n1 = pb.IF((20, 10), 10)
        n2 = pb.IF((10, 10), 10)
        s1 = pb.MatMul2d(n1, n2, np.random.randint(-128, 128, (20, 10), dtype=np.int8))
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        s2.source = n1
        s2.target = n2

        new_target1 = pb.LIF((10, 10), 7)
        s2.target = new_target1

        assert s1.target != s2.target

    def test_Conv2d_copy(self):
        n1 = pb.IF((8, 28, 28), 10)
        n2 = pb.IF((16, 14, 14), 10)
        s1 = pb.Conv2d(
            n1,
            n2,
            np.random.randint(-128, 128, (16, 8, 3, 3), dtype=np.int8),
            stride=2,
            padding=1,
        )
        s2 = s1.copy()

        assert isinstance(s2, FullConnectedSyn)
        assert id(s1) != id(s2)
        assert s1 != s2
        assert s1.source == s2.source
        assert s1.target == s2.target
        assert s1.weights is not s2.weights
        assert np.array_equal(s1.connectivity, s2.connectivity)

        s2.source = n1
        s2.target = n2

        new_target1 = pb.IF((16, 14, 14), 7)
        s2.target = new_target1

        assert s1.target != s2.target


class TestFullConn:
    @pytest.mark.parametrize(
        "n1, n2, scalar_weight, expected_wp",
        [
            (pb.IF(10, 3), pb.IF(10, 3), 1, WP.WEIGHT_WIDTH_1BIT),
            (pb.IF((3, 3), 3), pb.IF((3, 3), 3), 4, WP.WEIGHT_WIDTH_4BIT),
            (pb.IF((5,), 3), pb.IF((5,), 3), -1, WP.WEIGHT_WIDTH_2BIT),
            # TODO 3-dimension shape is correct for data flow?
            (pb.IF((10, 2, 3), 3), pb.IF((10, 2, 3), 3), 16, WP.WEIGHT_WIDTH_8BIT),
            (pb.IF((10, 2), 3), pb.IF((4, 5), 3), -100, WP.WEIGHT_WIDTH_8BIT),
            (pb.IF(10, 3), pb.IF((2, 5), 3), 7, WP.WEIGHT_WIDTH_4BIT),
        ],
    )
    def test_FullConn_One2One_scalar(self, n1, n2, scalar_weight, expected_wp):
        s1 = pb.FullConn(n1, n2, scalar_weight, conn_type=pb.SynConnType.One2One)

        assert np.array_equal(s1.weights, scalar_weight)
        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(
            s1.connectivity,
            scalar_weight * np.identity(n1.num_out, dtype=np.int8),
        )
        assert (
            s1.connectivity.dtype == np.int8
            if expected_wp > WP.WEIGHT_WIDTH_1BIT
            else np.bool_
        )
        assert s1.weight_precision is expected_wp

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.IF(10, 3), pb.IF(100, 4)),
            (pb.IF((10, 10), 3), pb.IF((5, 10), 4)),
            (pb.IF((10,), 3), pb.IF((5,), 4)),
            (pb.IF(10, 3), pb.IF((5, 10), 4)),
        ],
    )
    def test_FullConn_One2One_scalar_illegal(self, n1, n2):
        with pytest.raises(ShapeError):
            s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.One2One)

    def test_FullConn_One2One_matrix(self):
        weight = np.array([2, 3, 4], np.int8)
        s1 = pb.FullConn(
            pb.IF((3,), 3), pb.IF((3,), 3), weight, conn_type=pb.SynConnType.One2One
        )

        assert (s1.num_in, s1.num_out) == (3, 3)
        assert np.array_equal(s1.weights, weight)
        assert np.array_equal(
            s1.connectivity, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.int8)
        )
        assert s1.connectivity.dtype == np.int8
        assert s1.weight_precision is WP.WEIGHT_WIDTH_4BIT

        weight = np.array([1, 0, 1, 0], np.int8)
        s2 = pb.FullConn(
            pb.IF((2, 2), 3), pb.IF((2, 2), 3), weight, conn_type=pb.SynConnType.One2One
        )

        assert (s2.num_in, s2.num_out) == (4, 4)
        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(
            s2.connectivity,
            np.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.bool_
            ),
        )
        assert s2.connectivity.dtype == np.int8
        assert s2.weight_precision is WP.WEIGHT_WIDTH_1BIT

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.IF(10, 3), pb.IF(10, 3)),
            (pb.IF((3, 3), 3), pb.IF((3, 3), 3)),
            (pb.IF((5,), 3), pb.IF((5,), 3)),
            (pb.IF(10, 3), pb.IF(100, 3)),
            (pb.IF((10, 10), 3), pb.IF((5, 5), 3)),
        ],
    )
    def test_FullConn_All2All(self, n1, n2):
        s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.All2All)

        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert s1.connectivity.dtype == np.bool_
        assert np.array_equal(s1.weights, 1)
        assert np.array_equal(s1.connectivity, np.ones((n1.num_out, n2.num_in)))

    def test_FullConn_All2All_with_weights(self):
        n1 = pb.IF(3, 3)
        n2 = pb.IF(3, 3)

        """1. Single weight."""
        weight = 2
        s1 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert np.array_equal(s1.weights, weight)
        assert s1.connectivity.dtype == np.int8
        assert s1.weight_precision is WP.WEIGHT_WIDTH_4BIT

        """2. Weights matrix."""
        weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s2 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert s2.connectivity.dtype == np.int8
        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(s2.connectivity, weight)

        # Wrong shape
        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1, n2, np.array([1, 2, 3]), conn_type=pb.SynConnType.All2All
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2], [4, 5], [6, 7]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 2, 3]]),
                conn_type=pb.SynConnType.All2All,
            )


class TestMatMul2d:
    @pytest.mark.parametrize(
        "n1, n2, w_shape, expectation",
        [
            (pb.IF(10, 3), pb.IF(10, 3), (10, 10), nullcontext()),
            (pb.IF(10, 3), pb.IF((1, 10), 3), (10, 10), nullcontext()),
            (pb.IF((10, 2), 3), pb.IF((100,), 3), (2, 10), pytest.raises(ShapeError)),
            (pb.IF((2, 4, 6), 3), pb.IF((10,), 3), (12, 10), pytest.raises(ShapeError)),
            (pb.IF((8, 4), 3), pb.IF((4, 2), 3), (8, 2), nullcontext()),
        ],
    )
    def test_MatMul2d_instance(self, n1, n2, w_shape, expectation):
        weights = np.arange(shape2num(w_shape), dtype=np.int8).reshape(w_shape)

        with expectation:
            s = pb.MatMul2d(n1, n2, weights=weights)

            assert (s.num_in, s.num_out) == (n1.num_out, n2.num_in)
            assert s.connectivity.dtype == np.int8
            assert np.array_equal(s.weights, weights)


class TestConv:
    def test_Conv1d_instance(self):
        in_shape = (32,)
        kernel_size = (5,)
        stride = 2
        padding = 1
        out_shape = ((32 + 2 - 5) // 2 + 1,)
        in_channels = 8
        out_channels = 16
        korder = "IOL"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CL
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.Conv1d(
            n1, n2, weight, stride=stride, padding=padding, kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_Conv2d_instance(self):
        in_shape = (32, 32)
        kernel_size = (5, 5)
        padding = (1, 1)
        stride = 2
        out_shape = ((32 + 2 - 5) // 2 + 1, (32 + 2 - 5) // 2 + 1)
        in_channels = 8
        out_channels = 16
        korder = "IOHW"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CHW
        # Strict output shape is no need
        n2 = pb.IF((out_channels * out_shape[0] * out_shape[1],), 3)

        weight = np.random.randint(
            -8, 8, size=(in_channels, out_channels) + kernel_size, dtype=np.int32
        )
        s1 = pb.Conv2d(
            n1, n2, weight, stride=stride, padding=padding, kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_Conv1d_inchannel_omitted(self):
        in_shape = (32,)
        kernel_size = (5,)
        stride = 2
        out_shape = ((32 - 5) // 2 + 1,)
        in_channels = 1  # omit it
        out_channels = 4
        korder = "IOL"

        n1 = pb.IF(in_shape, 3)  # HW, (in_channels=1)
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int64
        )
        s1 = pb.Conv1d(n1, n2, weight, stride=stride, kernel_order=korder)

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_Conv2d_inchannel_omitted(self):
        in_shape = (32, 32)
        kernel_size = (5, 5)
        stride = 2
        out_shape = ((32 - 5) // 2 + 1, (32 - 5) // 2 + 1)
        in_channels = 1  # omit it
        out_channels = 4
        korder = "IOHW"

        n1 = pb.IF(in_shape, 3)  # HW, (in_channels=1)
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.Conv2d(n1, n2, weight, stride=stride, kernel_order=korder)

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )


class TestConvTranspose2d:
    def test_ConvTranspose1d_instance(self):
        in_shape = (14,)
        kernel_size = (5,)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 * 1 + 1,)
        in_channels = 16
        out_channels = 8
        korder = "IOL"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CL
        n2 = pb.IF((out_channels * out_shape[0],), 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.ConvTranspose1d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_ConvTranspose2d_instance(self):
        in_shape = (14, 14)
        kernel_size = (5, 5)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 + 1, (14 - 1) * 2 + 5 - 2 + 1)
        in_channels = 8
        out_channels = 16
        korder = "IOHW"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CHW
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -8, 8, size=(in_channels, out_channels) + kernel_size, dtype=np.int32
        )
        s1 = pb.ConvTranspose2d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_ConvTranspose1d_inchannel_omitted(self):
        in_shape = (14,)
        kernel_size = (5,)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 * 1 + 1,)
        in_channels = 1  # omit it
        out_channels = 4
        korder = "IOL"

        n1 = pb.IF(in_shape, 3)  # L, (in_channels=1)
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int64
        )
        s1 = pb.ConvTranspose1d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_ConvTranspose2d_inchannel_omitted(self):
        in_shape = (14, 14)
        kernel_size = (5, 5)
        stride = 2
        padding = 1
        output_padding = 1
        out_shape = ((14 - 1) * 2 + 5 - 2 + 1, (14 - 1) * 2 + 5 - 2 + 1)
        in_channels = 1  # omit it
        out_channels = 4
        korder = "IOHW"

        n1 = pb.IF(in_shape, 3)  # HW, (in_channels=1)
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.ConvTranspose2d(
            n1,
            n2,
            weight,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            kernel_order=korder,
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.dtype == np.int8
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )
