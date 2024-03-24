import numpy as np
import pytest
from paicorelib import WeightPrecision as WP

import paibox as pb
from paibox.exceptions import ShapeError
from paibox.utils import shape2num


def test_SynSys_Attrs():
    n1 = pb.TonicSpiking(3, 3)
    n2 = pb.TonicSpiking(3, 3)
    s1 = pb.FullConn(
        n1,
        n2,
        weights=np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1]], dtype=np.int8),
        conn_type=pb.SynConnType.MatConn,
    )

    assert np.array_equal(s1.n_axon_each, np.array([1, 3, 2]))
    assert s1.num_axon == 3
    assert s1.num_dendrite == 3
    assert s1.weight_precision is WP.WEIGHT_WIDTH_1BIT
    assert s1.weights.dtype == np.int8


class TestFullConn:
    @pytest.mark.parametrize(
        "n1, n2, scalar_weight, expected_wp",
        [
            (
                pb.TonicSpiking(10, 3),
                pb.TonicSpiking(10, 3),
                1,
                WP.WEIGHT_WIDTH_1BIT,
            ),
            (
                pb.TonicSpiking((3, 3), 3),
                pb.TonicSpiking((3, 3), 3),
                4,
                WP.WEIGHT_WIDTH_4BIT,
            ),
            (
                pb.TonicSpiking((5,), 3),
                pb.TonicSpiking((5,), 3),
                -1,
                WP.WEIGHT_WIDTH_2BIT,
            ),
            # TODO 3-dimension shape is correct for data flow?
            (
                pb.TonicSpiking((10, 2, 3), 3),
                pb.TonicSpiking((10, 2, 3), 3),
                16,
                WP.WEIGHT_WIDTH_8BIT,
            ),
            (
                pb.TonicSpiking((10, 2), 3),
                pb.TonicSpiking((4, 5), 3),
                -100,
                WP.WEIGHT_WIDTH_8BIT,
            ),
            (
                pb.TonicSpiking(10, 3),
                pb.TonicSpiking((2, 5), 3),
                7,
                WP.WEIGHT_WIDTH_4BIT,
            ),
        ],
    )
    def test_FullConn_One2One_scalar(self, n1, n2, scalar_weight, expected_wp):
        s1 = pb.FullConn(n1, n2, scalar_weight, conn_type=pb.SynConnType.One2One)

        assert np.array_equal(s1.weights, scalar_weight)
        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(
            s1.connectivity,
            scalar_weight * np.eye(n1.num_out, n2.num_in, dtype=np.int8),
        )
        assert s1.weight_precision is expected_wp

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (
                pb.TonicSpiking(10, 3),
                pb.TonicSpiking(100, 4),
            ),
            (
                pb.TonicSpiking((10, 10), 3),
                pb.TonicSpiking((5, 10), 4),
            ),
            (
                pb.IF((10,), 3),
                pb.TonicSpiking((5,), 4),
            ),
            (
                pb.TonicSpiking(10, 3),
                pb.TonicSpiking((5, 10), 4),
            ),
        ],
    )
    def test_FullConn_One2One_scalar_illegal(self, n1, n2):
        with pytest.raises(ShapeError):
            s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.One2One)

    def test_FullConn_One2One_matrix(self):
        weight = np.array([2, 3, 4], np.int8)
        s1 = pb.FullConn(
            pb.TonicSpiking((3,), 3),
            pb.TonicSpiking((3,), 3),
            weight,
            conn_type=pb.SynConnType.One2One,
        )

        assert (s1.num_in, s1.num_out) == (3, 3)
        assert np.array_equal(s1.weights, weight)
        assert np.array_equal(
            s1.connectivity, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.int8)
        )
        assert s1.weight_precision is WP.WEIGHT_WIDTH_4BIT

        weight = np.array([1, 0, 1, 0], np.int8)
        s2 = pb.FullConn(
            pb.TonicSpiking((2, 2), 3),
            pb.TonicSpiking((2, 2), 3),
            weight,
            conn_type=pb.SynConnType.One2One,
        )

        assert (s2.num_in, s2.num_out) == (4, 4)
        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(
            s2.connectivity,
            np.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.bool_
            ),
        )
        assert s2.weight_precision is WP.WEIGHT_WIDTH_1BIT

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.TonicSpiking(10, 3), pb.TonicSpiking(10, 3)),
            (
                pb.TonicSpiking((3, 3), 3),
                pb.TonicSpiking((3, 3), 3),
            ),
            (
                pb.TonicSpiking((5,), 3),
                pb.TonicSpiking((5,), 3),
            ),
            (
                pb.TonicSpiking(10, 3),
                pb.TonicSpiking(100, 3),
            ),
            (
                pb.TonicSpiking((10, 10), 3),
                pb.TonicSpiking((5, 5), 3),
            ),
        ],
    )
    def test_FullConn_All2All(self, n1, n2):
        s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.All2All)

        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(s1.weights, 1)
        assert np.array_equal(s1.connectivity, np.ones((n1.num_out, n2.num_in)))

    def test_FullConn_All2All_with_weights(self):
        n1 = pb.TonicSpiking(3, 3)
        n2 = pb.TonicSpiking(3, 3)

        """1. Single weight."""
        weight = 2
        s1 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert np.array_equal(s1.weights, weight)
        assert s1.weight_precision is WP.WEIGHT_WIDTH_4BIT

        """2. Weights matrix."""
        weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s2 = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.All2All)

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

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.TonicSpiking(10, 3), pb.TonicSpiking(10, 3)),
            (
                pb.TonicSpiking((3, 3), 3),
                pb.TonicSpiking((3, 3), 3),
            ),
            (
                pb.TonicSpiking((5,), 3),
                pb.TonicSpiking((5,), 3),
            ),
        ],
    )
    def test_FullConn_MatConn(self, n1, n2):
        weight = np.random.randint(
            -128, 128, size=(n1.num_out, n2.num_in), dtype=np.int8
        )

        s = pb.FullConn(n1, n2, weight, conn_type=pb.SynConnType.MatConn)

        assert np.array_equal(s.weights, weight)
        assert (s.num_in, s.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(s.connectivity, weight)

        # Wrong weight type
        with pytest.raises(TypeError):
            s = pb.FullConn(n1, n2, 1, conn_type=pb.SynConnType.MatConn)

        # Wrong shape
        with pytest.raises(ShapeError):
            s = pb.FullConn(
                n1, n2, np.array([1, 2, 3]), conn_type=pb.SynConnType.MatConn
            )

        # Wrong shape
        with pytest.raises(ShapeError):
            s = pb.FullConn(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6]]),
                conn_type=pb.SynConnType.MatConn,
            )


class TestConv2d:
    def test_Conv1d_instance(self):
        in_shape = (32,)
        kernel_size = (5,)
        stride = 2
        out_shape = ((32 - 5) // 2 + 1,)
        in_channels = 8
        out_channels = 16
        korder = "IOL"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CL
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.Conv1d(
            n1, n2, weight, stride=stride, fm_order="CL", kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )

    def test_Conv2d_instance(self):
        in_shape = (32, 32)
        kernel_size = (5, 5)
        stride = 2
        out_shape = ((32 - 5) // 2 + 1, (32 - 5) // 2 + 1)
        in_channels = 8
        out_channels = 16
        korder = "IOHW"

        n1 = pb.IF((in_channels,) + in_shape, 3)  # CHW
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.Conv2d(
            n1, n2, weight, stride=stride, fm_order="CHW", kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
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

        n1 = pb.IF(in_shape, 3)  # L, (in_channels=1)
        n2 = pb.IF((out_channels,) + out_shape, 3)

        weight = np.random.randint(
            -128, 128, size=(in_channels, out_channels) + kernel_size, dtype=np.int8
        )
        s1 = pb.Conv1d(
            n1, n2, weight, stride=stride, fm_order="CL", kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
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
        s1 = pb.Conv2d(
            n1, n2, weight, stride=stride, fm_order="CHW", kernel_order=korder
        )

        assert s1.num_in == in_channels * shape2num(in_shape)
        assert s1.connectivity.shape == (
            in_channels * shape2num(in_shape),
            out_channels * shape2num(out_shape),
        )
