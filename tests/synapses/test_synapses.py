import numpy as np
import pytest
from paicorelib import WeightPrecision as WP

import paibox as pb
from paibox.exceptions import ShapeError


def test_SynSys_Attrs():
    n1 = pb.neuron.TonicSpiking(3, 3)
    n2 = pb.neuron.TonicSpiking(3, 3)
    s1 = pb.synapses.NoDecay(
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


class TestNoDecay:
    @pytest.mark.parametrize(
        "n1, n2, scalar_weight, expected_wp",
        [
            (
                pb.neuron.TonicSpiking(10, 3),
                pb.neuron.TonicSpiking(10, 3),
                1,
                WP.WEIGHT_WIDTH_1BIT,
            ),
            (
                pb.neuron.TonicSpiking((3, 3), 3),
                pb.neuron.TonicSpiking((3, 3), 3),
                4,
                WP.WEIGHT_WIDTH_4BIT,
            ),
            (
                pb.neuron.TonicSpiking((5,), 3),
                pb.neuron.TonicSpiking((5,), 3),
                -1,
                WP.WEIGHT_WIDTH_2BIT,
            ),
            # TODO 3-dimension shape is correct for data flow?
            (
                pb.neuron.TonicSpiking((10, 2, 3), 3),
                pb.neuron.TonicSpiking((10, 2, 3), 3),
                16,
                WP.WEIGHT_WIDTH_8BIT,
            ),
            (
                pb.neuron.TonicSpiking((10, 2), 3),
                pb.neuron.TonicSpiking((4, 5), 3),
                -100,
                WP.WEIGHT_WIDTH_8BIT,
            ),
            (
                pb.neuron.TonicSpiking(10, 3),
                pb.neuron.TonicSpiking((2, 5), 3),
                7,
                WP.WEIGHT_WIDTH_4BIT,
            ),
        ],
    )
    def test_NoDecay_One2One_scalar(self, n1, n2, scalar_weight, expected_wp):
        s1 = pb.synapses.NoDecay(
            n1, n2, scalar_weight, conn_type=pb.SynConnType.One2One
        )

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
                pb.neuron.TonicSpiking(10, 3),
                pb.neuron.TonicSpiking(100, 4),
            ),
            (
                pb.neuron.TonicSpiking((10, 10), 3),
                pb.neuron.TonicSpiking((5, 10), 4),
            ),
            (
                pb.neuron.IF((10,), 3),
                pb.neuron.TonicSpiking((5,), 4),
            ),
            (
                pb.neuron.TonicSpiking(10, 3),
                pb.neuron.TonicSpiking((5, 10), 4),
            ),
        ],
    )
    def test_NoDecay_One2One_scalar_illegal(self, n1, n2):
        with pytest.raises(ShapeError):
            s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.SynConnType.One2One)

    def test_NoDecay_One2One_matrix(self):
        weight = np.array([2, 3, 4], np.int8)
        s1 = pb.synapses.NoDecay(
            pb.neuron.TonicSpiking((3,), 3),
            pb.neuron.TonicSpiking((3,), 3),
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
        s2 = pb.synapses.NoDecay(
            pb.neuron.TonicSpiking((2, 2), 3),
            pb.neuron.TonicSpiking((2, 2), 3),
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
            (pb.neuron.TonicSpiking(10, 3), pb.neuron.TonicSpiking(10, 3)),
            (
                pb.neuron.TonicSpiking((3, 3), 3),
                pb.neuron.TonicSpiking((3, 3), 3),
            ),
            (
                pb.neuron.TonicSpiking((5,), 3),
                pb.neuron.TonicSpiking((5,), 3),
            ),
            (
                pb.neuron.TonicSpiking(10, 3),
                pb.neuron.TonicSpiking(100, 3),
            ),
            (
                pb.neuron.TonicSpiking((10, 10), 3),
                pb.neuron.TonicSpiking((5, 5), 3),
            ),
        ],
    )
    def test_NoDecay_All2All(self, n1, n2):
        s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.SynConnType.All2All)

        assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(s1.weights, 1)
        assert np.array_equal(s1.connectivity, np.ones((n1.num_out, n2.num_in)))

    def test_NoDecay_All2All_with_weights(self):
        n1 = pb.neuron.TonicSpiking(3, 3)
        n2 = pb.neuron.TonicSpiking(3, 3)

        """1. Single weight."""
        weight = 2
        s1 = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert np.array_equal(s1.weights, weight)
        assert s1.weight_precision is WP.WEIGHT_WIDTH_4BIT

        """2. Weights matrix."""
        weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s2 = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.SynConnType.All2All)

        assert np.array_equal(s2.weights, weight)
        assert np.array_equal(s2.connectivity, weight)

        # Wrong shape
        with pytest.raises(ShapeError):
            s3 = pb.synapses.NoDecay(
                n1, n2, np.array([1, 2, 3]), conn_type=pb.SynConnType.All2All
            )

        with pytest.raises(ShapeError):
            s3 = pb.synapses.NoDecay(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.synapses.NoDecay(
                n1,
                n2,
                np.array([[1, 2], [4, 5], [6, 7]]),
                conn_type=pb.SynConnType.All2All,
            )

        with pytest.raises(ShapeError):
            s3 = pb.synapses.NoDecay(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 2, 3]]),
                conn_type=pb.SynConnType.All2All,
            )

    @pytest.mark.parametrize(
        "n1, n2",
        [
            (pb.neuron.TonicSpiking(10, 3), pb.neuron.TonicSpiking(10, 3)),
            (
                pb.neuron.TonicSpiking((3, 3), 3),
                pb.neuron.TonicSpiking((3, 3), 3),
            ),
            (
                pb.neuron.TonicSpiking((5,), 3),
                pb.neuron.TonicSpiking((5,), 3),
            ),
        ],
    )
    def test_NoDecay_MatConn(self, n1, n2):
        weight = np.random.randint(
            -128, 128, size=(n1.num_out, n2.num_in), dtype=np.int8
        )

        s = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.SynConnType.MatConn)

        assert np.array_equal(s.weights, weight)
        assert (s.num_in, s.num_out) == (n1.num_out, n2.num_in)
        assert np.array_equal(s.connectivity, weight)

        # Wrong weight type
        with pytest.raises(TypeError):
            s = pb.synapses.NoDecay(n1, n2, 1, conn_type=pb.SynConnType.MatConn)

        # Wrong shape
        with pytest.raises(ShapeError):
            s = pb.synapses.NoDecay(
                n1, n2, np.array([1, 2, 3]), conn_type=pb.SynConnType.MatConn
            )

        # Wrong shape
        with pytest.raises(ShapeError):
            s = pb.synapses.NoDecay(
                n1,
                n2,
                np.array([[1, 2, 3], [4, 5, 6]]),
                conn_type=pb.SynConnType.MatConn,
            )
