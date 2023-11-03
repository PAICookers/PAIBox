import numpy as np
import pytest

import paibox as pb


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
            pb.neuron.TonicSpiking((10, 2, 3), 3),
            pb.neuron.TonicSpiking((10, 2, 3), 3),
        ),
        (
            pb.neuron.TonicSpiking((10, 2), 3),
            pb.neuron.TonicSpiking((4, 5), 3),
        ),
        (
            pb.neuron.TonicSpiking(10, 3),
            pb.neuron.TonicSpiking((2, 5), 3),
        ),
    ],
)
def test_NoDecay_One2One_scalar(n1, n2):
    s1 = pb.synapses.NoDecay(n1, n2, 1, conn_type=pb.synapses.ConnType.One2One)

    assert np.array_equal(s1.weights, 1)
    assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
    assert np.array_equal(s1.connectivity, np.eye(n1.num_out, n2.num_in, dtype=np.int8))


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
def test_NoDecay_One2One_scalar_illegal(n1, n2):
    with pytest.raises(ValueError):
        s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.One2One)


def test_NoDecay_One2One_matrix():
    weight = np.array([2, 3, 4], np.int8)
    s2 = pb.synapses.NoDecay(
        pb.neuron.TonicSpiking((3,), 3),
        pb.neuron.TonicSpiking((3,), 3),
        weight,
        conn_type=pb.synapses.ConnType.One2One,
    )

    assert (s2.num_in, s2.num_out) == (3, 3)
    assert np.array_equal(s2.weights, weight)
    assert np.array_equal(
        s2.connectivity, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.int8)
    )

    weight = np.array([1, 0, 1, 0], np.int8)
    s2 = pb.synapses.NoDecay(
        pb.neuron.TonicSpiking((2, 2), 3),
        pb.neuron.TonicSpiking((2, 2), 3),
        weight,
        conn_type=pb.synapses.ConnType.One2One,
    )

    assert (s2.num_in, s2.num_out) == (4, 4)
    assert np.array_equal(s2.weights, weight)
    assert np.array_equal(
        s2.connectivity,
        np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.bool_
        ),
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
def test_NoDecay_All2All(n1, n2):
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.All2All)

    assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
    assert np.array_equal(s1.weights, 1)
    assert np.array_equal(s1.connectivity, np.ones((n1.num_out, n2.num_in)))


def test_NoDecay_All2All_with_weights():
    n1 = pb.neuron.TonicSpiking(3, 3)
    n2 = pb.neuron.TonicSpiking(3, 3)

    """1. Single weight."""
    weight = 2
    s1 = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.synapses.ConnType.All2All)

    assert np.array_equal(s1.weights, weight)

    """2. Weights matrix."""
    weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s2 = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.synapses.ConnType.All2All)

    assert np.array_equal(s2.weights, weight)
    assert np.array_equal(s2.connectivity, weight)

    with pytest.raises(ValueError):
        # Wrong shape
        s3 = pb.synapses.NoDecay(
            n1, n2, np.array([1, 2, 3]), conn_type=pb.synapses.ConnType.All2All
        )

    with pytest.raises(ValueError):
        s3 = pb.synapses.NoDecay(
            n1,
            n2,
            np.array([[1, 2, 3], [4, 5, 6]]),
            conn_type=pb.synapses.ConnType.All2All,
        )

    with pytest.raises(ValueError):
        s3 = pb.synapses.NoDecay(
            n1,
            n2,
            np.array(
                [
                    [
                        1,
                        2,
                    ],
                    [
                        4,
                        5,
                    ],
                    [6, 7],
                ]
            ),
            conn_type=pb.synapses.ConnType.All2All,
        )

    with pytest.raises(ValueError):
        s3 = pb.synapses.NoDecay(
            n1,
            n2,
            np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8], [1, 2, 3]]),
            conn_type=pb.synapses.ConnType.All2All,
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
def test_NoDecay_MatConn(n1, n2):
    weight = np.random.randint(-128, 128, size=(n1.num_out, n2.num_in), dtype=np.int8)

    s = pb.synapses.NoDecay(n1, n2, weight, conn_type=pb.synapses.ConnType.MatConn)

    assert np.array_equal(s.weights, weight)
    assert (s.num_in, s.num_out) == (n1.num_out, n2.num_in)
    assert np.array_equal(s.connectivity, weight)

    with pytest.raises(TypeError):
        # Wrong weight type
        s = pb.synapses.NoDecay(n1, n2, 1, conn_type=pb.synapses.ConnType.MatConn)

    with pytest.raises(ValueError):
        s = pb.synapses.NoDecay(
            n1, n2, np.array([1, 2, 3]), conn_type=pb.synapses.ConnType.MatConn
        )

    with pytest.raises(ValueError):
        s = pb.synapses.NoDecay(
            n1,
            n2,
            np.array([[1, 2, 3], [4, 5, 6]]),
            conn_type=pb.synapses.ConnType.MatConn,
        )
