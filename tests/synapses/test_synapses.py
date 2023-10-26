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
    ],
)
def test_NoDecay_One2One_scalar(n1: pb.neuron.TonicSpiking, n2: pb.neuron.TonicSpiking):
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.One2One())

    assert np.array_equal(s1.weights, np.eye(n1.num_out, n2.num_in, dtype=np.int8))
    assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
    # assert np.array_equal(s1.connectivity, np.eye(n1.num_out, n2.num_in, dtype=np.int8))


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
    ],
)
def test_NoDecay_One2One_scalar_illegal(
    n1: pb.neuron.TonicSpiking, n2: pb.neuron.TonicSpiking
):
    with pytest.raises(ValueError):
        s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.One2One())


def test_NoDecay_One2One_matrix():
    weight = np.array([2, 3, 4], np.int8)
    s2 = pb.synapses.NoDecay(
        pb.neuron.TonicSpiking((3,), 3),
        pb.neuron.TonicSpiking((3,), 3),
        pb.synapses.One2One(),
        weights=weight,
    )

    assert (s2.num_in, s2.num_out) == (3, 3)
    assert np.array_equal(
        s2.weights, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.int8)
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
def test_NoDecay_All2All(n1: pb.neuron.TonicSpiking, n2: pb.neuron.TonicSpiking):
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All())

    assert (s1.num_in, s1.num_out) == (n1.num_out, n2.num_in)
    assert np.array_equal(s1.weights, np.ones((n1.num_out, n2.num_in)))


def test_NoDecay_All2All_with_weights():
    n1 = pb.neuron.TonicSpiking(3, 3)
    n2 = pb.neuron.TonicSpiking(3, 3)

    """1. Single weight."""
    weight = 2
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All(), weights=weight)

    assert np.array_equal(s1.weights, weight * np.ones((n1.num_out, n2.num_in)))

    """2. Weights matrix."""
    weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s2 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All(), weights=weight)

    assert np.array_equal(s2.weights, weight)
    # assert np.array_equal(s2.connectivity, weight)

    with pytest.raises(ValueError):
        # Wrong shape
        s3 = pb.synapses.NoDecay(
            n1, n2, pb.synapses.All2All(), weights=np.array([1, 2, 3])
        )
