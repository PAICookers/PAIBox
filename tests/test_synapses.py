import pytest
import paibox as pb
import numpy as np


@pytest.mark.parametrize(
    "n1, n2",
    [
        (pb.neuron.TonicSpikingNeuron(10, 3), pb.neuron.TonicSpikingNeuron(10, 3)),
        (
            pb.neuron.TonicSpikingNeuron((3, 3), 3),
            pb.neuron.TonicSpikingNeuron((3, 3), 3),
        ),
        (
            pb.neuron.TonicSpikingNeuron((5,), 3),
            pb.neuron.TonicSpikingNeuron((5,), 3),
        ),
    ],
)
def test_NoDecay_One2One(
    n1: pb.neuron.TonicSpikingNeuron, n2: pb.neuron.TonicSpikingNeuron
):
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.One2One())

    assert s1.comm.weights == 1
    assert (s1.num_in, s1.num_out) == (n1.num, n2.num)
    assert np.allclose(s1.connectivity, np.eye(n1.num, n2.num))


@pytest.mark.parametrize(
    "n1, n2",
    [
        (
            pb.neuron.TonicSpikingNeuron(10, 3),
            pb.neuron.TonicSpikingNeuron(100, 4),
        ),
        (
            pb.neuron.TonicSpikingNeuron((10, 10), 3),
            pb.neuron.TonicSpikingNeuron((5, 10), 4),
        ),
    ],
)
def test_NoDecay_One2One_illegal(
    n1: pb.neuron.TonicSpikingNeuron, n2: pb.neuron.TonicSpikingNeuron
):
    with pytest.raises(ValueError):
        s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.One2One())


@pytest.mark.parametrize(
    "n1, n2",
    [
        (pb.neuron.TonicSpikingNeuron(10, 3), pb.neuron.TonicSpikingNeuron(10, 3)),
        (
            pb.neuron.TonicSpikingNeuron((3, 3), 3),
            pb.neuron.TonicSpikingNeuron((3, 3), 3),
        ),
        (
            pb.neuron.TonicSpikingNeuron((5,), 3),
            pb.neuron.TonicSpikingNeuron((5,), 3),
        ),
        (
            pb.neuron.TonicSpikingNeuron(10, 3),
            pb.neuron.TonicSpikingNeuron(100, 3),
        ),
        (
            pb.neuron.TonicSpikingNeuron((10, 10), 3),
            pb.neuron.TonicSpikingNeuron((5, 5), 3),
        ),
    ],
)
def test_NoDecay_All2All(
    n1: pb.neuron.TonicSpikingNeuron, n2: pb.neuron.TonicSpikingNeuron
):
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All())

    assert s1.comm.weights == 1
    assert (s1.num_in, s1.num_out) == (n1.num, n2.num)
    assert np.allclose(s1.connectivity, np.ones((n1.num, n2.num)))


def test_NoDecay_All2All_with_weights():
    n1 = pb.neuron.TonicSpikingNeuron(3, 3)
    n2 = pb.neuron.TonicSpikingNeuron(3, 3)

    """1. Single weight."""
    weight = 2
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All(), weights=weight)

    assert s1.comm.weights == weight
    assert np.allclose(s1.connectivity, weight * np.ones((n1.num, n2.num)))

    """2. Weights matrix."""
    weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    s2 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All(), weights=weight)

    assert np.allclose(s2.comm.weights, weight)
    assert np.allclose(s2.connectivity, weight)

    with pytest.raises(ValueError):
        # Wrong shape
        s3 = pb.synapses.NoDecay(
            n1, n2, pb.synapses.All2All(), weights=np.array([1, 2, 3])
        )
