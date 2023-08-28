import pytest
import paibox as pb
from paibox.node import NodeDict


def test_Sequential_instance():
    n1 = pb.neuron.TonicSpikingNeuron(1, fire_step=3)
    n2 = pb.neuron.TonicSpikingNeuron(1, fire_step=5)
    sequential = pb.network.Sequential(n1, n2, name="Sequential_1")

    assert isinstance(sequential, pb.network.Sequential)


def test_Sequential_getitem():
    n1 = pb.neuron.TonicSpikingNeuron(10, fire_step=3, name="n1")
    n2 = pb.neuron.TonicSpikingNeuron(10, fire_step=5, name="n2")
    sequential = pb.network.Sequential(n1, n2, name="Sequential_1")

    assert isinstance(sequential.children, NodeDict)

    for str in ["n1", "n2"]:
        sequential[str]

    with pytest.raises(KeyError):
        sequential["n3"]

    for item in [0, 1]:
        sequential[item]

    seq = sequential[:1]

    assert seq != sequential

    seq = sequential[1:]
    seq = sequential[0:]
    seq = sequential[1:10]

    seq = sequential["n1"]

    sequential[1:2]  # legal


def test_Sequential_data_update():
    n1 = pb.neuron.TonicSpikingNeuron((10, 10), fire_step=3, name="n1")
    n2 = pb.neuron.TonicSpikingNeuron((10, 10), fire_step=5, name="n2")
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All())
    
    # List in order
    sequential = pb.network.Sequential(n1, s1, n2, name="Sequential_1")

    import numpy as np
    
    x = np.random.randint(0, 2, size=(100,))
    
    y = sequential.update(x)