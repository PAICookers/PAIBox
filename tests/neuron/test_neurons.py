import numpy as np
import pytest

import paibox as pb
from paibox.utils import as_shape, shape2num


@pytest.mark.parametrize(
    "shape",
    [5, (12,), (20, 20), (1, 2, 3)],
    ids=["scalar", "ndim=1", "ndim=2", "ndim=3"],
)
def test_neuron_instance(shape):
    # keep_shape = True
    n1 = pb.neuron.TonicSpiking(shape, 5, keep_shape=True)

    assert n1.shape_in == as_shape(shape)
    assert n1.shape_out == as_shape(shape)
    assert len(n1) == shape2num(shape)

    # keep_shape = False
    n2 = pb.neuron.TonicSpiking(shape, 5)

    assert n2.shape_in == as_shape(shape2num(shape))
    assert n2.shape_out == as_shape(shape2num(shape))
    assert len(n2) == shape2num(shape)


def fakeout(t):
    data = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        np.bool_,
    )

    return data[t]


class Net1(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(fakeout, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking((2,), 3)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.One2One
        )

        self.probe1 = pb.simulator.Probe(self.inp1, "output")
        self.probe2 = pb.simulator.Probe(self.s1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "output")


def test_neuron_behavior():
    net = Net1()
    sim = pb.Simulator(net)

    sim.run(10)

    print(sim.data[net.probe1])
