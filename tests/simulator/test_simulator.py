import numpy as np
import pytest

import paibox as pb


class Net1(pb.DynSysGroup):
    def __init__(self, n_neuron: int):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()

        self.inp = pb.projection.InputProj(pe, shape_out=(n_neuron,), keep_shape=True)
        self.n1 = pb.neuron.TonicSpiking(n_neuron, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(n_neuron, fire_step=2)
        self.s0 = pb.synapses.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.One2One
        )
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.One2One
        )

        # Probes inside
        self.n1_acti = pb.simulator.Probe(self.n1, "output")
        self.n2_acti = pb.simulator.Probe(self.n2, "output")


def test_probe():
    net = Net1(100)

    probe_outside = pb.simulator.Probe(net.inp, "state")

    sim = pb.Simulator(net)
    sim.add_probe(probe_outside)

    # Normalized data
    input_data = np.random.rand(10, 10).astype(np.float32)

    sim.run(10, input=input_data)

    inp_state = sim.data[probe_outside]
    assert type(inp_state) == np.ndarray

    inp_state2 = sim.get_raw(probe_outside)
    assert type(inp_state2) == list

    # Get the data at time=1
    inp_state_at_t = sim.get_raw_at_t(probe_outside, t=5)
    assert type(inp_state_at_t) == np.ndarray
