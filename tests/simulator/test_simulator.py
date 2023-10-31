import numpy as np
import pytest

import paibox as pb


class Net1(pb.DynSysGroup):
    def __init__(self):
        super().__init__()

        # Input inside
        class GenDataProcess(pb.base.Process):
            def __init__(self, shape_out):
                super().__init__(shape_out)

            def update(self, ts: int, *args, **kwargs):
                return np.ones((2,)) * ts

        self.inp = pb.projection.InputProj(GenDataProcess(2))
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(2, fire_step=2)
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
    net = Net1()

    probe_outside = pb.simulator.Probe(net.inp, "state")

    sim = pb.Simulator(net)
    sim.add_probe(probe_outside)

    sim.run(10)

    inp_state = sim.data[probe_outside]
    assert type(inp_state) == np.ndarray

    inp_state2 = sim.get_raw(probe_outside)
    assert type(inp_state2) == list

    # Get the data at time=1
    inp_state_at_t = sim.get_raw_at_t(probe_outside, t=5)
    assert type(inp_state_at_t) == np.ndarray
