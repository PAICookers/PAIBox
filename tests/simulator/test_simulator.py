import numpy as np
import pytest

import paibox as pb


class Net1(pb.DynSysGroup):
    def __init__(self, n_neuron: int):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()

        self.inp = pb.InputProj(pe, shape_out=(n_neuron,), keep_shape=True)
        self.n1 = pb.LIF(n_neuron, threshold=3, reset_v=0, tick_wait_start=1)
        self.n2 = pb.IF(n_neuron, threshold=3, reset_v=1, tick_wait_start=3)  # tws = 3
        self.s0 = pb.NoDecay(
            self.inp,
            self.n1,
            weights=np.random.randint(-128, 128, size=(n_neuron,), dtype=np.int8),
            conn_type=pb.synapses.ConnType.One2One,
        )
        self.s1 = pb.NoDecay(
            self.n1,
            self.n2,
            weights=np.random.randint(
                -128, 128, size=(n_neuron, n_neuron), dtype=np.int8
            ),
            conn_type=pb.synapses.ConnType.All2All,
        )

        # Probes inside
        self.n1_acti = pb.Probe(self.n1, "output", name="n1_acti")
        self.s1_weight = pb.Probe(self.s1, "weights", name="s1_weight")
        self.n2_acti = pb.Probe(self.n2, "output", name="n2_acti")


class TestSimulator:
    def test_probe(self):
        net = Net1(100)

        probe_outside = pb.Probe(net.inp, "state", name="inp_state")

        sim = pb.Simulator(net)
        sim.add_probe(probe_outside)

        # Normalized data
        input_data = np.random.rand(10, 10).astype(np.float32)

        sim.run(10, input=input_data)

        inp_state = sim.data[probe_outside]
        assert type(inp_state) == np.ndarray

        inp_state2 = sim.get_raw(probe_outside)
        assert type(inp_state2) == list

        # Get the data at time=5
        inp_state_at_t = sim.get_raw_at_t(probe_outside, t=5)
        assert type(inp_state_at_t) == np.ndarray

    def test_sim_behavior(self):
        net = Net1(100)
        sim = pb.Simulator(net, include_time_zero=True)
        sim.run(10, input=np.zeros(100, dtype=np.int8))

        assert sim.time == 10

        sim2 = pb.Simulator(net, include_time_zero=False)
        sim2.run(10, input=np.zeros(100, dtype=np.int8))

        assert sim2.time == 11
