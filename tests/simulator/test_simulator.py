import numpy as np
import pytest

import paibox as pb
from paibox.context import FRONTEND_ENV


class Net1(pb.DynSysGroup):
    def __init__(self, n_neuron: int):
        super().__init__()

        pe = pb.simulator.PoissonEncoder()

        self.inp = pb.InputProj(pe, shape_out=(n_neuron,), keep_shape=True)
        self.n1 = pb.LIF(n_neuron, threshold=3, reset_v=0, tick_wait_start=1)
        self.n2 = pb.IF(n_neuron, threshold=3, reset_v=1, tick_wait_start=2)
        self.s0 = pb.NoDecay(
            self.inp,
            self.n1,
            weights=np.random.randint(-128, 128, size=(n_neuron,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.NoDecay(
            self.n1,
            self.n2,
            weights=np.random.randint(
                -128, 128, size=(n_neuron, n_neuron), dtype=np.int8
            ),
            conn_type=pb.SynConnType.All2All,
        )

        # Probes inside
        self.n1_acti = pb.Probe(self.n1, "spike", name="n1_acti")
        self.s1_weight = pb.Probe(self.s1, "weights", name="s1_weight")
        self.n2_acti = pb.Probe(self.n2, "spike", name="n2_acti")


def fake_out_1(t, a, **kwargs):
    return t + a


def fake_out_2(t, b, **kwargs):
    return t + b


class Net2_with_multi_inpproj_func(pb.DynSysGroup):
    def __init__(self, n: int):
        super().__init__()

        self.inp1 = pb.InputProj(fake_out_1, shape_out=(n,), keep_shape=True)
        self.inp2 = pb.InputProj(fake_out_2, shape_out=(n,), keep_shape=True)
        self.n1 = pb.LIF(n, threshold=3, reset_v=0, tick_wait_start=1)
        self.s0 = pb.NoDecay(
            self.inp1,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.NoDecay(
            self.inp2,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )

        # Probes inside
        self.inp1_output = pb.Probe(self.inp1, "output")
        self.inp2_output = pb.Probe(self.inp2, "output")
        self.n1_output = pb.Probe(self.n1, "spike")


class Net2_with_multi_inpproj_encoder(pb.DynSysGroup):
    def __init__(self, n: int):
        super().__init__()

        pe1 = pb.simulator.PoissonEncoder(seed=21)
        pe2 = pb.simulator.PoissonEncoder(seed=42)

        self.inp1 = pb.InputProj(pe1, shape_out=(n,), keep_shape=True)
        self.inp2 = pb.InputProj(pe2, shape_out=(n,), keep_shape=True)
        self.n1 = pb.LIF(n, threshold=3, reset_v=0, tick_wait_start=1)
        self.s0 = pb.NoDecay(
            self.inp1,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )
        self.s1 = pb.NoDecay(
            self.inp2,
            self.n1,
            weights=np.ones((n,), dtype=np.int8),
            conn_type=pb.SynConnType.One2One,
        )

        # Probes inside
        self.inp1_output = pb.Probe(self.inp1, "output")
        self.inp2_output = pb.Probe(self.inp2, "output")
        self.n1_output = pb.Probe(self.n1, "spike")


class Nested_Net_level_1(pb.DynSysGroup):
    """Level 1 nested network: pre_n -> syn -> post_n"""

    def __init__(self):
        super().__init__()

        self.pre_n = pb.LIF((10,), 2, tick_wait_start=2)
        self.post_n = pb.LIF((10,), 10, tick_wait_start=3)

        w = np.ones((10, 10), dtype=np.int8)
        self.syn = pb.NoDecay(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=w
        )

        self.probe_in_subnet = pb.Probe(self.pre_n, "spike")


class Nested_Net_level_2(pb.DynSysGroup):
    """Level 2 nested network: -> s1 -> Nested_Net_level_1"""

    def __init__(self):
        self.inp1 = pb.InputProj(None, shape_out=(10,))
        self.n1 = pb.LIF((10,), 2, tick_wait_start=1)

        subnet = Nested_Net_level_1()

        self.s1 = pb.NoDecay(
            self.inp1,
            self.n1,
            conn_type=pb.SynConnType.One2One,
        )
        self.s2 = pb.NoDecay(
            self.n1,
            subnet.pre_n,
            conn_type=pb.SynConnType.One2One,
        )

        self.probe1 = pb.Probe(self.inp1, "spike")
        self.probe2 = pb.Probe(self.n1, "spike")
        self.probe3 = pb.Probe(self.n1, "voltage")
        self.probe4 = pb.Probe(subnet.pre_n, "spike")
        self.probe5 = pb.Probe(subnet.post_n, "spike")

        super().__init__(subnet)


class TestSimulator:
    def test_probe(self):
        net = Net1(100)

        probe_outside = pb.Probe(net.inp, "spike", name="inp_spike")

        sim = pb.Simulator(net)
        sim.add_probe(probe_outside)

        # Normalized data
        input_data = np.random.rand(10, 10).astype(np.float32)
        net.inp.input = input_data

        sim.run(10)

        inp_state = sim.data[probe_outside]
        assert type(inp_state) == np.ndarray

        inp_state2 = sim.get_raw(probe_outside)
        assert type(inp_state2) == list

        # Get the data at time=5
        inp_state_at_t = sim.get_raw_at_t(probe_outside, t=5)
        assert type(inp_state_at_t) == np.ndarray

    def test_sim_behavior(self):
        net = Net1(100)
        probe = pb.Probe(net.inp, "spike", name="inp_spike")
        probe2 = pb.Probe(net.inp, "spike", name="inp_spike2")

        sim = pb.Simulator(net, start_time_zero=True)
        sim.add_probe(probe)

        net.inp.input = np.zeros(100, dtype=np.int8)
        sim.run(10)  # Actually, 0~9

        assert len(sim.data["ts"] == 10)
        d = sim.get_raw_at_t(probe, 0)
        d = sim.get_raw_at_t(probe, 9)

        with pytest.raises(IndexError):
            d = sim.get_raw_at_t(probe, 10)

        with pytest.raises(IndexError):
            d = sim.get_raw_at_t(probe, -1)

        # Continue to run 5 timesteps
        net.inp.input = np.ones(100, dtype=np.int8)
        sim.run(5)
        assert len(sim.data["ts"] == 15)

        sim2 = pb.Simulator(net, start_time_zero=False)
        sim2.add_probe(probe2)

        net.inp.input = np.zeros(100, dtype=np.int8)
        sim2.run(10)  # Actually, 1-10

        assert len(sim2.data["ts"] == 11)
        d = sim2.get_raw_at_t(probe2, 1)
        d = sim2.get_raw_at_t(probe2, 10)

        with pytest.raises(IndexError):
            d = sim2.get_raw_at_t(probe2, 11)
        with pytest.raises(IndexError):
            d = sim2.get_raw_at_t(probe2, 0)

    def test_sim_specify_inputs_1(self):
        net = Net2_with_multi_inpproj_func(10)
        sim = pb.Simulator(net, start_time_zero=False)

        FRONTEND_ENV.save(a=1, b=2)
        sim.run(10)

        FRONTEND_ENV.save("a", -1, "b", -2)
        sim.run(3)

        sim.reset()

    def test_sim_specify_inputs_2(self):
        n = 10
        net = Net2_with_multi_inpproj_encoder(10)
        sim = pb.Simulator(net, start_time_zero=False)

        net.inp1.input = np.random.randint(-128, 128, size=(n,), dtype=np.int8)
        net.inp2.input = np.random.randint(-128, 128, size=(n,), dtype=np.int8)
        sim.run(10)

        net.inp1.input = np.ones((n,), dtype=np.int8)
        net.inp2.input = np.ones((n,), dtype=np.int8)
        sim.run(3)

        sim.reset()

    def test_sim_nested_net(self):
        net = Nested_Net_level_2()
        sim = pb.Simulator(net, start_time_zero=False)

        # The probes defined in the subnets cannot be discovered.
        assert len(sim.probes) == 5

        net.inp1.input = np.ones((10,), dtype=np.int8)
        sim.run(20)

        sim.reset()
