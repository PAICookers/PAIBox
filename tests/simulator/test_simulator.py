import numpy as np
import pytest

import paibox as pb

from .sim_networks import (
    Net1,
    Net2_with_multi_inpproj_func,
    Net2_with_multi_inpproj_encoder,
    Conv2d_Net,
)


class TestSimulator:
    def test_probe(self):
        net = Net1(100)

        probe_outside = pb.Probe(net.inp, "spike", name="out_probe")

        sim = pb.Simulator(net)
        sim.add_probe(probe_outside)

        # Normalized data
        input_data = np.random.rand(10, 10).astype(np.float32)
        net.inp.input = input_data

        sim.run(10)

        inp_state = sim.data[probe_outside]
        assert isinstance(inp_state, np.ndarray)

        inp_state2 = sim.get_raw(probe_outside)
        assert isinstance(inp_state2, list)

        # Get the data at time=5
        inp_state_at_t = sim.get_raw_at_t(probe_outside, t=5)
        assert isinstance(inp_state_at_t, np.ndarray)

    def test_sim_behavior(self):
        net = Net1(100)
        probe = pb.Probe(net.inp, "spike", name="inp_spike1")
        probe2 = pb.Probe(net.inp, "spike", name="inp_spike2")

        sim = pb.Simulator(net, start_time_zero=True)
        sim.add_probe(probe)

        net.inp.input = np.zeros(100, dtype=np.int8)
        sim.run(10)  # Actually, 0~9

        assert len(sim.data["ts"] == 10)
        d = sim.get_raw_at_t(probe, 0)
        d = sim.get_raw_at_t(probe, 9)

        with pytest.raises(IndexError):
            _ = sim.get_raw_at_t(probe, 10)

        with pytest.raises(IndexError):
            _ = sim.get_raw_at_t(probe, -1)

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
            _ = sim2.get_raw_at_t(probe2, 11)

        with pytest.raises(IndexError):
            _ = sim2.get_raw_at_t(probe2, 0)

    def test_sim_specify_inputs_1(self):
        net = Net2_with_multi_inpproj_func(10)
        sim = pb.Simulator(net, start_time_zero=False)

        pb.FRONTEND_ENV.save(a=1, b=2)
        sim.run(10)

        pb.FRONTEND_ENV.save("a", -1, "b", -2)
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

    def test_sim_nested_net(self, build_Nested_Net_L3):
        net = build_Nested_Net_L3
        sim = pb.Simulator(net, start_time_zero=False)

        # The probes defined in the subnets will be discovered.
        assert len(sim.probes) == 3 + 1 * 2 + 2 * 1

        net.inp1.input = np.ones((10,), dtype=np.int8)
        sim.run(20)

        sim.reset()

    def test_sim_conv2d_net(self):
        net = Conv2d_Net()
        sim = pb.Simulator(net, start_time_zero=False)

        net.inp1.input = np.random.rand(8, 24, 24)
        sim.run(10)

        sim.reset()
