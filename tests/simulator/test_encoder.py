import numpy as np
import pytest

import paibox as pb


class TestEncoder:
    def test_PeriodicEncoder(self):
        spike = np.full((5, 3), 0)
        spike[0, 1] = 1
        spike[1, 0] = 1
        spike[4, 2] = 1

        pe = pb.simulator.PeriodicEncoder(spike)
        out_spike = np.full((20, 3), 0)
        for t in range(20):
            out_spike[t] = pe()

        assert np.array_equal(spike, out_spike[:5])
        assert np.array_equal(spike, out_spike[5:10])
        assert np.array_equal(spike, out_spike[10:15])
        assert np.array_equal(spike, out_spike[15:20])

    def test_LatencyEncoder(self):
        N = 6
        x = np.random.rand(N)
        T = 20

        le1 = pb.simulator.LatencyEncoder(T, "linear")
        le2 = pb.simulator.LatencyEncoder(T, "log")

        out_spike1 = np.zeros((T, N), dtype=np.bool_)
        out_spike2 = np.zeros((T, N), dtype=np.bool_)
        for t in range(T):
            out_spike1[t] = le1(x)
            out_spike2[t] = le2(x)
        assert 1

    def test_PoissonEncoder(self):
        seed = 1
        rng = np.random.RandomState(seed=seed)
        x = rng.rand(10, 10).astype(np.float32)
        pe = pb.simulator.PoissonEncoder(seed=seed)

        out_spike = np.full((20, 10, 10), 0)
        for t in range(20):
            out_spike[t] = pe(x=x)
        for t in range(1, 20):
            assert not np.array_equal(out_spike[0], out_spike[t])


    def test_DirectEncoder(self):
        seed = 1
        rng = np.random.RandomState(seed=seed)
        ksize = np.random.uniform(-1, 1, size=(64, 3, 3, 3)).astype(np.float32)
        stride = (1, 1)
        padding = (1,1)
        outshape = (64, 10, 10)
        x = rng.rand(3, 10, 10).astype(np.float32)
        de = pb.simulator.DirectEncoder(ksize=ksize, stride=stride, padding=padding)
        for t in range(20):
            out_spike = de(x=x)
            assert out_spike.shape == outshape
            assert out_spike.dtype
