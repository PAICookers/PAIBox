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

    def test_PoissonEncoder(self):
        seed = 1
        rng = np.random.RandomState(seed=seed)
        x = rng.rand(10, 10).astype(np.float32)
        pe = pb.simulator.PoissonEncoder(seed=seed)

        out_spike = np.full((20, 10, 10), 0)

        for t in range(20):
            out_spike[t] = pe(input=x)

        for t in range(1, 20):
            assert not np.array_equal(out_spike[0], out_spike[t])
