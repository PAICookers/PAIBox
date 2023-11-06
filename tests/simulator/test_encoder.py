import numpy as np
import pytest

import paibox as pb


class TestEncoder:
    def test_PeriodicEncoder(self):
        spike = np.full((5, 3), 0)
        spike[0, 1] = 1
        spike[1, 0] = 1
        spike[4, 2] = 1

        pe = pb.simulator.encoder.PeriodicEncoder(spike)

        out_spike = np.full((20, 3), 0)

        for t in range(20):
            out_spike[t] = pe(spike)

        assert np.array_equal(spike, out_spike[:5])
        assert np.array_equal(spike, out_spike[5:10])
        assert np.array_equal(spike, out_spike[10:15])
        assert np.array_equal(spike, out_spike[15:20])

    def test_PoissonEncoder(self):
        pe = pb.simulator.encoder.PoissonEncoder(shape_out=(10, 10))
        x = np.random.randint(-128, 128, (10, 10))

        # out_spike = np.full((20, 10, 10), 0)

        # for t in range(20):
        #     out_spike[t] = pe(x)
            
        out = pe.run(20, x=x)
