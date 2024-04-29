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

    @pytest.mark.parametrize(
        "in_shape, in_channels, out_channels, kernel_size, stride, padding",
        # Padding is fixed at (0, 0)
        [
            ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0)),
            ((28, 28), 24, 12, (3, 3), (2, 2), (1, 1)),
            ((28, 28), 16, 8, (3, 3), (1, 1), (2, 2)),
            ((28, 28), 24, 12, (3, 3), (2, 2), (1, 2)),
            ((16, 16), 8, 16, (3, 3), (2, 2), (0, 0)),
            ((28, 28), 16, 8, (3, 3), (1, 1), (0, 0)),
            ((24, 32), 8, 8, (3, 4), (2, 1), (1, 1)),
            ((24, 24), 8, 16, (7, 7), (2, 2), (2, 2)),
            ((32, 16), 4, 12, (5, 7), (1, 2), (2, 1)),
            ((24, 24), 8, 16, (7, 7), (2, 2), (0, 0)),
        ],
    )
    def test_Conv2dEncoder(
        self, in_shape, in_channels, out_channels, kernel_size, stride, padding
    ):
        kernel = np.random.uniform(
            -1, 1, size=(out_channels, in_channels, *kernel_size)
        ).astype(np.float32)

        out_shape = (
            out_channels,
            (in_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
            (in_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
        )

        de = pb.simulator.Conv2dEncoder(
            kernel, stride, padding, tau=2, decay_input=True, v_reset=0.2
        )
        x = np.random.uniform(-1, 1, size=(in_channels, *in_shape)).astype(np.float32)

        for t in range(20):
            out_spike = de(x)
            assert out_spike.shape == out_shape
            assert out_spike.dtype == np.bool_
