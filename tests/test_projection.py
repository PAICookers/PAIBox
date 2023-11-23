import numpy as np
import pytest

import paibox as pb
from paibox import FRONTEND_ENV


class TestInputProj:
    @pytest.mark.parametrize(
        "keep_shape, expected_shape", [(True, (4, 4)), (False, (16,))]
    )
    def test_input_scalar(
        self,
        keep_shape,
        expected_shape,
    ):
        inp = pb.InputProj(1, shape_out=(4, 4), keep_shape=keep_shape)
        prob = pb.simulator.Probe(inp, "feature_map")

        sim = pb.Simulator(inp)
        sim.add_probe(prob)
        sim.run(10)

        assert len(sim.data[prob]) == 10
        assert sim.data[prob][0].shape == expected_shape

    def test_input_array(self):
        shape_out = (4, 4)

        inp1 = pb.InputProj(np.random.rand(*shape_out), shape_out, keep_shape=True)

        prob = pb.simulator.Probe(inp1, "feature_map")
        sim = pb.Simulator(inp1)
        sim.add_probe(prob)
        sim.run(10)

        assert sim.data[prob][0].shape == shape_out

    def test_input_callable(self):
        def fakeout(*args, **kwargs):
            return np.ones((10, 10), dtype=np.int8)

        def fakeout_with_t(t, *args, **kwargs):
            return np.ones((10, 10), dtype=np.int8) * t

        def fakeout_with_args(t, bias, *args, **kwargs):
            return np.ones((10, 10), dtype=np.int8) * bias

        inp1 = pb.InputProj(input=fakeout, shape_out=(10, 10), keep_shape=False)
        prob1 = pb.simulator.Probe(inp1, "output")

        sim1 = pb.Simulator(inp1)
        sim1.add_probe(prob1)
        sim1.run(10)
        assert len(sim1.data[prob1]) == 10
        assert sim1.data[prob1][0].shape == (100,)

        inp2 = pb.InputProj(input=fakeout_with_t, shape_out=(10, 10), keep_shape=True)
        prob2 = pb.simulator.Probe(inp2, "feature_map")

        sim2 = pb.Simulator(inp2)
        sim2.add_probe(prob2)
        sim2.run(10)
        assert len(sim2.data[prob2]) == 10
        assert sim2.data[prob2][0].shape == (10, 10)

        inp3 = pb.InputProj(
            input=fakeout_with_args, shape_out=(10, 10), keep_shape=False
        )
        prob3 = pb.simulator.Probe(inp3, "output")
        sim3 = pb.Simulator(inp3)
        sim3.add_probe(prob3)
        sim3.run(10, bias=3)

        assert len(sim3.data[prob3]) == 10
        assert sim3.data[prob3][0].shape == (100,)

    def test_input_PoissonEncoder(self):
        # Normalized data
        input_data1 = np.random.rand(10, 10).astype(np.float32)
        input_data2 = np.random.rand(10, 10).astype(np.float32)

        assert not np.allclose(input_data1, input_data2)

        pe = pb.simulator.PoissonEncoder()
        inp = pb.InputProj(pe, shape_out=(10, 10), keep_shape=False)

        sim = pb.Simulator(inp)
        prob = pb.simulator.Probe(inp, "output")
        sim.add_probe(prob)

        sim.run(10, input=input_data1)
        assert len(sim.data[prob]) == 10

        # 10 different results after poisson encoding.
        assert not all(
            np.array_equal(sim.data[prob][0], data) for data in sim.data[prob]
        )

        # Change the input and continue. Unsafe!
        FRONTEND_ENV["input"] = input_data2
        sim.run(10)
        assert len(sim.data[prob]) == 20

        # 20 different results after poisson encoding.
        assert not all(
            np.array_equal(sim.data[prob][0], data) for data in sim.data[prob]
        )

        sim.reset()  # FRONTEND_ENV["t"] reset to zero as well.
        # Clear the input you stored in the frontend environment.
        FRONTEND_ENV.clear_ctx("input")

    def test_input_PeriodicEncoder(self):
        spike = np.full((5, 3), 0)
        spike[0, 1] = 1
        spike[1, 0] = 1
        spike[4, 2] = 1

        pe = pb.simulator.PeriodicEncoder(spike)
        inp = pb.InputProj(pe, shape_out=(3,), keep_shape=False)

        sim = pb.Simulator(inp)
        prob = pb.simulator.Probe(inp, "output")
        sim.add_probe(prob)

        sim.run(10)
        assert len(sim.data[prob]) == 10
