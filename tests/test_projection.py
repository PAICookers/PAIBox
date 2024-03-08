import numpy as np
import pytest

import paibox as pb
from paibox import FRONTEND_ENV
from paibox.exceptions import ShapeError, SimulationError


class TestInputProj:
    @pytest.mark.parametrize(
        "keep_shape, expected_shape", [(True, (4, 4)), (False, (16,))]
    )
    def test_numeric_input_scalar(
        self,
        keep_shape,
        expected_shape,
    ):
        inp = pb.InputProj(1, shape_out=(4, 4), keep_shape=keep_shape)
        prob = pb.simulator.Probe(inp, "feature_map")

        sim = pb.Simulator(inp)
        sim.add_probe(prob)
        sim.run(10, reset=True)

        assert len(sim.data[prob]) == 10
        assert sim.data[prob][0].shape == expected_shape

    def test_numeric_input_array(self):
        shape_out = (4, 4)
        inp1 = pb.InputProj(
            np.random.randint(-10, 10, shape_out, dtype=np.int8),
            shape_out,
            keep_shape=True,
        )

        prob = pb.simulator.Probe(inp1, "feature_map")
        sim = pb.Simulator(inp1)
        sim.add_probe(prob)
        sim.run(10)

        assert sim.data[prob][0].shape == shape_out

    def test_None_input(self):
        shape_out = (4, 4)
        inp1 = pb.InputProj(None, shape_out, keep_shape=True)

        # Set the numeric input after claim
        inp1.input = np.ones(shape_out, dtype=np.int8)

        prob = pb.simulator.Probe(inp1, "feature_map")
        sim = pb.Simulator(inp1)
        sim.add_probe(prob)
        sim.run(10)

        assert sim.data[prob][0].shape == shape_out

    def test_functional_input(self):
        def fakeout_without_t(*args, **kwargs):
            return np.ones((10, 10), dtype=np.int8)

        def fakeout_with_t(t):
            return np.ones((10, 10), dtype=np.int8) * t

        def fakeout_with_args(t, bias, *args, **kwargs):
            return np.ones((10, 10), dtype=np.int8) * bias

        inp1 = pb.InputProj(
            input=fakeout_without_t, shape_out=(10, 10), keep_shape=False
        )
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

        FRONTEND_ENV.save(bias=3)  # Pass the extra arguments
        sim3.run(10)

        assert len(sim3.data[prob3]) == 10
        assert sim3.data[prob3][0].shape == (100,)

    def test_passing_args_through_run(self):
        def fakeout_with_args(t, bias, *args, **kwargs):
            return np.ones((10, 10), dtype=np.int8) * bias

        FRONTEND_ENV.clear_ctx("bias")

        inp = pb.InputProj(
            input=fakeout_with_args, shape_out=(10, 10), keep_shape=False
        )
        prob = pb.simulator.Probe(inp, "output")
        sim = pb.Simulator(inp)
        sim.add_probe(prob)

        with pytest.warns(DeprecationWarning):
            sim.run(10, bias=3)

    def test_input_PoissonEncoder(self):
        # Normalized data
        input_data1 = np.random.rand(10, 10).astype(np.float32)
        input_data2 = np.random.rand(10, 10).astype(np.float32)

        assert not np.allclose(input_data1, input_data2)

        pe = pb.simulator.PoissonEncoder()
        inp = pb.InputProj(pe, shape_out=(10, 10), keep_shape=False)
        inp.input = input_data1

        sim = pb.Simulator(inp)
        prob = pb.simulator.Probe(inp, "output")
        sim.add_probe(prob)
        sim.run(10)
        assert len(sim.data[prob]) == 10

        # 10 different results after poisson encoding.
        assert not all(
            np.array_equal(sim.data[prob][0], data) for data in sim.data[prob]
        )

        # Change the input & test again.
        inp.input = input_data2
        sim.run(5)
        assert len(sim.data[prob]) == 15

        # 15 different results after poisson encoding.
        assert not all(
            np.array_equal(sim.data[prob][0], data) for data in sim.data[prob]
        )

        sim.reset()  # FRONTEND_ENV["t"] reset to initial value as well.

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

    def test_illegal_input(self):
        def fakeout_with_t(t, **kwargs):
            return np.ones((10, 10), dtype=np.int8) * t

        inp1 = pb.InputProj(None, shape_out=(4, 4), keep_shape=True)
        with pytest.raises(TypeError):
            inp1.input = fakeout_with_t  # type: ignore

        sim = pb.Simulator(inp1)

        # Didn't claim the numeric input nor the functional input
        with pytest.raises(SimulationError):
            sim.run(10)

        # Wrong input shape
        inp1.input = np.ones((3, 4), dtype=np.int8)
        with pytest.raises(ShapeError):
            sim.run(10)

    def test_inputproj_in_Network(self):
        class Net(pb.Network):
            def __init__(self):
                super().__init__()
                self.inp1 = pb.InputProj(
                    input=np.ones((10, 10), dtype=np.int8),
                    shape_out=(10, 10),
                    keep_shape=False,
                )
                self.inp2 = pb.InputProj(
                    input=pb.simulator.PoissonEncoder(),
                    shape_out=(10, 10),
                    keep_shape=False,
                )

                self.probe1 = pb.Probe(self.inp1, "output")
                self.probe2 = pb.Probe(self.inp2, "output")

        net = Net()
        sim = pb.Simulator(net, start_time_zero=False)

        # Set the input of nodes
        net.inp1.input = np.random.randint(-10, 10, size=(10, 10), dtype=np.int8)
        net.inp2.input = np.random.rand(10, 10).astype(np.float32)

        sim.run(5)
        assert len(sim.data[net.probe1]) == 5
