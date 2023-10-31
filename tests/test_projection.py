import numpy as np
import paibox as pb
import pytest


class TestInputProjUpdate:
    def test_input_None(self):
        inp = pb.InputProj(input=None, shape_out=(4, 4))
        prob = pb.simulator.Probe(inp, "output")

        sim = pb.Simulator(inp)
        sim.add_probe(prob)

        with pytest.raises(RuntimeError):
            sim.run(10)

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

        inp1 = pb.InputProj(np.random.rand(*shape_out), keep_shape=True)

        prob = pb.simulator.Probe(inp1, "output")
        sim = pb.Simulator(inp1)
        sim.add_probe(prob)
        sim.run(10)

        assert sim.data[prob][0].shape == shape_out

    def test_input_callable(self):
        def fakeout(*args):
            return np.ones((10, 10), dtype=np.int8)

        def fakeout_with_t(t):
            return np.ones((10, 10), dtype=np.int8) * t

        def fakeout_with_args(t, bias):
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
        sim3.run(10, bias=1)

    def test_input_Encoder(self):
        inp1 = pb.InputProj(shape_out=(10, 10), keep_shape=False)

        encoder = pb.simulator.PoissonEncoder((10, 10))
        input_data1 = np.random.randint(0, 2, (10, 10))
        input_data2 = np.random.rand(10, 10)

        inp1.input = encoder(input_data1)

        sim = pb.Simulator(inp1)
        prob = pb.simulator.Probe(inp1, "output")
        sim.add_probe(prob)

        sim.run(10)
        assert len(sim.data[prob]) == 10
        sim.reset()

        # Change the input and run again.
        inp1.input = encoder(input_data2)
        sim.run(10)
        assert len(sim.data[prob]) == 10

        with pytest.raises(ValueError):
            # when input is None, shape_out is required
            inp2 = pb.InputProj(input=None, keep_shape=False)

        inp3 = pb.InputProj(shape_out=(5, 5), keep_shape=False)
        with pytest.raises(ValueError):
            # Different from the shape when you claimed.
            inp3.input = encoder(input_data1)
