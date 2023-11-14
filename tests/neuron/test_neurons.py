import numpy as np
import pytest

import paibox as pb
from paibox.neuron.base import *
from paibox.utils import as_shape, shape2num


class TestMetaNeuronBehavior:
    sim = SIM.MODE_DETERMINISTIC
    lim = LIM.MODE_DETERMINISTIC
    ld = LDM.MODE_FORWARD
    lc = LCM.LEAK_AFTER_COMP
    leak_v = 0
    pos_thres = 10
    neg_thres = 0
    mask = 0
    reset_v = 5
    ntm = NTM.MODE_SATURATION
    reset_mode = RM.MODE_NORMAL
    bt = 0

    @pytest.mark.parametrize(
        "vjt_init, x, expected",
        [
            (0, np.array([[1, 0, 1], [0, 1, 1]]), np.array([2, 2])),
            (1, np.array([[1, 0], [1, 1]]), np.array([2, 3])),
            (0, np.array([1, 1]), np.array([1, 1])),
            (0, np.array(2), np.array([2, 2])),
        ],
    )
    def test_neuronal_charge(self, vjt_init, x, expected):
        n1 = MetaNeuron(
            2,
            self.reset_mode,
            self.reset_v,
            self.lc,
            self.mask,
            self.ntm,
            self.neg_thres,
            self.pos_thres,
            self.ld,
            self.lim,
            self.leak_v,
            self.sim,
            self.bt,
            vjt_init,
            keep_shape=True,
        )
        n1._neuronal_charge(x)

        assert np.array_equal(n1.voltage, expected)

    @pytest.mark.parametrize(
        "lim, ld, vjt, leak_v, expected",
        [
            (LIM.MODE_DETERMINISTIC, LDM.MODE_FORWARD, 1, 2, np.array([3, 3])),
            (LIM.MODE_DETERMINISTIC, LDM.MODE_REVERSAL, 1, 2, np.array([3, 3])),
            (LIM.MODE_DETERMINISTIC, LDM.MODE_REVERSAL, -2, 2, np.array([-4, -4])),
        ],
        # ids="path_1, path_2, path_3,path_4,path_5,path_6,path_7,path_8,path_9"
    )
    def test_neuronal_leak(self, lim, ld, vjt, leak_v, expected):
        n1 = MetaNeuron(
            2,
            self.reset_mode,
            self.reset_v,
            self.lc,
            self.mask,
            self.ntm,
            self.neg_thres,
            self.pos_thres,
            ld,
            lim,
            leak_v,
            self.sim,
            self.bt,
            vjt_init=vjt,
            keep_shape=True,
        )
        n1._neuronal_leak()

        assert np.array_equal(n1._vjt, expected)

    @pytest.mark.parametrize(
        "ntm, vjt, neg_thres, pos_thres, expected",
        [
            (NTM.MODE_SATURATION, 10, 10, 3, np.array([True])),
            (NTM.MODE_SATURATION, 5, 10, 3, np.array([False])),
            (NTM.MODE_SATURATION, -12, 10, 3, np.array([False])),
        ],
    )
    def test_neuronal_fire(self, ntm, vjt, neg_thres, pos_thres, expected):
        # mask=3
        n1 = MetaNeuron(
            1,
            self.reset_mode,
            self.reset_v,
            self.lc,
            3,
            ntm,
            neg_thres,
            pos_thres,
            self.ld,
            self.lim,
            2,
            self.sim,
            self.bt,
            vjt_init=vjt,
            keep_shape=True,
        )
        n1._neuronal_fire()

        assert np.array_equal(n1._spike, expected)

    @pytest.mark.parametrize(
        "ntm, thr_mode, reset_mode, expected",
        [
            (NTM.MODE_RESET, TM.EXCEED_POSITIVE, RM.MODE_NORMAL, np.array([5])),
            (NTM.MODE_RESET, TM.EXCEED_POSITIVE, RM.MODE_NONRESET, np.array([10])),
            (NTM.MODE_RESET, TM.EXCEED_NEGATIVE, RM.MODE_NORMAL, np.array([-5])),
            (NTM.MODE_RESET, TM.EXCEED_NEGATIVE, RM.MODE_NONRESET, np.array([10])),
            (NTM.MODE_SATURATION, TM.EXCEED_NEGATIVE, RM.MODE_NONRESET, np.array([-3])),
        ],
    )
    def test_neuronal_reset(self, ntm, thr_mode, reset_mode, expected):
        n1 = MetaNeuron(
            1,
            reset_mode,
            5,
            self.lc,
            self.mask,
            ntm,
            3,
            -2,
            self.ld,
            self.lim,
            self.leak_v,
            self.sim,
            self.bt,
            vjt_init=10,
            keep_shape=True,
        )
        n1._threshold_mode = thr_mode
        n1._neuronal_reset()

        assert np.array_equal(n1._vjt, expected)


@pytest.mark.parametrize(
    "shape",
    [5, (12,), (20, 20), (1, 2, 3)],
    ids=["scalar", "ndim=1", "ndim=2", "ndim=3"],
)
def test_neuron_instance(shape):
    # keep_shape = True
    n1 = pb.neuron.TonicSpiking(shape, 5, keep_shape=True)

    assert n1.shape_in == as_shape(shape)
    assert n1.shape_out == as_shape(shape)
    assert len(n1) == shape2num(shape)

    # keep_shape = False
    n2 = pb.neuron.TonicSpiking(shape, 5)

    assert n2.shape_in == as_shape(shape2num(shape))
    assert n2.shape_out == as_shape(shape2num(shape))
    assert len(n2) == shape2num(shape)


def fakeout(t):
    data = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        np.bool_,
    )

    return data[t]


class Net1(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(fakeout, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking((2,), 3)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.One2One
        )

        self.probe1 = pb.simulator.Probe(self.inp1, "output")
        self.probe2 = pb.simulator.Probe(self.s1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "output")


class TonicSpikingNet(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(fakeout, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking((2,), 3)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.One2One
        )

        self.probe1 = pb.simulator.Probe(self.s1, "output")
        self.probe2 = pb.simulator.Probe(self.n1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "voltage")


class TestNeuronBehavior:
    def test_TonicSpiking_simple_sim(self):
        n1 = pb.neuron.TonicSpiking(shape=1, fire_step=3)
        inp_data = np.ones((10,), dtype=np.bool_)
        output = np.full((10,), 0, dtype=np.bool_)
        voltage = np.full((10,), 0, dtype=np.int32)

        for t in range(10):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_PhasicSpiking_simple_sim(self):
        n1 = pb.neuron.PhasicSpiking(shape=1, time_to_fire=3)
        # [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        inp_data = np.concatenate((np.zeros((2,), np.bool_), np.ones((10,), np.bool_)))
        output = np.full((12,), 0, dtype=np.bool_)
        voltage = np.full((12,), 0, dtype=np.int32)

        for t in range(12):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_neuron_behavior(self):
        net = Net1()
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])

    def test_TonicSpiking_behavior(self):
        net = TonicSpikingNet()
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])
