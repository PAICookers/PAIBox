import json

import numpy as np
import pytest
from paicorelib import *

import paibox as pb
from paibox.utils import as_shape, shape2num


def test_NeuronParams_instance(ensure_dump_dir):
    n1 = pb.neuron.LIF((100,), 3)

    attrs = NeuronAttrs.model_validate(n1.export_params(), strict=True)

    attrs_dict = attrs.model_dump(by_alias=True)

    with open(ensure_dump_dir / f"ram_model_{n1.name}.json", "w") as f:
        json.dump({n1.name: attrs_dict}, f, indent=4, ensure_ascii=True)


def test_NeuronParams_check():
    with pytest.raises(ValueError):
        n1 = pb.neuron.LIF((100,), threshold=-1)

    with pytest.raises(ValueError):
        n2 = pb.neuron.IF((100,), 1, delay=-1)

    with pytest.raises(ValueError):
        n3 = pb.neuron.IF((100,), 1, delay=1, tick_wait_start=-1, tick_wait_end=100)


class TestNeuronBehavior:
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
        "vjt, x, expected",
        [
            (0, np.array([[1, 0, 1], [0, 1, 1]]), np.array([2, 2])),
            (0, np.array([1, 1]), np.array([1, 1])),
            (0, np.array([2, 2]), np.array([2, 2])),
        ],
    )
    def test_neuronal_charge(self, vjt, x, expected):
        n1 = pb.neuron.Neuron(
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
            0,
            keep_shape=True,
        )
        vjt = n1._neuronal_charge(x, vjt)

        assert np.array_equal(vjt, expected)

    @pytest.mark.parametrize(
        "lim, ld, vjt, leak_v, expected",
        [
            (
                LIM.MODE_DETERMINISTIC,
                LDM.MODE_FORWARD,
                np.array([1, 1]),
                2,
                np.array([3, 3]),
            ),
            (
                LIM.MODE_DETERMINISTIC,
                LDM.MODE_REVERSAL,
                np.array([1, 1]),
                2,
                np.array([3, 3]),
            ),
            (
                LIM.MODE_DETERMINISTIC,
                LDM.MODE_REVERSAL,
                np.array([-2, -2]),
                2,
                np.array([-4, -4]),
            ),
        ],
    )
    def test_neuronal_leak(self, lim, ld, vjt, leak_v, expected):
        n1 = pb.neuron.Neuron(
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
        leaked_vjt = n1._neuronal_leak(vjt)

        assert np.array_equal(leaked_vjt, expected)

    @pytest.mark.parametrize(
        "ntm, vjt, neg_thres, pos_thres, expected",
        [
            (NTM.MODE_SATURATION, np.array([10, 10]), -10, 3, np.array([True, True])),
            (NTM.MODE_SATURATION, np.array([5, 10]), -10, 3, np.array([True, True])),
            (NTM.MODE_SATURATION, np.array([-12, 10]), -10, 3, np.array([False, True])),
        ],
    )
    def test_neuronal_fire(self, ntm, vjt, neg_thres, pos_thres, expected):
        mask = 3
        leak_v = 2

        n1 = pb.neuron.Neuron(
            2,
            self.reset_mode,
            self.reset_v,
            self.lc,
            mask,
            ntm,
            neg_thres,
            pos_thres,
            self.ld,
            self.lim,
            leak_v,
            self.sim,
            self.bt,
            vjt_init=vjt,
            keep_shape=True,
        )
        spike = n1._neuronal_fire(vjt)

        assert np.array_equal(spike, expected)

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
    def test_neuronal_reset(self, monkeypatch, ntm, thr_mode, reset_mode, expected):
        reset_v = 5
        neg_thres = -3
        pos_thres = 2
        vjt_init = 0
        incoming_v = 10

        n1 = pb.neuron.Neuron(
            1,
            reset_mode,
            reset_v,
            self.lc,
            self.mask,
            ntm,
            neg_thres,
            pos_thres,
            self.ld,
            self.lim,
            self.leak_v,
            self.sim,
            self.bt,
            vjt_init=vjt_init,
            keep_shape=True,
        )

        # Set the threshold mode manually
        monkeypatch.setattr(n1, "thres_mode", thr_mode)
        vjt = n1._neuronal_reset(np.array((incoming_v,), dtype=np.int32))

        assert np.array_equal(vjt, expected)


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


def test_neuron_keep_shape():
    n1 = pb.neuron.TonicSpiking((4, 4), 5, keep_shape=True)
    n2 = pb.neuron.TonicSpiking((4, 4), 5, keep_shape=False)

    assert n1.spike.shape == (16,)
    assert n1.voltage.shape == (4, 4)
    assert n1.output.shape == (256, 16)
    assert n1.feature_map.shape == (4, 4)

    assert n2.spike.shape == (16,)
    assert n2.voltage.shape == (16,)
    assert n2.output.shape == (256, 16)
    assert n2.feature_map.shape == (16,)


class TestNeuronSim:
    def test_TonicSpiking_simple_sim(self):
        n1 = pb.neuron.TonicSpiking(shape=1, fire_step=3)
        inp_data = np.ones((10,), dtype=np.bool_)
        output = np.full((10, 1), 0, dtype=np.bool_)
        voltage = np.full((10, 1), 0, dtype=np.int32)

        for t in range(10):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_PhasicSpiking_simple_sim(self):
        n1 = pb.neuron.PhasicSpiking(shape=1, time_to_fire=3)
        # [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        inp_data = np.concatenate((np.zeros((2,), np.bool_), np.ones((10,), np.bool_)))
        output = np.full((12, 1), 0, dtype=np.bool_)
        voltage = np.full((12, 1), 0, dtype=np.int32)

        for t in range(12):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_IF_simple_sim(self):
        n1 = pb.neuron.IF(shape=1, threshold=5, reset_v=2)
        # [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        inp_data = np.concatenate((np.zeros((2,), np.bool_), np.ones((10,), np.bool_)))
        # inp_data = np.ones((12,), dtype=np.bool_)
        output = np.full((12, 1), 0, dtype=np.bool_)
        voltage = np.full((12, 1), 0, dtype=np.int32)

        for t in range(12):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_LIF_simple_sim(self):
        n1 = pb.neuron.LIF(shape=1, threshold=5, reset_v=2, leaky_v=1)  # leak + 1
        # [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        inp_data = np.concatenate((np.zeros((2,), np.bool_), np.ones((10,), np.bool_)))
        # inp_data = np.ones((12,), dtype=np.bool_)
        output = np.full((12, 1), 0, dtype=np.bool_)
        voltage = np.full((12, 1), 0, dtype=np.int32)

        for t in range(12):
            output[t] = n1(inp_data[t])
            voltage[t] = n1.voltage

        print(output)

    def test_neuron_behavior(self, build_Net1):
        net = build_Net1
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])

    def test_TonicSpiking_behavior(self, build_TonicSpikingNet):
        net = build_TonicSpikingNet
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])

    def test_sum_inputs_behavior(self, build_Net2):
        net = build_Net2
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])
        print(sim.data[net.probe2])

    def test_tick_attr_behavior(self, monkeypatch, build_Net3):
        net = build_Net3
        sim = pb.Simulator(net)

        # n1 works on 1 <= T <= 1+5-1
        # n2 works on 2 <= T <= 2+6-1
        monkeypatch.setattr(net.n1, "_tws", 1)
        monkeypatch.setattr(net.n1, "_twe", 5)
        monkeypatch.setattr(net.n2, "_tws", 2)
        monkeypatch.setattr(net.n2, "_twe", 6)

        sim.run(10)
        sim.reset()

        # n1 works on T >= 1
        # n2 won't work
        monkeypatch.setattr(net.n1, "_tws", 1)
        monkeypatch.setattr(net.n1, "_twe", 0)
        monkeypatch.setattr(net.n2, "_tws", 0)
        monkeypatch.setattr(net.n2, "_twe", 0)

        sim.run(10)
        sim.reset()

        # n1 works on T >= 5
        # n2 works on T >= 1
        monkeypatch.setattr(net.n1, "_tws", 5)
        monkeypatch.setattr(net.n1, "_twe", 2)
        monkeypatch.setattr(net.n2, "_tws", 1)
        monkeypatch.setattr(net.n2, "_twe", 0)

        sim.run(10)
        sim.reset()

        # TODO can add new test items here
