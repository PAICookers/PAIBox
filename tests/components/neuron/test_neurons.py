import json
from copy import copy

import numpy as np
import pytest
from paicorelib import LCM, LDM, LIM, NTM, RM, SIM, TM, NeuronAttrs

import paibox as pb
from paibox.components import Neuron
from paibox.utils import as_shape, shape2num

from .conftest import Net_Test_Neuron_Behavior


def test_NeuronParams_instance(ensure_dump_dir):
    n1 = pb.LIF((100,), 3, reset_v=-20, leak_v=-2)

    attrs = NeuronAttrs.model_validate(n1.export_params(), strict=True)

    attrs_dict = attrs.model_dump(by_alias=True)

    with open(ensure_dump_dir / f"ram_model_{n1.name}.json", "w") as f:
        json.dump({n1.name: attrs_dict}, f, indent=2)


def test_NeuronParams_check():
    with pytest.raises(ValueError):
        n1 = pb.LIF((100,), threshold=-1)

    with pytest.raises(ValueError):
        n2 = pb.IF((100,), 1, delay=-1)

    with pytest.raises(ValueError):
        n3 = pb.IF((100,), 1, delay=1, tick_wait_start=-1, tick_wait_end=100)


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
        "incoming_v, x, expected",
        [
            (0, np.array([[1, 0, 1], [0, 1, 1]]), np.array([2, 2])),
            (0, np.array([1, 1]), np.array([1, 1])),
            (0, np.array([2, 2]), np.array([2, 2])),
        ],
    )
    def test_neuronal_charge(self, incoming_v, x, expected):
        n1 = Neuron(
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
            keep_shape=True,
        )
        v_charged = n1._neuronal_charge(x, incoming_v)

        assert np.array_equal(v_charged, expected)

    @pytest.mark.parametrize(
        "lim, ld, incoming_v, leak_v, expected",
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
    def test_neuronal_leak(self, lim, ld, incoming_v, leak_v, expected):
        n1 = Neuron(
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
            keep_shape=True,
        )
        v_leaked = n1._neuronal_leak(incoming_v)

        assert np.array_equal(v_leaked, expected)

    @pytest.mark.parametrize(
        "ntm, incoming_v, neg_thres, pos_thres, expected",
        [
            (NTM.MODE_SATURATION, np.array([10, 10]), -10, 3, np.array([True, True])),
            (NTM.MODE_SATURATION, np.array([5, 10]), -10, 3, np.array([True, True])),
            (NTM.MODE_SATURATION, np.array([-12, 10]), -10, 3, np.array([False, True])),
        ],
    )
    def test_neuronal_fire(self, ntm, incoming_v, neg_thres, pos_thres, expected):
        mask = 3
        leak_v = 2

        n1 = Neuron(
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
            keep_shape=True,
        )
        spike = n1._neuronal_fire(incoming_v)

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
    def test_neuronal_reset(self, ntm, thr_mode, reset_mode, expected):
        reset_v = 5
        neg_thres = -3
        pos_thres = 2
        incoming_v = 10

        n1 = Neuron(
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
            keep_shape=True,
        )

        # Set the threshold mode manually
        setattr(n1, "thres_mode", thr_mode)
        v_reset = n1._neuronal_reset(np.array((incoming_v,), dtype=np.int32))

        assert np.array_equal(v_reset, expected)

    @pytest.mark.parametrize(
        "incoming_v, expected_v, expected_spike",
        [
            (
                np.array([2**30], dtype=np.int32),
                np.array([2**30 - 2**30], dtype=np.int32),
                np.array([False], dtype=np.bool_),
            ),
            (
                np.array([-(2**31)], dtype=np.int32),
                np.array([0]),  # Reset
                # Exceeded the negative threshold but no spike
                np.array([False], dtype=np.bool_),
            ),
        ],
        ids=["positive overflow", "negative overflow"],
    )
    def test_vjt_overflow(self, incoming_v, expected_v, expected_spike):
        pb.FRONTEND_ENV["t"] = 0
        reset_v = 0
        neg_thres = -(1 << 29)
        pos_thres = 1 << 29

        n1 = Neuron(
            1,
            RM.MODE_NORMAL,
            reset_v,
            self.lc,
            self.mask,
            NTM.MODE_RESET,
            neg_thres,
            pos_thres,
            self.ld,
            self.lim,
            self.leak_v,
            self.sim,
            self.bt,
        )

        pb.FRONTEND_ENV["t"] += 1  # Only update when n1 starts working
        n1.update(incoming_v)

        assert np.array_equal(n1.voltage, expected_v)
        assert np.array_equal(n1.spike, expected_spike)


@pytest.mark.parametrize(
    "shape",
    [5, (12,), (20, 20), (1, 2, 3)],
    ids=["scalar", "ndim=1", "ndim=2", "ndim=3"],
)
def test_neuron_instance(shape):
    # keep_shape = True
    n1 = pb.TonicSpiking(shape, 5, keep_shape=True)

    assert n1.shape_in == as_shape(shape)
    assert n1.shape_out == as_shape(shape)
    assert len(n1) == shape2num(shape)

    # keep_shape = False
    n2 = pb.TonicSpiking(shape, 5)

    assert n2.shape_in == as_shape(shape)
    assert n2.shape_out == as_shape(shape)
    assert len(n2) == shape2num(shape)


def test_neuron_keep_shape():
    n1 = pb.TonicSpiking((4, 4), 5, keep_shape=True)
    n2 = pb.TonicSpiking((4, 4), 5, keep_shape=False)

    assert n1.spike.shape == (16,)
    assert n1.voltage.shape == (4, 4)
    assert n1.output.shape == (256, 16)
    assert n1.feature_map.shape == (4, 4)

    assert n2.spike.shape == (16,)
    assert n2.voltage.shape == (16,)
    assert n2.output.shape == (256, 16)
    assert n2.feature_map.shape == (16,)


def test_neuron_copy():
    # Deepcopy is the same
    n1 = pb.LIF(
        (4, 4),
        5,
        keep_shape=True,
        delay=1,
        tick_wait_start=0,
        tick_wait_end=3,
        unrolling_factor=4,
        name="n1",
    )
    n2 = copy(n1)

    n2.unrolling_factor = 2
    n2._tws = 10

    assert id(n1) != id(n2)
    assert isinstance(n2, Neuron)
    assert n1.name != n2.name
    assert n1.unrolling_factor != n2.unrolling_factor
    assert n1._tws != n2._tws
    assert id(n1.voltage) != id(n2.voltage)


class TestNeuron:
    def test_IF_hard_reset(self):
        n1 = pb.IF(1, 5, 2)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array(
            [[0], [0], [0], [1], [0], [1], [1], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[2], [1], [4], [2], [3], [2], [2], [0]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_IF_soft_reset(self):
        n1 = pb.IF(1, 5, None)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array(
            [[0], [0], [0], [1], [1], [0], [1], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[2], [1], [4], [4], [0], [2], [1], [-1]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_LIF_hard_reset(self):
        # hard reset + leak before comparison
        n1 = pb.LIF(shape=1, threshold=5, reset_v=2, leak_v=-1)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array(
            [[0], [0], [0], [1], [0], [0], [1], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[1], [-1], [1], [2], [2], [3], [2], [-1]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_LIF_soft_reset(self):
        n1 = pb.LIF(1, 5, reset_v=None, leak_v=-1)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array(
            [[0], [0], [0], [1], [0], [0], [0], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[1], [-1], [1], [0], [0], [1], [4], [1]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_LIF_with_bias(self):
        # Hard reset, bias. leak_v is ignored.
        n1 = pb.LIF(shape=1, threshold=6, reset_v=1, leak_v=10, bias=2)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)
        assert n1.leak_v == n1.bias == 2

        inp_data = np.array([1, 1, 0, 1, 0, 1], dtype=np.bool_)
        expected_spike = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.bool_)
        expected_vol = np.array([[3], [1], [3], [1], [3], [1]], dtype=np.int32)

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_TonicSpiking(self):
        n1 = pb.TonicSpiking(1, fire_step=3)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
        expected_spike = np.array(
            [[0], [0], [1], [0], [0], [0], [0], [1], [0], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[1], [2], [0], [1], [1], [2], [2], [0], [0], [1]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_PhasicSpiking(self):
        n1 = pb.PhasicSpiking(1, fire_step=3, neg_floor=-2)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
        expected_spike = np.array(
            [[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]], dtype=np.bool_
        )
        expected_vol = np.array(
            [[2], [4], [-3], [-2], [-2], [-2], [-2], [-2], [-2], [-2]], dtype=np.int32
        )

        for i in range(inp_data.size):
            net.inp1.input = inp_data[i]
            sim.run(1)

        assert np.array_equal(sim.data[net.pb_n_spike], expected_spike)
        assert np.array_equal(sim.data[net.pb_n_volage], expected_vol)

    def test_SpikingRelu(self):
        n1 = pb.SpikingRelu(1)
        net = Net_Test_Neuron_Behavior(n1)
        sim = pb.Simulator(net)

        inp_data = np.random.randint(0, 2, size=(20, 1), dtype=np.bool_)

        for t in range(inp_data.size):
            net.inp1.input = inp_data[t]
            sim.run(1)

        for t in range(20):
            assert np.array_equal(sim.data[net.pb_n_spike][t], inp_data[t])

    def test_sum_inputs_behavior(self, build_Net2):
        net = build_Net2
        sim = pb.Simulator(net)

        sim.run(10)

        print(sim.data[net.probe1])
        print(sim.data[net.probe2])

    def test_tick_attr_behavior(self, monkeypatch, build_Net3):
        net = build_Net3
        sim = pb.Simulator(net)

        # n1 works on T in [1, 1+5-1]
        # n2 works on T in [2, 2+6-1]
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

    @pytest.mark.skipif(
        not hasattr(pb, "Always1Neuron"),
        reason="'Always1Neuron' is not exported to paibox.",
    )
    def test_Always1Neuron_behavior(self) -> None:
        class Net(pb.Network):
            def __init__(self):
                super().__init__()
                self.inp1 = pb.InputProj(input=None, shape_out=(1,))
                self.n1 = pb.Always1Neuron(shape=(1,), tick_wait_start=1)
                self.s1 = pb.FullConn(
                    self.inp1, self.n1, weights=0, conn_type=pb.SynConnType.One2One
                )

                self.probe1 = pb.Probe(self.n1, "spike")

        net = Net()
        sim = pb.Simulator(net)

        for i in range(20):
            net.inp1.input = np.random.randint(0, 2, size=(1,), dtype=np.bool_)
            sim.run(1)

        assert np.array_equal(
            sim.data[net.probe1], 20 * [np.ones((1,), dtype=np.bool_)]
        )

    @pytest.mark.parametrize("n_window", [4, 6, 8, 9, 12, 16, 25, 32, 36, 49])
    def test_AvgPool_Neuron(self, n_window):
        """
        NOTE: This neuron is used in `functional.SpikingAvgPool2d` & its basic functions need   \
            to be verified here.
        """
        from paibox.utils import typical_round

        class Net(pb.Network):
            def __init__(self):
                super().__init__()
                self.inp1 = pb.InputProj(input=None, shape_out=(n_window,))

                self.n1 = Neuron(
                    shape=(1,),
                    leak_v=1 - typical_round(n_window / 2),
                    neg_threshold=0,
                    tick_wait_start=1,
                )
                self.s1 = pb.FullConn(self.inp1, self.n1)
                self.probe1 = pb.Probe(self.n1, "spike")

        net = Net()
        sim = pb.Simulator(net)

        # Generate upper triangular matrix where the number of 1's increases in sequence.
        inp = np.tril(np.ones((1 + n_window, n_window), dtype=np.bool_))

        for t in range(1 + n_window):
            net.inp1.input = inp[t]
            sim.run(1)

            expected = (t + 1) >= typical_round(n_window / 2)
            assert sim.data[net.probe1][t][0] == expected
