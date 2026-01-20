import json
from enum import Enum
from typing import Any, Literal

import numpy as np
import pytest
from paicorelib import (
    LCM,
    LDM,
    LIM,
    LUT_DTYPE,
    NTM,
    RM,
    SIM,
    CoreMode,
    OfflineNeuAttrs,
    OfflineNeuRegLim,
    OnlineNeuAttrs,
)
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox.components import OfflineNeuron
from paibox.components.neuron.base import bit_truncate
from paibox.components.neuron.utils import NeuFireState as TM
from paibox.exceptions import ShapeError
from paibox.types import (
    NEUOUT_SPIKE_DTYPE,
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    NeuOutType,
    VoltageType,
)
from paibox.utils import as_shape, shape2num


class NeuCfgJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, Enum):
            return o.value
        return super().default(o)


def test_NeuronParams_check():
    with pytest.raises(ValueError):
        _ = pb.LIF((100,), threshold=-1)

    with pytest.raises(ValueError):
        _ = pb.IF((100,), 1, delay=-1)

    with pytest.raises(ValueError):
        _ = pb.IF((100,), 1, delay=1, tick_wait_start=-1, tick_wait_end=100)

    with pytest.raises(ShapeError):
        _ = pb.LIF((10, 20), 1, bias=np.ones((100,)))

    # If CoreMode specifies all configurations, there will be no invalid situations.
    if len(CoreMode) < 8:
        with pytest.raises(ValueError):
            _ = pb.LIF((100,), 10, input_width=8, spike_width=8, snn_en=True)


def test_neuron_keep_shape():
    n1 = pb.TonicSpiking((4, 4), 5, keep_shape=True)
    n2 = pb.TonicSpiking((4, 4), 5, keep_shape=False)

    assert n1.spike.shape == (16,)
    assert n1.spike.shape == n1.output.shape
    assert n1.voltage.shape == (4, 4)
    assert n1.feature_map.shape == (4, 4)

    assert n2.spike.shape == (16,)
    assert n2.voltage.shape == (16,)
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
    n1_copy = n1.copy()

    n1_copy.unrolling_factor = 2
    n1_copy._tws = 10

    assert id(n1) != id(n1_copy)
    assert isinstance(n1_copy, OfflineNeuron)
    assert n1.name != n1_copy.name
    assert n1.unrolling_factor != n1_copy.unrolling_factor
    assert n1._tws != n1_copy._tws
    assert id(n1.voltage) != id(n1_copy.voltage)

    n2 = pb.STDPLIF(
        (4, 4), 10, -1, -2, 0, neg_threshold=-10, lateral_inhi_value=-1, init_v=3
    )
    n2_copy = n2.copy()

    assert id(n2) != id(n2_copy)
    assert n2.shape_out == n2_copy.shape_out
    assert n2.neg_threshold == n2_copy.neg_threshold


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


def _reg_kwds(
    iw: Literal[1, 8], sw: Literal[1, 8], snn_en: Literal[0, 1]
) -> dict[str, Any]:
    return {"input_width": iw, "spike_width": sw, "snn_en": bool(snn_en)}


_reg000_kwds = _reg_kwds(1, 1, 0)
_reg001_kwds = _reg_kwds(1, 1, 1)
_reg010_kwds = _reg_kwds(1, 8, 0)
_reg011_kwds = _reg_kwds(1, 8, 1)
_reg100_kwds = _reg_kwds(8, 1, 0)
_reg110_kwds = _reg_kwds(8, 8, 0)
_bann_kwds = _reg000_kwds
_ann_kwds = _reg110_kwds
_snn_kwds = _reg001_kwds


def ann_bit_trunc(vj: VoltageType, neu: OfflineNeuron) -> NeuOutType:
    return np.where(vj >= neu.pos_threshold, bit_truncate(vj, neu.bit_trunc), 0).astype(
        NEUOUT_U8_DTYPE
    )


class TestOfflineNeuron:
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
        n1 = OfflineNeuron(
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
            **_snn_kwds,
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
        n1 = OfflineNeuron(
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
            **_snn_kwds,
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

        n1 = OfflineNeuron(
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
            **_snn_kwds,
        )
        spike = n1._neuronal_fire(incoming_v)

        assert np.array_equal(spike, expected)

    @pytest.mark.parametrize(
        "ntm, thr_mode, reset_mode, expected",
        [
            (NTM.MODE_RESET, TM.FIRING_POS, RM.MODE_NORMAL, np.array([5])),
            (NTM.MODE_RESET, TM.FIRING_POS, RM.MODE_NONRESET, np.array([10])),
            (NTM.MODE_RESET, TM.FIRING_NEG, RM.MODE_NORMAL, np.array([-5])),
            (NTM.MODE_RESET, TM.FIRING_NEG, RM.MODE_NONRESET, np.array([10])),
            (NTM.MODE_SATURATION, TM.FIRING_NEG, RM.MODE_NONRESET, np.array([-3])),
        ],
    )
    def test_neuronal_reset(self, ntm, thr_mode, reset_mode, expected):
        reset_v = 5
        neg_thres = -3
        pos_thres = 2
        incoming_v = 10

        n1 = OfflineNeuron(
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
            **_snn_kwds,
        )

        # Set the threshold mode manually
        setattr(n1, "thres_mode", thr_mode)
        v_reset = n1._neuronal_reset(np.array((incoming_v,), dtype=VOLTAGE_DTYPE))

        assert np.array_equal(v_reset, expected)

    @pytest.mark.parametrize(
        "incoming_v, expected_v, expected_spike",
        [
            (
                np.array([OfflineNeuRegLim.VOLTAGE_MAX + 1], dtype=VOLTAGE_DTYPE),
                np.array([OfflineNeuRegLim.VOLTAGE_MIN + 1], dtype=VOLTAGE_DTYPE),
                # Exceeded the positive threshold but no spike
                np.array([False], dtype=bool),
            ),
            (
                np.array([OfflineNeuRegLim.VOLTAGE_MIN - 1], dtype=VOLTAGE_DTYPE),
                np.array([OfflineNeuRegLim.VOLTAGE_MAX - 1], dtype=VOLTAGE_DTYPE),
                # Exceeded the negative threshold but no spike
                np.array([False], dtype=bool),
            ),
        ],
        ids=["positive overflow", "negative overflow"],
    )
    def test_vjt_overflow(self, incoming_v, expected_v, expected_spike):
        pb.FRONTEND_ENV["t"] = 0
        neg_thres = OfflineNeuRegLim.VOLTAGE_MIN
        pos_thres = OfflineNeuRegLim.VOLTAGE_MAX

        n1 = OfflineNeuron(
            1,
            RM.MODE_NORMAL,
            0,
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
            **_snn_kwds,
        )

        pb.FRONTEND_ENV["t"] += 1  # Only update when n1 starts working
        n1.update(incoming_v)

        assert np.array_equal(n1.voltage, expected_v)
        assert np.array_equal(n1.spike, expected_spike)

    def test_IF_hard_reset(self):
        n1 = pb.IF(1, 5, 2)

        incoming_v = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array([[0], [0], [0], [1], [0], [1], [1], [0]], dtype=bool)
        expected_vol = np.array(
            [[2], [1], [4], [2], [3], [2], [2], [0]], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_IF_soft_reset(self):
        n1 = pb.IF(1, 5, None)

        incoming_v = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array([[0], [0], [0], [1], [1], [0], [1], [0]], dtype=bool)
        expected_vol = np.array(
            [[2], [1], [4], [4], [0], [2], [1], [-1]], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_LIF_hard_reset(self):
        # hard reset + leak before comparison
        n1 = pb.LIF(shape=1, threshold=5, reset_v=2, leak_v=-1)

        incoming_v = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array([[0], [0], [0], [1], [0], [0], [1], [0]], dtype=bool)
        expected_vol = np.array(
            [[1], [-1], [1], [2], [2], [3], [2], [-1]], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_LIF_soft_reset(self):
        n1 = pb.LIF(1, 5, reset_v=None, leak_v=-1)

        incoming_v = np.array([2, -1, 3, 5, 1, 2, 4, -2], dtype=np.int8)
        expected_spike = np.array([[0], [0], [0], [1], [0], [0], [0], [0]], dtype=bool)
        expected_vol = np.array(
            [[1], [-1], [1], [0], [0], [1], [4], [1]], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_LIF_with_bias(self):
        # Hard reset, bias, scalar.
        n1 = pb.LIF(shape=1, threshold=6, reset_v=1, leak_v=0, bias=2)
        assert n1.leak_v == n1.bias == 2

        incoming_v = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        expected_spike = np.array([[0], [1], [0], [1], [0], [1]], dtype=bool)
        expected_vol = np.array([[3], [1], [3], [1], [3], [1]], dtype=VOLTAGE_DTYPE)

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_LIF_with_bias_vector(self):
        # Soft reset, bias.
        n1 = pb.LIF(
            shape=(3, 2),
            threshold=6,
            reset_v=0,
            bias=np.array([1, 2, 2], dtype=VOLTAGE_DTYPE),
        )

        incoming_v = np.array([[[0, 0], [1, 1], [0, 0]]], dtype=bool)
        expected_vol = np.array([[[3, 3], [3, 3], [0, 0]]], dtype=VOLTAGE_DTYPE)

        for _ in range(3):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[0].ravel())

        assert np.array_equal(n1.voltage, expected_vol[0])

    def test_LIF_both_leak_bias(self):
        # Soft reset, leak & bias.
        n1 = pb.LIF(shape=1, threshold=6, leak_v=-1, bias=2)
        assert n1.leak_v == n1.bias == 1

        incoming_v = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        expected_spike = np.array([[0], [0], [0], [1], [0], [0]], dtype=bool)
        expected_vol = np.array([[2], [4], [5], [1], [2], [4]], dtype=VOLTAGE_DTYPE)

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_TonicSpiking(self):
        n1 = pb.TonicSpiking(1, fire_step=3)

        incoming_v = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        expected_spike = np.array(
            [[0], [0], [1], [0], [0], [0], [0], [1], [0], [0]], dtype=bool
        )
        expected_vol = np.array(
            [[1], [2], [0], [1], [1], [2], [2], [0], [0], [1]], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_PhasicSpiking(self):
        n1 = pb.PhasicSpiking(1, fire_step=3, neg_floor=-2)

        incoming_v = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        expected_spike = np.array(
            [[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]], dtype=bool
        )
        expected_vol = np.array(
            [[2], [4], [-3], [-2], [-2], [-2], [-2], [-2], [-2], [-2]],
            dtype=VOLTAGE_DTYPE,
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, expected_spike[i])
            assert np.array_equal(n1.voltage, expected_vol[i])

    def test_BypassNeuron(self):
        n1 = pb.BypassNeuron(1, **_snn_kwds)

        incoming_v = np.random.randint(0, 2, size=(20, 1), dtype=bool)

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike, incoming_v[i])

    def test_sum_inputs_behavior(self, build_Net2):
        net = build_Net2
        sim = pb.Simulator(net)

        _always_spike = np.full((net.n1.num_out,), 1, dtype=bool)

        for i in range(10):
            sim.run(1)
            assert np.array_equal(sim.data[net.probe2][i], _always_spike)

    def test_max_inputs_behavior(self):
        """Only check the voltage result after the `sum_inputs` of neuron."""
        incoming_v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=VOLTAGE_DTYPE)
        incoming_v2 = np.array([-1, 7, -3, 8, -5, -6, 1, 2], dtype=VOLTAGE_DTYPE)
        incoming_v3 = np.array([2, 3, 1, -8, 0, 8, 4, 7], dtype=VOLTAGE_DTYPE)
        incoming_v = [incoming_v1, incoming_v2, incoming_v3]

        v_poolmax = np.zeros_like(incoming_v1)
        for v in incoming_v:
            if v_poolmax is None:
                v_poolmax = v.copy()
            else:
                v_poolmax = np.maximum(v_poolmax, v)

        assert v_poolmax.shape == incoming_v1.shape
        assert np.array_equal(v_poolmax, np.array([2, 7, 3, 8, 5, 8, 7, 8]))

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
        n1 = pb.Always1Neuron((1,))  # type: ignore

        for i in range(10):
            pb.FRONTEND_ENV["t"] += 1
            n1.update()

            assert np.array_equal(n1.spike, np.ones((1,), dtype=NEUOUT_SPIKE_DTYPE))

    @pytest.mark.parametrize("n_window", [4, 6, 8, 9, 12, 16, 25, 32, 36, 49])
    def test_AvgPool_Neuron(self, n_window):
        # This neuron is used in `functional.SpikingAvgPool2d`.
        from paibox.utils import typical_round

        n1 = OfflineNeuron(
            shape=(1,), leak_v=1 - typical_round(n_window / 2), neg_threshold=0
        )

        # Generate upper triangular matrix where the number of 1's increases in sequence.
        incoming_v = np.tril(
            np.ones((1 + n_window, n_window), dtype=NEUOUT_SPIKE_DTYPE)
        )

        for i in range(1 + n_window):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(np.sum(incoming_v[i]))

            expected = (i + 1) >= typical_round(n_window / 2)
            assert np.array_equal(n1.spike[0], expected)

    def test_ANNNeuron(self):
        n1 = pb.ANNNeuron(1, 0, bit_trunc=8)

        incoming_v = np.random.randint(-128, 128, size=(20, 1), dtype=VOLTAGE_DTYPE)

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(
                n1.spike,
                (
                    np.array([0], dtype=NEUOUT_U8_DTYPE)
                    if incoming_v[i] < 0
                    else incoming_v[i]
                ),
            )

    @pytest.mark.parametrize(
        "bit_trunc, expected_v",
        [
            (8, np.array([10, 255, 255, 90, 110 & 255, 255, 0, 0], dtype=np.uint8)),
            (
                9,
                np.array(
                    [
                        (10 >> 1) & 255,
                        (390 >> 1) & 255,
                        255,
                        (90 >> 1) & 255,
                        (110 >> 1) & 255,
                        (468 >> 1) & 255,
                        0,
                        0,
                    ],
                    dtype=np.uint8,
                ),
            ),
        ],
        ids=["8_bit", "9_bit"],
    )
    def test_ANNNeuron_bit_trunc(self, bit_trunc, expected_v):
        n1 = pb.ANNNeuron(1, -10, bit_trunc=bit_trunc)

        incoming_v = np.array(
            [20, 400, 1000, 100, 120, 478, 0, -10], dtype=VOLTAGE_DTYPE
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            assert np.array_equal(n1.spike[0], expected_v[i])

    @pytest.mark.parametrize(
        "reg_kwds", [_reg010_kwds, _reg110_kwds], ids=["010", "ann"]
    )
    def test_IF_ss10(self, reg_kwds):
        n1 = pb.IF(1, 0, 0, bit_trunc=8, **reg_kwds)

        incoming_v = np.random.randint(
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max,
            size=(8,),
            dtype=VOLTAGE_DTYPE,
        )

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])
            v_bt = ann_bit_trunc(np.atleast_1d(incoming_v[i]), n1)

            assert np.array_equal(n1.spike, v_bt)

    def test_LIF_ss11(self):
        pos_thres = 8000
        n1 = pb.LIF(1, pos_thres, bit_trunc=12, **_reg011_kwds)

        incoming_v = np.random.randint(-10000, 10000, size=(20,), dtype=VOLTAGE_DTYPE)
        pre_vjt = 0

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            pre_vjt += incoming_v[i]
            spike = pre_vjt >= pos_thres

            v_bt = ann_bit_trunc(np.atleast_1d(pre_vjt), n1)

            if spike:
                pre_vjt -= pos_thres

            assert np.array_equal(n1.spike, v_bt)

    @pytest.mark.parametrize("reg_kwds", [_reg000_kwds, _reg100_kwds])
    def test_LIF_ss00(self, reg_kwds):
        pos_thres = 8000
        n1 = pb.LIF(1, pos_thres, reset_v=2000, bit_trunc=10, **reg_kwds)

        incoming_v = np.random.randint(-10000, 10000, size=(20,), dtype=VOLTAGE_DTYPE)
        pre_vjt = 0

        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            pre_vjt = incoming_v[i]
            spike = pre_vjt >= pos_thres

            if spike:
                pre_vjt = 2000

            assert np.array_equal(n1.spike[0], spike)

    def test_attrs_export(self, ensure_dump_dir):
        n1 = pb.LIF((100,), 3, reset_v=-20, leak_v=-2)

        attrs = OfflineNeuAttrs.model_validate(n1.attrs(for_copy=True), strict=True)
        attrs_dict = attrs.model_dump(by_alias=True)

        fp = ensure_dump_dir / f"ram_model_{n1.name}.json"
        with open(fp, "w") as f:
            json.dump({n1.name: attrs_dict}, f, indent=2)

        # leak_v is an array
        n2 = pb.LIF((4, 4, 4), 3, reset_v=-20, leak_v=-2, bias=np.arange(4))

        attrs = OfflineNeuAttrs.model_validate(
            n2._slice_attrs(slice(2 * 4 * 4 - 10, 3 * 4 * 4 + 2, 1)),
            strict=True,
        )
        attrs_dict = attrs.model_dump(by_alias=True)

        fp2 = ensure_dump_dir / f"ram_model_{n2.name}.json"
        with open(fp2, "w") as f:
            json.dump({n2.name: attrs_dict}, f, indent=2, cls=NeuCfgJsonEncoder)


class TestOnlineNeuron:
    def test_neuron_lateral_inhi(self):
        n1 = pb.STDPLIF(
            (4, 4),
            10,
            leak_v=-1,
            neg_threshold=-10,
            lateral_inhi_value=-2,
            init_v=np.ones((4, 4), dtype=VOLTAGE_DTYPE),
        )

        incoming_v = np.full(n1.num_out, 3, dtype=VOLTAGE_DTYPE)
        for i in range(4):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v)

            if n1.has_spike():
                assert i == 3  # First spike at 4

    def test_neuron_lateral_inhi_multi_layers(self):
        # 3 layers. n1 inhibits n2 and n2 inhibits n1 & n3
        class InhiNetwork(pb.Network):
            def __init__(self):
                super().__init__()
                self.input1 = pb.InputProj(input=None, shape_out=(50,))
                self.n1 = pb.STDPLIF(
                    50,
                    1,
                    0,
                    -1,
                    0,
                    neg_threshold=-10,
                    lateral_inhi_value=-2,
                    tick_wait_start=1,
                )
                self.n3 = pb.STDPLIF(
                    10, 1, lateral_inhi_value=-3, init_v=1, tick_wait_start=3
                )
                self.n2 = pb.STDPLIF(
                    36,
                    1,
                    -3,
                    -1,
                    lateral_inhi_value=-1,
                    lateral_inhi_target=[self.n1, self.n3],
                    tick_wait_start=2,
                )
                self.n1.set_lateral_inhi_target(self.n2)

                self.s1 = pb.STDPFullConn(
                    self.input1,
                    self.n1,
                    np.ones((self.input1.num_out, self.n1.num_in), dtype=np.int8),
                )
                self.s2 = pb.STDPFullConn(
                    self.n1,
                    self.n2,
                    np.ones((self.n1.num_out, self.n2.num_in), dtype=np.int8),
                )
                self.s3 = pb.STDPFullConn(
                    self.n2,
                    self.n3,
                    np.ones((self.n2.num_out, self.n3.num_in), dtype=np.int8),
                )

        net = InhiNetwork()
        sim = pb.Simulator(net)

        net.input1.input = np.ones((50,), dtype=VOLTAGE_DTYPE)

        while 1:
            sim.run(1)
            if net.n2.has_spike() > 0:
                sim.run(1)
                assert net.n1.need_lateral_inhi
                assert net.n3.need_lateral_inhi
                break

    def test_attrs_export(self, ensure_dump_dir):
        n1 = pb.STDPLIF(
            (100,), 3, reset_v=0, leak_v=-2, init_v=np.arange(100, dtype=VOLTAGE_DTYPE)
        )

        attrs = OnlineNeuAttrs.model_validate(
            n1.attrs(for_copy=True),
            strict=True,
            context={"weight_width": WW.WEIGHT_WIDTH_8BIT},
        )
        attrs_dict = attrs.model_dump(by_alias=True)

        fp = ensure_dump_dir / f"ram_model_{n1.name}.json"
        with open(fp, "w") as f:
            json.dump({n1.name: attrs_dict}, f, indent=2, cls=NeuCfgJsonEncoder)

        # leak_v is an array
        n2 = pb.STDPLIF((4, 4, 4), 3, reset_v=0, leak_v=-2, bias=np.arange(4), init_v=1)

        attrs = OnlineNeuAttrs.model_validate(
            n2._slice_attrs(slice(2 * 4 * 4 - 10, 3 * 4 * 4 + 2, 1)),
            context={"weight_width": WW.WEIGHT_WIDTH_8BIT},
            strict=True,
        )
        attrs_dict = attrs.model_dump(by_alias=True)

        fp2 = ensure_dump_dir / f"ram_model_{n2.name}.json"
        with open(fp2, "w") as f:
            json.dump({n2.name: attrs_dict}, f, indent=2, cls=NeuCfgJsonEncoder)

    def test_attrs_stdp_syn_export(self):
        n1 = pb.IF(100)
        n2 = pb.STDPLIF(
            (100,), 3, leak_v=-2, init_v=np.arange(100, dtype=VOLTAGE_DTYPE)
        )
        lut = np.zeros((60,), dtype=LUT_DTYPE)
        lut[:30] = -1
        lut[30:] = 2
        s1 = pb.STDPFullConn(
            n1,
            n2,
            np.ones((n1.num_out, n2.num_in), dtype=np.int8),
            upper_weight=99,
            lower_weight=-100,
            weight_decay=-1,
            lut=lut,
            random_seed=2,
        )

        attrs = n2.attrs()
        assert attrs["weight_decay_value"] == -1
        assert attrs["upper_weight"] == 99

        # Check `n2._set_syn_attrs`
        n3 = pb.IF(100)
        with pytest.raises(ValueError):
            _ = pb.STDPFullConn(
                n3, n2, np.ones((n1.num_out, n2.num_in), dtype=np.int8), lut=lut
            )

        # Set an invalid attribute to n2
        with pytest.raises(ValueError):
            n2._set_syn_attrs(weight_decay=-1)  # type: ignore


class TestSpecialTypeNeuron:
    @pytest.mark.parametrize("leak_v", [0, 10, -10])
    def test_StoreVoltageNeuron(self, leak_v):
        n1 = pb.StoreVoltageNeuron(1, leak_v=leak_v)
        incoming_v = np.random.randint(-100, 100, size=(100,), dtype=VOLTAGE_DTYPE)

        expected_v = 0
        for i in range(incoming_v.size):
            pb.FRONTEND_ENV["t"] += 1
            n1.update(incoming_v[i])

            expected_v += incoming_v[i] + leak_v
            assert np.array_equal(n1.voltage[0], expected_v)
            assert n1.spike.all() == 0  # not spiking to effect the output receiving
