import numpy as np
import pytest
from paibox.neuron.base import *

_sim = SIM.MODE_DETERMINISTIC
_lim = LIM.MODE_DETERMINISTIC
_ld = LDM.MODE_FORWARD
_lc = LCM.LEAK_AFTER_COMP
_leak_v = 0
_pos_thres = 10
_neg_thres = 0
_mask = 0
_reset_v = 5
_ntm = NTM.MODE_SATURATION
_reset_mode = RM.MODE_NORMAL
_bt = 0


@pytest.mark.parametrize(
    "vjt_init, x, expected",
    [
        (0, np.array([[1, 0, 1], [0, -1, 1]]), np.array([2, 0])),
        (1, np.array([[1, 0], [-1, -2]]), np.array([2, -2])),
        (0, np.array([1, 2]), np.array([1, 2])),
        (0, np.array(2), np.array([2, 2])),
    ]
)
def test_neuronal_charge(vjt_init, x, expected):
    n1 = MetaNeuron(2,
                    _reset_mode,
                    _reset_v,
                    _lc,
                    _mask,
                    _ntm,
                    _neg_thres,
                    _pos_thres,
                    _ld,
                    _lim,
                    _leak_v,
                    _sim,
                    _bt,
                    vjt_init,
                    keep_shape=True, )
    n1._neuronal_charge(x)
    assert np.array_equal(n1._vjt, expected)


@pytest.mark.parametrize(
    "_lim, _ld, vjt, leak_v, expected",
    [
        (LIM.MODE_DETERMINISTIC, LDM.MODE_FORWARD, 1, 2, np.array([3, 3])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_FORWARD, 2, 2, np.array([3, 3])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_FORWARD, 2, 1, np.array([2, 2])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_FORWARD, 2, -2, np.array([1, 1])),
        (LIM.MODE_DETERMINISTIC, LDM.MODE_REVERSAL, 1, 2, np.array([3, 3])),
        (LIM.MODE_DETERMINISTIC, LDM.MODE_REVERSAL, -2, 2, np.array([-4, -4])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_REVERSAL, 2, 2, np.array([3, 3])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_REVERSAL, -1, 1, np.array([-1, -1])),
        (LIM.MODE_STOCHASTIC, LDM.MODE_REVERSAL, 2, 1, np.array([2, 2])),
    ],
    # ids="path_1, path_2, path_3,path_4,path_5,path_6,path_7,path_8,path_9"
)
def test__neuronal_leak(vjt, leak_v, _lim, _ld, expected):
    n1 = MetaNeuron(2,
                    _reset_mode,
                    _reset_v,
                    _lc,
                    _mask,
                    _ntm,
                    _neg_thres,
                    _pos_thres,
                    _ld,
                    _lim,
                    leak_v,
                    _sim,
                    _bt,
                    vjt_init=vjt,
                    keep_shape=True, )
    n1._neuronal_leak()
    assert np.array_equal(n1._vjt, expected)


@pytest.mark.parametrize(
    " ntm, vjt, neg_thres, pos_thres, expected",
    [
        (NTM.MODE_RESET, 10, 5, 3, np.array([True])),
        (NTM.MODE_SATURATION, 10, 10, 3, np.array([True])),
        (NTM.MODE_SATURATION, 5, 10, 3, np.array([False])),
        (NTM.MODE_SATURATION, -12, 10, 3, np.array([False])),
    ]
)
def test_neuronal_fire(vjt, neg_thres, pos_thres, ntm, expected):
    # mask=3
    n1 = MetaNeuron(1,
                    _reset_mode,
                    _reset_v,
                    _lc,
                    3,
                    ntm,
                    neg_thres,
                    pos_thres,
                    _ld,
                    _lim,
                    2,
                    _sim,
                    _bt,
                    vjt_init=vjt,
                    keep_shape=True, )
    n1._neuronal_fire()
    assert np.array_equal(n1._spike, expected)

@pytest.mark.parametrize(
    "ntm, thr_mode, reset_mode, expected",
    [
        (NTM.MODE_RESET, TM.EXCEED_POSITIVE, RM.MODE_NORMAL, np.array([5])),
        (NTM.MODE_RESET, TM.EXCEED_POSITIVE, RM.MODE_LINEAR, np.array([12])),
        (NTM.MODE_RESET, TM.EXCEED_POSITIVE, RM.MODE_NONRESET, np.array([10])),
        (NTM.MODE_RESET, TM.EXCEED_NEGATIVE, RM.MODE_NORMAL, np.array([-5])),
        (NTM.MODE_RESET, TM.EXCEED_NEGATIVE, RM.MODE_LINEAR, np.array([13])),
        (NTM.MODE_RESET, TM.EXCEED_NEGATIVE, RM.MODE_NONRESET, np.array([10])),
        (NTM.MODE_SATURATION, TM.EXCEED_NEGATIVE, RM.MODE_NONRESET, np.array([-3])),
    ],
)

def test_neuronal_reset(ntm, thr_mode,reset_mode, expected):
    n1 = MetaNeuron(1,
                    reset_mode,
                    5,
                    _lc,
                    _mask,
                    ntm,
                    3,
                    -2,
                    _ld,
                    _lim,
                    _leak_v,
                    _sim,
                    _bt,
                    vjt_init=10,
                    keep_shape=True, )
    n1._threshold_mode = thr_mode
    n1._neuronal_reset()
    assert np.array_equal(n1._vjt, expected)