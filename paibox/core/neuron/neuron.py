from ._neuron import Neuron
from .ram_types import (
    ResetMode as RM,
    LeakingComparisonMode as LCM,
    NegativeThresholdMode as NTM,
    LeakingDirectionMode as LDM,
    LeakingIntegrationMode as LIM,
    SynapticIntegrationMode as SIM,
)


class BasicFireNeuron(Neuron):
    def __init__(
        self,
        fire_rate: int,
        vjt_init: int = 0,
        *,
        tick_relative: int,
        addr_axon: int,
        addr_core_x: int,
        addr_core_y: int,
        addr_core_x_ex: int,
        addr_core_y_ex: int,
        addr_chip_x: int,
        addr_chip_y: int,
        chip_x: int,
        chip_y: int,
        core_x: int,
        core_y: int,
        nid: int,
    ):
        _weight = 1
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = 0
        _pos_thres = fire_rate
        _neg_thres = 0
        _mask = 0
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            tick_relative,
            addr_axon,
            addr_core_x,
            addr_core_y,
            addr_core_x_ex,
            addr_core_y_ex,
            addr_chip_x,
            addr_chip_y,
            chip_x,
            chip_y,
            core_x,
            core_y,
            nid,
            _weight,
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
        )


class PeriodFireNeuron(Neuron):
    def __init__(
        self,
        time_to_fire: int,
        neg_floor: int = 10,
        vjt_init: int = 0,
        *,
        tick_relative: int,
        addr_axon: int,
        addr_core_x: int,
        addr_core_y: int,
        addr_core_x_ex: int,
        addr_core_y_ex: int,
        addr_chip_x: int,
        addr_chip_y: int,
        chip_x: int,
        chip_y: int,
        core_x: int,
        core_y: int,
        nid: int,
    ):
        _weight = 1
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_REVERSAL
        _lc = LCM.LEAK_BEFORE_COMP
        _leak_v = 1
        _pos_thres = (_weight + _leak_v) * time_to_fire
        _neg_thres = neg_floor
        _mask = 0
        _reset_v = -1 - _neg_thres
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            tick_relative,
            addr_axon,
            addr_core_x,
            addr_core_y,
            addr_core_x_ex,
            addr_core_y_ex,
            addr_chip_x,
            addr_chip_y,
            chip_x,
            chip_y,
            core_x,
            core_y,
            nid,
            _weight,
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
        )
