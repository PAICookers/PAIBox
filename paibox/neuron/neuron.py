from typing import ClassVar, Dict, List

from ._neuron import MetaNeuron
from .ram_model import ParamsRAM
from .ram_types import LeakingComparisonMode as LCM
from .ram_types import LeakingDirectionMode as LDM
from .ram_types import LeakingIntegrationMode as LIM
from .ram_types import NegativeThresholdMode as NTM
from .ram_types import ResetMode as RM
from .ram_types import SynapticIntegrationMode as SIM


class Neuron(MetaNeuron):
    """Father class of wrapped neurons.

    The parameters are always legal. This is public for user to define.
    """

    neuron_num: ClassVar[int] = 1

    def __init__(
        self,
        weights: List[int],
        reset_mode: RM,
        reset_v: int,
        leaking_comparison: LCM,
        threshold_mask_bits: int,
        neg_thres_mode: NTM,
        neg_threshold: int,
        pos_threshold: int,
        leaking_direction: LDM,
        leaking_integration_mode: LIM,
        leak_v: int,
        synaptic_integration_mode: SIM,
        bit_truncate: int,
        vjt_init: int,
    ) -> None:
        super().__init__(
            weights,
            reset_mode,
            reset_v,
            leaking_comparison,
            threshold_mask_bits,
            neg_thres_mode,
            neg_threshold,
            pos_threshold,
            leaking_direction,
            leaking_integration_mode,
            leak_v,
            synaptic_integration_mode,
            bit_truncate,
            vjt_init,
        )

    # def export_params_model(self) -> ParamsRAM:
    #     model = ParamsRAM(
    #         tick_relative=self._tick_relative,
    #         addr_axon=self._addr_axon,
    #         addr_core_x=self._addr_core_x,
    #         addr_core_y=self._addr_core_y,
    #         addr_core_x_ex=self._addr_core_x_ex,
    #         addr_core_y_ex=self._addr_core_y_ex,
    #         addr_chip_x=self._addr_chip_x,
    #         addr_chip_y=self._addr_chip_y,
    #         reset_mode=self._reset_mode,
    #         reset_v=self._reset_v,
    #         leaking_comparison=self._leaking_comparison,
    #         threshold_mask_bits=self._threshold_mask_bits,
    #         neg_thres_mode=self._neg_thres_mode,
    #         neg_threshold=self._neg_threshold,
    #         pos_threshold=self._pos_threshold,
    #         leaking_direction=self._leaking_direction,
    #         leaking_integration_mode=self._leaking_integration_mode,
    #         leak_v=self._leak_v,
    #         synaptic_integration_mode=self._synaptic_integration_mode,
    #         bit_truncate=self._bit_truncate,
    #         vjt_init=self._vjt_init,
    #     )

    #     return model

    # def export_params_dict(self) -> Dict[str, int]:
    #     model = self.export_params_model()
    #     return model.model_dump(by_alias=True)


class TonicSpikingNeuron(Neuron):
    """Tonic spiking neuron

    ## Arguments:
        - fire_step: every `N` spike, the neuron will fire positively.
        - vjt_init: initial membrane potential. Default is 0.

    ## Description:
        The neuron receives `N` spikes and fires, then resets to 0.
        `N` stands for firing steps.

        NOTE: the weight is 1.
    """

    def __init__(
        self,
        fire_step: int,
        vjt_init: int = 0,
    ):
        _weights = [
            1,
        ]
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = 0
        _pos_thres = _weights[0] * fire_step
        _neg_thres = 0
        _mask = 0
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _weights,
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


class PhasicSpikingNeuron(Neuron):
    """Phasic spiking neuron

    ## Arguments:
        - time_to_fire: after `time_to_fire` spikes, the neuron will fire positively.
        - neg_floor: the negative floor that the neuron stays once firing. Default is 10 (unsigned).
        - vjt_init: initial membrane potential. Default is 0.

    ## Description:
        The neuron receives `N` spikes and fires,
        then resets the membrane potential to 0,
        and never fires again.

        `N` stands for `time_to_fire`.

        NOTE: the weight is 1.
    """

    def __init__(
        self,
        time_to_fire: int,
        neg_floor: int = 10,
        vjt_init: int = 0,
    ):
        _weights = [
            1,
        ]
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_REVERSAL
        _lc = LCM.LEAK_BEFORE_COMP
        _leak_v = 1
        _pos_thres = (_weights[0] + _leak_v) * time_to_fire
        _neg_thres = neg_floor
        _mask = 0
        _reset_v = -1 - _neg_thres
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _weights,
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


class Class1ExcitableNeuron(Neuron):
    """Class 1 excitable neuron

    ## Arguments:
        - fire_step: after `time_to_fire` spikes, the neuron will fire positively.

    ## Description:

    """

    def __init__(
        self,
        fire_step: int,
        vjt_init: int = 0,
    ):
        _weights = [
            1,
        ]
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_REVERSAL
        _lc = LCM.LEAK_BEFORE_COMP
        _leak_v = 0
        _pos_thres = _weights[0] * fire_step
        _neg_thres = 0
        _mask = 0
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _weights,
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


class SpikeLatencyNeuron(Neuron):
    """Neuron spiking with a lantency.

    ## Arguments:
        - fire_step: after `time_to_fire` spikes, the neuron will fire positively.

    ## Description:

    """

    def __init__(
        self,
        fire_step: int,
        vjt_init: int = 0,
    ):
        _weights = [
            1,
        ]
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_REVERSAL
        _lc = LCM.LEAK_BEFORE_COMP
        _leak_v = 1
        _pos_thres = 52
        _neg_thres = 0
        _mask = 0
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            _weights,
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
