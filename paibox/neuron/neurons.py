from typing import Optional

from paibox._types import Shape
from paibox.libpaicore import LCM, LDM, LIM, NTM, RM, SIM

from .base import Neuron


class IF(Neuron):
    """IF neuron"""

    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: int = 0,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape    : the shape of the neuron(s). It can be an integer, tuple or list.
            - Threshold: When the membrane potential exceeds the threshold, neurons will fire
            - reset_v  : Membrane potential after firing
            - vjt_init : initial membrane potential. Default is 0.

        Description:
            IF neuron : intergration + firing
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = 0
        _pos_thres = threshold
        _neg_thres = 0
        _mask = 0
        _reset_v = reset_v
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            shape,
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
            keep_shape=keep_shape,
            name=name,
        )


class LIF(Neuron):
    """LIF neuron"""

    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: int = 0,
        leaky_v: int = 0,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - threshold: When the membrane potential exceeds the threshold, neurons will fire
            - reset_v: Membrane potential after firing
            - leaky_v: The leakage value will be directly added to the membrane potential.
                If it is positive, the membrane potential will increase.
                If is is negative, the membrane potential will decrease.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            LIF: leaky + intergration + firing
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = leaky_v
        _pos_thres = threshold
        _neg_thres = 0
        _mask = 0
        _reset_v = reset_v
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL
        _bt = 0

        super().__init__(
            shape,
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
            keep_shape=keep_shape,
            name=name,
        )


class TonicSpiking(Neuron):
    """Tonic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - fire_step: every `N` spike, the neuron will fire positively.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            The neuron receives `N` spikes and fires, then resets to 0.
            `N` stands for firing steps.
        """
        super().__init__(
            shape,
            RM.MODE_NORMAL,
            0,
            LCM.LEAK_AFTER_COMP,
            0,
            NTM.MODE_SATURATION,
            0,
            fire_step,
            LDM.MODE_FORWARD,
            LIM.MODE_DETERMINISTIC,
            0,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class PhasicSpiking(Neuron):
    """Phasic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        time_to_fire: int,
        neg_floor: int = 10,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - time_to_fire: after `time_to_fire` spikes, the neuron will fire positively.
            - neg_floor: the negative floor that the neuron stays once firing. Default is 10 (unsigned).
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            The neuron receives `N` spikes and fires, then resets the membrane potential to 0,
            and never fires again.

            `N` stands for `time_to_fire`.
        """
        leak_v = 1
        pos_thres = (1 + leak_v) * time_to_fire
        _neg_thres = neg_floor
        reset_v = -1 - _neg_thres

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            reset_v,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            neg_floor,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            leak_v,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class SpikeLatency(Neuron):
    """Spike latency neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_time: int,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - fire_time: When receiving a spike, the neuron will fire positively \
                after `fire_time` timesteps,
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            The neuron receives a spike and fires after some timesteps, \
            then resets the membrane potential to 0, and never fires again.

            `N` stands for `fire_time`.

        NOTE: the weight is 10.
        """
        pos_thres = 11 + fire_time

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            0,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            0,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            1,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class SubthresholdOscillations(Neuron):
    """Subthreshold Oscillations"""

    def __init__(
        self,
        shape: Shape,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            After receiving a spike, neurons emit pulses and the membrane potential oscillates

        NOTE: the weight is 22.
        """
        leak_v = -2
        pos_thres = 16
        neg_thres = 30

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            1,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            neg_thres,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            leak_v,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class ResonatorNeuron(Neuron):
    """Resonator Neuron"""

    def __init__(
        self,
        shape: Shape,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            After being stimulated, neurons emit pulses and the membrane potential oscillates
            Implementation question: Continuous inputs with higher frequencies will cause neurons
                to emit directly. How can we achieve the distribution of determined frequencies?

        NOTE: the weight is 2.
        """
        leak_v = -1
        pos_thres = 2
        reset_v = 0

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            reset_v,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            0,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            leak_v,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class Integrator(Neuron):
    """Integrator Neuron"""

    def __init__(
        self,
        shape: Shape,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            After being stimulated, neurons emit pulses and the membrane potential oscillates.

        NOTE: the weight is 24.
        """
        leak_v = -1
        pos_thres = 32

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            0,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            0,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            leak_v,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )


class InhibitionInducedSpiking(Neuron):
    """Inhibition Induced Spiking Neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        vjt_init: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - fire_step: every `N` spike, the neuron will fire positively.
            - vjt_init: initial membrane potential. Default is 0.

        Description:
            After receiving some inhibition induced spikes (-1), the neuron will fire.

        NOTE: the weight is -10.
        """
        leak_v = -1
        pos_thres = 9 * fire_step
        neg_thres = 40
        reset_v = -10

        super().__init__(
            shape,
            RM.MODE_NORMAL,
            reset_v,
            LCM.LEAK_BEFORE_COMP,
            0,
            NTM.MODE_SATURATION,
            neg_thres,
            pos_thres,
            LDM.MODE_REVERSAL,
            LIM.MODE_DETERMINISTIC,
            leak_v,
            SIM.MODE_DETERMINISTIC,
            0,
            vjt_init,
            keep_shape=keep_shape,
            name=name,
        )
