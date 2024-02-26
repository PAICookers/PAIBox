from typing import Optional

from paicorelib import LCM, LDM, LIM, NTM, RM, SIM

from paibox.types import Shape

from .base import Neuron


class IF(Neuron):
    """IF neuron"""

    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: int = 0,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape    : the shape of the neuron(s). It can be an integer, tuple or list.
            - Threshold: When the membrane potential exceeds the threshold, neurons will fire
            - reset_v  : Membrane potential after firing

        Description:
            IF neuron : intergration + firing
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _pos_thres = threshold
        _reset_v = reset_v
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL

        super().__init__(
            shape,
            _reset_mode,
            _reset_v,
            _lc,
            0,
            _ntm,
            0,
            _pos_thres,
            _ld,
            _lim,
            0,
            _sim,
            0,
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
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
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
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

        Description:
            LIF: leaky + intergration + firing
        """
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_AFTER_COMP
        _leak_v = leaky_v
        _pos_thres = threshold
        _reset_v = reset_v
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NORMAL

        super().__init__(
            shape,
            _reset_mode,
            _reset_v,
            _lc,
            0,
            _ntm,
            0,
            _pos_thres,
            _ld,
            _lim,
            _leak_v,
            _sim,
            0,
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
            keep_shape=keep_shape,
            name=name,
        )


class TonicSpiking(Neuron):
    """Tonic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - fire_step: every `N` spike, the neuron will fire positively.

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
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
            keep_shape=keep_shape,
            name=name,
        )


class PhasicSpiking(Neuron):
    """Phasic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        time_to_fire: int,
        neg_floor: int = -10,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - shape: the shape of the neuron(s). It can be an integer, tuple or list.
            - time_to_fire: after `time_to_fire` spikes, the neuron will fire positively.
            - neg_floor: the negative floor that the neuron stays once firing. Default is -10.

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
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
            keep_shape=keep_shape,
            name=name,
        )
