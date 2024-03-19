from typing import Optional

from paicorelib import LCM, LDM, NTM, RM

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
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: reset membrane potential after firing
            - delay: delay between neurons. Default is 1.
            - tick_wait_start: set the neuron group to start at the `N`-th timestep. 0 means not to     \
                start. Default is 1.
            - tick_wait_end: set the neuron group to continue working for `M` timesteps, 0 means working\
                forever. Default is 0.
            - unrolling_factor: the argument is related to the backend. It means that neurons will be   \
                unrolled & deployed to more physical cores to reduce latency and increase throughput.   \
                Default is 1.
            - keep_shape: whether to maintain size information when recording data in the simulation.   \
                Default is `False`.
            - name: name of the object.
        """
        super().__init__(
            shape,
            reset_v=reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            pos_threshold=threshold,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class LIF(Neuron):
    """LIF neuron"""

    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: int = 0,
        leak_v: int = 0,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: reset membrane potential after firing
            - leak_v: the signed leak voltage will be added directly to the membrane potential.
                - If it is positive, the membrane potential will increase.
                - If is is negative, the membrane potential will decrease.
        """
        super().__init__(
            shape,
            reset_mode=RM.MODE_NORMAL,
            reset_v=reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            pos_threshold=threshold,
            leak_v=leak_v,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class TonicSpiking(Neuron):
    """Tonic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - shape: shape of neurons.
            - fire_step: every `N` spike, the neuron will fire positively.

        NOTE: The neuron receives `N` spikes and fires, then it will reset to 0.
        """
        super().__init__(
            shape,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            pos_threshold=fire_step,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class PhasicSpiking(Neuron):
    """Phasic spiking neuron"""

    def __init__(
        self,
        shape: Shape,
        time_to_fire: int,
        neg_floor: int = -10,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            - shape: shape of neurons.
            - time_to_fire: after `N` spikes, the neuron will fire positively.
            - neg_floor: once fired, the neurons will remain at this negative membrane potential.   \
                Default is -10.

        NOTE: Once the neuron receives `N` spikes and fires, it will reset to the negative floor &  \
            never fires again. `N` stands for `time_to_fire`.
        """
        leak_v = 1
        super().__init__(
            shape,
            reset_v=(-1 - neg_floor),
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=neg_floor,
            pos_threshold=(1 + leak_v) * time_to_fire,
            leak_direction=LDM.MODE_REVERSAL,
            leak_v=leak_v,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class Always1Neuron(Neuron):
    """This neuron will always output 1 as long as it starts working.

    FIXME There must be a forward synapse connected to it, otherwise    \
        the backend will go wrong.
    """

    def __init__(
        self,
        shape: Shape,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape,
            reset_v=1,
            leak_comparison=LCM.LEAK_BEFORE_COMP,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            pos_threshold=0,
            leak_v=(1 << 29) - 1,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )
