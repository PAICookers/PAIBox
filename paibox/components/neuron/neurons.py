from typing import Optional

from paicorelib import LDM, NTM, RM

from paibox.types import Shape

from .base import Neuron

__all__ = ["IF", "LIF", "TonicSpiking", "PhasicSpiking", "Always1Neuron", "SpikingRelu"]


class IF(Neuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: Optional[int] = None,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """IF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: If not set, neurons will do soft reset after firing, v - threshold. If set,  \
                neurons will do hard reset after firing, v = reset_v.
            - delay: delay between neurons. Default is 1.
            - tick_wait_start: set the moodule to start at timestep `T`. 0 means not working.       \
                Default is 1.
            - tick_wait_end: set the module to turn off at time `T`. 0 means always working.        \
                Default is 0.
            - unrolling_factor: argument related to the backend. It represents the degree to which  \
                modules are expanded. The larger the value, the more cores required for deployment, \
                but the lower the latency & the higher the throughput. Default is 1.
            - keep_shape: whether to maintain shape in the simulation. Default is `False`.
            - name: name of the neuron. Optional.
        """
        if isinstance(reset_v, int):
            # Hard reset
            _reset_v = reset_v
            _rm = RM.MODE_NORMAL
        else:
            # Soft reset
            _reset_v = 0
            _rm = RM.MODE_LINEAR

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            pos_threshold=threshold,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class LIF(Neuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: Optional[int] = None,
        leak_v: int = 0,
        bias: Optional[int] = None,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """LIF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: If not set, neurons will do soft reset after firing, v - `threshold`. If set,  \
                neurons will do hard reset after firing, v = `reset_v`.
            - leak_v: the signed leak voltage will be added directly to the membrane potential.
                - If it is positive, the membrane potential will increase.
                - If is is negative, the membrane potential will decrease.
            - bias: if signed bias is given, it will be used as `leak_v` and neuron will leak before  \
                threshold comparison. `leak_v` will be ignored.
            - keep_shape: whether to maintain shape in the simulation. Default is `False`.
            - name: name of the neuron. Optional.
        """
        if isinstance(reset_v, int):
            # Hard reset
            _reset_v = reset_v
            _rm = RM.MODE_NORMAL
        else:
            # Soft reset
            _reset_v = 0
            _rm = RM.MODE_LINEAR

        if isinstance(bias, int):
            _leak_v = bias
        else:
            _leak_v = leak_v

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            pos_threshold=threshold,
            leak_v=_leak_v,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class TonicSpiking(Neuron):
    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Tonic spiking neuron.

        Args:
            - shape: shape of neurons.
            - fire_step: every `N` spike, the neuron will fire positively.

        NOTE: The neuron receives `N` spikes and fires, then it will reset to 0.
        """
        super().__init__(
            shape, pos_threshold=fire_step, keep_shape=keep_shape, name=name, **kwargs
        )


class PhasicSpiking(Neuron):
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
        """Phasic spiking neuron.

        Args:
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
    def __init__(
        self,
        shape: Shape,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """A neuron that always outputs 1 as long as it starts working.

        FIXME There must be a forward synapse connected to it, otherwise the backend will go wrong.
        """
        super().__init__(
            shape,
            reset_v=1,
            neg_thres_mode=NTM.MODE_SATURATION,
            pos_threshold=0,
            leak_v=(1 << 29) - 1,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class SpikingRelu(Neuron):
    def __init__(
        self,
        shape: Shape,
        *,
        keep_shape: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Spiking relu neuron. Act exactly the way you think."""
        super().__init__(shape, keep_shape=keep_shape, name=name, **kwargs)
