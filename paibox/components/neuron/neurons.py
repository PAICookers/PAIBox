from typing import Optional

import numpy as np
from paicorelib import LDM, NTM, RM

from paibox.types import DataArrayType, Shape

from .base import Neuron
from .utils import LEAK_V_MAX, NEG_THRES_MIN

__all__ = ["IF", "LIF", "TonicSpiking", "PhasicSpiking", "SpikingRelu"]


class IF(Neuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int,
        reset_v: Optional[int] = None,
        neg_threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """IF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: If not specified, neurons will do soft reset after firing, v - threshold. If \
                specified, neurons will do hard reset after firing, v = reset_v.
            - neg_threshold: signed negative theshold. If not specified, it will be the smallest    \
                negative integer allowed by the hardware.
            - delay: delay between neurons. Default is 1.
            - tick_wait_start: set the moodule to start at timestep `T`. 0 means not working.       \
                Default is 1.
            - tick_wait_end: set the module to turn off at time `T`. 0 means always working.        \
                Default is 0.
            - unrolling_factor: argument related to the backend. It represents the degree to which  \
                modules are expanded. The larger the value, the more cores required for deployment, \
                but the lower the latency & the higher the throughput. Default is 1.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
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

        if isinstance(neg_threshold, int):
            _neg_threshold = neg_threshold
        else:
            _neg_threshold = NEG_THRES_MIN

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=_neg_threshold,
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
        bias: Optional[DataArrayType] = None,
        neg_threshold: Optional[int] = None,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """LIF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the membrane potential exceeds the threshold, neurons will fire.
            - reset_v: if not specified, neurons will do soft reset after firing, v - threshold. If \
                specified, neurons will do hard reset after firing, v = reset_v.
            - leak_v: the signed leak voltage will be added directly to the membrane potential.
                - If it is positive, the membrane potential will increase.
                - If is is negative, the membrane potential will decrease.
                - the final leak_v is leak_v + bias (default=0).
            - bias: if a signed bias is given, it will be added to `leak_v`. The neuron will leak   \
                before threshold comparison. `leak_v` will also be considered now.
            - neg_threshold: signed negative theshold. If not specified, it will be the smallest    \
                negative integer allowed by the hardware.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
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

        if isinstance(bias, (list, tuple, np.ndarray)):
            _bias = np.asarray(bias, dtype=np.int32)
        elif bias is not None:
            _bias = int(bias)
        else:
            _bias = 0

        # Support passing in bias & leak_v at the same time
        _leak_v = leak_v + _bias

        if isinstance(neg_threshold, int):
            _neg_threshold = neg_threshold
        else:
            _neg_threshold = NEG_THRES_MIN

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=_neg_threshold,
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
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Tonic spiking neuron.

        Args:
            - shape: shape of neurons.
            - fire_step: every `N` spike, the neuron will fire positively.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.

        NOTE: The neuron receives `N` spikes and fires, then it will reset to 0.
        """
        super().__init__(
            shape, pos_threshold=fire_step, keep_shape=keep_shape, name=name, **kwargs
        )


class PhasicSpiking(Neuron):
    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        neg_floor: int = -10,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Phasic spiking neuron. Once the neuron receives `N` spikes and fires, it will reset to   \
            the negative floor and never fires again. `N` is `fire_step`.

        Args:
            - shape: shape of neurons.
            - fire_step: after `N` spikes, the neuron will fire positively.
            - neg_floor: signed negative floor. once fired, the neurons will remain at this negative\
                membrane potential. Default is -10.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.
        """
        leak_v = 1
        super().__init__(
            shape,
            reset_v=neg_floor - 1,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=neg_floor,
            pos_threshold=(1 + leak_v) * fire_step,
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
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """A neuron that always outputs 1 as long as it starts working.

        Args:
            - shape: shape of neurons.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.

        FIXME There must be a forward synapse connected to it, otherwise the backend will go wrong. \
            Therefore, Always1Neuron is not exported to pb.__init__.
        """
        super().__init__(
            shape,
            reset_v=1,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=0,
            pos_threshold=0,
            leak_v=LEAK_V_MAX,
            keep_shape=keep_shape,
            name=name,
            **kwargs,
        )


class SpikingRelu(Neuron):
    def __init__(
        self,
        shape: Shape,
        *,
        keep_shape: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Spiking relu neuron. Act exactly the way you think.

        Args:
            - shape: shape of neurons.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.
        """
        super().__init__(
            shape, neg_threshold=0, keep_shape=keep_shape, name=name, **kwargs
        )
