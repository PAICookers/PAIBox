from collections.abc import Sequence
import sys
from typing import Optional, Union

import numpy as np
from paicorelib import LCM, LDM, NTM, RM, OffRAMDefs

from paibox.types import LEAK_V_DTYPE, DataType, LeakVType, Shape

from .base import OfflineNeuron, OnlineNeuron
from .utils import LEAK_V_MAX, CommonExtraNeuAttrKwds, ExtraNeuAttrKwds

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


__all__ = [
    "IF",
    "LIF",
    "TonicSpiking",
    "PhasicSpiking",
    "BypassNeuron",
    "StoreVoltageNeuron",
    "Always1Neuron",
    "ANNBypassNeuron",
    "ANNNeuron",
    "STDPNeuron",
]

POS_THRES_MAX = OffRAMDefs.POS_THRES_MAX


def _bias_to_leak_v(bias: DataType) -> Union[LeakVType, int]:
    if isinstance(bias, np.ndarray):
        return np.atleast_1d(bias).astype(LEAK_V_DTYPE)
    else:
        return int(bias)


class IF(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int = 1,
        reset_v: Optional[int] = None,
        neg_threshold: Optional[int] = None,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """IF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the voltage exceeds the threshold, neurons will fire.
            - reset_v: if not specified, neurons will do soft reset after firing, v - threshold. If \
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

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=neg_threshold,
            pos_threshold=threshold,
            name=name,
            **kwargs,
        )


class LIF(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int = 1,
        reset_v: Optional[int] = None,
        leak_v: int = 0,
        bias: DataType = 0,
        neg_threshold: Optional[int] = None,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """LIF neuron.

        Args:
            - shape: shape of neurons.
            - threshold: when the voltage exceeds the threshold, neurons will fire.
            - reset_v: if not specified, neurons will do soft reset after firing, v - threshold. If \
                specified, neurons will do hard reset after firing, v = reset_v.
            - leak_v: the signed leak voltage will be added directly to the voltage.
                - If it is positive, the voltage will increase.
                - If is is negative, the voltage will decrease.
                - The final leak_v is leak_v + bias (default=0).
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

        super().__init__(
            shape,
            reset_mode=_rm,
            reset_v=_reset_v,
            neg_thres_mode=NTM.MODE_SATURATION,
            neg_threshold=neg_threshold,
            pos_threshold=threshold,
            leak_v=leak_v + _bias_to_leak_v(bias),
            name=name,
            **kwargs,
        )


class TonicSpiking(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        fire_step: int = 1,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """Tonic spiking neuron.

        Args:
            - shape: shape of neurons.
            - fire_step: every `N` spike, the neuron will fire positively.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.

        NOTE: The neuron receives `N` spikes and fires, then it will reset to 0.
        """
        super().__init__(shape, pos_threshold=fire_step, name=name, **kwargs)


class PhasicSpiking(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        fire_step: int,
        neg_floor: int = -10,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """Phasic spiking neuron. Once the neuron receives `N` spikes and fires, it will reset to   \
            the negative floor and never fires again. `N` is `fire_step`.

        Args:
            - shape: shape of neurons.
            - fire_step: after `N` spikes, the neuron will fire positively.
            - neg_floor: signed negative floor. once fired, the neurons will remain at this negative\
                voltage. Default is -10.
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
            name=name,
            **kwargs,
        )


class Always1Neuron(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
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
            name=name,
            **kwargs,
        )


class BypassNeuron(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """Bypass neuron. Output is equal to input.

        Args:
            - shape: shape of neurons.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.

        NOTE: positive threshold = 1, negative threshold = 0, reset_v = 0, and leak_v = 0.
        """
        super().__init__(shape, neg_threshold=0, name=name, **kwargs)


class StoreVoltageNeuron(OfflineNeuron):
    def __init__(
        self,
        shape: Shape,
        leak_v: int = 0,
        bias: DataType = 0,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """The neuron that stores the voltage and never fires nor resets.

        Args:
            - shape: shape of neurons.
            - leak_v: the signed leak voltage will be added directly to the voltage.
                - If it is positive, the voltage will increase.
                - If is is negative, the voltage will decrease.
                - The final leak_v is leak_v + bias (default=0).
            - bias: if a signed bias is given, it will be added to `leak_v`. The neuron will leak   \
                before threshold comparison. `leak_v` will also be considered now.
            - keep_shape: whether to maintain shape in the simulation. Default is `True`.
            - name: name of the neuron. Optional.
        """
        super().__init__(
            shape,
            reset_mode=RM.MODE_NONRESET,
            neg_thres_mode=NTM.MODE_RESET,
            leak_v=leak_v + _bias_to_leak_v(bias),
            pos_threshold=POS_THRES_MAX,
            name=name,
            **kwargs,
        )


class ANNNeuron(LIF):
    def __init__(
        self,
        shape: Shape,
        bias: DataType = 0,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        """General neuron used in ANN mode. Positive threshold = 1, negative threshold = 0."""
        kwargs.setdefault("bit_trunc", 8)
        kwargs.setdefault("input_width", 8)
        kwargs.setdefault("spike_width", 8)
        kwargs.setdefault("snn_en", False)

        super().__init__(shape, 1, bias=bias, name=name, **kwargs)


class ANNBypassNeuron(ANNNeuron):
    def __init__(
        self,
        shape: Shape,
        *,
        name: Optional[str] = None,
        **kwargs: Unpack[ExtraNeuAttrKwds],
    ) -> None:
        super().__init__(shape, bias=0, name=name, **kwargs)


class STDPNeuron(OnlineNeuron):
    def __init__(
        self,
        shape: Shape,
        threshold: int = 1,
        reset_v: int = 0,
        leak_v: int = 0,
        bias: DataType = 0,
        leak_comparison: LCM = LCM.LEAK_BEFORE_COMP,
        neg_threshold: Optional[int] = None,
        lateral_inhi_value: int = 0,
        init_v: Union[int, np.ndarray] = 0,
        *,
        learn_by_default: bool = True,
        name: Optional[str] = None,
        **kwargs: Unpack[CommonExtraNeuAttrKwds],
    ) -> None:
        super().__init__(
            shape,
            reset_v,
            leak_v + _bias_to_leak_v(bias),
            neg_threshold,
            threshold,
            leak_comparison,
            lateral_inhi_value,
            init_v,
            learn_by_default=learn_by_default,
            name=name,
            **kwargs,
        )
