import warnings
from enum import IntEnum, unique
from typing import Literal, TypedDict, Union

import numpy as np
from paicorelib import (
    OnCoreCfg,
    OffCoreCfg,
    InputWidthFormat,
    MaxPoolingEnable,
    OffRAMDefs,
    SNNModeEnable,
    SpikeWidthFormat,
)
from paicorelib.framelib.utils import _mask

from paibox.exceptions import FunctionalError, PAIBoxWarning
from paibox.types import (
    NEUOUT_U8_DTYPE,
    NEUOUT_SPIKE_DTYPE,
    VOLTAGE_DTYPE,
    LeakVType,
    VoltageType,
)

BIT_TRUNC_MAX = OffRAMDefs.BIT_TRUNC_MAX
LEAK_V_BIT_MAX = OffRAMDefs.LEAK_V_BIT_MAX
LEAK_V_MAX = OffRAMDefs.LEAK_V_MAX
LEAK_V_MIN = OffRAMDefs.LEAK_V_MIN
NEG_THRES_MAX = OffRAMDefs.NEG_THRES_MAX
V_MAX = OffRAMDefs.VOLTAGE_MAX
V_MIN = OffRAMDefs.VOLTAGE_MIN
V_BIT_MAX = OffRAMDefs.VOLTAGE_BIT_MAX


SIGNED_PARAM_OVERFLOW_TEXT = "{0} overflow, beyond the range of {1}-bit signed integer."
V_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format("voltage", V_BIT_MAX)
LEAK_V_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format("leak voltage", LEAK_V_BIT_MAX)
V_RANGE_LIMIT = V_MAX - V_MIN


def _is_v_overflow(v: VoltageType, strict: bool = False) -> bool:
    # NOTE: In most cases, the voltage overflow won't occur, otherwise the result
    # may be incorrect.
    if np.any(v > V_MAX) or np.any(v < V_MIN):
        if strict:
            raise FunctionalError(V_OVERFLOW_TEXT)
        else:
            warnings.warn(V_OVERFLOW_TEXT, PAIBoxWarning)

        return False

    return True


def v_overflow(v: VoltageType, strict: bool = False) -> VoltageType:
    """Handle the overflow of the voltage.

    NOTE: If the incoming voltage (30-bit signed) overflows, the chip will automatically handle it. \
        This behavior needs to be implemented in simulation.
    """
    _is_v_overflow(v, strict)

    return np.where(
        v > V_MAX, v - V_RANGE_LIMIT, np.where(v < V_MIN, v + V_RANGE_LIMIT, v)
    ).astype(VOLTAGE_DTYPE)


def _leak_v_check(leak_v: Union[int, LeakVType]) -> None:
    if isinstance(leak_v, int):
        if leak_v > LEAK_V_MAX or leak_v < LEAK_V_MIN:
            raise FunctionalError(LEAK_V_OVERFLOW_TEXT)

    elif np.any(leak_v > LEAK_V_MAX) or np.any(leak_v < LEAK_V_MIN):
        raise FunctionalError(LEAK_V_OVERFLOW_TEXT)


L = Literal


def _input_width_format(iwf: Union[L[1, 8], InputWidthFormat]) -> InputWidthFormat:
    if isinstance(iwf, InputWidthFormat):
        return iwf

    if iwf == 1:
        return InputWidthFormat.WIDTH_1BIT
    else:
        return InputWidthFormat.WIDTH_8BIT


def _spike_width_format(swf: Union[L[1, 8], SpikeWidthFormat]) -> SpikeWidthFormat:
    if isinstance(swf, SpikeWidthFormat):
        return swf

    if swf == 1:
        return SpikeWidthFormat.WIDTH_1BIT
    else:
        return SpikeWidthFormat.WIDTH_8BIT


def _get_neu_out_dtype(
    swf: SpikeWidthFormat,
) -> type[Union[NEUOUT_SPIKE_DTYPE, NEUOUT_U8_DTYPE]]:
    if swf is SpikeWidthFormat.WIDTH_1BIT:
        return NEUOUT_SPIKE_DTYPE
    else:
        return NEUOUT_U8_DTYPE


class RTModeKwds(TypedDict):
    """A typed keywords for runtime mode. Only for checking if necessary."""

    input_width: InputWidthFormat
    spike_width: SpikeWidthFormat
    snn_en: SNNModeEnable


class ExtraNeuAttrKwds(TypedDict, total=False):
    """A typed keywords for extra neuron attributes."""

    bit_trunc: int  # For ANNNeuron
    delay: int
    tick_wait_start: int
    tick_wait_end: int
    input_width: Union[L[1, 8], InputWidthFormat]
    spike_width: Union[L[1, 8], SpikeWidthFormat]
    snn_en: Union[bool, SNNModeEnable]
    pool_max: Union[bool, MaxPoolingEnable]
    unrolling_factor: int
    overflow_strict: bool
    target_chip: int


@unique
class NeuFireState(IntEnum):
    """Auxiliary enum type to indicate whether the neuron is firing or not.
    - `NOT_FIRING`: not firing.
    - `FIRING_POS`: firing positive threshold.
    - `FIRING_NEG`: firing negative threshold.
    """

    NOT_FIRING = 0
    FIRING_POS = 1
    FIRING_NEG = 2
