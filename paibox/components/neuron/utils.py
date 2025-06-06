import warnings
from typing import Literal, TypedDict, Union

import numpy as np
from paicorelib import (
    InputWidthFormat,
    MaxPoolingEnable,
    SNNModeEnable,
    SpikeWidthFormat,
)
from paicorelib.framelib.utils import _mask
from paicorelib.ram_model import (
    BIT_TRUNCATE_MAX,
    LEAK_V_BIT_MAX,
    LEAK_V_MAX,
    LEAK_V_MIN,
)
from paicorelib.ram_model import NEG_THRES_MAX as NEG_THRES_UNSIGNED_MAX
from paicorelib.ram_model import VJT_MAX, VJT_MIN, VJT_PRE_BIT_MAX

from paibox.exceptions import FunctionalError, PAIBoxWarning
from paibox.types import (
    NEUOUT_U8_DTYPE,
    SPIKE_DTYPE,
    VOLTAGE_DTYPE,
    LeakVType,
    VoltageType,
)

NEG_THRES_MIN = -NEG_THRES_UNSIGNED_MAX


SIGNED_PARAM_OVERFLOW_TEXT = "{0} overflow, beyond the range of {1}-bit signed integer."
VJT_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format(
    "membrane potential", VJT_PRE_BIT_MAX
)
LEAK_V_OVERFLOW_TEXT = SIGNED_PARAM_OVERFLOW_TEXT.format("leak voltage", LEAK_V_BIT_MAX)
VJT_RANGE_LIMIT = VJT_MAX - VJT_MIN


def _is_vjt_overflow(vjt: VoltageType, strict: bool = False) -> bool:
    # NOTE: In most cases, membrane potential overflow won't occur, otherwise the result
    # may be incorrect.
    if np.any(vjt > VJT_MAX) or np.any(vjt < VJT_MIN):
        if strict:
            raise FunctionalError(VJT_OVERFLOW_TEXT)
        else:
            warnings.warn(VJT_OVERFLOW_TEXT, PAIBoxWarning)

        return False

    return True


def vjt_overflow(vjt: VoltageType, strict: bool = False) -> VoltageType:
    """Handle the overflow of the membrane potential.

    NOTE: If the incoming membrane potential (30-bit signed) overflows, the chip will   \
        automatically handle it. This behavior needs to be implemented in simulation.
    """
    _is_vjt_overflow(vjt, strict)

    return np.where(
        vjt > VJT_MAX,
        vjt - VJT_RANGE_LIMIT,
        np.where(
            vjt < VJT_MIN,
            vjt + VJT_RANGE_LIMIT,
            vjt,
        ),
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
) -> type[Union[SPIKE_DTYPE, NEUOUT_U8_DTYPE]]:
    if swf is SpikeWidthFormat.WIDTH_1BIT:
        return SPIKE_DTYPE
    else:
        return NEUOUT_U8_DTYPE


class RTModeKwds(TypedDict):
    """A typed keywords for runtime mode. Only for checking if necessary."""

    input_width: InputWidthFormat
    spike_width: SpikeWidthFormat
    snn_en: SNNModeEnable


class ExtraNeuAttrKwds(TypedDict, total=False):
    """A typed keywords for extra neuron attributes."""

    bit_truncation: int  # For ANNNeuron
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
