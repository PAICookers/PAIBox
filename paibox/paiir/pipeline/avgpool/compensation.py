"""Compensation formulas for AvgPool deployment."""

import math

from paicorelib import LeakMultiComparisonOrder, LeakMultiInputMode, LeakMultiMode

from ...ir.calc_params import LutData, NeuronParams
from ...ir.core_neuron import CoreNeuronV25
from ...ir.lut_activation import LutActivation

__all__ = [
    "apply_avgpool_leak_params",
    "apply_avgpool_lut_compensation",
    "apply_avgpool_snn_compensation",
    "apply_sumpool_snn_compensation",
    "compensate_avgpool_lut",
    "compensate_avgpool_lut_for_sumpool",
    "compensate_avgpool_neuron",
    "compensate_sumpool_neuron",
]


def _is_power_of_two(n: int) -> bool:
    """Return True when AvgPool division can be represented by an exact shift."""
    return n > 0 and (n & (n - 1)) == 0


def apply_avgpool_leak_params(
    params: NeuronParams, window_size: int, *, is_ann: bool = False
) -> NeuronParams:
    """Set leak-related registers used to emulate AvgPool division by shift.

    In ANN mode the division itself is approximated by ``2^-N`` on the shared
    datapath, so we explicitly program ``leak_tau = -N``.  In SNN/LIF mode the
    leak register already belongs to the neuron dynamics and must be preserved.
    """
    params.leak_multi_input = LeakMultiInputMode.ENABLE
    if is_ann:
        shift = round(math.log2(window_size))
        params.leak_tau = -shift
        params.leak_multi_mode = LeakMultiMode.DISABLE
        params.leak_multi_sequence = LeakMultiComparisonOrder.AFTER_COMPARE
    return params


def compensate_avgpool_lut(lut: LutData, window_size: int) -> LutData:
    """Rescale LUT thresholds for shared-core AvgPool + ANN deployment."""
    shift = round(math.log2(window_size))
    scale = window_size / (1 << shift)
    scaled = lut.thresholds.float() * scale
    if not lut.is_float:
        scaled.round_()
    lut.thresholds = scaled.to(lut.thresholds.dtype)
    return lut


def compensate_avgpool_neuron(
    params: NeuronParams,
    window_size: int,
    decay_input: bool,
    tau: float,
    *,
    is_float: bool = False,
) -> NeuronParams:
    """Rescale threshold distances for shared-core AvgPool + LIF.

    The compiled core integrates the pooled sum rather than the exact average.
    Thresholds are therefore moved into the same working domain while keeping
    the reset voltage fixed.
    """
    reset_v = params.reset_v
    factor = window_size if decay_input else window_size / tau
    params.thres_pos = reset_v + (params.thres_pos - reset_v) * factor
    params.thres_neg = reset_v + (params.thres_neg - reset_v) * factor
    if not is_float:
        params.thres_pos = round(params.thres_pos)
        params.thres_neg = round(params.thres_neg)
    return params


def compensate_avgpool_lut_for_sumpool(lut: LutData, window_size: int) -> LutData:
    """Move a LUT from AvgPool domain into SumPool domain."""
    lut.thresholds = lut.thresholds * window_size
    return lut


def compensate_sumpool_neuron(
    params: NeuronParams, window_size: int, *, is_float: bool = False
) -> NeuronParams:
    """Scale all voltage-domain params for split-core SumPool + LIF.

    Core 1 emits ``sum = window_size * avg`` in the split-core LIF path, so Core
    2 must run the neuron in that same scaled voltage domain.
    """
    params.thres_pos *= window_size
    params.thres_neg *= window_size
    params.reset_v *= window_size
    params.init_v *= window_size

    if not is_float:
        params.thres_pos = round(params.thres_pos)
        params.thres_neg = round(params.thres_neg)
        params.reset_v = round(params.reset_v)
        params.init_v = round(params.init_v)

    return params


def apply_avgpool_lut_compensation(lut: LutActivation, window_size: int) -> None:
    """Write shared-core ANN compensation back into the live LUT module."""
    if _is_power_of_two(window_size):
        # Exact power-of-two windows need no threshold correction.
        return

    shift = round(math.log2(window_size))
    scale = window_size / (1 << shift)
    compensated = lut.thresholds.float() * scale
    if not lut.is_float:
        compensated.round_()
    lut.thresholds.copy_(compensated.to(lut.thresholds.dtype))


def apply_avgpool_snn_compensation(
    neuron: CoreNeuronV25,
    window_size: int,
    *,
    decay_input: bool,
    calibrated: bool = False,
) -> None:
    """Write shared-core LIF compensation back into the live neuron module."""
    if not calibrated:
        reset_v = neuron.reset_v
        # Shared-core LIF runs in the sum domain, so thresholds move there while
        # reset voltage stays in place. For ``decay_input=False`` the ideal
        # current term is not divided by ``tau``, so the threshold distance only
        # needs to absorb ``window_size / tau``.
        factor = window_size if decay_input else window_size / neuron.tau
        neuron.thres_pos = round(reset_v + (neuron.thres_pos - reset_v) * factor)
        neuron.thres_neg = round(reset_v + (neuron.thres_neg - reset_v) * factor)

    # Shared-core AvgPool+LIF relies on the neuron leak path itself to supply the
    # division-by-shift behavior during simulation/export.
    neuron.leak_multi_input = LeakMultiInputMode.ENABLE
    neuron.leak_multi_mode = LeakMultiMode.ENABLE
    neuron.leak_multi_sequence = LeakMultiComparisonOrder.AFTER_COMPARE


def apply_sumpool_snn_compensation(
    neuron: CoreNeuronV25, window_size: int, *, is_float: bool = False
) -> None:
    """Write split-core SumPool + LIF scaling back into the live neuron module."""
    neuron.thres_pos *= window_size
    neuron.thres_neg *= window_size
    neuron.reset_v *= window_size
    neuron.init_v *= window_size

    if not is_float:
        neuron.thres_pos = round(neuron.thres_pos)
        neuron.thres_neg = round(neuron.thres_neg)
        neuron.reset_v = round(neuron.reset_v)
        neuron.init_v = round(neuron.init_v)
