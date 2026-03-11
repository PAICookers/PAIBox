"""AvgPool shared-core compensation functions.

Standalone pure functions that transform NeuronParams and LutData
to account for AvgPool's division-by-shift approximation on chip.
Called from SequentialOp/AccumulateOp properties, never mutate
the original activation module.
"""

import math

from paicorelib import LeakMultiComparisonOrder, LeakMultiInputMode, LeakMultiMode

from .calc_params import LutData, NeuronParams

__all__ = [
    "apply_avgpool_leak_params",
    "compensate_avgpool_lut",
    "compensate_avgpool_neuron",
    "compensate_splitcore_avgpool_threshold",
]


def apply_avgpool_leak_params(
    params: NeuronParams, window_size: int, *, is_ann: bool = False
) -> NeuronParams:
    """Set leak registers for AvgPool shared-core division-by-shift.

    AvgPool on chip is implemented as sum pooling followed by a
    division-by-shift in the neuron's leak stage.  The leak mechanism
    performs a right-shift by N bits, approximating division by the
    pooling window size::

        AvgPool(x) = SumPool(x) / window_size
                   ≈ SumPool(x) >> N       where N = round(log2(window_size))

    Register configuration:

    - ``leak_multi_input = ENABLE``: activate the multiplicative leak path
      so the shift is applied to the accumulated membrane potential.
    - ``leak_tau = -N``: negative value encodes a right-shift by N bits.
    - ANN mode additionally requires ``leak_multi_mode = DISABLE`` and
      ``leak_multi_sequence = AFTER_COMPARE`` to ensure the shift occurs
      after threshold comparison (LUT lookup).

    Args:
        params: Neuron parameters to modify (mutated in place).
        window_size: Number of elements in the pooling window (product of
            kernel dimensions).
        is_ann: Whether the core runs in ANN mode (LUT activation).

    Returns:
        The modified *params* (same object, for chaining).
    """
    N = round(math.log2(window_size))
    params.leak_multi_input = LeakMultiInputMode.ENABLE
    params.leak_tau = -N
    if is_ann:
        params.leak_multi_mode = LeakMultiMode.DISABLE
        params.leak_multi_sequence = LeakMultiComparisonOrder.AFTER_COMPARE
    return params


def compensate_avgpool_lut(lut: LutData, window_size: int) -> LutData:
    """Scale LUT bin boundaries to compensate for shift approximation error.

    The hardware divides by ``2^N`` (right-shift), but the true divisor is
    ``window_size``.  When ``window_size`` is not a power of two the neuron
    sees a scaled input::

        v_hw = SumPool(x) >> N = SumPool(x) / 2^N

    The ideal input to the LUT would be::

        v_ideal = SumPool(x) / window_size

    Their ratio gives the compensation factor::

        v_hw / v_ideal = window_size / 2^N   (= 1.0 when window_size is a power of 2)

    To recover the correct LUT bin assignment we scale each threshold::

        threshold' = threshold * (window_size / 2^N)

    so that ``v_hw >= threshold'`` iff ``v_ideal >= threshold``.
    For non-float LUTs (integer register mode) the result is rounded to
    the nearest integer.  Output values are unchanged — only the bin
    boundaries shift.

    Args:
        lut: LUT data to modify (mutated in place).
        window_size: Number of elements in the pooling window (product of
            kernel dimensions).

    Returns:
        The modified *lut* (same object, for chaining).
    """
    N = round(math.log2(window_size))
    scale = window_size / (1 << N)
    scaled = lut.thresholds.float() * scale
    if not lut.is_float:
        scaled.round_()
    lut.thresholds = scaled.to(lut.thresholds.dtype)
    return lut


def compensate_avgpool_neuron(
    params: NeuronParams, window_size: int, decay_input: bool, tau: float
) -> NeuronParams:
    """Adjust SNN neuron thresholds for AvgPool shared-core deployment.

    In a shared core the neuron receives the *sum-pooled* input (not the
    average).  The leak stage will right-shift by N to approximate
    division, but the neuron's threshold comparison sees the accumulated
    (pre-shift) value.  We must scale the threshold so that the neuron
    fires at the same logical point as it would with the true average.

    Let ``r`` = reset voltage, ``θ`` = original threshold.

    **Case 1 — decay_input = True** (input is scaled by tau internally):

    The membrane potential accumulates as ``v = window_size * x`` (sum of
    all elements in the pooling window).  The threshold must scale by the
    same factor::

        θ' = r + (θ - r) * window_size

    **Case 2 — decay_input = False** (raw input, no tau scaling):

    The membrane potential accumulates as ``v = window_size * x / τ``.
    The threshold scales by the reduced factor::

        θ' = r + (θ - r) * window_size / τ

    Both positive and negative thresholds are compensated symmetrically.

    Args:
        params: Neuron parameters to modify (mutated in place).
        window_size: Number of elements in the pooling window (product of
            kernel dimensions).
        decay_input: Whether the neuron decays incoming input
            (``leak_multi_input == ENABLE``).
        tau: Neuron time constant.  Only used when ``decay_input`` is False.

    Returns:
        The modified *params* (same object, for chaining).
    """
    r = params.reset_v
    if decay_input:
        factor = window_size
    else:
        factor = window_size / tau

    params.thres_pos = r + (params.thres_pos - r) * factor
    params.thres_neg = r + (params.thres_neg - r) * factor
    return params


def compensate_splitcore_avgpool_threshold(
    params: NeuronParams, window_size: int
) -> NeuronParams:
    """Scale thresholds for split-core AvgPool + IF deployment.

    In split-core mode, Core 1 (AvgPool) divides by ``2^N`` (right-shift)
    instead of ``window_size``.  Core 2 (IF neuron) therefore sees a signal
    scaled by ``window_size / 2^N``.  The threshold must scale by the same
    factor so that the neuron fires at the correct logical point.

    Formula::

        theta' = r + (theta - r) * window_size / 2^N

    When ``window_size`` is a power of 2, the factor is 1.0 and no
    compensation is needed.

    Args:
        params: Neuron parameters to modify (mutated in place).
        window_size: Number of elements in the pooling window (product of
            kernel dimensions).

    Returns:
        The modified *params* (same object, for chaining).
    """
    N = round(math.log2(window_size))
    factor = window_size / (1 << N)
    if factor == 1.0:
        return params

    r = params.reset_v
    params.thres_pos = r + (params.thres_pos - r) * factor
    params.thres_neg = r + (params.thres_neg - r) * factor
    return params
