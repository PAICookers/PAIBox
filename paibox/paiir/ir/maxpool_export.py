"""Helpers for standalone MaxPool export-strategy selection.

This module owns the compile-time decision of how a VALUE-domain standalone
MaxPool is exported:

- keep spike-domain identity semantics with a neuron path, or
- switch to ANN mode and emit an explicit identity LUT.
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, TypeGuard

import torch
from paicorelib import DataSign, DataWidth, SNNMode, ThresholdNegMode
from torch import nn

from .calc_params import LUT_TABLE_SIZE, LutData, NeuronParams
from .core_neuron import ANNNodeV25, IFNodeV25
from .lut_activation import LutCustom, _build_hw_lut_data_from_logical_lut
from .signal_domain import SignalDomain
from .value_code import ValueCodeRange, code_range_for_format

if TYPE_CHECKING:
    from .op_node import StandaloneCompOp

__all__ = [
    "MaxPoolExportKind",
    "build_identity_hw_lut_data",
    "build_identity_lut_neuron_params",
    "build_spike_identity_neuron_params",
    "is_maxpool_comp",
    "refresh_maxpool_export_kind",
    "select_maxpool_export_kind",
]


class MaxPoolExportKind(Enum):
    """Concrete export strategies for standalone MaxPool VALUE outputs."""

    U_SPIKE = auto()
    S_SPIKE = auto()
    LUT = auto()


def is_maxpool_comp(
    comp: nn.Module,
) -> TypeGuard[
    nn.MaxPool1d | nn.MaxPool2d | nn.AdaptiveMaxPool1d | nn.AdaptiveMaxPool2d
]:
    """Return whether ``comp`` is a standalone MaxPool compute op."""
    return isinstance(
        comp,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
        ),
    )


def select_maxpool_export_kind(
    input_sign: DataSign,
    input_width: DataWidth,
    known_input_code_range: ValueCodeRange | None,
) -> MaxPoolExportKind:
    """Choose the export strategy from resolved input encoding information.

    ``known_input_code_range`` is the strongest signal when available. If the
    exact integer alphabet is not known, the function falls back to the input
    sign/width envelope and chooses the most conservative export strategy that
    stays correct for that format.
    """
    if known_input_code_range is not None:
        lo, hi = known_input_code_range
        if 0 <= lo and hi <= 1:
            return MaxPoolExportKind.U_SPIKE
        if -1 <= lo and hi <= 1:
            return MaxPoolExportKind.S_SPIKE
        return MaxPoolExportKind.LUT

    if (input_sign, input_width) == (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT):
        return MaxPoolExportKind.U_SPIKE
    if (input_sign, input_width) == (DataSign.SIGNED, DataWidth.WIDTH_2BIT):
        return MaxPoolExportKind.S_SPIKE
    return MaxPoolExportKind.LUT


def refresh_maxpool_export_kind(node: "StandaloneCompOp") -> MaxPoolExportKind | None:
    """Refresh MaxPool export-side core params and return the chosen strategy.

    Today the refresh is limited to keeping ``core_params.snn_mode`` aligned
    with the resolved export strategy. The return value lets callers reuse the
    same decision for neuron-params/LUT export without recalculating policy in
    multiple places.
    """
    if node.signal_semantics.output_domain is not SignalDomain.VALUE:
        return None

    kind = select_maxpool_export_kind(
        node.core_params.input_sign,
        node.core_params.input_width,
        node.signal_semantics.known_code_range,
    )
    node.core_params.snn_mode = (
        SNNMode.ANN if kind is MaxPoolExportKind.LUT else SNNMode.SNN
    )
    return kind


def build_spike_identity_neuron_params(export: MaxPoolExportKind) -> NeuronParams:
    """Build exact identity neuron params for spike-code standalone MaxPool."""
    if export is MaxPoolExportKind.U_SPIKE:
        act = IFNodeV25(thres_neg_mode=ThresholdNegMode.FLOOR, thres_neg=0)
    elif export is MaxPoolExportKind.S_SPIKE:
        act = IFNodeV25(thres_neg_mode=ThresholdNegMode.FIRE, thres_neg=-1)
    else:
        raise ValueError(f"'{export.name}' is not a spike-identity export kind")

    return act.to_neuron_params()


def _build_identity_lut_data(code_min: int, code_max: int) -> tuple[LutData, bool]:
    if code_min > code_max:
        raise ValueError(
            f"identity LUT requires code_min <= code_max, got {code_min} > {code_max}"
        )

    code_count = code_max - code_min + 1
    if code_count > LUT_TABLE_SIZE:
        raise ValueError(f"identity LUT code range too large: ({code_min}, {code_max})")

    codes = torch.arange(code_min, code_max + 1, dtype=torch.int32)
    thresholds = codes
    values = codes
    if code_count < LUT_TABLE_SIZE:
        pad_count = LUT_TABLE_SIZE - code_count
        thresholds = torch.cat(
            (
                thresholds,
                torch.full((pad_count,), code_max, dtype=torch.int32),
            )
        )
        values = torch.cat(
            (
                values,
                torch.full((pad_count,), code_max, dtype=torch.int32),
            )
        )

    return LutData(thresholds, values, is_float=False), code_min < 0


def _build_identity_lut(code_min: int, code_max: int) -> LutCustom:
    logical_lut, output_signed = _build_identity_lut_data(code_min, code_max)
    return LutCustom(
        logical_lut.thresholds,
        logical_lut.values,
        output_signed,
        is_float=False,
    )


def build_identity_hw_lut_data(
    sign: DataSign,
    width: DataWidth,
    known_range: ValueCodeRange | None,
    output_data_sign: DataSign,
    output_width: DataWidth,
) -> LutData:
    """Build hardware SRAM LUT data for standalone MaxPool identity export."""
    lo, hi = code_range_for_format(sign, width) if known_range is None else known_range
    logical_lut, _ = _build_identity_lut_data(lo, hi)
    return _build_hw_lut_data_from_logical_lut(
        logical_lut, output_data_sign, output_width
    )


def build_identity_lut_neuron_params(
    sign: DataSign, width: DataWidth, known_range: ValueCodeRange | None
) -> NeuronParams:
    """Build ANN-mode neuron params that pair with the identity LUT export path."""
    lo, hi = code_range_for_format(sign, width) if known_range is None else known_range
    return ANNNodeV25(_build_identity_lut(lo, hi)).to_neuron_params()
