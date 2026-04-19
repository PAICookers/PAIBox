"""Fusion helpers for AvgPool-specific deployment patterns."""

import math
from typing import cast

import torch
import torch.nn as nn
from paicorelib import (
    DataSign,
    DataWidth,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
)

from ...exceptions import UnsupportedFusionError
from ...ir.core_neuron import ANNNodeV25
from ...ir.graph import PAIIRGraph
from ...ir.lut_activation import LutCustom
from ...ir.op_node import (
    OfflineCoreOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from ...nn import SumPool1d, SumPool2d
from ..data_format import infer_output_format
from ..fusion_utils import _materialize_shared_sequential
from .compensation import (
    apply_avgpool_lut_compensation,
    apply_avgpool_snn_compensation,
    apply_sumpool_snn_compensation,
)
from .deploy_scheme import AvgPoolDeployScheme, select_avgpool_lif_candidate
from .metadata import AvgPoolDeployMetadata
from .utils import _get_avgpool_divisor, _get_pool_window_size, _is_avgpool

__all__ = [
    "_try_handle_avgpool_activation",
]


def _prepare_shared_avgpool_ann_params(act: ANNNodeV25, avg_divisor: int) -> None:
    """Write shared-core ANN AvgPool deployment parameters into the live node."""
    assert act.lut is not None
    apply_avgpool_lut_compensation(act.lut, avg_divisor)
    shift = round(math.log2(avg_divisor))
    act.leak_tau = -shift
    act.leak_multi_input = LeakMultiInputMode.ENABLE
    act.leak_multi_mode = LeakMultiMode.DISABLE
    act.leak_multi_sequence = LeakMultiComparisonOrder.AFTER_COMPARE


def _prepare_shared_avgpool_lif_params(
    node: SequentialOp, avg_divisor: int, uses_calibration: bool
) -> None:
    """Write shared-core LIF deployment params or calibration prerequisites."""
    source_decay_input = node.act.leak_multi_input == LeakMultiInputMode.ENABLE
    node.avgpool_deploy_metadata = AvgPoolDeployMetadata(
        source_decay_input, uses_calibration
    )
    apply_avgpool_snn_compensation(
        node.act,
        avg_divisor,
        decay_input=source_decay_input,
        calibrated=uses_calibration,
    )


def _prepare_split_avgpool_lif_core2_params(
    act_node: StandaloneActOp, avg_divisor: int
) -> None:
    """Write split-core LIF Core 2 params into the sum domain."""
    apply_sumpool_snn_compensation(act_node.act, avg_divisor)


def _infer_avgpool_pred_output_format(
    graph: PAIIRGraph, avgpool_name: str
) -> tuple[DataSign, DataWidth]:
    """Infer the predecessor activation's intrinsic output format.

    AvgPool topology choice runs before the graph-wide data-format propagation
    pass.  For the current split/shared decision we only need the predecessor
    activation's own output sign/width, which can be inferred locally here.
    """
    avgpool_preds = graph.predecessors(avgpool_name)
    if len(avgpool_preds) == 0:
        raise ValueError(
            f"AvgPool node '{avgpool_name}' has no predecessor; "
            "cannot determine upstream output format"
        )

    avgpool_pred = graph.nodes[avgpool_preds[0]]
    pred_act = getattr(avgpool_pred, "act", None)
    if pred_act is None:
        raise ValueError(
            f"AvgPool predecessor '{avgpool_preds[0]}' has no activation; "
            "cannot determine output format"
        )

    return infer_output_format(pred_act)


def _build_split_avgpool_core1_lut(
    out_width: DataWidth, avg_divisor: int, *, emit_exact_sum_code: bool
) -> LutCustom:
    """Build the Core 1 LUT used by split-core AvgPool deployment."""
    if emit_exact_sum_code:
        # LIF split-core path: Core 1 emits the exact pooled sum code itself.
        # Core 2 then restores the original AvgPool+LIF dynamics by scaling all
        # voltage-domain parameters into the same sum domain.
        if out_width == DataWidth.WIDTH_1BIT:
            thresholds = torch.arange(256, dtype=torch.int32)
            values = torch.arange(256, dtype=torch.uint8)
            return LutCustom(thresholds, values, output_sign=0, is_float=False)

        if out_width == DataWidth.WIDTH_2BIT:
            thresholds = torch.arange(-128, 128, dtype=torch.int32)
            values = torch.arange(-128, 128, dtype=torch.int8)
            return LutCustom(thresholds, values, output_sign=1, is_float=False)

        raise UnsupportedFusionError(
            "exact-sum split-core AvgPool LUT is only supported for 1-bit or 2-bit "
            "spike predecessors"
        )

    if out_width == DataWidth.WIDTH_1BIT:
        # IF split-core path: Core 1 converts the pooled sum into the downstream
        # IF input domain. The LUT therefore behaves like a thresholded mapper
        # rather than a raw sum pass-through.
        thresholds = torch.arange(256, dtype=torch.int32) * avg_divisor
        values = torch.cat(
            [torch.zeros(1, dtype=torch.int8), torch.ones(255, dtype=torch.int8)]
        )
        return LutCustom(thresholds, values, output_sign=0, is_float=False)

    if out_width == DataWidth.WIDTH_2BIT:
        raise UnsupportedFusionError(
            "AvgPool after signed spike (WIDTH_2BIT) is not yet supported for IF "
            "split-core deployment"
        )

    raise UnsupportedFusionError(
        f"AvgPool after ANN layer ({out_width}) is not yet supported for IF "
        "split-core deployment"
    )


def _try_handle_avgpool_activation(
    graph: PAIIRGraph,
    act_name: str,
    consumed: set[str],
    node_remap: dict[str, str],
    *,
    enable_split_avgpool_lif: bool = False,
    enable_avgpool_calibration: bool = False,
) -> SequentialOp | list[OfflineCoreOp] | None:
    """Handle ``AvgPool -> activation`` patterns during fusion.

    The policy is activation-dependent:

    - ANN / LUT activations stay shared-core.
    - IF-like spike neurons must split, because shared-core AvgPool would
      introduce multiplicative leak and change IF semantics.
    - LIF-like spike neurons default to shared-core, but may use the
      experimental split-core path when the exact sum-domain code can be
      transmitted losslessly from Core 1 to Core 2.
    """
    act_node = graph.nodes[act_name]
    assert isinstance(act_node, StandaloneActOp)

    preds = graph.predecessors(act_name)
    if len(preds) != 1:
        return None

    pred_name = preds[0]
    if pred_name in consumed:
        return None

    pred = graph.nodes[pred_name]
    if not isinstance(pred, StandaloneCompOp):
        return None
    if not _is_avgpool(pred.comp):
        return None
    if len(graph.successors(pred_name)) != 1:
        return None

    if not act_node.act.is_snn:
        # ANN activations only need static AvgPool gain compensation, so the
        # regular shared-core SequentialOp remains the preferred topology.
        avg_divisor = _get_avgpool_divisor(pred.comp)
        shared = _materialize_shared_sequential(
            pred_name, pred, act_name, act_node, consumed, node_remap
        )
        _prepare_shared_avgpool_ann_params(cast(ANNNodeV25, shared.act), avg_divisor)
        return shared

    if act_node.act.has_if_dynamics:
        # IF must split: shared-core AvgPool would enable leak on the neuron
        # datapath and turn the downstream behavior into LIF-like dynamics.
        _, out_width = _infer_avgpool_pred_output_format(graph, pred_name)
        avg_divisor = _get_avgpool_divisor(pred.comp)

        core1_lut = _build_split_avgpool_core1_lut(
            out_width, avg_divisor, emit_exact_sum_code=False
        )
        return _materialize_split_avgpool_pair(
            pred_name, pred, act_name, act_node, consumed, node_remap, core1_lut
        )

    if not act_node.act.has_lif_dynamics:
        raise UnsupportedFusionError(
            "AvgPool after SNN activation is only supported for IF-like or "
            "LIF-like neurons"
        )

    _, out_width = _infer_avgpool_pred_output_format(graph, pred_name)
    sum_window_size = _get_pool_window_size(pred.comp)
    avg_divisor = _get_avgpool_divisor(pred.comp)

    # LIF is the only case where both shared-core & split-core can be valid.
    # Delegate the final topology choice to the AvgPool deployment policy.
    best_candidate = select_avgpool_lif_candidate(
        act_node.act,
        out_width,
        sum_window_size,
        enable_split_avgpool_lif,
        try_calibration=enable_avgpool_calibration,
        avg_divisor=avg_divisor,
    )
    if best_candidate.scheme == AvgPoolDeployScheme.SHARED_CORE:
        # Default/compatible case: keep AvgPool and LIF on the same core and
        # write the shared-core deployment params immediately.
        shared = _materialize_shared_sequential(
            pred_name, pred, act_name, act_node, consumed, node_remap
        )
        _prepare_shared_avgpool_lif_params(
            shared, avg_divisor, best_candidate.uses_calibration
        )
        return shared

    # SPLIT_CORE_LIF_EXACT_SUM
    core1_lut = _build_split_avgpool_core1_lut(
        out_width, avg_divisor, emit_exact_sum_code=True
    )
    split_pair = _materialize_split_avgpool_pair(
        pred_name, pred, act_name, act_node, consumed, node_remap, core1_lut
    )
    _prepare_split_avgpool_lif_core2_params(act_node, avg_divisor)
    return split_pair


def _materialize_split_avgpool_pair(
    pred_name: str,
    pred: StandaloneCompOp,
    act_name: str,
    act_node: StandaloneActOp,
    consumed: set[str],
    node_remap: dict[str, str],
    core1_lut: LutCustom,
) -> list[OfflineCoreOp]:
    """Build the two-core split topology for AvgPool patterns.

    Core 1 is always ``SumPool + ANNNodeV25(lut)`` and Core 2 reuses the
    original activation node.
    """
    if isinstance(pred.comp, nn.AvgPool1d):
        sumpool = SumPool1d(
            pred.comp.kernel_size,
            pred.comp.stride,
            pred.comp.padding,
            pred.comp.ceil_mode,
        )
    elif isinstance(pred.comp, nn.AvgPool2d):
        sumpool = SumPool2d(
            pred.comp.kernel_size,
            pred.comp.stride,
            pred.comp.padding,
            pred.comp.ceil_mode,
        )
    else:
        raise TypeError("split AvgPool pair requires AvgPool1d/2d")

    core1_act = ANNNodeV25(core1_lut)
    core1 = SequentialOp(sumpool, core1_act)
    core1.input_layouts = pred.input_layouts
    core1.output_layouts = act_node.input_layouts

    consumed.add(pred_name)
    consumed.add(act_name)
    node_remap[pred_name] = core1.name
    node_remap[act_name] = act_node.name
    return [core1, act_node]
