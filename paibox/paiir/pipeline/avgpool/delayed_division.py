"""Delayed AvgPool division rewrite for backend-safe exact-sum chains.

This pass handles one narrow but useful case:

- the graph contains a standalone ``AvgPool`` that is not followed by an
  activation immediately, so fusion cannot absorb the division locally
- the downstream path is a single-consumer linear chain
- every delayed ``AvgPool`` can be replaced by an exact-sum 8-bit carrier
- the final ``LUT / IF / LIF`` endpoint can absorb the full divisor safely

When all of those checks pass we keep the pooled value in the sum domain for
longer, then materialize the division exactly once at the endpoint by scaling
its thresholds / neuron voltage-domain parameters.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from paicorelib import OfflineNeuRegLim

from ...ir.core_neuron import ANNNodeV25, CoreNeuronV25
from ...ir.graph import PAIIRGraph
from ...ir.ir_base import InputNode, OutputNode, PAIIRNode
from ...ir.lut_activation import LutActivation, LutCustom
from ...ir.op_node import OfflineCoreOp, SequentialOp, StandaloneActOp, StandaloneCompOp
from ...ir.signal_domain import SignalDomain
from ..data_format import fits_value_code_range
from ..graph_utils import (
    collect_effective_predecessor_values,
    is_format_transparent_routing_node,
)
from .utils import build_sum_pool, get_avgpool_divisor, get_pool_window_size, is_avgpool

__all__ = ["rewrite_delayed_avgpool_division"]


_INT32_MIN = torch.iinfo(torch.int32).min
_INT32_MAX = torch.iinfo(torch.int32).max
_INT8_MIN = torch.iinfo(torch.int8).min
_INT8_MAX = torch.iinfo(torch.int8).max
_UINT8_MAX = torch.iinfo(torch.uint8).max


@dataclass(frozen=True, slots=True)
class _DelayedDivisionPlan:
    avgpool_names: tuple[str, ...]
    endpoint_name: str
    pending_divisor: int
    carrier_signed: bool


def rewrite_delayed_avgpool_division(graph: PAIIRGraph) -> PAIIRGraph:
    """Rewrite eligible standalone AvgPool chains before standalone fallback.

    The pass is intentionally conservative:

    - only one-consumer chains are considered
    - every delayed AvgPool must stay representable on the 8-bit VALUE path
    - the final endpoint must be able to absorb the divisor without register
      overflow

    If any check fails, the graph is left unchanged and the later standalone
    AvgPool rewrite pass remains free to apply its existing fallback policy.
    """
    rewritten = graph.clone_shallow()
    handled_avgpools: set[str] = set()
    changed = False

    for name in graph.topo_sort():
        if name in handled_avgpools:
            continue

        node = graph.nodes[name]
        if not (isinstance(node, StandaloneCompOp) and is_avgpool(node.comp)):
            continue

        plan = _analyze_delayed_division_chain(graph, name)
        if plan is None:
            continue

        _apply_delayed_division_plan(rewritten, graph, plan)
        handled_avgpools.update(plan.avgpool_names)
        changed = True

    return rewritten if changed else graph


def _analyze_delayed_division_chain(
    graph: PAIIRGraph, start_name: str
) -> _DelayedDivisionPlan | None:
    """Return one delayed-division plan or ``None`` when any safety check fails.

    The analysis walks forward from the first standalone AvgPool and tracks two
    independent quantities:

    - ``pending_divisor``: the product of all logical AvgPool divisors that must
      eventually be absorbed at the endpoint
    - ``code_min/code_max``: the exact sum-domain code range carried between
      nodes, which grows by pooling *window size* rather than divisor

    Those are only equal for the common ``divisor == window_size`` case, so the
    pass keeps them separate to handle ``divisor_override`` correctly.
    """
    code_ranges = _collect_effective_source_code_ranges(graph, start_name)
    if len(code_ranges) != 1:
        return None

    code_min, code_max = code_ranges[0]
    pending_divisor = 1
    avgpool_names: list[str] = []
    current_name = start_name

    while True:
        current = graph.nodes[current_name]

        if isinstance(current, StandaloneCompOp) and is_avgpool(current.comp):
            succs = graph.successors(current_name)
            if len(succs) != 1:
                return None

            # Division is postponed by multiplying the eventual endpoint scale
            # factor, while the on-wire exact-sum code range grows with the
            # physical pooling window size.
            pending_divisor *= get_avgpool_divisor(current.comp)
            code_min *= get_pool_window_size(current.comp)
            code_max *= get_pool_window_size(current.comp)
            if not fits_value_code_range(code_min, code_max):
                return None

            avgpool_names.append(current_name)
            current_name = succs[0]
            continue

        if is_format_transparent_routing_node(current):
            succs = graph.successors(current_name)
            if len(succs) != 1:
                return None
            current_name = succs[0]
            continue

        if not avgpool_names:
            return None

        if not _is_supported_endpoint_node(current):
            return None

        if not _can_materialize_pending_scale(current, pending_divisor):
            return None

        return _DelayedDivisionPlan(
            avgpool_names=tuple(avgpool_names),
            endpoint_name=current_name,
            pending_divisor=pending_divisor,
            carrier_signed=code_min < 0,
        )


def _apply_delayed_division_plan(
    rewritten: PAIIRGraph, graph: PAIIRGraph, plan: _DelayedDivisionPlan
) -> None:
    """Apply one pre-validated plan on the cloned graph.

    Each delayed AvgPool is replaced 1:1 by ``SumPool + exact identity LUT`` so
    the graph topology stays stable. The terminal node is then deep-copied and
    rewritten in-place to absorb the accumulated divisor.
    """
    for avgpool_name in plan.avgpool_names:
        avgpool_node = graph.nodes[avgpool_name]
        assert isinstance(avgpool_node, StandaloneCompOp)
        assert is_avgpool(avgpool_node.comp)
        replacement = _build_exact_sum_carrier(avgpool_node.comp, plan.carrier_signed)
        replacement.name = avgpool_name
        replacement.input_layouts = avgpool_node.input_layouts
        replacement.output_layouts = avgpool_node.output_layouts
        rewritten.replace_node(avgpool_name, replacement)

    endpoint = copy.deepcopy(graph.nodes[plan.endpoint_name])
    endpoint.name = plan.endpoint_name
    _materialize_pending_scale(endpoint, plan.pending_divisor)
    rewritten.replace_node(plan.endpoint_name, endpoint)


def _build_exact_sum_carrier(
    comp: nn.AvgPool1d | nn.AvgPool2d, signed: bool
) -> SequentialOp:
    """Build the exact-sum carrier that replaces one standalone AvgPool."""
    sum_pool = build_sum_pool(comp)
    return SequentialOp(sum_pool, ANNNodeV25(_build_identity_lut(signed)))


def _build_identity_lut(signed: bool) -> LutCustom:
    """Build the 8-bit identity LUT used to carry exact sum-domain codes."""
    if signed:
        thresholds = torch.arange(_INT8_MIN, _INT8_MAX + 1, dtype=torch.int32)
        values = torch.arange(_INT8_MIN, _INT8_MAX + 1, dtype=torch.int8)
        return LutCustom(thresholds, values, output_sign=1, is_float=False)

    thresholds = torch.arange(_UINT8_MAX + 1, dtype=torch.int32)
    values = torch.arange(_UINT8_MAX + 1, dtype=torch.uint8)
    return LutCustom(thresholds, values, output_sign=0, is_float=False)


def _collect_effective_source_code_ranges(
    graph: PAIIRGraph, node_name: str
) -> list[tuple[int, int]]:
    """Collect code ranges from effective VALUE-domain producers upstream."""
    return collect_effective_predecessor_values(
        graph,
        node_name,
        _resolve_effective_source_code_ranges,
        _is_source_transparent_node,
    )


def _resolve_effective_source_code_ranges(
    node: PAIIRNode, _node_name: str
) -> list[tuple[int, int]] | None:
    if _is_source_transparent_node(node):
        return None

    if isinstance(node, InputNode):
        code_range = node.signal_semantics.known_code_range
        return [] if code_range is None else [code_range]

    code_range = _get_effective_output_code_range(node)
    if code_range is not None:
        return [code_range]

    if isinstance(node, OfflineCoreOp):
        return []

    return None


def _get_effective_output_code_range(node: PAIIRNode) -> tuple[int, int] | None:
    if not isinstance(node, OfflineCoreOp):
        return None
    if node.signal_semantics.output_domain is not SignalDomain.VALUE:
        return None
    return node.signal_semantics.known_code_range


def _is_source_transparent_node(node: PAIIRNode) -> bool:
    return isinstance(node, OutputNode) or is_format_transparent_routing_node(node)


def _is_supported_endpoint_node(node: PAIIRNode) -> bool:
    if isinstance(node, SequentialOp):
        return _is_supported_weighted_comp(node.comp)
    if isinstance(node, StandaloneActOp):
        return True
    return False


def _is_supported_weighted_comp(comp: nn.Module) -> bool:
    return isinstance(comp, (nn.Conv1d, nn.Conv2d, nn.Linear))


def _can_materialize_pending_scale(node: PAIIRNode, factor: int) -> bool:
    """Check whether the endpoint can absorb the postponed divisor safely."""
    if factor <= 0:
        return False

    if isinstance(node, SequentialOp):
        bias = _get_comp_bias(node.comp)
        if not _bias_leak_is_within_bounds(node.act.leak_v, bias, factor):
            return False
        if not node.act.is_snn:
            return _lut_thresholds_within_bounds(node.act.lut, factor)
        return _neuron_params_within_bounds(node.act, factor)

    if isinstance(node, StandaloneActOp):
        if not _bias_leak_is_within_bounds(node.act.leak_v, None, factor):
            return False
        if not node.act.is_snn:
            return _lut_thresholds_within_bounds(node.act.lut, factor)
        return _neuron_params_within_bounds(node.act, factor)

    return False


def _lut_thresholds_within_bounds(lut: LutActivation | None, factor: int) -> bool:
    if lut is None:
        return False

    thresholds = lut.thresholds.detach().to(torch.float64) * factor
    if not lut.is_float:
        return True
    return bool(
        torch.all((thresholds >= _INT32_MIN) & (thresholds <= _INT32_MAX)).item()
    )


def _neuron_params_within_bounds(act: CoreNeuronV25, factor: int) -> bool:
    if isinstance(act.thres_pos, torch.Tensor):
        raise ValueError(
            "Delayed AvgPool division does not support per-channel thres_pos"
        )

    threshold_values = (
        torch.tensor([act.thres_pos, act.thres_neg], dtype=torch.float64) * factor
    )
    if not torch.all(
        (threshold_values >= -OfflineNeuRegLim.NEG_THRES_MAX)
        & (threshold_values <= OfflineNeuRegLim.POS_THRES_MAX)
    ).item():
        return False

    reset_values = torch.tensor([act.reset_v], dtype=torch.float64) * factor
    if not torch.all(
        (reset_values >= OfflineNeuRegLim.RESET_V_MIN)
        & (reset_values <= OfflineNeuRegLim.RESET_V_MAX)
    ).item():
        return False

    init_values = torch.tensor([act.init_v], dtype=torch.float64) * factor
    return bool(
        torch.all(
            (init_values >= OfflineNeuRegLim.VOLTAGE_MIN)
            & (init_values <= OfflineNeuRegLim.VOLTAGE_MAX)
        ).item()
    )


def _bias_leak_is_within_bounds(
    leak_v: float | torch.Tensor, bias: torch.Tensor | None, factor: int
) -> bool:
    leak_tensor = _to_tensor(leak_v)
    if bias is not None:
        leak_tensor = leak_tensor + bias.detach().to(torch.float64)
    scaled = leak_tensor * factor
    return bool(
        torch.all(
            (scaled >= OfflineNeuRegLim.LEAK_V_MIN)
            & (scaled <= OfflineNeuRegLim.LEAK_V_MAX)
        ).item()
    )


def _to_tensor(value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float64)
    return torch.tensor(value, dtype=torch.float64)


def _materialize_pending_scale(node: PAIIRNode, factor: int) -> None:
    """Write the delayed divisor into the terminal node's live parameters.

    For weighted endpoints we keep weights unchanged. The divisor is absorbed by
    scaling:

    - compute bias, because it lives in the same pre-activation domain
    - LUT thresholds for ANN endpoints
    - voltage-domain neuron parameters for IF/LIF endpoints
    """
    if isinstance(node, SequentialOp):
        _scale_comp_bias(node.comp, factor)
        if node.act.is_snn:
            _scale_neuron(node.act, factor)
        else:
            _scale_lut_thresholds(node.act, factor)
        return

    if isinstance(node, StandaloneActOp):
        if node.act.is_snn:
            _scale_neuron(node.act, factor)
        else:
            _scale_lut_thresholds(node.act, factor)


def _scale_comp_bias(comp: nn.Module, factor: int) -> None:
    bias = _get_comp_bias(comp)
    if bias is None:
        return
    with torch.no_grad():
        bias.mul_(factor)


def _get_comp_bias(comp: nn.Module) -> torch.Tensor | None:
    bias = getattr(comp, "bias", None)
    if torch.is_tensor(bias):
        return bias
    return None


def _scale_lut_thresholds(act: CoreNeuronV25, factor: int) -> None:
    """Scale one ANN endpoint into the delayed sum domain."""
    lut = act.lut
    assert lut is not None
    with torch.no_grad():
        scaled = lut.thresholds.detach().to(torch.float64) * factor
        lut.thresholds.copy_(scaled.to(lut.thresholds.dtype))
    act.leak_v = act.leak_v * factor


def _scale_neuron(act: CoreNeuronV25, factor: int) -> None:
    """Scale all neuron voltage-domain parameters by the pending divisor."""
    if isinstance(act.thres_pos, torch.Tensor):
        raise ValueError(
            "Delayed AvgPool division does not support per-channel thres_pos"
        )

    act.thres_pos *= factor
    act.thres_neg *= factor
    act.reset_v *= factor
    act.init_v *= factor
    act.leak_v = act.leak_v * factor
