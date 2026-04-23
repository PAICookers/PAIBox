"""Standalone AvgPool rewrite modes."""

import torch
from paicorelib import DataSign, DataWidth, SNNMode, ThresholdNegMode
from torch import nn

from ...ir.core_neuron import ANNNodeV25, IFNodeV25
from ...ir.graph import PAIIRGraph
from ...ir.ir_base import InputNode, OutputNode, PAIIRNode
from ...ir.lut_activation import LutCustom
from ...ir.op_node import OfflineCoreOp, SequentialOp, StandaloneActOp, StandaloneCompOp
from ...ir.signal_domain import SignalDomain
from ..graph_utils import (
    collect_effective_predecessor_values,
    is_format_transparent_routing_node,
    is_standalone_maxpool,
)
from .utils import build_sum_pool, get_avgpool_divisor, get_pool_window_size, is_avgpool

__all__ = ["rewrite_standalone_avgpools"]


def rewrite_standalone_avgpools(graph: PAIIRGraph) -> PAIIRGraph:
    """Rewrite standalone AvgPool nodes according to *mode*.

    This pass expects graph-wide signal-domain and data-format propagation to
    have already run once. It inspects each standalone AvgPool node's resolved
    input format and its effective upstream producer modes, and rewrites only
    the cases whose source semantics are known well enough to preserve exactly.
    All other standalone AvgPool nodes are left unchanged.
    """

    rewritten = graph.clone_shallow()
    changed = False

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not (isinstance(node, StandaloneCompOp) and is_avgpool(node.comp)):
            continue

        replacement = _rewrite_avgpool_node(graph, name, node)
        if replacement is None:
            continue

        replacement.input_layouts = node.input_layouts
        replacement.output_layouts = node.output_layouts
        rewritten.replace_node(name, replacement)
        changed = True

    return rewritten if changed else graph


def _rewrite_avgpool_node(
    graph: PAIIRGraph, node_name: str, node: StandaloneCompOp
) -> SequentialOp | None:
    assert is_avgpool(node.comp)

    source_modes = _collect_effective_source_modes(graph, node_name)
    if len(source_modes) != 1:
        return None

    source_mode = next(iter(source_modes))
    input_format = (node.core_params.input_sign, node.core_params.input_width)

    if source_mode is SNNMode.SNN and input_format == (
        DataSign.UNSIGNED,
        DataWidth.WIDTH_1BIT,
    ):
        if get_avgpool_divisor(node.comp) != get_pool_window_size(node.comp):
            return None
        return _build_binary_majority_avgpool(node.comp)

    if source_mode is SNNMode.ANN and input_format[1] == DataWidth.WIDTH_8BIT:
        return _build_exact_ann_avgpool(node.comp, input_format)

    return None


def _collect_effective_source_modes(
    graph: PAIIRGraph, node_name: str
) -> frozenset[SNNMode]:
    modes = collect_effective_predecessor_values(
        graph, node_name, _resolve_effective_source_modes, _is_source_transparent_node
    )
    return frozenset(modes)


def _resolve_effective_source_modes(
    node: PAIIRNode, _node_name: str
) -> list[SNNMode] | None:
    if _is_source_transparent_node(node):
        return None

    if isinstance(node, InputNode):
        return []

    source_mode = _get_effective_value_source_mode(node)
    if source_mode is not None:
        return [source_mode]

    if isinstance(node, OfflineCoreOp):
        return []

    return None


def _get_effective_value_source_mode(node: PAIIRNode) -> SNNMode | None:
    if not isinstance(node, OfflineCoreOp):
        return None

    if node.signal_semantics.output_domain is not SignalDomain.VALUE:
        return None

    if isinstance(node, (SequentialOp, StandaloneActOp)):
        return node.core_params.snn_mode

    act = getattr(node, "act", None)
    if act is not None:
        return node.core_params.snn_mode

    return None


def _is_source_transparent_node(node: PAIIRNode) -> bool:
    return (
        isinstance(node, OutputNode)
        or is_format_transparent_routing_node(node)
        or is_standalone_maxpool(node)
    )


def _build_binary_majority_avgpool(comp: nn.AvgPool1d | nn.AvgPool2d) -> SequentialOp:
    window_size = get_pool_window_size(comp)
    divisor = get_avgpool_divisor(comp)
    if divisor != window_size:
        raise ValueError(
            "binary-majority standalone AvgPool requires divisor == window_size, "
            f"got divisor={divisor}, window_size={window_size}"
        )
    threshold = window_size // 2 + 1
    sum_pool = build_sum_pool(comp)

    act = IFNodeV25(
        thres_neg_mode=ThresholdNegMode.FLOOR, leak_v=-(threshold - 1), thres_neg=0
    )
    return SequentialOp(sum_pool, act)


def _build_exact_ann_avgpool(
    comp: nn.AvgPool1d | nn.AvgPool2d, input_format: tuple[DataSign, DataWidth]
) -> SequentialOp:
    sum_pool = build_sum_pool(comp)

    lut = _build_exact_avg_round_lut(
        get_avgpool_divisor(comp), input_format, get_pool_window_size(comp)
    )
    return SequentialOp(sum_pool, ANNNodeV25(lut))


def _build_exact_avg_round_lut(
    divisor: int, input_format: tuple[DataSign, DataWidth], window_size: int
) -> LutCustom:
    sign, width = input_format
    if width is not DataWidth.WIDTH_8BIT:
        raise ValueError(
            f"exact ANN standalone AvgPool only supports WIDTH_8BIT input, got {width}"
        )

    if sign is DataSign.UNSIGNED:
        codes = torch.arange(256, dtype=torch.int32)
        min_val, max_val = 0, 255
        output_sign = 0
    else:
        codes = torch.arange(-128, 128, dtype=torch.int32)
        min_val, max_val = -128, 127
        output_sign = 1

    min_sum = min_val * window_size
    max_sum = max_val * window_size
    sums = torch.arange(min_sum, max_sum + 1, dtype=torch.int32)
    rounded = torch.round(sums.to(torch.float64) / divisor).to(torch.int32)

    thresholds: list[int] = []
    for code in codes.tolist():
        idx = (rounded == code).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise ValueError(
                f"exact ANN standalone AvgPool cannot build LUT for code {code}: "
                f"divisor={divisor}, input_format={input_format}"
            )
        thresholds.append(int(sums[int(idx[0])].item()))

    return LutCustom(torch.tensor(thresholds, dtype=torch.int32), codes, output_sign)
