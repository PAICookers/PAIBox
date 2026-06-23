"""Standalone AvgPool rewrite modes.

This pass handles AvgPool nodes that remain after normal core fusion. Most
rewrites preserve the original graph semantics. The output-boundary
``sum_approx_if_avgpool`` mode is explicit because it changes the exported
value from an average to an unnormalized sum/count for host-side handling.
"""

import warnings
from typing import Literal, TypeAlias

import torch
from paicorelib import DataSign, DataWidth, SNNMode, ThresholdNegMode
from torch import nn

from ...exceptions import OutputApproxWarning
from ...ir.calc_params import LUT_TABLE_SIZE
from ...ir.core_neuron import ANNNodeV25, IFNodeV25
from ...ir.graph import PAIIRGraph
from ...ir.ir_base import InputNode, OutputNode, PAIIRNode
from ...ir.lut_activation import LutCustom
from ...ir.op_node import OfflineCoreOp, SequentialOp, StandaloneActOp, StandaloneCompOp
from ...ir.signal_domain import SignalDomain
from ..data_format import DataFormat, fits_value_code_range, infer_output_format
from ..graph_utils import (
    collect_effective_predecessor_values,
    collect_effective_value_code_ranges,
    is_format_transparent_routing_node,
    is_standalone_maxpool,
)
from .utils import (
    build_integer_identity_lut,
    build_integer_interval_lut,
    build_sum_pool,
    get_avgpool_divisor,
    get_pool_window_size,
    is_avgpool,
)

__all__ = ["rewrite_standalone_avgpools"]

OutputApprox: TypeAlias = Literal["default", "sum_approx_if_avgpool"]
_OUTPUT_APPROX_MODES: tuple[str, ...] = ("default", "sum_approx_if_avgpool")


def rewrite_standalone_avgpools(
    graph: PAIIRGraph, output_approx: OutputApprox = "default"
) -> PAIIRGraph:
    """Rewrite standalone AvgPool nodes according to the selected policy.

    This pass expects graph-wide signal-domain and data-format propagation to
    have already run once. It inspects each standalone AvgPool node's resolved
    input format and its effective upstream producer modes, and rewrites only
    the cases whose source semantics are known well enough to preserve exactly.

    ``output_approx="sum_approx_if_avgpool"`` additionally allows eligible
    output-layer AvgPool nodes to export sums/counts instead of averages. All
    unsupported or unsafe AvgPool nodes are left unchanged.
    """
    if output_approx not in _OUTPUT_APPROX_MODES:
        supported = ", ".join(_OUTPUT_APPROX_MODES)
        raise ValueError(
            f"unsupported output_approx={output_approx!r}; supported: {supported}"
        )

    rewritten = graph.clone_shallow()
    changed = False

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not (isinstance(node, StandaloneCompOp) and is_avgpool(node.comp)):
            continue

        replacement = _rewrite_avgpool_node(graph, name, node, output_approx)
        if replacement is None:
            continue

        replacement.input_layouts = node.input_layouts
        replacement.output_layouts = node.output_layouts
        rewritten.replace_node(name, replacement)
        changed = True

    return rewritten if changed else graph


def _rewrite_avgpool_node(
    graph: PAIIRGraph,
    node_name: str,
    node: StandaloneCompOp,
    output_approx: OutputApprox,
) -> SequentialOp | None:
    """Return a deployable replacement for one standalone AvgPool, if safe."""
    assert is_avgpool(node.comp)

    output_boundary = _resolve_output_boundary(graph, node_name)
    if (
        output_approx == "sum_approx_if_avgpool"
        and output_boundary is not None
        and (replacement := _build_output_sum_approx_avgpool(graph, node_name, node))
        is not None
    ):
        return replacement

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
        replacement = _build_binary_majority_avgpool(node.comp)
        if output_boundary is not None:
            _warn_output_majority_fallback(node_name, node, replacement)
        return replacement

    if source_mode is SNNMode.ANN and input_format[1] == DataWidth.WIDTH_8BIT:
        return _build_exact_ann_avgpool(node.comp, input_format)

    return None


def _resolve_output_boundary(graph: PAIIRGraph, node_name: str) -> str | None:
    """Return the terminal OutputNode for a single transparent output path."""
    current_name = node_name

    while True:
        succs = graph.successors(current_name)
        if len(succs) != 1:
            return None

        succ_name = succs[0]
        succ = graph.nodes[succ_name]
        if isinstance(succ, OutputNode):
            return succ_name

        if not is_format_transparent_routing_node(succ):
            return None

        if len(graph.predecessors(succ_name)) != 1:
            return None
        current_name = succ_name


def _build_output_sum_approx_avgpool(
    graph: PAIIRGraph, node_name: str, node: StandaloneCompOp
) -> SequentialOp | None:
    """Build the explicit output-layer sum/count approximation.

    The replacement keeps the chip-side value in the integer sum domain:
    ``AvgPool(x)`` becomes ``SumPool(x) + identity LUT``. This is only valid for
    full, non-padded windows with a known VALUE-domain input range that fits the
    deployable 8-bit data path. The caller emits an ``OutputApproxWarning`` so
    the application knows it must divide by the logical divisor or accumulate
    counts before applying task-level decisions such as ``argmax``.
    """
    assert is_avgpool(node.comp)

    if not _has_exact_full_pool_windows(node.comp):
        return None

    code_ranges = collect_effective_value_code_ranges(graph, node_name)
    if len(code_ranges) != 1:
        return None

    input_min, input_max = code_ranges[0]
    window_size = get_pool_window_size(node.comp)
    code_range = input_min * window_size, input_max * window_size
    code_min, code_max = code_range
    if code_min > code_max:
        return None
    if not fits_value_code_range(code_min, code_max):
        return None
    if code_max - code_min + 1 > LUT_TABLE_SIZE:
        return None

    sum_pool = build_sum_pool(node.comp)
    code_min, code_max = code_range
    act = ANNNodeV25(build_integer_identity_lut(code_min, code_max))
    replacement = SequentialOp(sum_pool, act)
    replacement.name = node_name

    data_format = _format_data_format(infer_output_format(act))
    exported_form = f"{type(sum_pool).__name__} + identity LUT"
    logical_divisor = get_avgpool_divisor(node.comp)
    cpu_responsibility = (
        f"divide by divisor {logical_divisor} to recover average, or accumulate "
        "counts before argmax"
    )
    # TODO(cpu-runtime): when PAIIR/backend can export explicit CPU postprocess
    # tasks, attach this divide/accumulate responsibility to the graph instead
    # of communicating it only through a warning.
    warnings.warn(
        (
            f"Output-layer node '{node_name}' ({type(node.comp).__name__}) is "
            f"exported as {exported_form}; exported range={code_range}, "
            f"format={data_format}. Original AvgPool average is replaced by an "
            f"unnormalized sum/count. CPU side must {cpu_responsibility}."
        ),
        OutputApproxWarning,
        stacklevel=3,
    )
    return replacement


def _collect_effective_source_modes(
    graph: PAIIRGraph, node_name: str
) -> frozenset[SNNMode]:
    """Collect effective upstream execution modes through transparent nodes."""
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
    """Return the execution mode of a VALUE-producing offline core."""
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


def _has_exact_full_pool_windows(comp: nn.AvgPool1d | nn.AvgPool2d) -> bool:
    """Return whether each output aggregates exactly one full unpadded window."""
    if comp.ceil_mode:
        return False

    padding = comp.padding
    if isinstance(padding, int):
        return padding == 0
    return all(p == 0 for p in padding)


def _build_binary_majority_avgpool(comp: nn.AvgPool1d | nn.AvgPool2d) -> SequentialOp:
    """Build the default SNN binary-majority fallback for spike AvgPool."""
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


def _warn_output_majority_fallback(
    node_name: str, node: StandaloneCompOp, replacement: SequentialOp
) -> None:
    """Warn that the default output-layer fallback exports majority spikes."""
    data_format = _format_data_format(infer_output_format(replacement.act))
    code_range = (0, 1)
    warnings.warn(
        (
            f"Output-layer node '{node_name}' ({type(node.comp).__name__}) is "
            f"exported as {type(replacement.comp).__name__} + IFNodeV25; "
            f"exported range={code_range}, format={data_format}. Default "
            "output-layer AvgPool fallback emits majority spikes rather than the "
            "original average; use output_approx='sum_approx_if_avgpool' to export "
            "counts when the input range is supported. CPU side must interpret the "
            "result as majority spike output."
        ),
        OutputApproxWarning,
        stacklevel=3,
    )


def _format_data_format(data_format: DataFormat) -> str:
    sign, width = data_format
    prefix = "s" if sign is DataSign.SIGNED else "u"
    width_name = width.name
    if width_name.startswith("WIDTH_") and width_name.endswith("BIT"):
        bits = width_name.removeprefix("WIDTH_").removesuffix("BIT")
    else:
        bits = str(width)
    return f"{prefix}{bits} DATA"


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
        codes = torch.arange(LUT_TABLE_SIZE, dtype=torch.int32)
        min_val, max_val = 0, 255
        output_signed = False
    else:
        codes = torch.arange(-128, 128, dtype=torch.int32)
        min_val, max_val = -128, 127
        output_signed = True

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

    return build_integer_interval_lut(
        thresholds, codes.tolist(), output_signed=output_signed
    )
