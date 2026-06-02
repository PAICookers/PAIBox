"""Materialize quantized adapter IR into backend-ready PAIIR nodes."""

import copy

import torch
from paicorelib import DataSign, DataWidth

from ..ir.add_ops import PotentialAddOp
from ..ir.graph import Edge, PAIIRGraph
from ..ir.ir_base import TensorLayout
from ..ir.op_node import SequentialOp, StandaloneActOp, StandaloneCompOp
from ..ir.quantized_ops import (
    PotentialPassthroughNodeV25,
    QuantizedConvAddReLU2dOp,
    QuantizedSequentialOp,
)

__all__ = ["materialize_quantized_ops"]


def _clone_layout(layout: TensorLayout) -> TensorLayout:
    return TensorLayout(layout.shape, layout.dims)


def _build_shortcut_scale_conv(
    input_layout: TensorLayout, output_layout: TensorLayout, gain: int
) -> torch.nn.Conv2d:
    """Build an existing-backend compute op for shortcut ``x * gain``."""
    if not 0 < gain <= 255:
        raise ValueError(
            "QuantizedConvAddReLU2dOp shortcut gain must fit uint8 Conv2d "
            f"weights, got {gain}."
        )
    if len(input_layout.shape) != 4 or len(output_layout.shape) != 4:
        raise ValueError(
            "QuantizedConvAddReLU2dOp shortcut scale requires 4D NCHW layouts."
        )
    if tuple(input_layout.shape) != tuple(output_layout.shape):
        raise ValueError(
            "QuantizedConvAddReLU2dOp shortcut scale requires input/output shapes "
            f"to match, got {tuple(input_layout.shape)} -> {tuple(output_layout.shape)}."
        )

    channels = int(input_layout.shape[1])
    conv = torch.nn.Conv2d(
        channels,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=channels,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.fill_(int(gain))
    return conv


def _replace_edges(
    graph: PAIIRGraph,
    old_name: str,
    replacement_nodes: list[str],
    new_edges: list[Edge],
) -> None:
    """Replace one node by a prepared node fragment."""

    graph.remove_node(old_name)

    for edge in new_edges:
        graph.add_edge(
            edge.src,
            edge.dst,
            src_port=edge.src_port,
            dst_port=edge.dst_port,
        )

    # Keep this local helper honest during future edits.
    for name in replacement_nodes:
        if name not in graph.nodes:
            raise RuntimeError(f"replacement node '{name}' was not added")


def _materialize_quantized_sequential(
    graph: PAIIRGraph, name: str, node: QuantizedSequentialOp
) -> bool:
    replacement = SequentialOp(copy.deepcopy(node.comp), node.act.clone())
    replacement.input_layouts = node.input_layouts
    replacement.output_layouts = node.output_layouts
    graph.replace_node(name, replacement)
    return True


def _materialize_quantized_conv_add_relu(
    graph: PAIIRGraph, name: str, node: QuantizedConvAddReLU2dOp
) -> bool:
    incoming = graph.incoming_edges(name)
    outgoing = graph.outgoing_edges(name)
    if len(incoming) != 2:
        raise ValueError(
            f"QuantizedConvAddReLU2dOp '{name}' expects 2 inputs, got {len(incoming)}"
        )

    by_port = {edge.dst_port: edge for edge in incoming}
    if set(by_port) != {0, 1}:
        raise ValueError(
            f"QuantizedConvAddReLU2dOp '{name}' requires dst_port 0/1 inputs"
        )

    conv_in = by_port[0]
    shortcut_in = by_port[1]
    if len(node.input_layouts) != 2 or len(node.output_layouts) != 1:
        raise ValueError(
            f"QuantizedConvAddReLU2dOp '{name}' requires shape metadata from sample_inputs"
        )

    # Core A: conv branch. No activation here, so it naturally emits POTENTIAL.
    conv_core = StandaloneCompOp(copy.deepcopy(node.conv))
    conv_core.input_layouts = (_clone_layout(node.input_layouts[0]),)
    conv_core.output_layouts = (_clone_layout(node.output_layouts[0]),)

    # Core B: shortcut branch. Weight M is represented as a normal depthwise
    # 1x1 Conv2d, so backendv2 only sees an existing compute op. Shift n is
    # stored in leak_tau so backend writes the existing leak register.
    shortcut_conv = _build_shortcut_scale_conv(
        node.input_layouts[1],
        node.output_layouts[0],
        node.shortcut_m,
    )
    shortcut_core = SequentialOp(
        shortcut_conv,
        PotentialPassthroughNodeV25(node.shortcut_n),
    )
    shortcut_core.core_params.set_weight_format((DataSign.UNSIGNED, DataWidth.WIDTH_8BIT))
    shortcut_core.input_layouts = (_clone_layout(node.input_layouts[1]),)
    shortcut_core.output_layouts = (_clone_layout(node.output_layouts[0]),)

    # Core C part 1: membrane-potential add. The following normal PAIIR fusion
    # pass may combine this with the activation core into AccumulateOp.
    add = PotentialAddOp((1, 1))
    add.input_layouts = (
        _clone_layout(node.output_layouts[0]),
        _clone_layout(node.output_layouts[0]),
    )
    add.output_layouts = (_clone_layout(node.output_layouts[0]),)
    # 只有量化 residual 展开的膜电平加法允许在后续 fusion 中变成
    # direct-add AccumulateOp。普通 PyTorch add 仍保持原有保守策略。
    add._allow_direct_add_activation_fusion = True

    # Core C part 2: calibrated ReLU LUT.
    act = StandaloneActOp(node.act.clone())
    act.input_layouts = (_clone_layout(node.output_layouts[0]),)
    act.output_layouts = (_clone_layout(node.output_layouts[0]),)

    graph.add_node(conv_core)
    graph.add_node(shortcut_core)
    graph.add_node(add)
    graph.add_node(act)

    new_edges = [
        Edge(conv_in.src, conv_core.name, conv_in.src_port, 0),
        Edge(shortcut_in.src, shortcut_core.name, shortcut_in.src_port, 0),
        Edge(conv_core.name, add.name, 0, 0),
        Edge(shortcut_core.name, add.name, 0, 1),
        Edge(add.name, act.name, 0, 0),
    ]
    new_edges.extend(Edge(act.name, edge.dst, 0, edge.dst_port) for edge in outgoing)

    _replace_edges(
        graph,
        name,
        [conv_core.name, shortcut_core.name, add.name, act.name],
        new_edges,
    )
    return True


def materialize_quantized_ops(graph: PAIIRGraph) -> PAIIRGraph:
    """Lower quantized adapter nodes to the normal backend-ready IR subset.

    量化 IR 只是前端适配层。这个 pass 结束后，图里不应该再保留
    ``Quantized*`` 节点，后续 pass 和 backendv2 继续使用已有基础算子。
    """

    rewritten = graph.clone_shallow()
    changed = False

    for name in list(graph.topo_sort()):
        if name not in rewritten.nodes:
            continue
        node = rewritten.nodes[name]
        if isinstance(node, QuantizedSequentialOp):
            changed |= _materialize_quantized_sequential(rewritten, name, node)
            continue
        if isinstance(node, QuantizedConvAddReLU2dOp):
            changed |= _materialize_quantized_conv_add_relu(rewritten, name, node)

    return rewritten if changed else graph
