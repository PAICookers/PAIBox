"""PAIIR-level folding for explicit zero padding.

Only local, provably equivalent ``PadOp -> Conv`` patterns are folded here.
Everything else remains an explicit ``PadOp`` for later pipeline/backend
handling.
"""

import copy

from torch import nn

from ..ir.graph import PAIIRGraph
from ..ir.op_node import PadOp, StandaloneCompOp

__all__ = ["fold_zero_pad_into_convs"]


def fold_zero_pad_into_convs(graph: PAIIRGraph) -> PAIIRGraph:
    """Fold eligible ``PadOp -> Conv`` patterns into the Conv padding field.

    The fold is intentionally conservative: one producer, one consumer,
    spatial-only symmetric padding, and a following zero-padding Conv1d/Conv2d.
    Asymmetric pads, shared pads, and non-Conv consumers stay explicit.
    """
    changed = False
    for name in list(graph.topo_sort()):
        node = graph.nodes.get(name)
        if not isinstance(node, PadOp):
            continue
        if _try_fold_pad_into_conv(graph, name, node):
            changed = True

    return graph.clone_shallow() if changed else graph


def _try_fold_pad_into_conv(graph: PAIIRGraph, pad_name: str, pad: PadOp) -> bool:
    incoming = graph.incoming_edges(pad_name)
    outgoing = graph.outgoing_edges(pad_name)
    if len(incoming) != 1 or len(outgoing) != 1:
        return False

    conv_name = outgoing[0].dst
    conv_node = graph.nodes.get(conv_name)
    if not isinstance(conv_node, StandaloneCompOp):
        return False

    comp = conv_node.comp
    folded_padding = _folded_conv_padding(comp, pad.padding)
    if folded_padding is None:
        return False

    updated_comp = copy.deepcopy(comp)
    updated_comp.padding = folded_padding
    conv_node.comp = updated_comp
    conv_node.input_layouts = pad.input_layouts

    graph.remove_node_and_reconnect(pad_name, source_name=incoming[0].src)
    return True


def _folded_conv_padding(
    comp: nn.Module, pad: tuple[int, ...]
) -> tuple[int, ...] | None:
    if isinstance(comp, nn.Conv1d):
        if comp.padding_mode != "zeros" or len(pad) != 2:
            return None
        left, right = pad
        if left != right:
            return None
        base = _conv_padding_tuple(comp.padding, 1)
        if base is None:
            return None
        return (base[0] + left,)

    if isinstance(comp, nn.Conv2d):
        if comp.padding_mode != "zeros" or len(pad) != 4:
            return None
        left, right, top, bottom = pad
        if left != right or top != bottom:
            return None
        base = _conv_padding_tuple(comp.padding, 2)
        if base is None:
            return None
        return base[0] + top, base[1] + left

    return None


def _conv_padding_tuple(
    padding: str | int | tuple[int, ...], ndim: int
) -> tuple[int, ...] | None:
    if isinstance(padding, str):
        return None
    if isinstance(padding, int):
        return (padding,) * ndim
    if len(padding) != ndim:
        return None
    return tuple(int(p) for p in padding)
