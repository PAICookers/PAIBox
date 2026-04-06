"""Split-specific FX lowering helpers.

This module keeps ``converter.py`` focused on orchestration while centralizing
the front-end analysis and IR-node construction for ``torch.split``-style
patterns.
"""

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx

from ..ir.op_node import SplitOp
from .dims_prop import DimsType
from .fx_utils import (
    get_call_arg,
    get_fx_call_target_name,
    get_output_dims,
    get_output_shape,
)

SplitSections = int | tuple[int, ...]

SPLIT_FUNCTIONS = (torch.split,)
SUPPORTED_SPLIT_METHODS = ("split",)


@dataclass(frozen=True, slots=True)
class SplitProducerInfo:
    data_input: fx.Node
    sections: SplitSections
    dim: int


def is_split_like_node(node: fx.Node) -> bool:
    if node.op == "call_function":
        return node.target in SPLIT_FUNCTIONS
    if node.op == "call_method":
        return node.target in SUPPORTED_SPLIT_METHODS
    return False


def _split_like_name(node: fx.Node) -> str:
    return get_fx_call_target_name(node)


def _normalize_split_sections(value: Any) -> SplitSections | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(int(item) for item in value)
    return None


def _is_identity_dims(dims: DimsType) -> bool:
    return not dims or dims == tuple(range(len(dims)))


def _extract_split_producer_args(
    node: fx.Node,
) -> tuple[fx.Node, SplitSections, int] | None:
    if not node.args or not isinstance(node.args[0], fx.Node):
        return None

    data_input = node.args[0]
    raw_sections = get_call_arg(node, 1, "split_size_or_sections")
    if raw_sections is None and node.op == "call_method":
        raw_sections = node.kwargs.get("split_size")
    sections = _normalize_split_sections(raw_sections)

    raw_dim = get_call_arg(node, 2, "dim", 0)
    if sections is None or isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        return None

    return data_input, sections, raw_dim


def _extract_split_getitem_index(node: fx.Node, producer: fx.Node) -> int | None:
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    if len(node.args) < 2 or node.args[0] is not producer:
        return None

    index = node.args[1]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None

    return index


def analyze_split_producer(
    node: fx.Node,
) -> tuple[SplitProducerInfo | None, str | None]:
    split_name = _split_like_name(node)
    if split_name != "split":
        return (
            None,
            "is not supported by PAIIR split V1; only torch.split / Tensor.split are currently supported",
        )

    parsed = _extract_split_producer_args(node)
    if parsed is None:
        return (
            None,
            "requires static Python split_size_or_sections and integer dim arguments",
        )

    data_input, sections, dim = parsed
    input_dims = get_output_dims(data_input)
    if input_dims and not _is_identity_dims(input_dims):
        return (
            None,
            "does not support pending transpose/permute layout before split; "
            "materialize or remove the layout transform first",
        )

    for user in node.users:
        if _extract_split_getitem_index(user, node) is None:
            return (
                None,
                "only supports direct non-negative integer getitem consumers",
            )

    return SplitProducerInfo(data_input, sections, dim), None


def describe_unsupported_split_like(
    node: fx.Node,
) -> str:
    kind = "function" if node.op == "call_function" else "method"
    split_name = _split_like_name(node)
    _, description = analyze_split_producer(node)
    if description is None:
        description = "is not supported"
    return f"{kind} '{split_name}' {description}"


def apply_split_analysis_rule(
    gm: fx.GraphModule,
    split_producers: dict[fx.Node, SplitProducerInfo],
    split_consumers: dict[fx.Node, tuple[SplitProducerInfo, int]],
) -> None:
    for node in gm.graph.nodes:
        if not is_split_like_node(node):
            continue

        split_info, description = analyze_split_producer(node)
        if split_info is None or description is not None:
            continue

        split_producers[node] = split_info
        for user in node.users:
            output_index = _extract_split_getitem_index(user, node)
            assert output_index is not None
            split_consumers[user] = (split_info, output_index)


def build_split_ir_node(
    split_info: SplitProducerInfo,
) -> tuple[SplitOp, tuple[fx.Node, ...]]:
    ir_node = SplitOp(split_info.sections, split_info.dim)
    # The SplitOp itself records only the split contract. Per-consumer branch
    # selection is attached later on outgoing edges via `src_port`.
    ir_node.input_shapes = [get_output_shape(split_info.data_input)]
    ir_node.input_dims = [get_output_dims(split_info.data_input)]
    ir_node.output_dims = ir_node.input_dims[0] if ir_node.input_dims else ()
    return ir_node, (split_info.data_input,)
