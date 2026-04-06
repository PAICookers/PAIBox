"""FX-level analysis for reshape-like sinks and shape-only helper subgraphs.

This module keeps shape reasoning separate from the main PAIIR lowering flow.
It identifies reshape/flatten sinks, extracts the FX nodes that participate
only in output-size computation, and returns analysis results that the
converter can consume when building executable ``ReshapeOp`` routing nodes.
"""

import operator
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from torch import fx

from ..ir.reshape_semantics import (
    FLATTEN_FUNCTION_TARGETS,
    RESHAPE_FUNCTION_TARGETS,
    RESHAPE_LEAF_MODULE_TYPES,
    RESHAPE_METHOD_NAMES,
    SQUEEZE_FUNCTION_TARGETS,
    UNSQUEEZE_FUNCTION_TARGETS,
    is_identity_repeat_values,
)
from .fx_utils import get_call_arg

__all__ = ["ShapeAnalysisResult", "ReshapeSinkInfo", "analyze_shape_helpers"]

_SHAPE_EXPR_FUNCTION_TARGETS = (
    operator.getitem,
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.remainder,
)
_SHAPE_EXPR_UNARY_FUNCTION_TARGETS = (operator.neg, operator.pos, torch.neg)

ShapeExprKind: TypeAlias = Literal["scalar", "shape_container", "unknown"]


@dataclass(frozen=True, slots=True)
class ReshapeSinkInfo:
    """Normalized description of one reshape-like FX node.

    Attributes:
        kind: The semantic reshape family. ``"flatten"`` means reshape driven by
            ``start_dim`` / ``end_dim``. ``"reshape"`` means the output shape is
            already available from FX metadata and the converter can materialize
            a fixed-shape ``ReshapeOp``.
        data_input: The real tensor input that should become the sole PAIIR
            predecessor.
        shape_seed_nodes: FX nodes that contribute only to output-size
            computation. These seed the shape-helper subgraph collection.
        output_shape: Output shape propagated by Torch metadata when available.
        start_dim: Flatten start dimension for ``kind == "flatten"``.
        end_dim: Flatten end dimension for ``kind == "flatten"``.
    """

    kind: Literal["flatten", "reshape"]
    data_input: fx.Node
    shape_seed_nodes: tuple[fx.Node, ...]
    output_shape: torch.Size
    start_dim: int = 0
    end_dim: int = -1


@dataclass(frozen=True, slots=True)
class ShapeAnalysisResult:
    """Public analysis result consumed by the main converter pipeline.

    Attributes:
        reshape_sinks: FX nodes that should lower to ``ReshapeOp`` and their
            normalized sink metadata.
        aux_nodes: FX nodes that participate only in reshape-size reasoning and
            must be ignored by PAIIR data-flow lowering/wiring.
    """

    reshape_sinks: dict[fx.Node, ReshapeSinkInfo]
    aux_nodes: set[fx.Node]

    def sink_for(self, node: fx.Node) -> ReshapeSinkInfo | None:
        return self.reshape_sinks.get(node)

    def is_aux(self, node: fx.Node) -> bool:
        return node in self.aux_nodes


def analyze_shape_helpers(gm: fx.GraphModule) -> ShapeAnalysisResult:
    """Analyze reshape-like sinks and the FX nodes used only for shape math."""
    reshape_sinks = _analyze_reshape_sinks(gm)
    shape_aux_nodes = _collect_shape_aux_nodes(reshape_sinks)
    return ShapeAnalysisResult(reshape_sinks, shape_aux_nodes)


def _analyze_reshape_sinks(gm: fx.GraphModule) -> dict[fx.Node, ReshapeSinkInfo]:
    sinks: dict[fx.Node, ReshapeSinkInfo] = {}

    for node in gm.graph.nodes:
        info = _build_reshape_sink_info(gm, node)
        if info is not None:
            sinks[node] = info

    return sinks


def _build_reshape_sink_info(
    gm: fx.GraphModule, node: fx.Node
) -> ReshapeSinkInfo | None:
    if node.op == "call_module":
        torch_module = gm.get_submodule(str(node.target))
        if not isinstance(torch_module, RESHAPE_LEAF_MODULE_TYPES):
            return None
        if not node.args or not isinstance(node.args[0], fx.Node):
            return None

        return ReshapeSinkInfo(
            "flatten",
            node.args[0],
            (),
            _extract_tensor_output_shape(node),
            torch_module.start_dim,
            torch_module.end_dim,
        )

    if not node.args or not isinstance(node.args[0], fx.Node):
        return None

    data_input = node.args[0]

    if node.op == "call_method":
        if node.target == "flatten":
            return ReshapeSinkInfo(
                "flatten",
                data_input,
                (),
                _extract_tensor_output_shape(node),
                get_call_arg(node, 1, "start_dim", 0),
                get_call_arg(node, 2, "end_dim", -1),
            )

        if node.target == "unsqueeze":
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                (),
                _extract_tensor_output_shape(node),
            )

        if node.target == "squeeze":
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                (),
                _extract_tensor_output_shape(node),
            )

        if node.target == "repeat" and _is_identity_repeat(node):
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                (),
                _extract_tensor_output_shape(node),
            )

        if node.target in RESHAPE_METHOD_NAMES:
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                tuple(_reshape_shape_seed_nodes(node)),
                _extract_tensor_output_shape(node),
            )
        return None

    if node.op == "call_function":
        if node.target in FLATTEN_FUNCTION_TARGETS:
            return ReshapeSinkInfo(
                "flatten",
                data_input,
                (),
                _extract_tensor_output_shape(node),
                get_call_arg(node, 1, "start_dim", 0),
                get_call_arg(node, 2, "end_dim", -1),
            )

        if node.target in SQUEEZE_FUNCTION_TARGETS:
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                (),
                _extract_tensor_output_shape(node),
            )

        if node.target in UNSQUEEZE_FUNCTION_TARGETS:
            return ReshapeSinkInfo(
                "reshape", data_input, (), _extract_tensor_output_shape(node)
            )

        if node.target in RESHAPE_FUNCTION_TARGETS:
            return ReshapeSinkInfo(
                "reshape",
                data_input,
                tuple(_reshape_shape_seed_nodes(node)),
                _extract_tensor_output_shape(node),
            )

    return None


def _is_identity_repeat(node: fx.Node) -> bool:
    repeat_values = [*node.args[1:], *node.kwargs.values()]
    return is_identity_repeat_values(repeat_values)


def _reshape_shape_seed_nodes(node: fx.Node) -> list[fx.Node]:
    """Return the non-data FX arguments that define a reshape output size."""
    args = list(node.args)
    kwargs = list(node.kwargs.values())
    non_data_values = [*args[1:], *kwargs]

    seeds: list[fx.Node] = []
    seen: set[fx.Node] = set()
    for value in non_data_values:
        for maybe_node in _iter_nested_fx_nodes(value):
            if maybe_node not in seen:
                seeds.append(maybe_node)
                seen.add(maybe_node)
    return seeds


def _iter_nested_fx_nodes(value: Any) -> list[fx.Node]:
    """Collect FX nodes recursively from tuple/list/dict-shaped arguments."""
    if isinstance(value, fx.Node):
        return [value]
    if isinstance(value, (tuple, list)):
        nodes: list[fx.Node] = []
        for item in value:
            nodes.extend(_iter_nested_fx_nodes(item))
        return nodes
    if isinstance(value, dict):
        nodes: list[fx.Node] = []
        for item in value.values():
            nodes.extend(_iter_nested_fx_nodes(item))
        return nodes
    return []


def _extract_tensor_output_shape(node: fx.Node) -> torch.Size:
    """Return output tensor shape using Torch-propagated metadata when available."""
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None and hasattr(tensor_meta, "shape"):
        return torch.Size(tensor_meta.shape)

    val = node.meta.get("val")
    if torch.is_tensor(val):
        return val.shape

    return torch.Size()


def _collect_shape_aux_nodes(
    reshape_sinks: dict[fx.Node, ReshapeSinkInfo],
) -> set[fx.Node]:
    """Collect FX nodes used only to compute reshape-like output sizes.

    The collection is conservative: after gathering the backward slice from
    reshape shape-seeds, a user-closure pruning step removes any node whose
    value is also consumed by a non-shape user elsewhere in the FX graph.
    """
    shape_aux_nodes: set[fx.Node] = set()
    kind_cache: dict[fx.Node, ShapeExprKind] = {}

    for info in reshape_sinks.values():
        for seed in info.shape_seed_nodes:
            _collect_shape_aux_nodes_from_seed(seed, shape_aux_nodes, kind_cache)

    return _prune_nonlocal_shape_users(shape_aux_nodes, set(reshape_sinks))


def _collect_shape_aux_nodes_from_seed(
    seed: fx.Node, collected: set[fx.Node], kind_cache: dict[fx.Node, ShapeExprKind]
) -> None:
    """Recursively collect upstream shape-only helpers from one shape seed."""
    if seed in collected or _infer_shape_expr_kind(seed, kind_cache) == "unknown":
        return

    collected.add(seed)
    for inp in seed.all_input_nodes:
        _collect_shape_aux_nodes_from_seed(inp, collected, kind_cache)


def _prune_nonlocal_shape_users(
    candidate_nodes: set[fx.Node], reshape_sink_nodes: set[fx.Node]
) -> set[fx.Node]:
    """Drop candidates that are also consumed by non-shape users.

    A node is shape-aux only if every FX user stays inside the candidate
    subgraph or directly belongs to a reshape sink. This prevents shared nodes
    from being silently ignored when they also feed a real computation/output
    path elsewhere in the graph.
    """
    kept = set(candidate_nodes)
    changed = True

    while changed:
        changed = False
        for node in tuple(kept):
            if any(
                user not in kept and user not in reshape_sink_nodes
                for user in node.users
            ):
                kept.remove(node)
                changed = True

    return kept


def _infer_shape_expr_kind(
    node: fx.Node, cache: dict[fx.Node, ShapeExprKind]
) -> ShapeExprKind:
    """Infer whether an FX node semantically behaves like a shape expression.

    Torch-propagated metadata is the first information source. If metadata is
    not specific enough, fall back to a lightweight semantic analysis of the
    node's operator and the inferred kinds of its inputs.
    """
    if node in cache:
        return cache[node]

    kind = _shape_expr_kind_from_value(node.meta.get("val"))
    if kind == "unknown":
        if _is_shape_getattr(node):
            kind = "shape_container"
        elif _is_size_method(node):
            kind = "scalar"
        elif node.op == "call_function":
            kind = _infer_call_function_shape_expr_kind(node, cache)

    cache[node] = kind
    return kind


def _infer_call_function_shape_expr_kind(
    node: fx.Node, cache: dict[fx.Node, ShapeExprKind]
) -> ShapeExprKind:
    """Infer shape-expression kind for a call_function node from its semantics."""
    if node.target is operator.getitem:
        base_kind = _shape_expr_kind_from_value(get_call_arg(node, 0, "input"), cache)
        index_kind = _shape_expr_kind_from_value(get_call_arg(node, 1, "index"), cache)
        if base_kind == "shape_container" and index_kind in {"scalar", "unknown"}:
            return "scalar"
        return "unknown"

    if node.target in _SHAPE_EXPR_UNARY_FUNCTION_TARGETS:
        operand_kind = _shape_expr_kind_from_value(
            get_call_arg(node, 0, "input"), cache
        )
        return "scalar" if operand_kind == "scalar" else "unknown"

    if node.target in _SHAPE_EXPR_FUNCTION_TARGETS:
        operand_values = [*node.args, *node.kwargs.values()]
        if operand_values and all(
            _shape_expr_kind_from_value(value, cache) == "scalar"
            for value in operand_values
        ):
            return "scalar"

    return "unknown"


def _shape_expr_kind_from_value(
    value: Any, cache: dict[fx.Node, ShapeExprKind] | None = None
) -> ShapeExprKind:
    """Classify a Python/FX value as scalar shape, shape container, or unknown."""
    if isinstance(value, fx.Node):
        if cache is None:
            cache = {}
        return _infer_shape_expr_kind(value, cache)

    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, (int, torch.SymInt)):
        return "scalar"
    if isinstance(value, torch.Size):
        return "shape_container"
    if isinstance(value, (tuple, list)):
        if all(_shape_expr_kind_from_value(item, cache) == "scalar" for item in value):
            return "shape_container"
    return "unknown"


def _is_shape_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "shape"
    )


def _is_size_method(node: fx.Node) -> bool:
    return node.op == "call_method" and node.target == "size"
