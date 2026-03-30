"""Conv-family lowering helpers for both module-form and function-form convs."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, fx, nn

from ..ir.op_node import StandaloneCompOp

__all__ = [
    "ConvSpec",
    "build_conv_ir_node",
    "extract_functional_conv_spec",
    "extract_module_conv_spec",
]


@dataclass(frozen=True)
class ConvSpec:
    """Normalized conv-family lowering result independent of FX source form."""

    input_node: fx.Node
    module: nn.Conv1d | nn.Conv2d
    aux_nodes: frozenset[fx.Node] = field(default_factory=frozenset)


@dataclass(frozen=True)
class _FunctionalConvLoweringSpec:
    target_name: str
    spatial_ndim: int


_FUNCTIONAL_CONV_SPECS = {
    "conv1d": _FunctionalConvLoweringSpec("conv1d", 1),
    "conv2d": _FunctionalConvLoweringSpec("conv2d", 2),
}


def _get_functional_conv_spec(target: Any) -> _FunctionalConvLoweringSpec | None:
    return _FUNCTIONAL_CONV_SPECS.get(getattr(target, "__name__", ""))


def extract_module_conv_spec(
    node: fx.Node, module: nn.Module
) -> ConvSpec | None:
    if not isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return None
    if not node.args or not isinstance(node.args[0], fx.Node):
        return None
    return ConvSpec(node.args[0], module)


def _is_dtype_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "dtype"
    )


def _resolve_attr_value(gm: fx.GraphModule, target: str) -> Any:
    value: Any = gm
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _get_call_arg(node: fx.Node, index: int, name: str, default: Any = None) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _infer_normalize_arg_type(value: Any) -> Any:
    if isinstance(value, fx.Node):
        return Tensor
    if isinstance(value, tuple):
        return tuple(type(item) for item in value)
    if isinstance(value, list):
        return list[Any]
    return type(value)


def _get_normalized_call_kwargs(
    node: fx.Node, root: fx.GraphModule
) -> dict[str, Any] | None:
    arg_types = tuple(_infer_normalize_arg_type(arg) for arg in node.args)
    try:
        normalized = node.normalized_arguments(
            root, arg_types=arg_types, normalize_to_only_use_kwargs=True
        )
    except RuntimeError:
        return None

    if normalized is None:
        return None

    return dict(normalized.kwargs)


def _resolve_to_aux_nodes(
    gm: fx.GraphModule, node: fx.Node
) -> tuple[set[fx.Node] | None, bool]:
    extra_nodes = {node}
    for arg in (*node.args[1:], *node.kwargs.values()):
        if isinstance(arg, fx.Node) and _is_dtype_getattr(arg):
            extra_nodes.add(arg)
            continue
        if isinstance(arg, fx.Node):
            resolved, extra = _resolve_constant_value(gm, arg)
            if resolved is None:
                return None, False
            extra_nodes |= extra
    return extra_nodes, True


def _resolve_constant_value(
    gm: fx.GraphModule, value: Any
) -> tuple[Any | None, set[fx.Node]]:
    if isinstance(value, fx.Node):
        if value.op == "get_attr":
            return _resolve_attr_value(gm, str(value.target)), {value}
        if value.op == "call_method" and value.target == "to" and value.args:
            resolved, nodes = _resolve_constant_value(gm, value.args[0])
            if resolved is None:
                return None, set()

            extra_nodes, ok = _resolve_to_aux_nodes(gm, value)
            if not ok or extra_nodes is None:
                return None, set()
            return resolved, nodes | extra_nodes
        if value.op == "call_function" and value.target in (operator.mul, torch.mul):
            lhs, lhs_nodes = _resolve_constant_value(gm, value.args[0])
            rhs, rhs_nodes = _resolve_constant_value(gm, value.args[1])
            if lhs is None or rhs is None:
                return None, set()
            return lhs * rhs, lhs_nodes | rhs_nodes | {value}
        return None, set()

    if isinstance(value, (Tensor, int, float)):
        return value, set()

    return None, set()


def _match_quantized_functional_conv_weight_expr(
    gm: fx.GraphModule, value: Any
) -> tuple[Tensor, Tensor | float | int | None, set[fx.Node]] | None:
    """Match the quantized-weight expression used by customer functional convs."""
    if isinstance(value, fx.Node):
        if value.op == "get_attr":
            resolved = _resolve_attr_value(gm, str(value.target))
            if isinstance(resolved, Tensor):
                return resolved.detach().clone(), None, {value}
            return None

        if value.op == "call_method" and value.target == "to" and value.args:
            extracted = _match_quantized_functional_conv_weight_expr(gm, value.args[0])
            if extracted is None:
                return None
            weight, scale, nodes = extracted
            extra_nodes, ok = _resolve_to_aux_nodes(gm, value)
            if not ok or extra_nodes is None:
                return None
            return weight, scale, nodes | extra_nodes

        if value.op == "call_function" and value.target in (operator.mul, torch.mul):
            lhs = _match_quantized_functional_conv_weight_expr(gm, value.args[0])
            rhs = _match_quantized_functional_conv_weight_expr(gm, value.args[1])
            lhs_const, lhs_nodes = _resolve_constant_value(gm, value.args[0])
            rhs_const, rhs_nodes = _resolve_constant_value(gm, value.args[1])

            if lhs is not None and rhs_const is not None:
                weight, scale, nodes = lhs
                merged_scale = rhs_const if scale is None else scale * rhs_const
                return weight, merged_scale, nodes | rhs_nodes | {value}

            if rhs is not None and lhs_const is not None:
                weight, scale, nodes = rhs
                merged_scale = lhs_const if scale is None else scale * lhs_const
                return weight, merged_scale, nodes | lhs_nodes | {value}

    return None


def _infer_functional_conv_channels_and_kernel(
    weight: Tensor, groups: int, spatial_ndim: int
) -> tuple[int, int, tuple[int, ...]]:
    raw_weight = weight.detach()
    kernel_size = tuple(int(v) for v in raw_weight.shape[-spatial_ndim:])
    in_channels = int(raw_weight.shape[1]) * groups
    out_channels = int(raw_weight.shape[0])
    return in_channels, out_channels, kernel_size


def _attach_functional_conv_metadata(
    conv: nn.Conv1d | nn.Conv2d,
    weight: Tensor,
    weight_scale: Tensor | float | int | None,
    weight_zero_point: Tensor | int | None,
) -> None:
    raw_weight = weight.detach().clone()
    scale = torch.as_tensor(
        1.0 if weight_scale is None else weight_scale, dtype=torch.float32
    ).detach()
    zero_point = torch.as_tensor(
        0 if weight_zero_point is None else weight_zero_point, dtype=torch.int32
    ).detach()

    conv.register_buffer("raw_weight", raw_weight)
    conv.register_buffer("scale", scale)
    conv.register_buffer("zero_point", zero_point)


def _build_functional_conv_module(
    spec: _FunctionalConvLoweringSpec,
    weight: Tensor,
    bias: Tensor | None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
    weight_scale: Tensor | float | int | None = None,
    weight_zero_point: Tensor | int | None = None,
) -> nn.Conv1d | nn.Conv2d:
    in_channels, out_channels, kernel_size = _infer_functional_conv_channels_and_kernel(
        weight, groups, spatial_ndim=spec.spatial_ndim
    )

    module_cls = nn.Conv1d if spec.spatial_ndim == 1 else nn.Conv2d
    conv = module_cls(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias is not None,
    )

    with torch.no_grad():
        conv.weight.copy_(weight.detach().to(conv.weight.dtype))
        conv.weight.requires_grad_(False)
        if bias is not None and conv.bias is not None:
            conv.bias.copy_(bias.detach().to(conv.bias.dtype))
            conv.bias.requires_grad_(False)

    _attach_functional_conv_metadata(conv, weight, weight_scale, weight_zero_point)
    return conv


def extract_functional_conv_spec(
    gm: fx.GraphModule, node: fx.Node
) -> ConvSpec | None:
    """Extract a normalized conv spec from a function-form conv FX node."""
    spec = _get_functional_conv_spec(node.target)
    if node.op != "call_function" or spec is None:
        return None

    normalized_kwargs = _get_normalized_call_kwargs(node, gm)
    if normalized_kwargs is not None:
        input_arg = normalized_kwargs.get("input")
        weight_arg = normalized_kwargs.get("weight")
        bias_arg = normalized_kwargs.get("bias")
        stride = normalized_kwargs.get("stride", 1)
        padding = normalized_kwargs.get("padding", 0)
        dilation = normalized_kwargs.get("dilation", 1)
        groups = normalized_kwargs.get("groups", 1)
    else:
        input_arg = _get_call_arg(node, 0, "input")
        weight_arg = _get_call_arg(node, 1, "weight")
        bias_arg = _get_call_arg(node, 2, "bias")
        stride = _get_call_arg(node, 3, "stride", 1)
        padding = _get_call_arg(node, 4, "padding", 0)
        dilation = _get_call_arg(node, 5, "dilation", 1)
        groups = _get_call_arg(node, 6, "groups", 1)

    if not isinstance(input_arg, fx.Node) or not isinstance(groups, int):
        return None

    weight_spec = _match_quantized_functional_conv_weight_expr(gm, weight_arg)
    if weight_spec is None:
        return None

    weight, weight_scale, aux_nodes = weight_spec
    bias_value, bias_nodes = _resolve_constant_value(gm, bias_arg)
    if bias_value is not None and not isinstance(bias_value, Tensor):
        return None

    module = _build_functional_conv_module(
        spec,
        weight,
        bias_value,
        stride,
        padding,
        dilation,
        groups,
        weight_scale,
    )
    return ConvSpec(input_arg, module, frozenset(aux_nodes | bias_nodes))


def build_conv_ir_node(spec: ConvSpec) -> tuple[StandaloneCompOp, tuple[fx.Node, ...]]:
    return StandaloneCompOp(comp=spec.module), (spec.input_node,)
