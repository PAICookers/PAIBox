"""Conv-family lowering helpers for both module-form and function-form convs."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, fx, nn
from torch.nn.modules.utils import _pair, _single

from ..ir.op_node import StandaloneCompOp
from .fx_utils import get_call_arg

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


def extract_module_conv_spec(node: fx.Node, module: nn.Module) -> ConvSpec | None:
    if not isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return None
    if not node.args or not isinstance(node.args[0], fx.Node):
        return None
    return ConvSpec(node.args[0], module)


def _resolve_attr_value(gm: fx.GraphModule, target: str) -> Any:
    value: Any = gm
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


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


def _is_dtype_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "dtype"
    )


def _resolve_supported_to_call(
    gm: fx.GraphModule, node: fx.Node
) -> tuple[Any | None, set[fx.Node]]:
    resolved, aux_nodes = _resolve_static_value(gm, node.args[0])
    if resolved is None:
        return None, set()

    to_args = node.args[1:]
    if not any(isinstance(arg, fx.Node) for arg in to_args) and not any(
        isinstance(arg, fx.Node) for arg in node.kwargs.values()
    ):
        if torch.is_tensor(resolved):
            converted = resolved.to(*to_args, **node.kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
            return converted, aux_nodes | {node}
        return resolved, aux_nodes | {node}

    # Keep function-form conv lowering intentionally narrow. We support only the
    # common deploy wrapper pattern `tensor.to(x.dtype)`, and reject mixed
    # static/dynamic `.to(...)` forms rather than trying to partially emulate
    # every PyTorch cast combination.
    if (
        torch.is_tensor(resolved)
        and len(to_args) == 1
        and not node.kwargs
        and isinstance(to_args[0], fx.Node)
        and _is_dtype_getattr(to_args[0])
    ):
        return resolved, aux_nodes | {node, to_args[0]}

    return None, set()


def _resolve_static_value(
    gm: fx.GraphModule, value: Any
) -> tuple[Any | None, set[fx.Node]]:
    # Keep this helper narrow: function-form conv lowering accepts directly
    # resolvable tensor values plus simple canonical tensor reshaping/casting.
    # Quantization or custom weight expressions belong in register_module(...)
    # converters, where the user states the intended canonical module.
    if isinstance(value, fx.Node):
        if value.op == "get_attr":
            resolved = _resolve_attr_value(gm, str(value.target))
            if torch.is_tensor(resolved):
                return resolved.detach().clone(), {value}
            return resolved, {value}
        if value.op == "call_method" and value.target == "to" and value.args:
            return _resolve_supported_to_call(gm, value)
        if (
            value.op == "call_method"
            and value.target in {"view", "reshape"}
            and value.args
        ):
            resolved, aux_nodes = _resolve_static_value(gm, value.args[0])
            if resolved is None or not torch.is_tensor(resolved):
                return None, set()
            if any(isinstance(arg, fx.Node) for arg in value.args[1:]) or any(
                isinstance(arg, fx.Node) for arg in value.kwargs.values()
            ):
                return None, set()

            tensor_method = getattr(resolved, str(value.target))
            return (
                tensor_method(*value.args[1:], **value.kwargs),
                aux_nodes | {value},
            )
        return None, set()

    if isinstance(value, (Tensor, int, float)):
        if torch.is_tensor(value):
            return value.detach().clone(), set()
        return value, set()

    return None, set()


def _resolve_static_tensor_value(
    gm: fx.GraphModule, value: Any
) -> tuple[Tensor, set[fx.Node]] | None:
    resolved, aux_nodes = _resolve_static_value(gm, value)
    if torch.is_tensor(resolved):
        return resolved.detach().clone(), aux_nodes
    return None


def _infer_functional_conv_channels(weight: Tensor, groups: int) -> tuple[int, int]:
    in_channels = int(weight.shape[1]) * groups
    out_channels = int(weight.shape[0])
    return in_channels, out_channels


def _build_functional_conv_module(
    spec: _FunctionalConvLoweringSpec,
    weight: Tensor,
    bias: Tensor | None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
) -> nn.Conv1d | nn.Conv2d | None:
    in_channels, out_channels = _infer_functional_conv_channels(weight, groups)

    if spec.spatial_ndim == 1:
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=int(weight.shape[-1]),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            groups=groups,
            bias=bias is not None,
        )
    elif spec.spatial_ndim == 2:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(weight.shape[-2]), int(weight.shape[-1])),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            groups=groups,
            bias=bias is not None,
        )
    else:
        return None

    with torch.no_grad():
        _ = conv.weight.copy_(weight.detach().to(conv.weight.dtype))
        if bias is not None and conv.bias is not None:
            _ = conv.bias.copy_(bias.detach().to(conv.bias.dtype))
    return conv


def extract_functional_conv_spec(gm: fx.GraphModule, node: fx.Node) -> ConvSpec | None:
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
        input_arg = get_call_arg(node, 0, "input")
        weight_arg = get_call_arg(node, 1, "weight")
        bias_arg = get_call_arg(node, 2, "bias")
        stride = get_call_arg(node, 3, "stride", 1)
        padding = get_call_arg(node, 4, "padding", 0)
        dilation = get_call_arg(node, 5, "dilation", 1)
        groups = get_call_arg(node, 6, "groups", 1)

    if not isinstance(input_arg, fx.Node) or not isinstance(groups, int):
        return None

    weight_spec = _resolve_static_tensor_value(gm, weight_arg)
    if weight_spec is None:
        return None

    weight, aux_nodes = weight_spec
    bias_value: Tensor | None = None
    bias_nodes: set[fx.Node] = set()
    if bias_arg is not None:
        bias_spec = _resolve_static_tensor_value(gm, bias_arg)
        if bias_spec is None:
            return None
        bias_value, bias_nodes = bias_spec

    module = _build_functional_conv_module(
        spec, weight, bias_value, stride, padding, dilation, groups
    )
    if module is None:
        return None
    return ConvSpec(input_arg, module, frozenset(aux_nodes | bias_nodes))


def build_conv_ir_node(spec: ConvSpec) -> tuple[StandaloneCompOp, tuple[fx.Node, ...]]:
    return StandaloneCompOp(comp=spec.module), (spec.input_node,)
