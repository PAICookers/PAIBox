"""Convert a PyTorch model to a :class:`PAIIRGraph`.

Pipeline:

1. FX symbolic trace with registered module types + bypass types as leaf modules.
2. ``_EraseModuleTransformer`` removes Dropout/Identity nodes from the graph.
3. ``ShapeProp`` for tensor shape propagation (when *sample_inputs* given).
4. ``DimsProp`` for axis ordering propagation (detects transpose/permute).
5. 1:1 node mapping to PAIIR nodes (no fusion at this stage).

Example::

    from paibox.paiir import torch_to_paiir, register_neuron, ANNNodeV25

    # Register a custom neuron type before conversion
    register_neuron(
        MyNeuron,
        converter=lambda mod: ANNNodeV25(LutCustom(...)),
    )

    # With shape inference
    graph = torch_to_paiir(model, torch.randn(1, 3, 32, 32))

    # Without shape inference
    graph = torch_to_paiir(model)
"""

import operator
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from spikingjelly.activation_based import neuron
from torch import Tensor, fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from ..exceptions import UnsupportedOpError, UnsupportedOpWarning
from ..ir.add_ops import AddOperandKind, AddOperandSpec, GeneralAddOp
from ..ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from ..ir.graph import PAIIRGraph
from ..ir.ir_base import InputNode, OutputNode
from ..ir.lut_activation import LutActivation, LutReLU, LutSigmoid, LutSoftsign, LutTanh
from ..ir.op_node import ConcatOp, OpNode, ReshapeOp, StandaloneActOp, StandaloneCompOp
from .dims_prop import DimsProp, DimsType

__all__ = ["torch_to_paiir", "register_neuron"]

ModuleMapper = dict[type[nn.Module], Callable[[nn.Module], OpNode]]
"""Module mapping type: ``nn.Module`` subclass -> converter function returning an :class:`OpNode`."""

# Modules that are kept as leaf nodes but produce no PAIIR node (bypass).
# Note: Flatten is a bypass - it changes tensor shape but is handled via shape propagation.
BYPASS_MODULE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.Flatten)

# Modules removed by _EraseModuleTransformer: each call_module node is replaced
# by its input, leaving no trace in the graph after dead-code elimination.
ERASE_MODULE_TYPES = (nn.Dropout, nn.Identity)

ADD_OPS = (operator.add, torch.add)
SUB_OPS = (operator.sub, torch.sub)
CAT_OPS = (torch.cat,)

# Known bypass targets (shape/dim ops that don't need IR nodes)
KNOWN_BYPASS_FUNCS = (operator.getitem,)
KNOWN_BYPASS_METHODS = (
    "flatten",
    "reshape",
    "view",
    "contiguous",
    "transpose",
    "permute",
)


class FunctionalConv2d(nn.Conv2d):
    """Thin ``nn.Conv2d`` adapter for FX graphs that materialize ``torch.conv2d``.

    This adapter is intentionally graph-first: it preserves the original
    exported quantized weight tensor and convolution hyperparameters so PAIIR
    can reconstruct the FX graph semantics accurately. Quantization parameters
    are kept only as optional metadata; exact dequantized simulation is not the
    priority here because the chip backend cannot execute dequantization.

    Preserved metadata:

    - ``raw_weight``: exported graph-side quantized weight tensor
    - ``scale``: optional quantization scale observed in the FX graph
    - ``zero_point``: optional quantization zero-point, defaulting to ``0``
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        weight_scale: Tensor | float | int | None = None,
        weight_zero_point: Tensor | int | None = None,
    ) -> None:
        raw_weight = weight.detach().clone()
        scale = torch.as_tensor(
            1.0 if weight_scale is None else weight_scale, dtype=torch.float32
        ).detach()
        zero_point = torch.as_tensor(
            0 if weight_zero_point is None else weight_zero_point, dtype=torch.int32
        ).detach()

        kernel_size = tuple(int(v) for v in raw_weight.shape[-2:])
        in_channels = int(raw_weight.shape[1]) * groups
        out_channels = int(raw_weight.shape[0])

        super().__init__(
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
            # Keep the native Conv2d surface available for downstream code, but
            # do not treat this float-cast compatibility weight as authoritative
            # graph information.
            self.weight.copy_(raw_weight.to(self.weight.dtype))
            self.weight.requires_grad_(False)
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.detach().to(self.bias.dtype))
                self.bias.requires_grad_(False)

        self.register_buffer("raw_weight", raw_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    @property
    def weight_scale(self) -> Tensor:
        """Compatibility alias for customer exports that use ``weight_scale``."""
        return self.scale


def _map_comp(mod: nn.Module, **kwargs) -> OpNode:
    return StandaloneCompOp(comp=mod, **kwargs)


def _map_sj_ifnode(mod: nn.Module, **kwargs) -> OpNode:
    assert isinstance(mod, neuron.IFNode)
    return StandaloneActOp(act=IFNodeV25(mod.v_threshold, mod.v_reset, **kwargs))


def _map_sj_lifnode(mod: nn.Module, **kwargs) -> OpNode:
    assert isinstance(mod, neuron.LIFNode)
    return StandaloneActOp(
        act=LIFNodeV25(mod.tau, mod.decay_input, mod.v_threshold, mod.v_reset, **kwargs)
    )


def _map_core_neuron(mod: nn.Module, **kwargs) -> OpNode:
    assert isinstance(mod, CoreNeuronV25)
    return StandaloneActOp(act=mod)


def _map_lut_activation(mod: nn.Module, **kwargs) -> OpNode:
    assert isinstance(mod, LutActivation)
    return StandaloneActOp(act=ANNNodeV25(mod))


_DEFAULT_MODULE_MAP: ModuleMapper = {
    # Compute ops
    nn.Conv1d: _map_comp,
    nn.Conv2d: _map_comp,
    nn.Linear: _map_comp,
    nn.MaxPool1d: _map_comp,
    nn.MaxPool2d: _map_comp,
    nn.AvgPool1d: _map_comp,
    nn.AvgPool2d: _map_comp,
    # SJ neuron -> chip-accurate neuron (IFNodeV25/LIFNodeV25 are CoreNeuronV25 subclasses)
    neuron.IFNode: _map_sj_ifnode,
    neuron.LIFNode: _map_sj_lifnode,
    # Standard activations -> ANNNodeV25 with LUT
    nn.ReLU: lambda _: StandaloneActOp(act=ANNNodeV25(LutReLU())),
    nn.Sigmoid: lambda _: StandaloneActOp(act=ANNNodeV25(LutSigmoid())),
    nn.Tanh: lambda _: StandaloneActOp(act=ANNNodeV25(LutTanh())),
    nn.Softsign: lambda _: StandaloneActOp(act=ANNNodeV25(LutSoftsign())),
    # PAIIR activation ops (pass through)
    CoreNeuronV25: _map_core_neuron,
    LutActivation: _map_lut_activation,
}


def _propagate_shapes(gm: fx.GraphModule, *inputs: Tensor) -> None:
    ShapeProp(gm).propagate(*inputs)


def _propagate_dims(gm: fx.GraphModule) -> None:
    """Propagate axis ordering through an FX graph.

    After propagation, ``node.meta["dims"]`` contains the output axis
    ordering as a ``tuple[int, ...]``.  Identity ordering ``(0, 1, ...)``
    means no transpose/permute has been applied.

    Requires ``ShapeProp`` to have been run first (needs ``tensor_meta``
    for determining the number of dimensions).
    """
    DimsProp().propagate(gm)


def _is_conv2d_target(target: Any) -> bool:
    return getattr(target, "__name__", "") == "conv2d"


def _is_dtype_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "dtype"
    )


def _is_shape_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "shape"
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
        return list[type(value[0])] if value else list[Any]
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
    # These dtype/getattr/... nodes belong to the auxiliary weight expression
    # feeding a functional conv; they should not later appear as real data-flow
    # predecessors of the lowered conv node.
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


def _collect_shape_aux_nodes(gm: fx.GraphModule) -> set[fx.Node]:
    """Collect FX nodes that participate only in tensor-shape arithmetic.

    These nodes are needed to validate reshape/view argument expressions but
    must not reappear as tensor predecessors when wiring the PAIIR data-flow
    graph. Otherwise shape queries such as ``x.shape[...]`` leak the original
    input tensor back into unrelated compute/activation nodes.
    """
    shape_aux_nodes = {node for node in gm.graph.nodes if _is_shape_getattr(node)}
    changed = True
    while changed:
        changed = False
        for node in gm.graph.nodes:
            if node in shape_aux_nodes:
                continue

            if node.op == "call_function" and node.target is operator.getitem:
                if any(inp in shape_aux_nodes for inp in node.all_input_nodes):
                    shape_aux_nodes.add(node)
                    changed = True
                    continue

            if node.op == "call_function" and node.target is operator.floordiv:
                if node.all_input_nodes and all(
                    inp in shape_aux_nodes for inp in node.all_input_nodes
                ):
                    shape_aux_nodes.add(node)
                    changed = True

    return shape_aux_nodes


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


def _resolve_add_coefficient(gm: fx.GraphModule, value: Any) -> int | None:
    def _normalize_scalar(scalar: Any) -> int | None:
        if isinstance(scalar, bool):
            return int(scalar)
        if isinstance(scalar, int):
            return scalar
        if isinstance(scalar, float) and scalar.is_integer():
            return int(scalar)
        return None

    normalized = _normalize_scalar(value)
    if normalized is not None:
        return normalized

    resolved, _ = _resolve_constant_value(gm, value)
    normalized = _normalize_scalar(resolved)
    if normalized is not None:
        return normalized
    if isinstance(resolved, Tensor) and resolved.numel() == 1:
        scalar = resolved.detach().cpu().item()
        normalized = _normalize_scalar(scalar)
        if normalized is not None:
            return normalized
    return None


def _build_general_add_node(
    gm: fx.GraphModule, node: fx.Node, *, subtract: bool
) -> tuple[GeneralAddOp, tuple[fx.Node, ...]] | None:
    if node.op == "call_method":
        lhs_raw = _get_call_arg(node, 0, "input")
        rhs_raw = _get_call_arg(node, 1, "other")
        alpha_raw = node.kwargs.get("alpha", 1)
    else:
        normalized_kwargs = _get_normalized_call_kwargs(node, gm)
        if (
            normalized_kwargs is not None
            and {"input", "other"} <= normalized_kwargs.keys()
        ):
            lhs_raw = normalized_kwargs.get("input")
            rhs_raw = normalized_kwargs.get("other")
            alpha_raw = normalized_kwargs.get("alpha", 1)
        else:
            lhs_raw = _get_call_arg(node, 0, "input")
            rhs_raw = _get_call_arg(node, 1, "other")
            alpha_raw = _get_call_arg(node, 2, "alpha", 1)

    alpha = _resolve_add_coefficient(gm, alpha_raw)
    if alpha is None:
        return None

    operand_specs: list[AddOperandSpec] = []
    tensor_inputs: list[fx.Node] = []

    def append_operand(raw_value: Any, coeff: int) -> bool:
        if isinstance(raw_value, fx.Node):
            const_value, _ = _resolve_constant_value(gm, raw_value)
            if const_value is not None and isinstance(
                const_value, (Tensor, int, float)
            ):
                operand_specs.append(
                    AddOperandSpec(
                        coeff,
                        AddOperandKind.CONST,
                        const_value=const_value.detach().clone()
                        if isinstance(const_value, Tensor)
                        else const_value,
                    )
                )
                return True

            tensor_port = len(tensor_inputs)
            tensor_inputs.append(raw_value)
            operand_specs.append(
                AddOperandSpec(coeff, AddOperandKind.TENSOR, tensor_port)
            )
            return True

        if isinstance(raw_value, (Tensor, int, float)):
            operand_specs.append(
                AddOperandSpec(
                    coeff,
                    AddOperandKind.CONST,
                    const_value=raw_value.detach().clone()
                    if isinstance(raw_value, Tensor)
                    else raw_value,
                )
            )
            return True

        return False

    if not append_operand(lhs_raw, 1):
        return None
    rhs_coeff = -alpha if subtract else alpha
    if not append_operand(rhs_raw, rhs_coeff):
        return None

    return GeneralAddOp(operand_specs), tuple(tensor_inputs)


def _lower_general_add_ir(
    gm: fx.GraphModule,
    paiir_graph: PAIIRGraph,
    node: fx.Node,
    ctx: "_LoweringContext",
    strict: bool,
    *,
    subtract: bool,
    description: str,
) -> None:
    built = _build_general_add_node(gm, node, subtract=subtract)
    if built is None:
        _mark_unsupported(ctx, node, description, strict)
        return

    ir_node, input_override = built
    _register_ir_node(
        paiir_graph, ctx, node, ir_node, input_nodes_override=input_override
    )


def _match_quantized_conv_weight_expr(
    gm: fx.GraphModule, value: Any
) -> tuple[Tensor, Tensor | float | int | None, set[fx.Node]] | None:
    """Match the quantized-weight expression used by customer functional convs.

    Supported forms are intentionally narrow and mirror the exported
    ``QuantizedConv2d.forward()`` pattern:

    - ``get_attr(weight_int8)``
    - ``weight_expr.to(x.dtype)``
    - ``weight_expr * constant`` or ``constant * weight_expr``

    Returns the raw graph-side weight tensor, an optional scale-like metadata
    factor collected from ``mul(...)``, and the FX nodes that belong to this
    auxiliary weight-construction expression. This is a pattern matcher for the
    current functional-conv export style, not a general FX expression evaluator.
    """
    if isinstance(value, fx.Node):
        if value.op == "get_attr":
            resolved = _resolve_attr_value(gm, str(value.target))
            if isinstance(resolved, Tensor):
                return resolved.detach().clone(), None, {value}
            return None

        if value.op == "call_method" and value.target == "to" and value.args:
            extracted = _match_quantized_conv_weight_expr(gm, value.args[0])
            if extracted is None:
                return None
            weight, scale, nodes = extracted
            extra_nodes, ok = _resolve_to_aux_nodes(gm, value)
            if not ok or extra_nodes is None:
                return None
            return weight, scale, nodes | extra_nodes

        if value.op == "call_function" and value.target in (operator.mul, torch.mul):
            lhs = _match_quantized_conv_weight_expr(gm, value.args[0])
            rhs = _match_quantized_conv_weight_expr(gm, value.args[1])
            lhs_const, lhs_nodes = _resolve_constant_value(gm, value.args[0])
            rhs_const, rhs_nodes = _resolve_constant_value(gm, value.args[1])

            # Only accept weight_expr * constant. Ordinary tensor-tensor mul
            # should stay visible as a separate FX operator instead of being
            # absorbed into a conv-weight pattern.
            if lhs is not None and rhs_const is not None:
                weight, scale, nodes = lhs
                merged_scale = rhs_const if scale is None else scale * rhs_const
                return weight, merged_scale, nodes | rhs_nodes | {value}

            if rhs is not None and lhs_const is not None:
                weight, scale, nodes = rhs
                merged_scale = lhs_const if scale is None else scale * lhs_const
                return weight, merged_scale, nodes | lhs_nodes | {value}

    return None


def _build_functional_conv2d_node(
    gm: fx.GraphModule,
    node: fx.Node,
) -> tuple[OpNode, set[fx.Node]] | None:
    """Build a PAIIR compute node for a supported function-form ``conv2d``."""
    if node.op != "call_function" or not _is_conv2d_target(node.target):
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

    weight_spec = _match_quantized_conv_weight_expr(gm, weight_arg)
    if weight_spec is None:
        return None

    weight, weight_scale, aux_nodes = weight_spec
    bias_value, bias_nodes = _resolve_constant_value(gm, bias_arg)
    if bias_value is not None and not isinstance(bias_value, Tensor):
        return None

    comp = FunctionalConv2d(
        weight, bias_value, stride, padding, dilation, groups, weight_scale
    )
    ir_node = StandaloneCompOp(comp=comp)
    return ir_node, aux_nodes | bias_nodes


def _get_output_shape(node: fx.Node) -> tuple[int, ...]:
    """Extract output shape from an FX node's meta."""
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return ()
    if hasattr(meta, "shape"):
        return tuple(meta.shape)
    return ()


def _get_input_shapes(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[tuple[int, ...]]:
    """Extract input shapes from an FX node's predecessor meta."""
    source_nodes = (
        input_nodes if input_nodes is not None else tuple(node.all_input_nodes)
    )
    return [_get_output_shape(inp) for inp in source_nodes]


def _get_output_dims(node: fx.Node) -> DimsType:
    """Get output dims from an FX node's meta."""
    return node.meta.get(DimsProp.KEY, ())


def _get_input_dims(
    node: fx.Node, input_nodes: tuple[fx.Node, ...] | None = None
) -> list[DimsType]:
    """Get input dims from an FX node's predecessor meta."""
    source_nodes = (
        input_nodes if input_nodes is not None else tuple(node.all_input_nodes)
    )
    return [_get_output_dims(inp) for inp in source_nodes]


class _PAIIRTracer(fx.Tracer):
    """Custom FX Tracer that treats registered, bypass, and erase types as
    leaf modules so they are not traced into.
    """

    def __init__(
        self, custom_leaf_modules: tuple[type[nn.Module], ...] = (), **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.custom_leaf_modules = custom_leaf_modules

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if self.custom_leaf_modules and isinstance(m, self.custom_leaf_modules):
            return True
        if getattr(m, "_is_leaf_module", False):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class _EraseModuleTransformer(fx.Transformer):
    """Remove ``ERASE_MODULE_TYPES`` (Dropout, Identity) nodes from a traced graph.

    Each matching ``call_module`` node is replaced by its single input node.
    Dead-code elimination is run automatically so no orphaned nodes remain.
    """

    def call_module(self, target: str, args: tuple, kwargs: dict):
        if isinstance(self.submodules[target], ERASE_MODULE_TYPES):
            return args[0]
        return super().call_module(target, args, kwargs)

    def transform(self) -> fx.GraphModule:
        gm = super().transform()
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm


def register_neuron(
    module_type: type[nn.Module],
    converter: Callable[[nn.Module], CoreNeuronV25],
) -> None:
    """Register a custom neuron type for PAIIR conversion.

    The converter receives the PyTorch module and must return a
    :class:`CoreNeuronV25` with appropriate chip parameters.
    """
    if module_type in _DEFAULT_MODULE_MAP:
        raise ValueError(
            f"Module type {module_type} is already registered. "
            f"Overriding existing registrations is not allowed."
        )

    _DEFAULT_MODULE_MAP[module_type] = lambda mod: StandaloneActOp(act=converter(mod))


def torch_to_paiir(
    model: nn.Module,
    *sample_inputs: Tensor,
    concrete_args: dict[str, Any] | None = None,
    strict: bool = True,
) -> PAIIRGraph:
    """Convert a PyTorch model to a :class:`PAIIRGraph`.

    Performs 1:1 node mapping only — no fusion.  Use
    :func:`fuse_to_offline_cores` afterwards to fuse nodes into
    offline-core units.

    Custom neuron types can be registered via :func:`register_neuron`
    before calling this function.

    Args:
        model: The PyTorch model.
        *sample_inputs: Example input tensor(s) for shape and dims inference.
            Omit entirely to skip shape propagation.
        concrete_args: Concrete arguments forwarded to ``fx.Tracer.trace``.
        strict: If True (default), raise :exc:`~paibox.paiir.exceptions.UnsupportedOpError`
            for unsupported ops. If False, emit a warning and bypass
            unsupported ops.

    Returns:
        A :class:`PAIIRGraph` with atomic (unfused) nodes.

    Raises:
        UnsupportedOpError: If ``strict=True`` and an unsupported operator
            is encountered. See :exc:`~paibox.paiir.exceptions.UnsupportedOpError`.
    """
    model.eval()
    full_map = {**_DEFAULT_MODULE_MAP}

    # Edge case: root module itself is in module_map
    # FX trace always decomposes the root module. Wrap in Sequential
    # so it becomes a submodule and is treated as a leaf module.
    if type(model) in full_map:
        model = nn.Sequential(model)

    # Leaf types = module_map keys + bypass types
    leaf_types = tuple(full_map.keys()) + BYPASS_MODULE_TYPES
    tracer = _PAIIRTracer(custom_leaf_modules=leaf_types)
    traced = tracer.trace(model, concrete_args)
    gm = fx.GraphModule(tracer.root, traced)
    gm = _EraseModuleTransformer(gm).transform()

    if sample_inputs:
        for i, inp in enumerate(sample_inputs):
            if inp.shape[0] != 1:
                raise ValueError(
                    f"sample_inputs[{i}] has batch size {inp.shape[0]}, expected 1. "
                    f"Chip deployment processes one sample at a time."
                )
        _propagate_shapes(gm, *sample_inputs)
        _propagate_dims(gm)

    return _fx_graph_to_paiir(gm, full_map, strict)


def _is_bypass_module(mod: nn.Module) -> bool:
    return isinstance(mod, BYPASS_MODULE_TYPES)


def _fill_shape_dims(
    ir_node: OpNode,
    fx_node: fx.Node,
    *,
    input_nodes_override: tuple[fx.Node, ...] | None = None,
) -> None:
    """Copy shape and axis-ordering info from FX node meta into a PAIIR node."""
    ir_node.input_shapes = _get_input_shapes(fx_node, input_nodes_override)
    ir_node.output_shape = _get_output_shape(fx_node)
    ir_node.input_dims = _get_input_dims(fx_node, input_nodes_override)
    ir_node.output_dims = _get_output_dims(fx_node)


@dataclass
class _LoweringContext:
    """Shared lowering state for ``FX -> PAIIR`` conversion."""

    fx_to_ir: dict[str, str] = field(default_factory=dict)
    bypass_nodes: set[fx.Node] = field(default_factory=set)
    aux_bypass_nodes: set[fx.Node] = field(default_factory=set)
    ignored_nodes: set[fx.Node] = field(default_factory=set)
    unsupported_ops: list[tuple[str, str]] = field(default_factory=list)
    prebuilt_ir_nodes: dict[fx.Node, OpNode] = field(default_factory=dict)
    input_nodes_overrides: dict[fx.Node, tuple[fx.Node, ...]] = field(
        default_factory=dict
    )


def _iter_output_args(node: fx.Node) -> tuple[Any, ...]:
    out_args = (
        node.args[0] if isinstance(node.args[0], (tuple, list)) else [node.args[0]]
    )
    return tuple(out_args)


def _register_ir_node(
    paiir_graph: PAIIRGraph,
    ctx: _LoweringContext,
    fx_node: fx.Node,
    ir_node: OpNode,
    *,
    fill_meta: bool = True,
    input_nodes_override: tuple[fx.Node, ...] | None = None,
) -> None:
    """Register an IR node produced from an FX node.

    By default, the IR node inherits shape/dims metadata directly from the FX
    node via :func:`_fill_shape_dims`.

    ``fill_meta=False`` is kept as an explicit extension hook for future
    lowering paths where metadata should be populated later or from a source
    other than the current FX node. The current converter paths all use the
    default behavior.
    """
    if fill_meta:
        _fill_shape_dims(ir_node, fx_node, input_nodes_override=input_nodes_override)
    paiir_graph.add_node(ir_node)
    ctx.fx_to_ir[fx_node.name] = ir_node.name


def _mark_unsupported(
    ctx: _LoweringContext, node: fx.Node, description: str, strict: bool
) -> None:
    ctx.unsupported_ops.append((node.name, description))
    if strict:
        raise UnsupportedOpError(node.name, description)
    ctx.bypass_nodes.add(node)


def _apply_shape_aux_rule(gm: fx.GraphModule, ctx: _LoweringContext) -> None:
    ctx.aux_bypass_nodes |= _collect_shape_aux_nodes(gm)


def _apply_functional_conv_rule(gm: fx.GraphModule, ctx: _LoweringContext) -> None:
    # Pre-identify function-form conv2d nodes and the helper nodes that build
    # their weight expressions, so lowering can materialize the conv itself
    # while edge wiring ignores these helper nodes as real data producers.
    for node in gm.graph.nodes:
        built = _build_functional_conv2d_node(gm, node)
        if built is None:
            continue

        ir_node, aux_nodes = built
        ctx.prebuilt_ir_nodes[node] = ir_node
        ctx.aux_bypass_nodes |= aux_nodes


def _analyze_graph(gm: fx.GraphModule, ctx: _LoweringContext) -> None:
    """Run non-mutating lowering analysis rules."""
    _apply_shape_aux_rule(gm, ctx)
    _apply_functional_conv_rule(gm, ctx)


def _create_placeholder_node(
    paiir_graph: PAIIRGraph, ctx: _LoweringContext, node: fx.Node
) -> None:
    ir_node = InputNode(shape=_get_output_shape(node))
    paiir_graph.add_node(ir_node)
    ctx.fx_to_ir[node.name] = ir_node.name


def _create_output_nodes(
    paiir_graph: PAIIRGraph, ctx: _LoweringContext, node: fx.Node
) -> None:
    for i, arg in enumerate(_iter_output_args(node)):
        out_shape = _get_output_shape(arg) if isinstance(arg, fx.Node) else ()
        ir_node = OutputNode(shape=out_shape)
        paiir_graph.add_node(ir_node)
        ctx.fx_to_ir[f"{node.name}_{i}"] = ir_node.name


def _apply_module_lowering_rule(
    gm: fx.GraphModule,
    paiir_graph: PAIIRGraph,
    node: fx.Node,
    ctx: _LoweringContext,
    module_map: ModuleMapper,
    strict: bool,
) -> bool:
    if node.op != "call_module":
        return False

    torch_module = gm.get_submodule(str(node.target))
    input_override = ctx.input_nodes_overrides.get(node)

    if _is_bypass_module(torch_module):
        ctx.bypass_nodes.add(node)
        return True

    if (mod_type := type(torch_module)) in module_map:
        ir_node = module_map[mod_type](torch_module)
        if isinstance(ir_node, OpNode):
            _register_ir_node(
                paiir_graph, ctx, node, ir_node, input_nodes_override=input_override
            )
        else:
            ctx.bypass_nodes.add(node)
        return True

    mod_name = type(torch_module).__name__
    _mark_unsupported(ctx, node, f"nn.Module '{mod_name}'", strict)
    return True


def _apply_builtin_function_lowering_rule(
    gm: fx.GraphModule,
    paiir_graph: PAIIRGraph,
    node: fx.Node,
    ctx: _LoweringContext,
    strict: bool,
) -> bool:
    if node.op != "call_function":
        return False

    if node in ctx.aux_bypass_nodes:
        ctx.bypass_nodes.add(node)
        return True

    if node in ctx.prebuilt_ir_nodes:
        _register_ir_node(
            paiir_graph,
            ctx,
            node,
            ctx.prebuilt_ir_nodes[node],
            input_nodes_override=ctx.input_nodes_overrides.get(node),
        )
        return True

    if node.target in ADD_OPS or node.target in SUB_OPS:
        if node.target in ADD_OPS:
            subtract = False
            description = "add with unsupported alpha/operand form"
        else:
            subtract = True
            description = "sub with unsupported alpha/operand form"

        _lower_general_add_ir(
            gm,
            paiir_graph,
            node,
            ctx,
            strict,
            subtract=subtract,
            description=description,
        )
        return True

    if node.target in CAT_OPS:
        raw_dim = node.kwargs.get("dim", 0)
        assert isinstance(raw_dim, int), f"cat dim must be int, got {type(raw_dim)}"
        _register_ir_node(paiir_graph, ctx, node, ConcatOp(dim=raw_dim))
        return True

    if node.target in KNOWN_BYPASS_FUNCS:
        ctx.bypass_nodes.add(node)
        return True

    if callable(node.target) and getattr(node.target, "__name__", "").startswith(
        "_assert"
    ):
        ctx.ignored_nodes.add(node)
        return True

    func_name = getattr(node.target, "__name__", str(node.target))
    _mark_unsupported(ctx, node, f"function '{func_name}'", strict)
    return True


def _apply_builtin_method_lowering_rule(
    gm: fx.GraphModule,
    paiir_graph: PAIIRGraph,
    node: fx.Node,
    ctx: _LoweringContext,
    strict: bool,
) -> bool:
    if node.op != "call_method":
        return False

    if node in ctx.aux_bypass_nodes:
        ctx.bypass_nodes.add(node)
        return True

    if node.target == "add" or node.target == "sub":
        _lower_general_add_ir(
            gm,
            paiir_graph,
            node,
            ctx,
            strict,
            subtract=(node.target == "sub"),
            description=f"method '{node.target}' with unsupported alpha/operand form",
        )
        return True

    if node.target == "flatten":
        # flatten method - create ReshapeOp for simulation
        # On chip, flatten is implicit (no computation)
        _register_ir_node(paiir_graph, ctx, node, ReshapeOp())
        return True

    if node.target in KNOWN_BYPASS_METHODS:
        ctx.bypass_nodes.add(node)
        return True

    method_name = str(node.target)
    _mark_unsupported(ctx, node, f"method '{method_name}'", strict)
    return True


def _lower_graph(
    gm: fx.GraphModule,
    paiir_graph: PAIIRGraph,
    module_map: ModuleMapper,
    ctx: _LoweringContext,
    strict: bool,
) -> None:
    """Lower FX nodes into atomic PAIIR nodes without wiring edges yet."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            _create_placeholder_node(paiir_graph, ctx, node)
            continue

        if node.op == "output":
            _create_output_nodes(paiir_graph, ctx, node)
            continue

        if node in ctx.ignored_nodes or node in ctx.bypass_nodes:
            continue

        if _apply_module_lowering_rule(gm, paiir_graph, node, ctx, module_map, strict):
            continue

        if _apply_builtin_function_lowering_rule(gm, paiir_graph, node, ctx, strict):
            continue

        if _apply_builtin_method_lowering_rule(gm, paiir_graph, node, ctx, strict):
            continue

        if node.op == "get_attr":
            ctx.bypass_nodes.add(node)


def _emit_unsupported_warnings(ctx: _LoweringContext, strict: bool) -> None:
    if not strict and ctx.unsupported_ops:
        warnings.warn(UnsupportedOpWarning(ctx.unsupported_ops), stacklevel=2)


class _SourceResolver:
    """Resolve FX predecessors into real PAIIR source nodes."""

    def __init__(self, ctx: _LoweringContext) -> None:
        self.ctx = ctx

    def resolve(self, node: fx.Node) -> set[str]:
        if node in self.ctx.ignored_nodes:
            return set()

        if node in self.ctx.aux_bypass_nodes:
            # Weight-construction helpers such as dtype/get_attr/to/mul are not
            # tensor producers for the main graph. Recurse through them and the
            # functional conv picks up bogus duplicate predecessors.
            return set()

        if node in self.ctx.bypass_nodes:
            sources: set[str] = set()
            for inp in node.all_input_nodes:
                sources |= self.resolve(inp)
            return sources

        if node.name in self.ctx.fx_to_ir:
            return {self.ctx.fx_to_ir[node.name]}

        return set()


def _wire_graph(
    gm: fx.GraphModule, paiir_graph: PAIIRGraph, ctx: _LoweringContext
) -> None:
    resolver = _SourceResolver(ctx)

    for node in gm.graph.nodes:
        if node in ctx.ignored_nodes or node in ctx.bypass_nodes:
            continue

        if node.op == "output":
            for i, arg in enumerate(_iter_output_args(node)):
                dst_name = ctx.fx_to_ir[f"{node.name}_{i}"]
                if isinstance(arg, fx.Node):
                    for src in resolver.resolve(arg):
                        paiir_graph.add_edge(src, dst_name)
            continue

        if node.name not in ctx.fx_to_ir:
            continue

        dst_name = ctx.fx_to_ir[node.name]
        source_nodes = ctx.input_nodes_overrides.get(node, tuple(node.all_input_nodes))
        for port_idx, inp_node in enumerate(source_nodes):
            for src in resolver.resolve(inp_node):
                paiir_graph.add_edge(src, dst_name, dst_port=port_idx)


def _fx_graph_to_paiir(
    gm: fx.GraphModule, module_map: ModuleMapper, strict: bool = False
) -> PAIIRGraph:
    """Convert an FX graph to a :class:`PAIIRGraph` (1:1 mapping, no fusion).

    Two-pass approach: first pass creates nodes, second pass wires edges while
    skipping bypass and ignored nodes.

    Raises:
        UnsupportedOpError: If ``strict=True`` and an unsupported operator
            is encountered. See :exc:`~paibox.paiir.exceptions.UnsupportedOpError`.
    """
    paiir_graph = PAIIRGraph(type(gm).__name__)
    ctx = _LoweringContext()

    _analyze_graph(gm, ctx)
    _lower_graph(gm, paiir_graph, module_map, ctx, strict)
    _emit_unsupported_warnings(ctx, strict)
    _wire_graph(gm, paiir_graph, ctx)

    paiir_graph.eval()
    return paiir_graph
