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

import math
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
from ..ir.reshape_semantics import RESHAPE_LEAF_MODULE_TYPES
from .conv_lowering import (
    build_conv_ir_node,
    extract_functional_conv_spec,
    extract_module_conv_spec,
)
from .dims_prop import DimsProp, DimsType
from .shape_analysis import (
    ReshapeSinkInfo,
    ShapeAnalysisResult,
    analyze_shape_helpers,
)

__all__ = ["torch_to_paiir", "register_neuron"]

ModuleMapper = dict[type[nn.Module], Callable[[nn.Module], OpNode]]
"""Module mapping type: ``nn.Module`` subclass -> converter function returning an :class:`OpNode`."""

# Modules that are kept in the FX graph but intentionally disappear from PAIIR
# data flow during lowering.
LOWERING_BYPASS_MODULE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)

# Modules that should remain leaf nodes during FX tracing so lowering can decide
# how to handle them later.
#
# - BatchNorm stays a true bypass at lowering time.
# - Flatten is preserved as a leaf for tracing, but lowering materializes it as
#   a ``ReshapeOp`` so graph-level simulation keeps the shape transition.
TRACE_LEAF_MODULE_TYPES = LOWERING_BYPASS_MODULE_TYPES + RESHAPE_LEAF_MODULE_TYPES

# Modules removed by _EraseModuleTransformer: each call_module node is replaced
# by its input, leaving no trace in the graph after dead-code elimination.
ERASE_MODULE_TYPES = (nn.Dropout, nn.Identity)

ADD_OPS = (operator.add, torch.add)
SUB_OPS = (operator.sub, torch.sub)
CAT_OPS = (torch.cat,)

# Known bypass targets (shape/dim ops that don't need IR nodes)
KNOWN_BYPASS_FUNCS = (operator.getitem,)
KNOWN_BYPASS_METHODS = ("size", "contiguous", "transpose", "permute")


def _map_comp(mod: nn.Module, **kwargs) -> OpNode:
    return StandaloneCompOp(comp=mod, **kwargs)


def _has_nonzero_padding(padding: Any) -> bool:
    if isinstance(padding, tuple):
        return any(int(p) != 0 for p in padding)
    return int(padding) != 0


def _unsupported_avgpool_description(mod: nn.Module) -> str | None:
    if not isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d)):
        return None

    if mod.count_include_pad is False and _has_nonzero_padding(mod.padding):
        return (
            f"nn.Module '{type(mod).__name__}' with count_include_pad=False and "
            "padding>0"
        )

    return None


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


def _is_dtype_getattr(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) >= 2
        and node.args[1] == "dtype"
    )


def _make_fixed_shape_fn(
    target_shape: tuple[int, ...],
) -> Callable[[torch.Size], torch.Size]:
    return lambda _input_shape, target=target_shape: torch.Size(target)


def _normalize_dim(ndim: int, dim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def _flatten_output_shape(
    input_shape: torch.Size, start_dim: int, end_dim: int
) -> torch.Size:
    ndim = len(input_shape)
    if ndim == 0:
        return torch.Size((1,))

    start = _normalize_dim(ndim, start_dim)
    end = _normalize_dim(ndim, end_dim)
    if start < 0 or end < 0 or start >= ndim or end >= ndim or start > end:
        raise ValueError(
            f"invalid flatten range start_dim={start_dim}, end_dim={end_dim}, ndim={ndim}"
        )

    flat_size = math.prod(input_shape[start : end + 1])
    return torch.Size((*input_shape[:start], flat_size, *input_shape[end + 1 :]))


def _make_flatten_shape_fn(
    start_dim: int = 0, end_dim: int = -1
) -> Callable[[torch.Size], torch.Size]:
    return lambda input_shape, s=start_dim, e=end_dim: _flatten_output_shape(
        input_shape, s, e
    )


def _build_reshape_op(
    output_shape: tuple[int, ...],
    shape_fn: Callable[[torch.Size], torch.Size] | None = None,
) -> ReshapeOp | None:
    """Create a ``ReshapeOp`` from analyzed shape metadata or a shape function."""
    if shape_fn is not None:
        return ReshapeOp(shape_fn)

    if output_shape:
        return ReshapeOp(shape_fn=_make_fixed_shape_fn(output_shape))

    return None


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
                        const_value=(
                            const_value.detach().clone()
                            if isinstance(const_value, Tensor)
                            else const_value
                        ),
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
                    const_value=(
                        raw_value.detach().clone()
                        if isinstance(raw_value, Tensor)
                        else raw_value
                    ),
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


def _build_flatten_ir_node(
    data_input: fx.Node, start_dim: int, end_dim: int
) -> tuple[ReshapeOp, tuple[fx.Node, ...]] | None:
    """Build the normalized PAIIR form for any flatten-style operation."""
    return (
        ReshapeOp(shape_fn=_make_flatten_shape_fn(start_dim, end_dim)),
        (data_input,),
    )


def _build_reshape_like_ir_node(
    sink_info: ReshapeSinkInfo,
) -> tuple[ReshapeOp, tuple[fx.Node, ...]] | None:
    """Lower one analyzed reshape sink to ``ReshapeOp`` and its data input."""
    if sink_info.kind == "flatten":
        return _build_flatten_ir_node(
            sink_info.data_input, sink_info.start_dim, sink_info.end_dim
        )

    ir_node = _build_reshape_op(sink_info.output_shape)
    if ir_node is None:
        return None
    return ir_node, (sink_info.data_input,)


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

    # Leaf types = module_map keys + modules that need special post-trace handling.
    leaf_types = tuple(full_map.keys()) + TRACE_LEAF_MODULE_TYPES
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


def _is_lowering_bypass_module(mod: nn.Module) -> bool:
    """Return whether *mod* should be elided during PAIIR lowering."""
    return isinstance(mod, LOWERING_BYPASS_MODULE_TYPES)


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
    shape_analysis: ShapeAnalysisResult | None = None
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

    if input_nodes_override is not None:
        # Persist the normalized data-input view for the later edge-wiring pass.
        # Without this, shape-only FX operands such as ``view_as(ref)`` still
        # appear in ``all_input_nodes`` and can be mis-wired as real data preds.
        ctx.input_nodes_overrides[fx_node] = input_nodes_override

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
    """Pre-mark reshape-size helper nodes so they never enter PAIIR data flow."""
    analysis = analyze_shape_helpers(gm)
    ctx.shape_analysis = analysis
    ctx.aux_bypass_nodes |= analysis.aux_nodes


def _get_reshape_sink_info(
    ctx: _LoweringContext, node: fx.Node
) -> ReshapeSinkInfo | None:
    analysis = ctx.shape_analysis
    if analysis is None:
        return None
    return analysis.sink_for(node)


def _apply_functional_conv_rule(gm: fx.GraphModule, ctx: _LoweringContext) -> None:
    # Pre-identify function-form conv nodes and the helper nodes that build
    # their weight expressions, so lowering can materialize the conv itself
    # while edge wiring ignores these helper nodes as real data producers.
    for node in gm.graph.nodes:
        spec = extract_functional_conv_spec(gm, node)
        if spec is None:
            continue

        ir_node, input_override = build_conv_ir_node(spec)
        ctx.prebuilt_ir_nodes[node] = ir_node
        ctx.input_nodes_overrides[node] = input_override
        ctx.aux_bypass_nodes |= set(spec.aux_nodes)


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

    conv_spec = extract_module_conv_spec(node, torch_module)
    if conv_spec is not None:
        ir_node, conv_input_override = build_conv_ir_node(conv_spec)
        _register_ir_node(
            paiir_graph,
            ctx,
            node,
            ir_node,
            input_nodes_override=conv_input_override,
        )
        return True

    sink_info = _get_reshape_sink_info(ctx, node)
    if sink_info is not None:
        built = _build_reshape_like_ir_node(sink_info)
        if built is None:
            _mark_unsupported(
                ctx,
                node,
                f"nn.Module '{type(torch_module).__name__}' with unsupported reshape arguments",
                strict,
            )
            return True

        ir_node, reshape_override = built
        _register_ir_node(
            paiir_graph,
            ctx,
            node,
            ir_node,
            input_nodes_override=reshape_override,
        )
        return True

    if _is_lowering_bypass_module(torch_module):
        ctx.bypass_nodes.add(node)
        return True

    if (
        unsupported_avgpool := _unsupported_avgpool_description(torch_module)
    ) is not None:
        _mark_unsupported(ctx, node, unsupported_avgpool, strict)
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

    sink_info = _get_reshape_sink_info(ctx, node)
    if sink_info is not None:
        built = _build_reshape_like_ir_node(sink_info)
        if built is None:
            func_name = getattr(node.target, "__name__", str(node.target))
            _mark_unsupported(
                ctx,
                node,
                f"function '{func_name}' with unsupported reshape arguments",
                strict,
            )
            return True

        ir_node, input_override = built
        _register_ir_node(
            paiir_graph,
            ctx,
            node,
            ir_node,
            input_nodes_override=input_override,
        )
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

    sink_info = _get_reshape_sink_info(ctx, node)
    if sink_info is not None:
        built = _build_reshape_like_ir_node(sink_info)
        if built is None:
            _mark_unsupported(
                ctx,
                node,
                f"method '{node.target}' with unsupported reshape arguments",
                strict,
            )
            return True

        ir_node, input_override = built
        _register_ir_node(
            paiir_graph,
            ctx,
            node,
            ir_node,
            input_nodes_override=input_override,
        )
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
