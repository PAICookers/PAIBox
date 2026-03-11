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
from typing import Any

import torch
from spikingjelly.activation_based import neuron
from torch import Tensor, fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from .core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from .dims_prop import DimsProp, DimsType
from .exceptions import UnsupportedOpError, UnsupportedOpWarning
from .graph import PAIIRGraph
from .ir_base import InputNode, OutputNode
from .lut_activation import LutActivation, LutReLU, LutSigmoid, LutSoftsign, LutTanh
from .op_node import AddOp, ConcatOp, OpNode, StandaloneActOp, StandaloneCompOp

__all__ = ["torch_to_paiir", "register_neuron"]

ModuleMapper = dict[type[nn.Module], Callable[[nn.Module], OpNode]]
"""Module mapping type: ``nn.Module`` subclass -> converter function returning an :class:`OpNode`."""

# Modules that are kept as leaf nodes but produce no PAIIR node (bypass).
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
    CoreNeuronV25: lambda mod: StandaloneActOp(act=mod),
    LutActivation: lambda mod: StandaloneActOp(act=ANNNodeV25(mod)),
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


def _get_output_shape(node: fx.Node) -> tuple[int, ...]:
    """Extract output shape from an FX node's meta."""
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return ()
    if hasattr(meta, "shape"):
        return tuple(meta.shape)
    return ()


def _get_input_shapes(node: fx.Node) -> list[tuple[int, ...]]:
    """Extract input shapes from an FX node's predecessor meta."""
    return [_get_output_shape(inp) for inp in node.all_input_nodes]


def _get_output_dims(node: fx.Node) -> DimsType:
    """Get output dims from an FX node's meta."""
    return node.meta.get(DimsProp.KEY, ())


def _get_input_dims(node: fx.Node) -> list[DimsType]:
    """Get input dims from an FX node's predecessor meta."""
    return [_get_output_dims(inp) for inp in node.all_input_nodes]


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
    if type(model) in full_map:
        ir_node = full_map[type(model)](model)
        graph = PAIIRGraph(type(model).__name__)
        inp = InputNode()
        out = OutputNode()
        graph.add_node(inp)
        if isinstance(ir_node, OpNode):
            graph.add_node(ir_node)
            graph.add_edge(inp.name, ir_node.name)
            graph.add_node(out)
            graph.add_edge(ir_node.name, out.name)
        else:
            graph.add_node(out)
            graph.add_edge(inp.name, out.name)
        return graph

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


def _fill_shape_dims(ir_node: OpNode, fx_node: fx.Node) -> None:
    """Copy shape and axis-ordering info from FX node meta into a PAIIR node."""
    ir_node.input_shapes = _get_input_shapes(fx_node)
    ir_node.output_shape = _get_output_shape(fx_node)
    ir_node.input_dims = _get_input_dims(fx_node)
    ir_node.output_dims = _get_output_dims(fx_node)


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

    fx_to_ir: dict[str, str] = {}
    bypass_nodes: set[fx.Node] = set()
    ignored_nodes: set[fx.Node] = set()
    unsupported_ops: list[tuple[str, str]] = []  # (node_name, op_description)

    # Pass 1: create PAIIR nodes
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ir_node = InputNode(shape=_get_output_shape(node))
            paiir_graph.add_node(ir_node)
            fx_to_ir[node.name] = ir_node.name

        elif node.op == "output":
            out_args = (
                node.args[0]
                if isinstance(node.args[0], (tuple, list))
                else [node.args[0]]
            )
            out_shape = ()
            for arg in out_args:
                if isinstance(arg, fx.Node):
                    out_shape = _get_output_shape(arg)
                    break
            ir_node = OutputNode(shape=out_shape)
            paiir_graph.add_node(ir_node)
            fx_to_ir[node.name] = ir_node.name

        elif node.op == "call_module":
            torch_module = gm.get_submodule(str(node.target))
            if _is_bypass_module(torch_module):
                bypass_nodes.add(node)
                continue
            if (mod_type := type(torch_module)) in module_map:
                ir_node = module_map[mod_type](torch_module)
                if isinstance(ir_node, OpNode):
                    _fill_shape_dims(ir_node, node)
                    paiir_graph.add_node(ir_node)
                    fx_to_ir[node.name] = ir_node.name
                else:
                    bypass_nodes.add(node)
            else:
                # Unsupported module
                mod_name = type(torch_module).__name__
                unsupported_ops.append((node.name, f"nn.Module '{mod_name}'"))
                if strict:
                    raise UnsupportedOpError(node.name, f"nn.Module '{mod_name}'")
                bypass_nodes.add(node)

        elif node.op == "call_function":
            if node.target in ADD_OPS:
                ir_node = AddOp(op_signs=(1, 1))
                _fill_shape_dims(ir_node, node)
                paiir_graph.add_node(ir_node)
                fx_to_ir[node.name] = ir_node.name
            elif node.target in SUB_OPS:
                ir_node = AddOp(op_signs=(1, -1))
                _fill_shape_dims(ir_node, node)
                paiir_graph.add_node(ir_node)
                fx_to_ir[node.name] = ir_node.name
            elif node.target in CAT_OPS:
                raw_dim = node.kwargs.get("dim", 0)
                assert isinstance(raw_dim, int), (
                    f"cat dim must be int, got {type(raw_dim)}"
                )
                ir_node = ConcatOp(dim=raw_dim)
                _fill_shape_dims(ir_node, node)
                paiir_graph.add_node(ir_node)
                fx_to_ir[node.name] = ir_node.name
            elif node.target in KNOWN_BYPASS_FUNCS:
                bypass_nodes.add(node)
            # Ignore assertion functions inserted by torch.fx for concrete_args validation
            elif callable(node.target) and getattr(
                node.target, "__name__", ""
            ).startswith("_assert"):
                ignored_nodes.add(node)
            else:
                # Unsupported function
                func_name = getattr(node.target, "__name__", str(node.target))
                unsupported_ops.append((node.name, f"function '{func_name}'"))
                if strict:
                    raise UnsupportedOpError(node.name, f"function '{func_name}'")
                bypass_nodes.add(node)

        elif node.op == "call_method":
            if node.target == "add":
                ir_node = AddOp(op_signs=(1, 1))
                _fill_shape_dims(ir_node, node)
                paiir_graph.add_node(ir_node)
                fx_to_ir[node.name] = ir_node.name
            elif node.target == "sub":
                ir_node = AddOp(op_signs=(1, -1))
                _fill_shape_dims(ir_node, node)
                paiir_graph.add_node(ir_node)
                fx_to_ir[node.name] = ir_node.name
            elif node.target in KNOWN_BYPASS_METHODS:
                # Known bypass methods (flatten, reshape, view, etc.)
                bypass_nodes.add(node)
            else:
                # Unknown method - may be unsupported
                method_name = str(node.target)
                unsupported_ops.append((node.name, f"method '{method_name}'"))
                if strict:
                    raise UnsupportedOpError(node.name, f"method '{method_name}'")
                bypass_nodes.add(node)

        elif node.op == "get_attr":
            bypass_nodes.add(node)

    # Emit warnings for unsupported ops in non-strict mode
    if not strict and unsupported_ops:
        warnings.warn(UnsupportedOpWarning(unsupported_ops), stacklevel=2)

    # Pass 2: create edges

    def _resolve_source(n: fx.Node) -> set[str]:
        """Recursively find non-bypass/ignored source PAIIR node names."""
        if n in ignored_nodes:
            return set()
        if n in bypass_nodes:
            sources: set[str] = set()
            for inp in n.all_input_nodes:
                sources |= _resolve_source(inp)
            return sources
        if n.name in fx_to_ir:
            return {fx_to_ir[n.name]}
        return set()

    for node in gm.graph.nodes:
        if node in ignored_nodes or node in bypass_nodes:
            continue
        if node.name not in fx_to_ir:
            continue

        dst_name = fx_to_ir[node.name]

        if node.op == "output":
            out_args = (
                node.args[0]
                if isinstance(node.args[0], (tuple, list))
                else [node.args[0]]
            )
            for arg in out_args:
                if isinstance(arg, fx.Node):
                    for src in _resolve_source(arg):
                        paiir_graph.add_edge(src, dst_name)
        else:
            for port_idx, inp_node in enumerate(node.all_input_nodes):
                for src in _resolve_source(inp_node):
                    paiir_graph.add_edge(src, dst_name, dst_port=port_idx)

    paiir_graph.eval()
    return paiir_graph
