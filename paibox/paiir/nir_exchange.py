"""Import and export PAIIR graphs through NIR.

This module is intentionally not imported from :mod:`paibox.paiir` so the
optional ``nir`` dependency is loaded only when users opt into this exchange
path.
"""

import graphlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import nir
import numpy as np
import torch
from paicorelib import (
    RM,
    LateralInhibitionMode,
    LeakAddMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    ThresholdNegMode,
    ThresholdPosMode,
)
from torch import Tensor, nn

from .exceptions import UnsupportedNIRNodeError
from .ir import (
    AccumulateOp,
    ANNNodeV25,
    ConcatOp,
    CoreNeuronV25,
    CPUOp,
    GeneralAddOp,
    IFNodeV25,
    InputNode,
    LIFNodeV25,
    OfflineCoreOp,
    OnlineCoreOp,
    OutputNode,
    PadOp,
    PAIIRGraph,
    PAIIRNode,
    PotentialAddOp,
    SequentialOp,
    ShapeStage,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TensorLayout,
    TransformOp,
)
from .nn import SumPool2d
from .pipeline.avgpool.standalone_rewrite import OutputApprox
from .pipeline.compile import CompileConfig, _compile_paiir_graph
from .pipeline.data_format import DataFormat

__all__ = ["compile_from_nir", "export_to_nir", "import_from_nir"]


_T = TypeVar("_T", bound=PAIIRNode)
GraphInput = nir.NIRGraph | str | Path


def import_from_nir(graph_or_path: GraphInput, *, dt: float = 1e-4) -> PAIIRGraph:
    """Convert a supported NIR graph into an unfused :class:`PAIIRGraph`."""
    nir_graph = _load_nir_graph(graph_or_path)
    nir_graph.check_types()
    topo_order = _check_nir_graph_structure(nir_graph)

    graph = PAIIRGraph("nir")
    for name in topo_order:
        node = nir_graph.nodes[name]
        paiir_node = _import_nir_node(name, node, dt)
        graph.add_node(paiir_node)

    for src, dst in nir_graph.edges:
        graph.add_edge(src, dst)

    graph.lint()
    return graph


def compile_from_nir(
    graph_or_path: GraphInput,
    *,
    dt: float = 1e-4,
    timesteps: int | None = None,
    auto_reset: bool | None = None,
    input_formats: dict[str, DataFormat] | None = None,
    compile_config: CompileConfig | None = None,
    enable_avgpool_calibration: bool | None = None,
    enable_split_avgpool_lif: bool | None = None,
    enable_delayed_avgpool_division: bool | None = None,
    output_approx: OutputApprox | None = None,
) -> PAIIRGraph:
    """Import NIR and run the standard PAIIR compile-time pipeline."""
    cfg = compile_config or CompileConfig()
    graph = import_from_nir(graph_or_path, dt=dt)
    return _compile_paiir_graph(
        graph,
        timesteps=cfg.timesteps if timesteps is None else timesteps,
        auto_reset=cfg.auto_reset if auto_reset is None else auto_reset,
        input_formats=cfg.input_formats if input_formats is None else input_formats,
        enable_avgpool_calibration=(
            cfg.enable_avgpool_calibration
            if enable_avgpool_calibration is None
            else enable_avgpool_calibration
        ),
        enable_split_avgpool_lif=(
            cfg.enable_split_avgpool_lif
            if enable_split_avgpool_lif is None
            else enable_split_avgpool_lif
        ),
        enable_delayed_avgpool_division=(
            cfg.enable_delayed_avgpool_division
            if enable_delayed_avgpool_division is None
            else enable_delayed_avgpool_division
        ),
        output_approx=cfg.output_approx if output_approx is None else output_approx,
    )


def export_to_nir(graph: PAIIRGraph, *, dt: float = 1e-4) -> nir.NIRGraph:
    """Export a supported PAIIR graph as a NIR graph."""
    graph.lint()

    nodes: dict[str, Any] = {}
    edges: list[tuple[str, str]] = []
    heads: dict[str, str] = {}
    tails: dict[str, str] = {}

    for name in graph.topo_sort():
        paiir_node = graph.nodes[name]
        exported_nodes, internal_edges = _export_paiir_node(name, paiir_node, dt)
        nodes.update(exported_nodes)
        edges.extend(internal_edges)
        exported_names = list(exported_nodes)
        heads[name] = exported_names[0]
        tails[name] = exported_names[-1]

    for edge in graph.edges:
        if edge.src_port != 0 or edge.dst_port != 0:
            _unsupported_export(
                edge.dst,
                graph.nodes[edge.dst],
                field="edge",
                value=edge,
                reason="NIR edges do not carry port indices",
            )
        edges.append((tails[edge.src], heads[edge.dst]))

    return nir.NIRGraph(nodes, edges, type_check=True)


def _load_nir_graph(graph_or_path: GraphInput) -> nir.NIRGraph:
    if isinstance(graph_or_path, (str, Path)):
        loaded = nir.read(graph_or_path)
    else:
        loaded = graph_or_path

    if not isinstance(loaded, nir.NIRGraph):
        raise TypeError(f"expected nir.NIRGraph or path, got {type(loaded).__name__}")
    return loaded


def _unsupported_import(
    name: str,
    node: Any,
    *,
    reason: str,
    field: str | None = None,
    value: Any | None = None,
) -> NoReturn:
    raise UnsupportedNIRNodeError(
        direction="import",
        node_name=name,
        node_type=type(node).__name__,
        field=field,
        value=value,
        reason=reason,
    )


def _unsupported_export(
    name: str,
    node: PAIIRNode,
    *,
    reason: str,
    field: str | None = None,
    value: Any | None = None,
) -> NoReturn:
    raise UnsupportedNIRNodeError(
        direction="export",
        node_name=name,
        node_type=type(node).__name__,
        field=field,
        value=value,
        reason=reason,
    )


def _check_nir_graph_structure(graph: nir.NIRGraph) -> tuple[str, ...]:
    for name, node in graph.nodes.items():
        if isinstance(node, nir.NIRGraph):
            _unsupported_import(
                name, node, reason="nested NIRGraph nodes are not supported"
            )

    incoming: dict[str, int] = {name: 0 for name in graph.nodes}
    for src, dst in graph.edges:
        if src not in graph.nodes:
            _unsupported_import(
                dst,
                graph.nodes.get(dst, graph),
                field="edge",
                value=(src, dst),
                reason="edge source is missing from graph nodes",
            )
        if dst not in graph.nodes:
            _unsupported_import(
                src,
                graph.nodes[src],
                field="edge",
                value=(src, dst),
                reason="edge destination is missing from graph nodes",
            )
        incoming[dst] += 1

    for name, count in incoming.items():
        if count > 1:
            _unsupported_import(
                name,
                graph.nodes[name],
                field="fan_in",
                value=count,
                reason="NIR fan-in has no PAIIR meaning without an explicit op",
            )

    return _topological_nir_order(graph)


def _topological_nir_order(graph: nir.NIRGraph) -> tuple[str, ...]:
    predecessors: dict[str, set[str]] = {name: set() for name in graph.nodes}
    for src, dst in graph.edges:
        predecessors[dst].add(src)

    try:
        return tuple(graphlib.TopologicalSorter(predecessors).static_order())
    except graphlib.CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        value = tuple(str(name) for name in cycle)
        raise UnsupportedNIRNodeError(
            direction="import",
            node_name="<graph>",
            node_type="NIRGraph",
            field="cycle",
            value=value,
            reason="recurrent or cyclic NIR graphs are not supported",
        ) from exc


def _import_nir_node(name: str, node: Any, dt: float) -> PAIIRNode:
    node_type = type(node)
    if node_type is nir.Input:
        return _named_node(InputNode(_layout_shape(_nir_input_shape(name, node))), name)
    if node_type is nir.Output:
        return _named_node(
            OutputNode(_layout_shape(_nir_output_shape(name, node))), name
        )

    importer = _NIR_IMPORTERS.get(node_type)
    if importer is not None:
        return importer(name, node)

    if node_type is nir.IF:
        return _import_if(name, node, dt)
    if node_type is nir.LIF:
        return _import_lif(name, node, dt)

    _unsupported_import(
        name, node, reason="NIR node type is not in the PAIIR import whitelist"
    )


def _named_node(node: _T, name: str) -> _T:
    node.name = name
    return node


def _layout_shape(nir_shape: Sequence[int]) -> torch.Size:
    return torch.Size((1, *tuple(int(dim) for dim in nir_shape)))


def _layout(nir_shape: Sequence[int]) -> TensorLayout:
    shape = _layout_shape(nir_shape)
    return TensorLayout(shape, tuple(range(len(shape))))


def _without_batch(layout: TensorLayout, name: str, node: PAIIRNode) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in layout.shape)
    if not shape:
        _unsupported_export(
            name, node, field="shape", value=shape, reason="missing shape"
        )
    if shape[0] != 1:
        _unsupported_export(
            name,
            node,
            field="shape",
            value=shape,
            reason="NIR exchange expects PAIIR batch dimension to be 1",
        )
    return shape[1:]


def _shape_from_type(
    name: str, node: Any, type_dict: dict[str, Any] | None, key: str, *, field: str
) -> tuple[int, ...]:
    if type_dict is None or key not in type_dict or type_dict[key] is None:
        _unsupported_import(
            name,
            node,
            field=field,
            value=type_dict,
            reason="dynamic or missing tensor shape is not supported",
        )

    value = type_dict[key]
    if isinstance(value, (int, np.integer)):
        dims = (int(value),)
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            dims = (int(array.item()),)
        else:
            dims = tuple(int(dim) for dim in array.tolist())

    if any(dim < 0 for dim in dims):
        _unsupported_import(
            name,
            node,
            field=field,
            value=dims,
            reason="negative or symbolic tensor dimensions are not supported",
        )
    return dims


def _nir_input_shape(name: str, node: Any) -> tuple[int, ...]:
    return _shape_from_type(name, node, node.input_type, "input", field="input_type")


def _nir_output_shape(name: str, node: Any) -> tuple[int, ...]:
    return _shape_from_type(name, node, node.output_type, "output", field="output_type")


def _set_layouts(
    op: StandaloneCompOp | StandaloneActOp | TransformOp, name: str, nir_node: Any
) -> None:
    op.input_layouts = (_layout(_nir_input_shape(name, nir_node)),)
    op.output_layouts = (_layout(_nir_output_shape(name, nir_node)),)


def _copy_parameter(param: Tensor, array: np.ndarray) -> None:
    with torch.no_grad():
        param.copy_(torch.as_tensor(array, dtype=torch.float32).to(dtype=param.dtype))


def _import_linear(name: str, node: nir.Linear) -> StandaloneCompOp:
    weight = np.asarray(node.weight)
    if weight.ndim != 2:
        _unsupported_import(
            name,
            node,
            field="weight",
            value=weight.shape,
            reason="only 2D Linear weight is supported",
        )
    comp = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    _copy_parameter(comp.weight, weight)
    return _finish_comp_node(name, node, comp)


def _import_affine(name: str, node: nir.Affine) -> StandaloneCompOp:
    weight = np.asarray(node.weight)
    bias = np.asarray(node.bias)
    if weight.ndim != 2:
        _unsupported_import(
            name,
            node,
            field="weight",
            value=weight.shape,
            reason="only 2D Affine weight is supported",
        )
    if bias.shape != (weight.shape[0],):
        _unsupported_import(
            name,
            node,
            field="bias",
            value=bias.shape,
            reason="Affine bias must match output feature dimension",
        )
    comp = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    _copy_parameter(comp.weight, weight)
    assert comp.bias is not None
    _copy_parameter(comp.bias, bias)
    return _finish_comp_node(name, node, comp)


def _import_conv1d(name: str, node: nir.Conv1d) -> StandaloneCompOp:
    _reject_string_padding(name, node, node.padding)
    weight = np.asarray(node.weight)
    bias = np.asarray(node.bias)
    if weight.ndim != 3:
        _unsupported_import(
            name,
            node,
            field="weight",
            value=weight.shape,
            reason="Conv1d weight must have shape (out_channels, in_channels, kernel)",
        )
    if node.groups != 1:
        _unsupported_import(
            name,
            node,
            field="groups",
            value=node.groups,
            reason="grouped NIR Conv1d is not supported",
        )
    if bias.shape != (weight.shape[0],):
        _unsupported_import(
            name,
            node,
            field="bias",
            value=bias.shape,
            reason="Conv1d bias must match output channels",
        )

    comp = nn.Conv1d(
        weight.shape[1],
        weight.shape[0],
        kernel_size=weight.shape[2],
        stride=int(node.stride),
        padding=int(node.padding),
        dilation=int(node.dilation),
        groups=1,
        bias=True,
    )
    _copy_parameter(comp.weight, weight)
    assert comp.bias is not None
    _copy_parameter(comp.bias, bias)
    return _finish_comp_node(name, node, comp)


def _import_conv2d(name: str, node: nir.Conv2d) -> StandaloneCompOp:
    _reject_string_padding(name, node, node.padding)
    weight = np.asarray(node.weight)
    bias = np.asarray(node.bias)
    if weight.ndim != 4:
        _unsupported_import(
            name,
            node,
            field="weight",
            value=weight.shape,
            reason=(
                "Conv2d weight must have shape "
                "(out_channels, in_channels, kernel_h, kernel_w)"
            ),
        )
    if node.groups != 1:
        _unsupported_import(
            name,
            node,
            field="groups",
            value=node.groups,
            reason="grouped NIR Conv2d is not supported",
        )
    if bias.shape != (weight.shape[0],):
        _unsupported_import(
            name,
            node,
            field="bias",
            value=bias.shape,
            reason="Conv2d bias must match output channels",
        )

    comp = nn.Conv2d(
        weight.shape[1],
        weight.shape[0],
        kernel_size=weight.shape[2:],
        stride=_to_pair(node.stride),
        padding=_to_pair(node.padding),
        dilation=_to_pair(node.dilation),
        groups=1,
        bias=True,
    )
    _copy_parameter(comp.weight, weight)
    assert comp.bias is not None
    _copy_parameter(comp.bias, bias)
    return _finish_comp_node(name, node, comp)


def _reject_string_padding(name: str, node: Any, padding: Any) -> None:
    if isinstance(padding, str):
        _unsupported_import(
            name,
            node,
            field="padding",
            value=padding,
            reason="string padding is deferred; only numeric padding is supported",
        )


def _import_avgpool2d(name: str, node: nir.AvgPool2d) -> StandaloneCompOp:
    comp = nn.AvgPool2d(
        kernel_size=_to_pair(node.kernel_size),
        stride=_to_pair(node.stride),
        padding=_to_pair(node.padding),
    )
    return _finish_comp_node(name, node, comp)


def _import_sumpool2d(name: str, node: nir.SumPool2d) -> StandaloneCompOp:
    comp = SumPool2d(
        kernel_size=_to_pair(node.kernel_size),
        stride=_to_pair(node.stride),
        padding=_to_pair(node.padding),
    )
    return _finish_comp_node(name, node, comp)


def _finish_comp_node(name: str, nir_node: Any, comp: nn.Module) -> StandaloneCompOp:
    comp.eval()
    op = StandaloneCompOp(comp)
    _set_layouts(op, name, nir_node)
    return _named_node(op, name)


def _import_flatten(name: str, node: nir.Flatten) -> TransformOp:
    output_shape = _layout_shape(_nir_output_shape(name, node))

    def shape_fn(_shape: torch.Size, target: torch.Size = output_shape) -> torch.Size:
        return target

    op = TransformOp((ShapeStage(shape_fn),))
    _set_layouts(op, name, node)
    return _named_node(op, name)


_NIR_IMPORTERS: dict[type[Any], Callable[[str, Any], PAIIRNode]] = {
    nir.Linear: _import_linear,
    nir.Affine: _import_affine,
    nir.Conv1d: _import_conv1d,
    nir.Conv2d: _import_conv2d,
    nir.AvgPool2d: _import_avgpool2d,
    nir.SumPool2d: _import_sumpool2d,
    nir.Flatten: _import_flatten,
}


def _import_if(name: str, node: nir.IF, dt: float) -> StandaloneActOp:
    expected_r = 1.0 / dt
    if not np.allclose(node.r, expected_r):
        _unsupported_import(
            name, node, field="r", value=node.r, reason="IF.r must equal 1 / dt"
        )

    act = IFNodeV25(
        v_threshold=_threshold_value(name, node, node.v_threshold),
        v_reset=_uniform_float(name, node, "v_reset", np.asarray(node.v_reset)),
    )
    return _finish_act_node(name, node, act)


def _import_lif(name: str, node: nir.LIF, dt: float) -> StandaloneActOp:
    tau_seconds = _uniform_float(name, node, "tau", node.tau)
    tau = tau_seconds / dt
    if tau <= 1:
        _unsupported_import(
            name,
            node,
            field="tau",
            value=tau_seconds,
            reason="LIF.tau / dt must be > 1",
        )

    if np.allclose(node.r, 1.0):
        decay_input = True
    elif np.allclose(node.r, tau):
        decay_input = False
    else:
        _unsupported_import(
            name,
            node,
            field="r",
            value=node.r,
            reason="LIF.r must be either 1 or LIF.tau / dt",
        )

    v_reset = _uniform_float(name, node, "v_reset", np.asarray(node.v_reset))
    v_leak = _uniform_float(name, node, "v_leak", np.asarray(node.v_leak))
    if not np.isclose(v_leak, v_reset):
        _unsupported_import(
            name,
            node,
            field="v_leak",
            value=v_leak,
            reason="PAIIR LIF import requires v_leak == v_reset",
        )

    act = LIFNodeV25(
        tau=float(tau),
        decay_input=decay_input,
        v_threshold=_threshold_value(name, node, node.v_threshold),
        v_reset=v_reset,
    )
    return _finish_act_node(name, node, act)


def _finish_act_node(name: str, nir_node: Any, act: CoreNeuronV25) -> StandaloneActOp:
    act.eval()
    op = StandaloneActOp(act)
    _set_layouts(op, name, nir_node)
    return _named_node(op, name)


def _uniform_float(name: str, node: Any, field: str, value: np.ndarray) -> float:
    array = value.astype(np.float64)
    if array.size == 0:
        _unsupported_import(name, node, field=field, value=value, reason="empty array")
    first = float(array.flat[0])
    if not np.allclose(array, first):
        _unsupported_import(
            name,
            node,
            field=field,
            value=value,
            reason="only uniform scalar values are supported",
        )
    return first


def _threshold_value(name: str, node: Any, value: np.ndarray) -> float | Tensor:
    array = value.astype(np.float32)
    if array.size == 0:
        _unsupported_import(
            name, node, field="v_threshold", value=value, reason="empty array"
        )
    # NIR fires on v > threshold; PAIIR/chip integer state fires on v >= threshold.
    array = np.floor(array) + 1
    first = float(array.flat[0])
    if np.allclose(array, first):
        return first

    if array.ndim == 0:
        return first
    by_channel = array.reshape(array.shape[0], -1)
    if np.allclose(by_channel, by_channel[:, :1]):
        return torch.as_tensor(by_channel[:, 0], dtype=torch.float32)

    _unsupported_import(
        name,
        node,
        field="v_threshold",
        value=value,
        reason="threshold must be scalar or constant per output channel",
    )


def _to_pair(value: Any) -> tuple[int, int]:
    array = np.asarray(value)
    if array.ndim == 0:
        item = int(array.item())
        return item, item
    items = tuple(int(v) for v in array.tolist())
    if len(items) == 1:
        return items[0], items[0]
    if len(items) == 2:
        return items
    raise ValueError(f"expected scalar or 2 values, got {value!r}")


def _to_single(value: Any) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array.item())
    items = tuple(int(v) for v in array.tolist())
    if len(items) == 1:
        return items[0]
    raise ValueError(f"expected scalar or 1 value, got {value!r}")


def _export_paiir_node(
    name: str, node: PAIIRNode, dt: float
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    if isinstance(node, InputNode):
        nir_node = nir.Input(
            {"input": np.asarray(_without_batch(node.layout, name, node))}
        )
        return {name: nir_node}, []

    if isinstance(node, OutputNode):
        nir_node = nir.Output(
            {"output": np.asarray(_without_batch(node.layout, name, node))}
        )
        return {name: nir_node}, []

    if isinstance(node, StandaloneCompOp):
        return {name: _export_comp(name, node, node.comp)}, []

    if isinstance(node, StandaloneActOp):
        return {name: _export_act(name, node, node.act, dt)}, []

    if isinstance(node, SequentialOp):
        comp_name = f"{name}.comp"
        act_name = f"{name}.act"
        return (
            {
                comp_name: _export_comp(comp_name, node, node.comp),
                act_name: _export_act(act_name, node, node.act, dt),
            },
            [(comp_name, act_name)],
        )

    if isinstance(node, TransformOp):
        return {name: _export_flatten(name, node)}, []

    _reject_known_paiir_node(name, node)


def _reject_known_paiir_node(name: str, node: PAIIRNode) -> NoReturn:
    if isinstance(node, (GeneralAddOp, PotentialAddOp, AccumulateOp)):
        reason = "add or accumulate nodes have no v1 NIR exchange mapping"
    elif isinstance(node, (ConcatOp, SplitOp)):
        reason = "concat and split need explicit port/routing semantics absent in NIR"
    elif isinstance(node, PadOp):
        reason = "standalone PadOp is not exported; use numeric Conv padding"
    elif isinstance(node, (OnlineCoreOp, CPUOp)):
        reason = "online core and CPU fallback nodes are not NIR-exportable"
    elif isinstance(node, OfflineCoreOp):
        reason = "offline-core wrapper is not in the NIR export whitelist"
    else:
        reason = "PAIIR node type is not in the NIR export whitelist"
    _unsupported_export(name, node, reason=reason)


def _export_comp(name: str, owner: PAIIRNode, comp: nn.Module) -> Any:
    if isinstance(comp, nn.Linear):
        weight = _parameter_numpy(comp.weight)
        if comp.bias is None:
            return nir.Linear(weight)
        return nir.Affine(weight, _parameter_numpy(comp.bias))

    if isinstance(comp, nn.Conv1d):
        _reject_torch_string_padding(name, owner, comp.padding)
        input_shape = _conv1d_input_shape(name, owner)
        bias = _conv_bias(comp.bias, comp.out_channels)
        return nir.Conv1d(
            input_shape=input_shape,
            weight=_parameter_numpy(comp.weight),
            stride=_to_single(comp.stride),
            padding=_to_single(comp.padding),
            dilation=_to_single(comp.dilation),
            groups=int(comp.groups),
            bias=bias,
        )

    if isinstance(comp, nn.Conv2d):
        _reject_torch_string_padding(name, owner, comp.padding)
        input_shape = _conv2d_input_shape(name, owner)
        bias = _conv_bias(comp.bias, comp.out_channels)
        return nir.Conv2d(
            input_shape=input_shape,
            weight=_parameter_numpy(comp.weight),
            stride=_to_pair(comp.stride),
            padding=_to_pair(comp.padding),
            dilation=_to_pair(comp.dilation),
            groups=int(comp.groups),
            bias=bias,
        )

    if isinstance(comp, nn.AvgPool2d):
        _check_avgpool2d_export(name, owner, comp)
        return nir.AvgPool2d(
            kernel_size=np.asarray(_to_pair(comp.kernel_size)),
            stride=np.asarray(_to_pair(comp.stride)),
            padding=np.asarray(_to_pair(comp.padding)),
        )

    if isinstance(comp, SumPool2d):
        _check_sumpool2d_export(name, owner, comp)
        return nir.SumPool2d(
            kernel_size=np.asarray(_to_pair(comp.kernel_size)),
            stride=np.asarray(_to_pair(comp.stride)),
            padding=np.asarray(_to_pair(comp.padding)),
        )

    _unsupported_export(
        name,
        owner,
        field="comp",
        value=type(comp).__name__,
        reason="compute module is not in the NIR export whitelist",
    )


def _export_act(name: str, owner: PAIIRNode, act: CoreNeuronV25, dt: float) -> Any:
    _check_snn_act_export(name, owner, act)
    shape = _single_output_shape(name, owner)

    threshold = _threshold_array(name, owner, act.thres_pos, shape)
    reset = np.full(shape, float(act.reset_v), dtype=np.float32)
    if act.has_if_dynamics:
        r = np.full(shape, 1.0 / dt, dtype=np.float32)
        return nir.IF(r=r, v_threshold=threshold, v_reset=reset)

    if act.has_lif_dynamics:
        _check_lif_act_export(name, owner, act)
        tau = float(act.tau)
        decay_input = act.leak_multi_input == LeakMultiInputMode.ENABLE
        r_value = 1.0 if decay_input else tau
        nir_node = nir.LIF(
            tau=np.full(shape, tau * dt, dtype=np.float32),
            r=np.full(shape, r_value, dtype=np.float32),
            v_leak=reset.copy(),
            v_threshold=threshold,
            v_reset=reset,
        )
        return nir_node

    _unsupported_export(
        name,
        owner,
        field="act",
        value=type(act).__name__,
        reason="neuron dynamics are not representable as NIR IF/LIF",
    )


def _check_snn_act_export(name: str, owner: PAIIRNode, act: CoreNeuronV25) -> None:
    if isinstance(act, ANNNodeV25) or act.lut is not None:
        _unsupported_export(
            name,
            owner,
            field="act",
            value=type(act).__name__,
            reason="ANN/LUT neurons are not representable as NIR IF/LIF",
        )
    for field in ("reset_v", "thres_neg", "leak_tau", "init_v", "tau"):
        value = getattr(act, field)
        if torch.is_tensor(value):
            _unsupported_export(
                name,
                owner,
                field=field,
                value=value,
                reason="vector neuron parameters are not supported by NIR export",
            )
    if act.reset_mode != RM.MODE_NORMAL:
        _unsupported_export(
            name,
            owner,
            field="reset_mode",
            value=act.reset_mode,
            reason="NIR export v1 supports hard reset only",
        )
    if act.thres_pos_mode != ThresholdPosMode.FIRE:
        _unsupported_export(
            name,
            owner,
            field="thres_pos_mode",
            value=act.thres_pos_mode,
            reason="positive threshold must fire",
        )
    if act.thres_neg_mode == ThresholdNegMode.FIRE:
        _unsupported_export(
            name,
            owner,
            field="thres_neg_mode",
            value=act.thres_neg_mode,
            reason="negative threshold firing is not representable in NIR IF/LIF",
        )
    if act.lateral_inhi != LateralInhibitionMode.DISABLE:
        _unsupported_export(
            name,
            owner,
            field="lateral_inhi",
            value=act.lateral_inhi,
            reason="lateral inhibition is PAICORE-specific",
        )


def _check_lif_act_export(name: str, owner: PAIIRNode, act: CoreNeuronV25) -> None:
    if float(act.tau) <= 1:
        _unsupported_export(
            name,
            owner,
            field="tau",
            value=act.tau,
            reason="NIR LIF requires tau / dt > 1 and cannot represent beta=0",
        )
    if act.leak_add_mode != LeakAddMode.FORWARD:
        _unsupported_export(
            name,
            owner,
            field="leak_add_mode",
            value=act.leak_add_mode,
            reason="only forward additive leak can map to NIR LIF",
        )
    if act.leak_multi_sequence != LeakMultiComparisonOrder.AFTER_COMPARE:
        _unsupported_export(
            name,
            owner,
            field="leak_multi_sequence",
            value=act.leak_multi_sequence,
            reason="only the standard LIFNodeV25 leak order is exported",
        )
    if _scalar_or_none(act.leak_v) not in (0, 0.0):
        _unsupported_export(
            name,
            owner,
            field="leak_v",
            value=act.leak_v,
            reason="additive leak voltage is not representable by this mapping",
        )


def _export_flatten(name: str, node: TransformOp) -> Any:
    if len(node.stages) != 1 or not isinstance(node.stages[0], ShapeStage):
        _unsupported_export(
            name,
            node,
            field="stages",
            value=node.stages,
            reason="only pure flatten TransformOp is supported",
        )

    input_shape = _single_input_shape(name, node)
    output_shape = _single_output_shape(name, node)
    if int(np.prod(input_shape)) != int(np.prod(output_shape)):
        _unsupported_export(
            name,
            node,
            field="shape",
            value=(input_shape, output_shape),
            reason="TransformOp changes element count",
        )

    nir_node = nir.Flatten({"input": np.asarray(input_shape)}, start_dim=0, end_dim=-1)
    return nir_node


def _parameter_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _conv_bias(bias: Tensor | None, channels: int) -> np.ndarray:
    if bias is None:
        return np.zeros((channels,), dtype=np.float32)
    return _parameter_numpy(bias)


def _single_input_shape(name: str, node: PAIIRNode) -> tuple[int, ...]:
    if not isinstance(
        node, (StandaloneCompOp, StandaloneActOp, SequentialOp, TransformOp)
    ):
        _unsupported_export(name, node, reason="node has no input layout")
    if len(node.input_layouts) != 1:
        _unsupported_export(
            name,
            node,
            field="input_layouts",
            value=node.input_layouts,
            reason="NIR export expects single-input nodes",
        )
    return _without_batch(node.input_layouts[0], name, node)


def _single_output_shape(name: str, node: PAIIRNode) -> tuple[int, ...]:
    if not isinstance(
        node, (StandaloneCompOp, StandaloneActOp, SequentialOp, TransformOp)
    ):
        _unsupported_export(name, node, reason="node has no output layout")
    if len(node.output_layouts) != 1:
        _unsupported_export(
            name,
            node,
            field="output_layouts",
            value=node.output_layouts,
            reason="NIR export expects single-output nodes",
        )
    return _without_batch(node.output_layouts[0], name, node)


def _conv1d_input_shape(name: str, node: PAIIRNode) -> int:
    shape = _single_input_shape(name, node)
    if len(shape) != 2:
        _unsupported_export(
            name,
            node,
            field="input_shape",
            value=shape,
            reason="Conv1d export expects N,C,L PAIIR layout",
        )
    return int(shape[-1])


def _conv2d_input_shape(name: str, node: PAIIRNode) -> tuple[int, int]:
    shape = _single_input_shape(name, node)
    if len(shape) != 3:
        _unsupported_export(
            name,
            node,
            field="input_shape",
            value=shape,
            reason="Conv2d export expects N,C,H,W PAIIR layout",
        )
    return int(shape[-2]), int(shape[-1])


def _check_avgpool2d_export(name: str, owner: PAIIRNode, comp: nn.AvgPool2d) -> None:
    if comp.ceil_mode:
        _unsupported_export(
            name,
            owner,
            field="ceil_mode",
            value=comp.ceil_mode,
            reason="NIR AvgPool2d has no ceil_mode field",
        )
    if not comp.count_include_pad:
        _unsupported_export(
            name,
            owner,
            field="count_include_pad",
            value=comp.count_include_pad,
            reason="NIR AvgPool2d cannot encode this PyTorch option",
        )
    if comp.divisor_override is not None:
        _unsupported_export(
            name,
            owner,
            field="divisor_override",
            value=comp.divisor_override,
            reason="NIR AvgPool2d cannot encode divisor_override",
        )


def _check_sumpool2d_export(name: str, owner: PAIIRNode, comp: SumPool2d) -> None:
    if comp.ceil_mode:
        _unsupported_export(
            name,
            owner,
            field="ceil_mode",
            value=comp.ceil_mode,
            reason="NIR SumPool2d has no ceil_mode field",
        )
    if tuple(comp.dilation) != (1, 1):
        _unsupported_export(
            name,
            owner,
            field="dilation",
            value=comp.dilation,
            reason="NIR SumPool2d has no dilation field",
        )


def _reject_torch_string_padding(name: str, owner: PAIIRNode, padding: Any) -> None:
    if isinstance(padding, str):
        _unsupported_export(
            name,
            owner,
            field="padding",
            value=padding,
            reason="string padding is deferred; only numeric padding is supported",
        )


def _threshold_array(
    name: str, node: PAIIRNode, threshold: float | Tensor, shape: tuple[int, ...]
) -> np.ndarray:
    if torch.is_tensor(threshold):
        # Center the strict NIR boundary in the integer bin below the PAIIR threshold.
        values = torch.ceil(threshold.detach().cpu().to(torch.float32)) - 0.5
        if values.ndim != 1 or len(shape) == 0 or values.numel() != shape[0]:
            _unsupported_export(
                name,
                node,
                field="thres_pos",
                value=tuple(values.shape),
                reason="per-channel threshold must match the first NIR axis",
            )
        reshape = (values.numel(), *(1 for _ in shape[1:]))
        return values.reshape(reshape).expand(shape).numpy()
    return np.full(shape, np.ceil(float(threshold)) - 0.5, dtype=np.float32)


def _scalar_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.item())
    return None
