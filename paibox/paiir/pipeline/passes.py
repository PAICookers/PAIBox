"""PAIIR graph-level pass entrypoints.

This module remains the public facade for the compilation pipeline. AvgPool-
specific implementation details live under :mod:`paibox.paiir.pipeline.avgpool`,
while generic graph validation, data-format propagation, and scheduling remain
here.

Two validation stages live in this module:

- :func:`validate_graph` runs early, immediately after node fusion. It may
  remove disconnected graph fragments and checks only the structural
  invariants required by later passes.
- :func:`validate_compiled_graph` runs at the end of compilation. It assumes
  all compile-time annotations should already be populated and validates the
  final graph that will be returned to callers.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from paicorelib import AddPotentialMode, DataSign, DataWidth, OutputType, SNNMode

from ..exceptions import GraphCleanupWarning, GraphValidationError
from ..ir.add_ops import AddOperandKind, AddOperandSpec, GeneralAddOp, PotentialAddOp
from ..ir.core_neuron import CoreNeuronV25
from ..ir.graph import Edge, PAIIRGraph
from ..ir.ir_base import FormatFlow, InputNode, OutputNode, PAIIRNode, TensorLayout
from ..ir.maxpool_export import refresh_maxpool_export_kind
from ..ir.op_node import (
    AccumulateOp,
    ConcatOp,
    OfflineCoreOp,
    OpNode,
    SequentialOp,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TransformOp,
)
from ..ir.reshape_semantics import shape_after_dims
from ..ir.signal_domain import SignalDomain
from ..ir.utils import infer_split_output_shapes
from ..ir.value_code import code_range_for_data_format, merge_code_ranges
from .avgpool import calibrate_avgpool_thresholds
from .avgpool.calibration import CalibrationResult
from .avgpool.fusion import _try_handle_avgpool_activation
from .avgpool.utils import is_value_avgpool
from .data_format import (
    DataFormat,
    infer_output_code_range,
    infer_output_format,
    infer_weight_format,
    merge_data_formats,
)
from .fusion_utils import _materialize_shared_sequential
from .graph_utils import (
    collect_effective_predecessor_values,
    is_format_transparent_routing_node,
    is_order_preserving_transform_node,
    is_standalone_maxpool,
)

__all__ = [
    "analyze_graph",
    "assign_tick_params",
    "calibrate_avgpool_thresholds",
    "CalibrationResult",
    "flatten_general_add_chains",
    "fuse_to_offline_cores",
    "propagate_signal_semantics",
    "propagate_data_format",
    "specialize_general_adds",
    "TickOverride",
    "validate_compiled_graph",
    "validate_deployable_graph",
    "validate_graph",
]

_DEPLOYABLE_GRAPH_NODE_TYPES = (
    InputNode,
    OutputNode,
    TransformOp,
    ConcatOp,
    SequentialOp,
    AccumulateOp,
    PotentialAddOp,
    StandaloneCompOp,
    StandaloneActOp,
)

KnownCodeRange = tuple[int, int] | None
NodeSignal = tuple[SignalDomain, KnownCodeRange]


@dataclass(frozen=True, slots=True)
class _PredSignalFacts:
    """Predecessor signal facts gathered once for one node.

    The semantics pass repeatedly needs the same three views of predecessor
    state:

    - raw predecessor domains, which may still contain ``None``
    - exact predecessor code ranges, which may also be partially unknown
    - filtered/merged views consumed by operator-specific rules
    """

    domains: tuple[SignalDomain | None, ...]
    code_ranges: tuple[KnownCodeRange, ...]

    @property
    def has_missing_domain(self) -> bool:
        return any(domain is None for domain in self.domains)

    @property
    def known_domains(self) -> list[SignalDomain]:
        return [domain for domain in self.domains if domain is not None]

    @property
    def present_code_ranges(self) -> list[tuple[int, int]]:
        return [code_range for code_range in self.code_ranges if code_range is not None]

    def single_domain(self) -> SignalDomain | None:
        if not self.domains:
            return None
        return self.domains[0]

    def single_code_range(self) -> KnownCodeRange:
        if not self.code_ranges:
            return None
        return self.code_ranges[0]

    def merged_code_range(self) -> KnownCodeRange:
        if not self.code_ranges or any(
            code_range is None for code_range in self.code_ranges
        ):
            return None
        return merge_code_ranges(
            [code_range for code_range in self.code_ranges if code_range is not None]
        )


def analyze_graph(
    graph: PAIIRGraph, input_formats: dict[str, DataFormat] | None = None
) -> PAIIRGraph:
    """Run the standard compile-time graph analyses in dependency order.

    The public contract stays intentionally small: validate structure first,
    then derive node-level signal semantics, then derive backend-facing data
    formats. The implementation keeps semantics and data-format propagation as
    two separate stages even though they share some operator-specific rules.
    """
    validate_graph(graph)
    propagate_signal_semantics(graph, input_formats)
    propagate_data_format(graph, input_formats)
    return graph


def _layout_shapes(node: OpNode) -> tuple[torch.Size, ...]:
    return tuple(layout.shape for layout in node.input_layouts)


def _layout_input_dims(node: OpNode) -> tuple[tuple[int, ...], ...]:
    return tuple(layout.dims for layout in node.input_layouts)


def _single_output_shape(node: OpNode) -> torch.Size:
    if node.num_outputs != 1:
        return torch.Size()
    return node.output_layouts[0].shape


@dataclass(frozen=True, slots=True)
class _FlatAddOperand:
    source_name: str
    source_port: int
    coeff: int
    layout: TensorLayout


@dataclass(frozen=True, slots=True)
class _FlatAddPlan:
    operands: tuple[_FlatAddOperand, ...]
    consumed_adds: frozenset[str]
    consumed_transforms: frozenset[str]
    changed: bool


def flatten_general_add_chains(graph: PAIIRGraph) -> PAIIRGraph:
    """Flatten single-use chains of :class:`GeneralAddOp` before specialization.

    FX represents ``a + b + c`` as a chain of binary adds.  Keeping that shape
    prevents later passes from seeing all signed paths at once.  Flattening
    serves both no-activation chains that specialize directly to an n-ary
    ``PotentialAddOp`` and activated chains that later fuse as
    ``PotentialAddOp -> ActivationOp -> AccumulateOp``.  This pass rewrites
    only the narrow subset that can still be specialized to deployable
    potential add:

    - tensor operands only
    - no broadcasted operands
    - per-path coefficients in ``{-1, +1}``
    - nested add nodes are single-use

    Order-preserving, shape-preserving ``TransformOp`` wrappers are treated as
    transparent when they are single-use.  Shape-changing transforms remain in
    the graph because ``AccumulateOp`` does not model per-path reshape stages.
    """
    flattened = graph.clone_shallow()
    changed_any = False
    changed = True

    while changed:
        changed = False
        for name in flattened.topo_sort():
            node = flattened.nodes.get(name)
            if not isinstance(node, GeneralAddOp):
                continue

            if _try_flatten_general_add_node(flattened, name):
                changed = True
                changed_any = True
                break

    return flattened if changed_any else graph


def _try_flatten_general_add_node(graph: PAIIRGraph, name: str) -> bool:
    node = graph.nodes.get(name)
    if not isinstance(node, GeneralAddOp):
        return False

    plan = _collect_flat_add_operands(graph, name, 1, frozenset())
    if plan is None or not plan.changed or not plan.consumed_adds:
        return False

    if _has_duplicate_flat_add_sources(plan.operands):
        return False

    output_shape = _single_output_shape(node)
    if output_shape and any(
        operand.layout.shape and operand.layout.shape != output_shape
        for operand in plan.operands
    ):
        return False

    replacement = GeneralAddOp(
        tuple(
            AddOperandSpec(operand.coeff, AddOperandKind.TENSOR, idx)
            for idx, operand in enumerate(plan.operands)
        )
    )
    replacement.input_layouts = tuple(operand.layout for operand in plan.operands)
    replacement.output_layouts = node.output_layouts

    outgoing = graph.outgoing_edges(name)
    graph.add_node(replacement)
    for idx, operand in enumerate(plan.operands):
        graph.add_edge(
            operand.source_name,
            replacement.name,
            src_port=operand.source_port,
            dst_port=idx,
        )
    for edge in outgoing:
        graph.add_edge(
            replacement.name, edge.dst, src_port=edge.src_port, dst_port=edge.dst_port
        )

    graph.remove_node(name)
    for consumed_name in sorted(plan.consumed_adds):
        if consumed_name in graph.nodes:
            graph.remove_node(consumed_name)
    for consumed_name in sorted(plan.consumed_transforms):
        if consumed_name in graph.nodes:
            graph.remove_node(consumed_name)

    return True


def _collect_flat_add_operands(
    graph: PAIIRGraph, add_name: str, outer_coeff: int, seen_adds: frozenset[str]
) -> _FlatAddPlan | None:
    if add_name in seen_adds:
        return None

    node = graph.nodes.get(add_name)
    if not isinstance(node, GeneralAddOp):
        return None
    if not _is_flattenable_general_add(node):
        return None

    incoming_by_port = _incoming_edges_by_port(graph, add_name)
    operands: list[_FlatAddOperand] = []
    consumed_adds: set[str] = set()
    consumed_transforms: set[str] = set()
    changed = False

    for operand in node.operands:
        if operand.kind is not AddOperandKind.TENSOR:
            return None
        assert operand.tensor_port is not None
        source_edge = incoming_by_port.get(operand.tensor_port)
        if source_edge is None:
            return None

        coeff = outer_coeff * operand.coeff
        if coeff not in (-1, 1):
            return None

        transparent_edge = _transparent_add_operand_edge(graph, source_edge)
        if transparent_edge is None:
            return None

        source_edge, consumed_transform = transparent_edge
        if consumed_transform is not None:
            consumed_transforms.add(consumed_transform)

        source_node = graph.nodes[source_edge.src]
        if isinstance(source_node, GeneralAddOp):
            if len(graph.successors(source_edge.src)) != 1:
                return None
            nested = _collect_flat_add_operands(
                graph, source_edge.src, coeff, seen_adds | {add_name}
            )
            if nested is None:
                return None

            operands.extend(nested.operands)
            consumed_adds.add(source_edge.src)
            consumed_adds.update(nested.consumed_adds)
            consumed_transforms.update(nested.consumed_transforms)
            changed = True
            continue

        layout = _edge_output_layout(graph, source_edge)
        if layout is None:
            return None
        operands.append(
            _FlatAddOperand(source_edge.src, source_edge.src_port, coeff, layout)
        )

    return _FlatAddPlan(
        tuple(operands),
        frozenset(consumed_adds),
        frozenset(consumed_transforms),
        changed,
    )


def _is_flattenable_general_add(node: GeneralAddOp) -> bool:
    if node.has_const_operands or node.has_broadcasted_operands:
        return False
    if len(node.tensor_operands) < 2:
        return False
    return all(coeff in (-1, 1) for coeff in node.tensor_coeffs)


def _incoming_edges_by_port(graph: PAIIRGraph, name: str) -> dict[int, Edge]:
    return {edge.dst_port: edge for edge in graph.incoming_edges(name)}


def _transparent_add_operand_edge(
    graph: PAIIRGraph, edge: Edge
) -> tuple[Edge, str | None] | None:
    node = graph.nodes.get(edge.src)
    if not isinstance(node, TransformOp):
        return edge, None
    if len(graph.successors(edge.src)) != 1:
        return None
    if len(graph.predecessors(edge.src)) != 1:
        return None
    if not is_order_preserving_transform_node(node):
        return None
    if node.input_layouts != node.output_layouts:
        return None

    return graph.incoming_edges(edge.src)[0], edge.src


def _edge_output_layout(graph: PAIIRGraph, edge: Edge) -> TensorLayout | None:
    node = graph.nodes.get(edge.src)
    if isinstance(node, InputNode):
        return node.layout
    if isinstance(node, OpNode) and edge.src_port < len(node.output_layouts):
        return node.output_layouts[edge.src_port]
    return None


def _has_duplicate_flat_add_sources(operands: Sequence[_FlatAddOperand]) -> bool:
    seen: set[tuple[str, int]] = set()
    for operand in operands:
        key = (operand.source_name, operand.source_port)
        if key in seen:
            return True
        seen.add(key)
    return False


def specialize_general_adds(graph: PAIIRGraph) -> PAIIRGraph:
    """Rewrite deployable expression-layer add nodes into deployable add IR.

    Only the narrow same-shape, tensor-only, no-broadcast subset is
    specialized. More general add semantics remain as :class:`GeneralAddOp`
    and must be handled before deployment or rejected at deployability
    validation time.
    """
    specialized_graph = graph.clone_shallow()

    for name in graph.topo_sort():
        if name not in specialized_graph.nodes:
            continue
        node = specialized_graph.nodes[name]

        if isinstance(node, GeneralAddOp):
            specialized = _try_specialize_general_add(node)
            if specialized is not None:
                specialized.input_layouts = node.input_layouts
                specialized.output_layouts = node.output_layouts
                specialized_graph.replace_node(name, specialized)

    return specialized_graph


def _try_specialize_general_add(node: GeneralAddOp) -> PotentialAddOp | None:
    if node.has_const_operands or node.has_broadcasted_operands:
        return None

    if len(node.tensor_operands) < 2:
        return None

    coeffs = node.tensor_coeffs
    if any(coeff not in (-1, 1) for coeff in coeffs):
        return None

    output_shape = _single_output_shape(node)
    input_shapes = _layout_shapes(node)
    if output_shape and input_shapes:
        if any(shape != output_shape for shape in input_shapes):
            return None

    return PotentialAddOp(op_signs=tuple(int(coeff) for coeff in coeffs))


def fuse_to_offline_cores(
    graph: PAIIRGraph,
    enable_split_avgpool_lif: bool = False,
    enable_avgpool_calibration: bool = False,
) -> PAIIRGraph:
    """Fuse atomic PAIIR nodes into offline-core units."""
    consumed: set[str] = set()
    node_remap: dict[str, str] = {}
    port_remap: dict[str, int] = {}
    fused_nodes: dict[str, PAIIRNode] = {}

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, StandaloneActOp):
            continue
        if name in consumed:
            continue

        result = _try_handle_avgpool_activation(
            graph,
            name,
            consumed,
            node_remap,
            enable_split_avgpool_lif=enable_split_avgpool_lif,
            enable_avgpool_calibration=enable_avgpool_calibration,
        )
        if result is not None:
            if isinstance(result, list):
                for fused in result:
                    fused_nodes[fused.name] = fused
            else:
                fused_nodes[result.name] = result
            continue

        result = _try_fuse_sequential(graph, name, consumed, node_remap)
        if result is not None:
            fused_nodes[result.name] = result
            continue

        result = _try_fuse_accumulate(graph, name, consumed, node_remap, port_remap)
        if result is not None:
            fused_nodes[result.name] = result
            continue

    new_graph = PAIIRGraph(graph.name)
    for name in graph.topo_sort():
        if name in consumed:
            fused_name = node_remap.get(name)
            if fused_name and fused_name in fused_nodes:
                new_graph.add_node(fused_nodes.pop(fused_name))
            continue
        new_graph.add_node(graph.nodes[name])
        node_remap[name] = name

    _rebuild_edges(graph, new_graph, node_remap, port_remap)
    return new_graph


def _try_fuse_sequential(
    graph: PAIIRGraph, act_name: str, consumed: set[str], node_remap: dict[str, str]
) -> SequentialOp | None:
    """Try to fuse ``StandaloneCompOp -> StandaloneActOp``."""
    act_node = graph.nodes[act_name]
    assert isinstance(act_node, StandaloneActOp)

    preds = graph.predecessors(act_name)
    if len(preds) != 1:
        return None

    pred_name = preds[0]
    if pred_name in consumed:
        return None

    pred = graph.nodes[pred_name]
    if not isinstance(pred, StandaloneCompOp):
        return None

    if len(graph.successors(pred_name)) != 1:
        return None

    # AvgPool patterns are handled first by the dedicated AvgPool fusion logic,
    # which may choose shared-core or split-core depending on activation type
    # and deployment constraints. Skip here to avoid bypassing that policy.
    if is_value_avgpool(pred.comp):
        return None

    return _materialize_shared_sequential(
        pred_name, pred, act_name, act_node, consumed, node_remap
    )


def _try_fuse_accumulate(
    graph: PAIIRGraph,
    act_name: str,
    consumed: set[str],
    node_remap: dict[str, str],
    port_remap: dict[str, int],
) -> AccumulateOp | None:
    """Try to fuse ``CompOps -> PotentialAddOp -> ActivationOp``."""
    act_node = graph.nodes[act_name]
    assert isinstance(act_node, StandaloneActOp)

    preds = graph.predecessors(act_name)
    if len(preds) != 1:
        return None

    add_name = preds[0]
    if add_name in consumed:
        return None

    add_node = graph.nodes[add_name]
    if not isinstance(add_node, PotentialAddOp):
        return None

    if len(graph.successors(add_name)) != 1:
        return None

    comp_preds = graph.predecessors(add_name)
    if any(p in consumed for p in comp_preds):
        return None
    if not all(isinstance(graph.nodes[p], StandaloneCompOp) for p in comp_preds):
        return None
    if not all(len(graph.successors(p)) == 1 for p in comp_preds):
        return None

    comps = [graph.nodes[p].comp for p in comp_preds]  # type: ignore[union-attr]
    op_signs = add_node.signs

    fused = AccumulateOp(comps=comps, act=act_node.act, op_signs=op_signs)
    fused_input_layouts = []
    for p in comp_preds:
        fused_input_layouts.extend(graph.nodes[p].input_layouts)  # type: ignore[union-attr]
    fused.input_layouts = tuple(fused_input_layouts)
    fused.output_layouts = act_node.output_layouts

    consumed.add(act_name)
    consumed.add(add_name)
    node_remap[act_name] = fused.name
    node_remap[add_name] = fused.name
    for i, p in enumerate(comp_preds):
        consumed.add(p)
        node_remap[p] = fused.name
        port_remap[p] = i

    return fused


def _rebuild_edges(
    old_graph: PAIIRGraph,
    new_graph: PAIIRGraph,
    node_remap: dict[str, str],
    port_remap: dict[str, int],
) -> None:
    """Rebuild edges in the new graph, remapping consumed nodes."""
    seen: set[tuple[str, str, int]] = set()
    for edge in old_graph.edges:
        src = node_remap.get(edge.src, edge.src)
        dst = node_remap.get(edge.dst, edge.dst)
        if src == dst:
            continue

        port = port_remap.get(edge.dst, edge.dst_port)
        key = (src, dst, port)
        if key in seen:
            continue

        seen.add(key)
        new_graph.add_edge(src, dst, dst_port=port)


def validate_graph(graph: PAIIRGraph) -> None:
    """Validate and clean up a fused PAIIR graph.

    This is the *mid-pipeline* structural validation step. It is intentionally
    limited to checks that make sense before data-format propagation and tick
    assignment run. In particular, it may remove disconnected nodes produced by
    earlier bypass / fusion decisions.

    Use :func:`validate_compiled_graph` for the final post-pass validation of a
    fully compiled graph.
    """
    graph.lint(allow_disconnected=True)

    to_remove = graph.disconnected_nodes()
    if to_remove:
        for name in to_remove:
            _remove_node(graph, name)
        warnings.warn(GraphCleanupWarning(to_remove))

    if not graph.input_nodes():
        raise GraphValidationError(
            ["graph has no input node after cleanup (all were disconnected)"]
        )
    if not graph.output_nodes():
        raise GraphValidationError(
            ["graph has no output node after cleanup (all were disconnected)"]
        )

    graph.lint()

    errors: list[str] = []
    missing_shape_nodes: list[str] = []

    for name, node in graph.nodes.items():
        if not isinstance(node, OpNode):
            continue
        if isinstance(node, SplitOp):
            if not node.input_layouts or any(
                not layout.shape for layout in node.input_layouts
            ):
                missing_shape_nodes.append(name)
            continue
        if not _single_output_shape(node):
            missing_shape_nodes.append(name)

    if missing_shape_nodes:
        names = ", ".join(missing_shape_nodes)
        errors.append(
            f"{len(missing_shape_nodes)} OpNode(s) missing shape info "
            f"(pass sample_inputs to torch_to_paiir): {names}"
        )

    for name, node in graph.nodes.items():
        if isinstance(node, PotentialAddOp):
            _validate_potential_add_contract(errors, graph, name, node)
        if isinstance(node, SplitOp):
            _validate_split_contract(errors, graph, name, node)
        if not isinstance(node, OfflineCoreOp):
            continue
        _validate_lut_mode_consistency(errors, name, node)

    if errors:
        raise GraphValidationError(errors)


def validate_compiled_graph(graph: PAIIRGraph) -> None:
    """Validate a fully compiled graph before returning it to callers.

    This is the *final* compile-stage validation step. Unlike
    :func:`validate_graph`, it does not perform cleanup. Instead it verifies the
    invariants that should hold after all compile-time passes have run:

    - every node lies on some input-to-output path
    - every :class:`OpNode` has shape and dims metadata
    - every :class:`OfflineCoreOp` has propagated data formats
    - every :class:`OfflineCoreOp` has valid tick parameters
    """
    graph.lint()

    errors: list[str] = []

    for name, node in graph.nodes.items():
        if isinstance(node, InputNode):
            if node.signal_semantics.output_domain is None:
                errors.append(f"InputNode '{name}' is missing output_domain")
            continue

        if isinstance(node, OutputNode):
            if node.signal_semantics.output_domain is None:
                errors.append(f"OutputNode '{name}' is missing output_domain")
            continue

        if not isinstance(node, OpNode):
            continue

        if isinstance(node, SplitOp):
            if not node.input_layouts or any(
                not layout.shape for layout in node.input_layouts
            ):
                errors.append(f"SplitOp '{name}' is missing input_layouts")
            if not node.input_layouts or any(
                not layout.dims for layout in node.input_layouts
            ):
                errors.append(f"SplitOp '{name}' is missing input layout dims")
            if not node.output_layouts:
                errors.append(f"SplitOp '{name}' is missing output_layouts")
            if node.signal_semantics.output_domain is None:
                errors.append(f"SplitOp '{name}' is missing output_domain")
            _validate_split_contract(errors, graph, name, node)
            continue

        if not node.input_layouts or any(
            not layout.shape for layout in node.input_layouts
        ):
            errors.append(f"OpNode '{name}' is missing input_layouts")
        if not node.output_layouts or any(
            not layout.shape for layout in node.output_layouts
        ):
            errors.append(f"OpNode '{name}' is missing output_layouts")
        if not node.input_layouts or any(
            not layout.dims for layout in node.input_layouts
        ):
            errors.append(f"OpNode '{name}' is missing input layout dims")
        if not node.output_layouts or any(
            not layout.dims for layout in node.output_layouts
        ):
            errors.append(f"OpNode '{name}' is missing output layout dims")
        if node.signal_semantics.output_domain is None:
            errors.append(f"OpNode '{name}' is missing output_domain")

        if isinstance(node, ConcatOp):
            _validate_concat_contract(errors, graph, name, node)

        if isinstance(node, TransformOp):
            _validate_reshape_contract(errors, graph, name, node)

        if isinstance(node, PotentialAddOp):
            _validate_potential_add_contract(errors, graph, name, node)

        if not isinstance(node, OfflineCoreOp):
            continue

        _validate_lut_mode_consistency(errors, name, node)
        _validate_per_channel_export_param_contract(
            errors, name, node, node.neuron_params.thres_pos, "thres_pos"
        )
        _validate_per_channel_export_param_contract(
            errors, name, node, node.neuron_params.leak_v, "leak_v"
        )
        _validate_output_domain_consistency(errors, name, node)
        _validate_32bit_input_contract(errors, name, node)

        try:
            node.core_params.validate_data_formats()
        except ValueError as exc:
            errors.append(f"'{name}' {exc}")

        try:
            node.core_params.validate_tick_params()
        except ValueError as exc:
            errors.append(f"'{name}' {exc}")

    if errors:
        raise GraphValidationError(errors)


def _validate_output_domain_consistency(
    errors: list[str], name: str, node: OfflineCoreOp
) -> None:
    if (domain := node.signal_semantics.output_domain) is None:
        return

    output_type = node.neuron_params.output_type
    expected = (
        OutputType.VALUE if domain is SignalDomain.VALUE else OutputType.POTENTIAL
    )
    if output_type != expected:
        errors.append(
            f"OfflineCoreOp '{name}' output_domain={domain.name} but "
            f"neuron_params.output_type={output_type.name}"
        )
        return


def _validate_per_channel_export_param_contract(
    errors: list[str],
    name: str,
    node: OfflineCoreOp,
    value: float | torch.Tensor,
    param_name: str,
) -> None:
    if not torch.is_tensor(value):
        return

    output_shape = _single_output_shape(node)
    if len(output_shape) < 2:
        errors.append(
            f"OfflineCoreOp '{name}' has per-channel {param_name} but "
            f"output_shape={tuple(output_shape)} has no channel dimension"
        )
        return

    if output_shape[0] != 1:
        errors.append(
            f"OfflineCoreOp '{name}' has per-channel {param_name} but "
            f"batch size {output_shape[0]} is not supported"
        )

    if value.ndim != 1:
        errors.append(
            f"OfflineCoreOp '{name}' has per-channel {param_name} but "
            f"shape={tuple(value.shape)} is not a 1D tensor"
        )
        return

    channel_count = output_shape[1]
    if value.numel() != channel_count:
        errors.append(
            f"OfflineCoreOp '{name}' {param_name} has {value.numel()} element(s) "
            f"but output channel count is {channel_count}"
        )


def _validate_32bit_input_contract(
    errors: list[str], name: str, node: OfflineCoreOp
) -> None:
    """Reject unsupported 32-bit membrane-input consumers before backend export.

    Backend frame packing only supports ``WIDTH_32BIT`` inputs for direct-add
    style consumers. In the current deploy contract that means:

    - ``PotentialAddOp``: explicit membrane add
    - ``StandaloneActOp``: activation-only core fed by an implicit identity path

    Other offline-core operators may still infer ``WIDTH_32BIT`` from a
    predecessor's membrane-potential output, but the backend cannot export
    those weighted-consumer shapes faithfully. Fail here with a clear compile-
    time error instead of surfacing a later backend packing failure.
    """
    if node.core_params.input_width != DataWidth.WIDTH_32BIT:
        return

    if isinstance(node, (PotentialAddOp, StandaloneActOp)):
        if node.core_params.add_potential != AddPotentialMode.DIRECT_ADD:
            errors.append(
                f"OfflineCoreOp '{name}' receives WIDTH_32BIT input but "
                "add_potential is not AddPotentialMode.DIRECT_ADD"
            )
        return

    errors.append(
        f"OfflineCoreOp '{name}' ({type(node).__name__}) receives WIDTH_32BIT "
        "membrane input, but only PotentialAddOp and StandaloneActOp are "
        "deployable 32-bit input consumers in the current backend contract"
    )


def _get_node_output_shape(node: PAIIRNode) -> torch.Size:
    if isinstance(node, InputNode):
        return node.shape
    if isinstance(node, OutputNode):
        return node.shape
    if isinstance(node, OpNode):
        return _single_output_shape(node)
    return torch.Size()


def _try_get_edge_output_shape(graph: PAIIRGraph, edge: Edge) -> torch.Size:
    """Best-effort shape probe for one concrete graph edge.

    This helper is intentionally non-throwing: validation code uses it to
    assemble richer error messages later, so malformed split metadata falls
    back to ``torch.Size()`` instead of aborting early.
    """
    node = graph.nodes[edge.src]
    if isinstance(node, SplitOp):
        if node.num_inputs != 1 or not node.input_layouts[0].shape:
            return torch.Size()

        input_shape = node.input_layouts[0].shape
        rank = len(input_shape)
        split_dim = node.dim if node.dim >= 0 else node.dim + rank
        if split_dim < 0 or split_dim >= rank:
            return torch.Size()

        input_extent = input_shape[split_dim]
        if isinstance(node.sections, tuple):
            if sum(node.sections) != input_extent:
                return torch.Size()
        elif node.sections <= 0:
            return torch.Size()

        # Split branch shapes are derived on demand rather than persisted on
        # the node because the backend does not consume them directly.
        if edge.src_port < 0 or edge.src_port >= node.num_outputs:
            return torch.Size()
        return node.output_layouts[edge.src_port].shape

    return _get_node_output_shape(node)


def _validate_concat_contract(
    errors: list[str], graph: PAIIRGraph, name: str, node: ConcatOp
) -> None:
    incoming = graph.incoming_edges(name)
    preds = [edge.src for edge in incoming]
    pred_shapes = [_try_get_edge_output_shape(graph, edge) for edge in incoming]
    input_shapes = _layout_shapes(node)
    output_shape = _single_output_shape(node)

    if len(preds) < 1:
        errors.append(f"ConcatOp '{name}' must define at least one input path")
        return

    if input_shapes and len(input_shapes) != len(preds):
        errors.append(
            f"ConcatOp '{name}' has {len(input_shapes)} input layouts but "
            f"{len(preds)} predecessor(s)"
        )
        return

    if input_shapes:
        mismatched = [
            (pred, pred_shape, expected_shape)
            for pred, pred_shape, expected_shape in zip(
                preds, pred_shapes, input_shapes
            )
            if pred_shape and pred_shape != expected_shape
        ]
        if mismatched:
            details = ", ".join(
                f"{pred}: pred_output_shape={pred_shape}, input_shape={expected_shape}"
                for pred, pred_shape, expected_shape in mismatched
            )
            errors.append(f"ConcatOp '{name}' predecessor shape mismatch: {details}")
            return

        ranks = {len(shape) for shape in input_shapes if shape}
        if len(ranks) > 1:
            errors.append(
                f"ConcatOp '{name}' requires same-rank operands, got {input_shapes}"
            )
            return

        if ranks:
            rank = next(iter(ranks))
            dim = node.dim if node.dim >= 0 else node.dim + rank
            if dim < 0 or dim >= rank:
                errors.append(
                    f"ConcatOp '{name}' has invalid concat dim={node.dim} for rank {rank}"
                )
                return

            base_shape = list(input_shapes[0])
            concat_extent = 0
            for shape in input_shapes:
                if any(
                    shape[axis] != base_shape[axis]
                    for axis in range(rank)
                    if axis != dim
                ):
                    errors.append(
                        f"ConcatOp '{name}' non-concat dims differ across inputs: {input_shapes}"
                    )
                    return
                concat_extent += shape[dim]

            expected_output = tuple(
                concat_extent if axis == dim else base_shape[axis]
                for axis in range(rank)
            )
            if output_shape and expected_output != output_shape:
                errors.append(
                    f"ConcatOp '{name}' output_shape mismatch: "
                    f"expected {expected_output}, got {output_shape}"
                )


def _validate_reshape_contract(
    errors: list[str], graph: PAIIRGraph, name: str, node: TransformOp
) -> None:
    incoming = graph.incoming_edges(name)
    preds = [edge.src for edge in incoming]
    if len(preds) != 1:
        errors.append(
            f"{type(node).__name__} '{name}' must have exactly one predecessor, got {len(preds)}"
        )
        return

    pred_shape = _try_get_edge_output_shape(graph, incoming[0])
    input_shapes = _layout_shapes(node)
    input_dims = _layout_input_dims(node)
    output_shape = _single_output_shape(node)
    if input_shapes and len(input_shapes) != 1:
        errors.append(
            f"{type(node).__name__} '{name}' has {len(input_shapes)} input layouts (expected 1)"
        )
        return

    if input_shapes and pred_shape:
        expected_input_shape = input_shapes[0]
        if pred_shape != expected_input_shape:
            logical_pred_shape = None
            if len(input_dims) == 1:
                logical_pred_shape = shape_after_dims(pred_shape, input_dims[0])

            if logical_pred_shape != expected_input_shape:
                details = f"pred_output_shape={pred_shape}, input_shape={expected_input_shape}"
                if logical_pred_shape is not None:
                    details += f", pred_shape_after_input_dims={logical_pred_shape}"
                errors.append(
                    f"{type(node).__name__} '{name}' predecessor shape mismatch: {details}"
                )
                return

    if input_shapes and output_shape:
        in_numel = math.prod(input_shapes[0])
        out_numel = math.prod(output_shape)
        if in_numel != out_numel:
            errors.append(
                f"{type(node).__name__} '{name}' changes element count: "
                f"input_shape={input_shapes[0]}, output_shape={output_shape}"
            )


def _validate_split_contract(
    errors: list[str], graph: PAIIRGraph, name: str, node: SplitOp
) -> None:
    incoming = graph.incoming_edges(name)
    preds = [edge.src for edge in incoming]
    if len(preds) != 1:
        errors.append(
            f"SplitOp '{name}' must have exactly one predecessor, got {len(preds)}"
        )
        return

    pred_shape = _try_get_edge_output_shape(graph, incoming[0])
    input_shapes = _layout_shapes(node)
    input_dims = _layout_input_dims(node)
    if input_shapes and len(input_shapes) != 1:
        errors.append(
            f"SplitOp '{name}' has {len(input_shapes)} input layouts (expected 1)"
        )
        return

    if input_shapes and pred_shape and pred_shape != input_shapes[0]:
        errors.append(
            f"SplitOp '{name}' predecessor shape mismatch: "
            f"pred_output_shape={pred_shape}, input_shape={input_shapes[0]}"
        )
        return

    if len(input_dims) == 1 and any(
        layout.dims != input_dims[0] for layout in node.output_layouts
    ):
        errors.append(
            f"SplitOp '{name}' output layout dims must match input layout dims, got "
            f"input_dims={input_dims[0]}, output_dims={[layout.dims for layout in node.output_layouts]}"
        )
        return

    if not input_shapes or not input_shapes[0]:
        return

    input_shape = input_shapes[0]
    try:
        expected_outputs = infer_split_output_shapes(
            input_shape, node.sections, node.dim
        )
    except ValueError as e:
        errors.append(f"SplitOp '{name}' has invalid split spec: {e}")
        return

    if node.output_layouts and len(node.output_layouts) != len(expected_outputs):
        errors.append(
            f"SplitOp '{name}' has {len(node.output_layouts)} output layouts but "
            f"{len(expected_outputs)} split output(s) were inferred"
        )
        return

    for idx, expected_shape in enumerate(expected_outputs):
        if node.output_layouts and node.output_layouts[idx].shape != expected_shape:
            errors.append(
                f"SplitOp '{name}' output layout mismatch at result {idx}: "
                f"expected {expected_shape}, got {node.output_layouts[idx].shape}"
            )
            return

    for edge in graph.outgoing_edges(name):
        if edge.src_port < 0 or edge.src_port >= len(expected_outputs):
            errors.append(
                f"SplitOp '{name}' edge to '{edge.dst}' "
                f"(dst_port={edge.dst_port}, src_port={edge.src_port}) exceeds "
                f"{len(expected_outputs)} split output(s)"
            )


def propagate_signal_semantics(
    graph: PAIIRGraph, input_formats: dict[str, DataFormat] | None = None
) -> None:
    """Infer and fill node-level semantic annotations for every graph node.

    This pass writes:

    - ``signal_semantics.output_domain``: coarse VALUE vs POTENTIAL semantics
    - ``signal_semantics.known_code_range``: optional exact VALUE code range when derivable

    The pass is intentionally operator-driven rather than format-driven:
    routing-like nodes mostly forward predecessor annotations, while deployable
    cores derive semantics from their own activation or from explicit operator
    contracts such as standalone MaxPool.
    """
    if input_formats is None:
        input_formats = {}

    input_node_names = {
        n for n, node in graph.nodes.items() if isinstance(node, InputNode)
    }
    for name in input_formats:
        if name not in input_node_names:
            warnings.warn(
                f"input_formats key '{name}' does not match any InputNode in the graph"
            )

    for node in graph.nodes.values():
        node.signal_semantics.output_domain = None
        node.signal_semantics.known_code_range = None

    errors: list[str] = []

    for name in graph.topo_sort():
        node = graph.nodes[name]

        if isinstance(node, InputNode):
            _set_node_signal_semantics(
                node,
                SignalDomain.VALUE,
                code_range_for_data_format(
                    _resolve_input_node_format(name, input_formats)
                ),
            )
            continue

        pred_facts = _gather_pred_signal_facts(graph, name)
        if pred_facts.has_missing_domain:
            errors.append(
                f"{type(node).__name__} '{name}' predecessor output_domain is missing"
            )
            continue

        if isinstance(node, (OutputNode, TransformOp, SplitOp)):
            inferred = _infer_passthrough_signal_semantics(pred_facts)
            if inferred is not None:
                _set_node_signal_semantics(node, *inferred)
            continue

        if isinstance(node, ConcatOp):
            if pred_facts.known_domains and any(
                domain != pred_facts.known_domains[0]
                for domain in pred_facts.known_domains
            ):
                errors.append(
                    f"ConcatOp '{name}' all predecessor domains must match, got "
                    f"{[domain.name for domain in pred_facts.known_domains]}"
                )
                continue
            inferred = _infer_concat_signal_semantics(pred_facts)
            if inferred is not None:
                _set_node_signal_semantics(node, *inferred)
            continue

        if isinstance(node, GeneralAddOp):
            inferred = _infer_general_add_signal_semantics(node, pred_facts)
            if inferred is None:
                errors.append(f"GeneralAddOp '{name}' cannot infer output_domain")
            elif inferred is False:
                errors.append(
                    f"GeneralAddOp '{name}' mixed tensor operand domains are not supported, got "
                    f"{[domain.name for domain in pred_facts.known_domains]}"
                )
            else:
                _set_node_signal_semantics(node, *inferred)
            continue

        if isinstance(node, PotentialAddOp):
            if pred_facts.known_domains and any(
                domain is not SignalDomain.POTENTIAL
                for domain in pred_facts.known_domains
            ):
                errors.append(
                    f"PotentialAddOp '{name}' all predecessor domains must be POTENTIAL, got "
                    f"{[domain.name for domain in pred_facts.known_domains]}"
                )
                continue
            _set_node_signal_semantics(node, SignalDomain.POTENTIAL, None)
            continue

        if is_standalone_maxpool(node):
            if pred_facts.known_domains and any(
                domain is not SignalDomain.VALUE for domain in pred_facts.known_domains
            ):
                errors.append(
                    f"Standalone MaxPool '{name}' requires VALUE-domain predecessor, got "
                    f"{[domain.name for domain in pred_facts.known_domains]}"
                )
                continue
            inferred = _infer_standalone_maxpool_signal_semantics(pred_facts)
            if inferred is not None:
                _set_node_signal_semantics(node, *inferred)
            continue

        if isinstance(node, StandaloneCompOp) and is_value_avgpool(node.comp):
            inferred = _infer_standalone_avgpool_signal_semantics(pred_facts)
            if inferred is not None:
                _set_node_signal_semantics(node, *inferred)
                continue

        if isinstance(node, OfflineCoreOp):
            _set_node_signal_semantics(
                node, *_infer_offline_core_signal_semantics(node, pred_facts)
            )
            continue

    if errors:
        raise GraphValidationError(errors)


def _infer_output_signal_domain(node: OfflineCoreOp) -> SignalDomain:
    """Infer the coarse output domain for a generic offline core."""
    if node.neuron_params.output_type == OutputType.POTENTIAL:
        return SignalDomain.POTENTIAL
    return SignalDomain.VALUE


def _set_node_signal_semantics(
    node: PAIIRNode, output_domain: SignalDomain, known_code_range: KnownCodeRange
) -> None:
    """Write both signal-semantics fields together to keep them in sync."""
    node.signal_semantics.output_domain = output_domain
    node.signal_semantics.known_code_range = known_code_range


def _gather_pred_signal_facts(graph: PAIIRGraph, node_name: str) -> _PredSignalFacts:
    """Gather predecessor domains and exact code ranges for one node."""
    preds = graph.predecessors(node_name)
    return _PredSignalFacts(
        domains=tuple(graph.nodes[p].signal_semantics.output_domain for p in preds),
        code_ranges=tuple(
            graph.nodes[p].signal_semantics.known_code_range for p in preds
        ),
    )


def _infer_passthrough_signal_semantics(
    pred_facts: _PredSignalFacts,
) -> NodeSignal | None:
    """Forward predecessor semantics through transparent single-input nodes."""
    domain = pred_facts.single_domain()
    if domain is None:
        return None
    return domain, pred_facts.single_code_range()


def _infer_concat_signal_semantics(pred_facts: _PredSignalFacts) -> NodeSignal | None:
    """Infer concat semantics once domain compatibility is already validated."""
    if not pred_facts.known_domains:
        return None
    return pred_facts.known_domains[0], pred_facts.merged_code_range()


def _infer_general_add_signal_semantics(
    node: GeneralAddOp, pred_facts: _PredSignalFacts
) -> tuple[SignalDomain, None] | Literal[False] | None:
    """Infer coarse add semantics without attempting exact value-range algebra.

    ``False`` is a sentinel for "mixed predecessor domains", letting the caller
    preserve the existing error wording without duplicating the rule.
    """
    if pred_facts.known_domains and any(
        domain != pred_facts.known_domains[0] for domain in pred_facts.known_domains
    ):
        return False
    if pred_facts.known_domains:
        return pred_facts.known_domains[0], None
    if node.has_const_operands:
        return SignalDomain.VALUE, None
    return None


def _infer_standalone_maxpool_signal_semantics(
    pred_facts: _PredSignalFacts,
) -> NodeSignal | None:
    """Infer standalone MaxPool semantics from VALUE-domain predecessors."""
    if not pred_facts.known_domains:
        return None
    return SignalDomain.VALUE, pred_facts.merged_code_range()


def _infer_standalone_avgpool_signal_semantics(
    pred_facts: _PredSignalFacts,
) -> NodeSignal | None:
    """Infer standalone AvgPool semantics conservatively.

    Standalone AvgPool computes a VALUE-domain average when its effective
    sources already live in the VALUE domain, but we do not attempt to derive
    an exact output code range here before the dedicated rewrite/materialization
    logic runs.
    """
    if not pred_facts.known_domains:
        return None
    if any(domain is not SignalDomain.VALUE for domain in pred_facts.known_domains):
        return None
    return SignalDomain.VALUE, None


def _infer_offline_core_signal_semantics(
    node: OfflineCoreOp, pred_facts: _PredSignalFacts
) -> NodeSignal:
    """Infer semantics for generic deployable offline cores."""
    domain = _infer_output_signal_domain(node)
    return domain, _infer_node_known_code_range(
        node, domain, pred_facts.present_code_ranges
    )


def _resolve_input_node_format(
    name: str, input_formats: dict[str, DataFormat]
) -> DataFormat:
    """Resolve external input format from explicit overrides or fixed default."""
    return input_formats[name] if name in input_formats else _DEFAULT_INPUT_FORMAT


def _infer_node_known_code_range(
    node: OfflineCoreOp,
    output_domain: SignalDomain,
    pred_code_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Infer the exact VALUE code range emitted by one offline core.

    The pass stays conservative: when a rule cannot guarantee an exact integer
    output range, it returns ``None`` and lets later stages fall back to format
    envelopes instead of pretending the range is known.
    """
    if output_domain is not SignalDomain.VALUE:
        return None
    if is_standalone_maxpool(node):
        return merge_code_ranges(pred_code_ranges)

    act = _get_node_act(node)
    if act is None:
        return None
    return infer_output_code_range(act)


def validate_deployable_graph(graph: PAIIRGraph) -> None:
    """Validate that a compiled graph contains only backend-ready IR nodes."""
    errors: list[str] = []

    for name, node in graph.nodes.items():
        if isinstance(node, GeneralAddOp):
            errors.append(
                f"GeneralAddOp '{name}' expression-layer add must be specialized before deployment"
            )
            continue

        if isinstance(node, SplitOp):
            errors.append(
                f"SplitOp '{name}' is frontend-only IR and is not part of the backend-ready PAIIR subset"
            )
            continue

        if not isinstance(node, _DEPLOYABLE_GRAPH_NODE_TYPES):
            errors.append(
                f"{type(node).__name__} '{name}' is not part of the backend-ready PAIIR subset"
            )
            continue

        if isinstance(node, PotentialAddOp):
            pred_domains = [
                graph.nodes[p].signal_semantics.output_domain
                for p in graph.predecessors(name)
            ]
            if any(domain is None for domain in pred_domains):
                errors.append(
                    f"PotentialAddOp '{name}' predecessor output_domain is missing"
                )
            else:
                known_pred_domains = [
                    domain for domain in pred_domains if domain is not None
                ]
                if any(
                    domain is not SignalDomain.POTENTIAL
                    for domain in known_pred_domains
                ):
                    errors.append(
                        f"PotentialAddOp '{name}' all predecessor domains must be POTENTIAL, got "
                        f"{[domain.name for domain in known_pred_domains]}"
                    )

        if isinstance(node, ConcatOp):
            pred_domains = [
                graph.nodes[p].signal_semantics.output_domain
                for p in graph.predecessors(name)
            ]
            if any(domain is None for domain in pred_domains):
                errors.append(f"ConcatOp '{name}' predecessor output_domain is missing")
            else:
                known_pred_domains = [
                    domain for domain in pred_domains if domain is not None
                ]
                if known_pred_domains and any(
                    domain != known_pred_domains[0] for domain in known_pred_domains
                ):
                    errors.append(
                        f"ConcatOp '{name}' all predecessor domains must match, got "
                        f"{[domain.name for domain in known_pred_domains]}"
                    )

        if isinstance(node, AccumulateOp):
            preds = graph.predecessors(name)
            expected_paths = len(node.comps)
            signs = list(node.signs)
            weights = node.weights

            if expected_paths < 2:
                errors.append(
                    f"AccumulateOp '{name}' must define at least two compute paths"
                )

            if len(preds) != expected_paths:
                errors.append(
                    f"AccumulateOp '{name}' has {len(preds)} predecessor(s) but "
                    f"{expected_paths} compute path(s)"
                )

            if len(signs) != expected_paths:
                errors.append(
                    f"AccumulateOp '{name}' has {len(signs)} sign entries but "
                    f"{expected_paths} compute path(s)"
                )

            if any(sign not in (-1, 1) for sign in signs):
                errors.append(
                    f"AccumulateOp '{name}' signs must be +/-1 only, got {tuple(signs)}"
                )

            if node.input_layouts and len(node.input_layouts) != expected_paths:
                errors.append(
                    f"AccumulateOp '{name}' has {len(node.input_layouts)} input layouts but "
                    f"{expected_paths} compute path(s)"
                )

            if node.input_layouts and len(node.input_layouts) != expected_paths:
                errors.append(
                    f"AccumulateOp '{name}' has {len(node.input_layouts)} input layouts but "
                    f"{expected_paths} compute path(s)"
                )

            if weights is not None and len(weights) != expected_paths:
                errors.append(
                    f"AccumulateOp '{name}' has {len(weights)} weight tensors but "
                    f"{expected_paths} compute path(s)"
                )

    if errors:
        raise GraphValidationError(errors)


def _remove_node(graph: PAIIRGraph, name: str) -> None:
    """Remove a node and all its edges from the graph."""
    graph.remove_node(name)


def _validate_potential_add_contract(
    errors: list[str], graph: PAIIRGraph, name: str, node: PotentialAddOp
) -> None:
    preds = graph.predecessors(name)
    expected_paths = len(node.signs)

    if expected_paths < 2:
        errors.append(
            f"PotentialAddOp '{name}' must define at least two signed input paths"
        )

    if len(preds) != expected_paths:
        errors.append(
            f"PotentialAddOp '{name}' has {len(preds)} predecessor(s) but "
            f"{expected_paths} sign/path entries"
        )

    if node.input_layouts and len(node.input_layouts) != expected_paths:
        errors.append(
            f"PotentialAddOp '{name}' has {len(node.input_layouts)} input layouts but "
            f"{expected_paths} sign/path entries"
        )

    input_shapes = _layout_shapes(node)
    output_shape = _single_output_shape(node)
    if output_shape and input_shapes:
        mismatched = [shape for shape in input_shapes if shape != output_shape]
        if mismatched:
            errors.append(
                f"PotentialAddOp '{name}' requires same-shape operands, got "
                f"input_shapes={input_shapes}, output_shape={output_shape}"
            )


def _validate_lut_mode_consistency(
    errors: list[str], name: str, node: OfflineCoreOp
) -> None:
    act = _get_node_act(node)
    if act is None:
        return

    has_lut = act.lut is not None
    if has_lut and node.core_params.snn_mode != SNNMode.ANN:
        errors.append(
            f"'{name}' has LUT activation but snn_mode is "
            f"{node.core_params.snn_mode!r} (expected SNNMode.ANN)"
        )
    elif not has_lut and node.core_params.snn_mode != SNNMode.SNN:
        errors.append(
            f"'{name}' has no LUT activation but snn_mode is "
            f"{node.core_params.snn_mode!r} (expected SNNMode.SNN)"
        )


_DEFAULT_INPUT_FORMAT = (DataSign.SIGNED, DataWidth.WIDTH_8BIT)
_WEIGHTLESS_WEIGHT_FORMAT = (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)


def propagate_data_format(
    graph: PAIIRGraph, input_formats: dict[str, DataFormat] | None = None
) -> None:
    """Infer and write data-format fields on every :class:`OfflineCoreOp`.

    This pass fills the three backend-facing format groups stored in
    :class:`~paibox.paiir.ir.calc_params.OfflineCoreParams`:

    - ``input_sign`` / ``input_width``
    - ``output_sign`` / ``output_width``
    - ``weight_sign`` / ``weight_width``

    The propagation is intentionally split into two topological passes:

    1. Seed an effective format for every external input, then assign each
       deployable core's intrinsic output format and weight format.
    2. Propagate those resolved formats through routing-only nodes
       (:class:`ConcatOp`, :class:`TransformOp`, :class:`SplitOp`, :class:`OutputNode`) and finally
       back-fill each deployable core's input format from its predecessors.

    The ordering matters because a core's input format depends on the already
    resolved output formats of its predecessors, while many cores can determine
    their own output format directly from their activation semantics.

    Args:
        graph: Fused PAIIR graph whose node connectivity has already been
            validated.
        input_formats: Optional mapping from ``InputNode`` name to explicit
            external input format. Any omitted input falls back to
            a fixed signed 8-bit default.
    """
    if input_formats is None:
        input_formats = {}

    input_node_names = {
        n for n, node in graph.nodes.items() if isinstance(node, InputNode)
    }
    # Warn early if the caller passed a boundary-name override that does not
    # correspond to the current graph, but keep processing valid entries.
    for name in input_formats:
        if name not in input_node_names:
            warnings.warn(
                f"input_formats key '{name}' does not match any InputNode in the graph"
            )

    resolved: dict[str, DataFormat] = {}

    # Pass 1: establish each deployable core's own output and weight formats.
    # This gives later consumers a stable predecessor-output view before we
    # compute per-core input formats.
    for name in graph.topo_sort():
        _seed_node_data_formats(graph, name, resolved, input_formats)

    # Pass 2: thread those resolved formats through non-deploy routing nodes
    # and then back-fill each deployable core's input format from predecessor
    # outputs.
    for name in graph.topo_sort():
        node = graph.nodes[name]

        routing_fmt = _infer_routing_resolved_format(graph, name, node, resolved)
        if routing_fmt is not None:
            resolved[name] = routing_fmt
            continue

        if not isinstance(node, OfflineCoreOp):
            continue

        _update_offline_core_input_format(graph, name, node, resolved)


def _seed_node_data_formats(
    graph: PAIIRGraph,
    node_name: str,
    resolved: dict[str, DataFormat],
    input_formats: dict[str, DataFormat],
) -> None:
    """Seed node-local output/weight formats before input back-fill starts."""
    node = graph.nodes[node_name]

    if isinstance(node, InputNode):
        resolved[node_name] = _resolve_input_node_format(node_name, input_formats)
        return
    if isinstance(node, OutputNode):
        return
    if is_format_transparent_routing_node(node):
        # Routing nodes have no `core_params`; their effective formats are
        # propagated in the second pass after predecessor outputs are known.
        return
    if not isinstance(node, OfflineCoreOp):
        return

    pred_formats = _collect_effective_predecessor_formats(graph, node_name, resolved)
    out_fmt = _infer_node_output_format(node, pred_formats)
    node.core_params.set_output_format(out_fmt)
    resolved[node_name] = out_fmt
    node.core_params.set_weight_format(_infer_node_weight_format(node))


def _infer_routing_resolved_format(
    graph: PAIIRGraph, node_name: str, node: PAIIRNode, resolved: dict[str, DataFormat]
) -> DataFormat | None:
    """Propagate already-resolved formats through non-deploy routing nodes."""
    match node.__format_flow__:
        case FormatFlow.PASS_THROUGH:
            return _single_resolved_predecessor_format(graph, node_name, resolved)
        case FormatFlow.MERGE:
            pred_formats = _resolved_predecessor_formats(graph, node_name, resolved)
            return merge_data_formats(pred_formats) if pred_formats else None
        case FormatFlow.NONE:
            return None


def _resolved_predecessor_formats(
    graph: PAIIRGraph, node_name: str, resolved: dict[str, DataFormat]
) -> list[DataFormat]:
    """Return already-resolved direct predecessor formats for one node."""
    return [resolved[p] for p in graph.predecessors(node_name) if p in resolved]


def _single_resolved_predecessor_format(
    graph: PAIIRGraph, node_name: str, resolved: dict[str, DataFormat]
) -> DataFormat | None:
    """Return the first resolved predecessor format for pass-through rules."""
    pred_formats = _resolved_predecessor_formats(graph, node_name, resolved)
    if not pred_formats:
        return None
    return pred_formats[0]


def _update_offline_core_input_format(
    graph: PAIIRGraph,
    node_name: str,
    node: OfflineCoreOp,
    resolved: dict[str, DataFormat],
) -> None:
    """Back-fill one deployable core's input format and dependent config."""
    pred_formats = _resolved_predecessor_formats(graph, node_name, resolved)
    in_fmt = merge_data_formats(pred_formats)
    node.core_params.set_input_format(in_fmt)
    _derive_input_add_potential_mode(graph, node_name, node)
    if is_standalone_maxpool(node):
        refresh_maxpool_export_kind(node)


def _collect_effective_predecessor_formats(
    graph: PAIIRGraph, node_name: str, resolved: dict[str, DataFormat]
) -> list[DataFormat]:
    """Collect predecessor output formats through transparent routing nodes."""
    return collect_effective_predecessor_values(
        graph,
        node_name,
        resolve=lambda _node, name: [resolved[name]] if name in resolved else None,
        passthrough=is_format_transparent_routing_node,
    )


def _infer_node_output_format(
    node: OfflineCoreOp, pred_formats: list[DataFormat] | None = None
) -> DataFormat:
    """Infer the backend-facing output format for one offline core."""
    if is_standalone_maxpool(node):
        if not pred_formats:
            raise ValueError(
                f"Standalone MaxPool '{node.name}' requires predecessor format"
            )
        return merge_data_formats(pred_formats)

    if (
        isinstance(node, StandaloneCompOp)
        and is_value_avgpool(node.comp)
        and node.signal_semantics.output_domain is SignalDomain.VALUE
    ):
        if not pred_formats:
            raise ValueError(
                f"Standalone AvgPool '{node.name}' requires predecessor format"
            )
        return merge_data_formats(pred_formats)

    output_type = node.neuron_params.output_type
    if output_type == OutputType.POTENTIAL:
        return DataSign.SIGNED, DataWidth.WIDTH_32BIT

    act = _get_node_act(node)
    if act is None:
        raise ValueError(
            f"OfflineCoreOp '{node.name}' declares {output_type.name} output but has no "
            "activation; nodes without act must emit POTENTIAL"
        )

    return infer_output_format(act)


def _infer_node_weight_format(node: OfflineCoreOp) -> DataFormat:
    """Infer the backend-facing weight format for one offline core."""
    weight_range = node.get_weight_value_range()
    if weight_range is not None:
        w_min, w_max = weight_range
        return infer_weight_format(w_min, w_max)

    weights = node.weights
    if weights is None:
        return _WEIGHTLESS_WEIGHT_FORMAT

    all_weights = torch.cat([w.flatten() for w in weights])
    w_min = int(all_weights.min().item())
    w_max = int(all_weights.max().item())
    return infer_weight_format(w_min, w_max)


def _derive_input_add_potential_mode(
    graph: PAIIRGraph, node_name: str, node: OfflineCoreOp
) -> None:
    """Derive hardware add-potential mode from predecessor signal semantics.

    Standalone activation cores synthesize an implicit identity connectivity
    path in the backend. When that path carries membrane potentials, the chip
    expects direct membrane accumulation rather than normal weighted
    accumulation.
    """
    if not isinstance(node, StandaloneActOp):
        return

    if (
        _gather_pred_signal_facts(graph, node_name).single_domain()
        is SignalDomain.POTENTIAL
    ):
        node.core_params.add_potential = AddPotentialMode.DIRECT_ADD
    else:
        node.core_params.add_potential = AddPotentialMode.NORMAL


def _get_node_act(node: OfflineCoreOp) -> CoreNeuronV25 | None:
    """Return a node's activation object when the node actually has one."""
    act = getattr(node, "act", None)
    return act if isinstance(act, CoreNeuronV25) else None


class TickOverride(TypedDict, total=False):
    """Per-node timing override for :func:`assign_tick_params`."""

    tick_start: int
    tick_duration: int
    auto_reset: bool


def _validate_overrides(graph: PAIIRGraph, overrides: dict[str, TickOverride]) -> None:
    for name, ovr in overrides.items():
        if name not in graph.nodes:
            raise KeyError(
                f"overrides key '{name}' does not match any node in the graph"
            )
        if "tick_start" in ovr and ovr["tick_start"] < 0:
            raise ValueError(
                f"overrides['{name}']['tick_start'] must be non-negative, "
                f"got {ovr['tick_start']}"
            )
        if "tick_duration" in ovr and ovr["tick_duration"] < 0:
            raise ValueError(
                f"overrides['{name}']['tick_duration'] must be non-negative, "
                f"got {ovr['tick_duration']}"
            )


def assign_tick_params(
    graph: PAIIRGraph,
    tick_duration: int = 0,
    auto_reset: bool = True,
    overrides: dict[str, TickOverride] | None = None,
) -> None:
    """Assign timing parameters on every :class:`OfflineCoreOp`."""
    if tick_duration < 0:
        raise ValueError(f"'tick_duration' must be non-negative, got {tick_duration}")
    if overrides is None:
        overrides = {}

    _validate_overrides(graph, overrides)

    depth: dict[str, int] = {}
    for name in graph.topo_sort():
        node = graph.nodes[name]
        if isinstance(node, InputNode):
            depth[name] = 0
            continue

        preds = graph.predecessors(name)
        pred_depth = max(depth.get(p, 0) for p in preds) if preds else 0
        depth[name] = pred_depth + node.__tick_depth__

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, OfflineCoreOp):
            continue

        cp = node.core_params
        ovr = overrides.get(name, {})

        if "tick_start" in ovr:
            cp.tick_start = ovr["tick_start"]
        elif cp.tick_start is None:
            cp.tick_start = depth[name]

        if "tick_duration" in ovr:
            cp.tick_duration = ovr["tick_duration"]
        elif cp.tick_duration == 0 and tick_duration != 0:
            cp.tick_duration = tick_duration

        if cp.snn_mode == SNNMode.ANN:
            cp.tick_initial = 1
        else:
            node_auto_reset = ovr.get("auto_reset", auto_reset)
            cp.tick_initial = (
                cp.tick_duration if node_auto_reset and cp.tick_duration > 0 else 0
            )

        cp.validate_tick_params()
