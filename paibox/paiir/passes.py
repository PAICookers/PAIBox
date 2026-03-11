"""PAIIR graph-level passes.

Operates on :class:`PAIIRGraph` to transform, annotate, and validate the
IR before backend lowering.

Passes:

- :func:`fuse_to_offline_cores` -- fuse atomic nodes into offline-core units.
- :func:`validate_graph` -- structural validation of a fused graph.
- :func:`propagate_data_format` -- infer and fill data format parameters on
  every :class:`OfflineCoreOp`.
- :func:`assign_tick_params` -- assign timing parameters (tick_start,
  tick_duration, tick_initial) on every :class:`OfflineCoreOp`.
"""

import warnings
from typing import TypedDict

import torch
from paicorelib import RM, DataSign, DataWidth, SNNMode

from .avgpool_compensation import compensate_splitcore_avgpool_threshold
from .core_neuron import ANNNodeV25, IFNodeV25
from .data_format import (
    DataFormat,
    infer_output_format,
    infer_weight_format,
    merge_data_formats,
)
from .exceptions import GraphCleanupWarning, GraphValidationError
from .graph import PAIIRGraph
from .ir_base import InputNode, OutputNode, PAIIRNode
from .lut_activation import LutCustom
from .op_node import (
    AccumulateOp,
    AddOp,
    ConcatOp,
    OfflineCoreOp,
    OpNode,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
    _get_pool_window_size,
    _is_avgpool,
)

__all__ = [
    "fuse_to_offline_cores",
    "validate_graph",
    "propagate_data_format",
    "assign_tick_params",
]


def fuse_to_offline_cores(graph: PAIIRGraph) -> PAIIRGraph:
    """Fuse atomic PAIIR nodes into offline-core units.

    Scans activation nodes and absorbs their compute predecessors:

    - ``StandaloneCompOp -> StandaloneActOp`` => ``SequentialOp``
    - ``CompOp_a + CompOp_b -> AddOp -> ActivationOp`` => ``AccumulateOp``
    - Remaining nodes stay unchanged.

    Args:
        graph: An unfused :class:`PAIIRGraph` (from :func:`torch_to_paiir`).

    Returns:
        A new :class:`PAIIRGraph` with fused nodes.
    """
    consumed: set[str] = set()
    node_remap: dict[str, str] = {}
    port_remap: dict[str, int] = {}
    fused_nodes: dict[str, PAIIRNode] = {}  # new fused name -> node

    # Pass 1: identify fusion groups (scan activation nodes)
    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, StandaloneActOp):
            continue
        if name in consumed:
            continue

        result = _try_fuse_sequential(graph, name, consumed, node_remap)
        if result is not None:
            fused_nodes[result.name] = result
            continue

        split = _try_split_avgpool_if(graph, name, consumed, node_remap)
        if split is not None:
            for n in split:
                fused_nodes[n.name] = n
            continue

        result = _try_fuse_accumulate(graph, name, consumed, node_remap, port_remap)
        if result is not None:
            fused_nodes[result.name] = result
            continue

    # Pass 2: build new graph
    new_graph = PAIIRGraph(graph.name)

    for name in graph.topo_sort():
        if name in consumed:
            # If this consumed node maps to a fused node not yet added
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

    # AvgPool + IF: shared-core would force IF into LIF behavior (qualitative
    # change). Skip here so _try_split_avgpool_if() handles this pattern.
    if _is_avgpool(pred.comp) and isinstance(act_node.act, IFNodeV25):
        return None

    fused = SequentialOp(pred.comp, act_node.act)
    fused.input_shapes = pred.input_shapes
    fused.output_shape = act_node.output_shape
    fused.input_dims = pred.input_dims
    fused.output_dims = act_node.output_dims

    consumed.add(act_name)
    consumed.add(pred_name)
    node_remap[act_name] = fused.name
    node_remap[pred_name] = fused.name

    return fused


def _try_split_avgpool_if(
    graph: PAIIRGraph, act_name: str, consumed: set[str], node_remap: dict[str, str]
) -> list[OfflineCoreOp] | None:
    """Split AvgPool + IFNodeV25 into two cores.

    Shared-core deployment is impossible because ``leak_tau=-N`` for AvgPool
    division also activates multiplicative leak, forcing IF into LIF behavior.
    Instead, create two separate cores:

    - Core 1: ``SequentialOp(AvgPool, ANNNodeV25(identity_lut))`` -- ANN mode,
      identity pass-through, handles division-by-shift.
    - Core 2: ``StandaloneActOp(compensated_IFNodeV25)`` -- SNN mode, identity
      weights, threshold compensated for the shift approximation error.
    """
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

    if not _is_avgpool(pred.comp):
        return None

    if not isinstance(act_node.act, IFNodeV25):
        return None

    if len(graph.successors(pred_name)) != 1:
        return None

    # When the predecessor outputs int8 (e.g. ANN layer), the sum-pooled value
    # after division by 2^N may exceed int8 range.  In that case Core 1 should
    # output membrane potential instead of going through the identity LUT.
    # For now we only handle the SNN case (spike input, sum <= window_size,
    # divided result fits in int8).
    # TODO: support wider-input case by configuring Core 1 with
    #   output_type=POTENTIAL and adjusting Core 2 threshold scaling accordingly.

    # Core 1: AvgPool + identity LUT (ANN mode)
    # Infer output_sign from the AvgPool's predecessor: if its activation
    # produces signed output (e.g. bidirectional IF), the identity LUT must
    # also be signed; otherwise unsigned suffices.
    avgpool_preds = graph.predecessors(pred_name)
    output_sign = 1  # default: signed (safe fallback)
    if avgpool_preds:
        avgpool_pred = graph.nodes[avgpool_preds[0]]
        pred_act = getattr(avgpool_pred, "act", None)
        if pred_act is not None:
            output_sign = pred_act.output_sign

    identity_lut = LutCustom(
        torch.arange(-128, 128), torch.arange(-128, 128), output_sign
    )
    core1 = SequentialOp(pred.comp, ANNNodeV25(identity_lut))
    core1.input_shapes = pred.input_shapes
    core1.output_shape = act_node.output_shape
    core1.input_dims = pred.input_dims
    core1.output_dims = act_node.output_dims

    # Core 2: compensated IF (SNN mode, identity weights)
    window_size = _get_pool_window_size(pred.comp)
    orig_act = act_node.act
    v_reset = orig_act.reset_v if orig_act.reset_mode == RM.MODE_NORMAL else None
    compensated_if = IFNodeV25(orig_act.thres_pos, v_reset)
    # Compensate threshold for shift approximation error
    compensated_params = compensated_if.to_neuron_params()
    compensate_splitcore_avgpool_threshold(compensated_params, window_size)
    compensated_if.thres_pos = compensated_params.thres_pos
    compensated_if.thres_neg = compensated_params.thres_neg

    core2 = StandaloneActOp(compensated_if)
    core2.input_shapes = [act_node.output_shape]
    core2.output_shape = act_node.output_shape
    core2.input_dims = [act_node.output_dims]
    core2.output_dims = act_node.output_dims

    # Mark original nodes as consumed, set remapping
    consumed.add(pred_name)
    consumed.add(act_name)
    node_remap[pred_name] = core1.name
    node_remap[act_name] = core2.name

    # Core 1 -> Core 2 edge will be created by _rebuild_edges via node_remap
    return [core1, core2]


def _try_fuse_accumulate(
    graph: PAIIRGraph,
    act_name: str,
    consumed: set[str],
    node_remap: dict[str, str],
    port_remap: dict[str, int],
) -> AccumulateOp | None:
    """Try to fuse ``CompOps -> AddOp -> ActivationOp``."""
    act_node = graph.nodes[act_name]
    assert isinstance(act_node, StandaloneActOp)

    preds = graph.predecessors(act_name)
    if len(preds) != 1:
        return None

    add_name = preds[0]
    if add_name in consumed:
        return None

    add_node = graph.nodes[add_name]
    if not isinstance(add_node, AddOp):
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

    fused.input_shapes = []
    fused.input_dims = []
    for p in comp_preds:
        fused.input_shapes.extend(graph.nodes[p].input_shapes)  # type: ignore[union-attr]
        fused.input_dims.extend(graph.nodes[p].input_dims)  # type: ignore[union-attr]
    fused.output_shape = act_node.output_shape
    fused.output_dims = act_node.output_dims

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

    Should be called **after** :func:`fuse_to_offline_cores` and **before**
    :func:`propagate_data_format`.

    Recoverable problems (auto-cleaned with a warning):

    - Disconnected :class:`InputNode` (no successors).
    - Disconnected :class:`OutputNode` (no predecessors).
    - Orphan :class:`OpNode` (missing predecessors or successors).

    Unrecoverable problems (raise :class:`GraphValidationError`):

    - No :class:`InputNode` in the graph.
    - No :class:`OutputNode` in the graph.
    - :class:`InputNode` has predecessors.
    - :class:`OutputNode` has successors.
    - ``snn_mode`` / LUT mismatch: LUT present but ``snn_mode != ANN``,
      or LUT absent but ``snn_mode != SNN``.

    Args:
        graph: A fused :class:`PAIIRGraph` (modified in place).

    Raises:
        GraphValidationError: If unrecoverable structural problems are found.
    """
    errors: list[str] = []
    to_remove: list[str] = []
    missing_shape_nodes: list[str] = []

    if not graph.input_nodes():
        errors.append("graph has no input node")
    if not graph.output_nodes():
        errors.append("graph has no output node")

    for name, node in graph.nodes.items():
        preds = graph.predecessors(name)
        succs = graph.successors(name)

        if isinstance(node, InputNode):
            if preds:
                errors.append(
                    f"InputNode '{name}' has predecessors {preds} (expected none)"
                )
            elif not succs:
                to_remove.append(name)

        elif isinstance(node, OutputNode):
            if succs:
                errors.append(
                    f"OutputNode '{name}' has successors {succs} (expected none)"
                )
            elif not preds:
                to_remove.append(name)

        elif isinstance(node, OpNode):
            if not preds or not succs:
                to_remove.append(name)
            elif not node.output_shape:
                missing_shape_nodes.append(name)

    if missing_shape_nodes:
        names = ", ".join(missing_shape_nodes)
        errors.append(
            f"{len(missing_shape_nodes)} OpNode(s) missing shape info "
            f"(pass sample_inputs to torch_to_paiir): {names}"
        )

    # Validate snn_mode vs lut_data consistency on OfflineCoreOp nodes
    for name, node in graph.nodes.items():
        if not isinstance(node, OfflineCoreOp):
            continue
        act = getattr(node, "act", None)
        if act is None:
            continue

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

    # Raise unrecoverable errors first (before modifying the graph)
    if errors:
        raise GraphValidationError(errors)

    # Auto-clean disconnected nodes
    if to_remove:
        for name in to_remove:
            _remove_node(graph, name)

        warnings.warn(GraphCleanupWarning(to_remove))

    # Post-cleanup: verify graph is still valid (has input and output)
    if not graph.input_nodes():
        raise GraphValidationError(
            ["graph has no input node after cleanup (all were disconnected)"]
        )
    if not graph.output_nodes():
        raise GraphValidationError(
            ["graph has no output node after cleanup (all were disconnected)"]
        )


def _remove_node(graph: PAIIRGraph, name: str) -> None:
    """Remove a node and all its edges from the graph."""
    del graph.nodes[name]
    graph.edges = [e for e in graph.edges if e.src != name and e.dst != name]


# Default input format for network entry points
_DEFAULT_SNN_INPUT: DataFormat = (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)
_DEFAULT_ANN_INPUT: DataFormat = (DataSign.SIGNED, DataWidth.WIDTH_8BIT)

# Weightless operators: most compact format that preserves correctness
_WEIGHTLESS_WEIGHT_FORMAT: DataFormat = (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)


def propagate_data_format(
    graph: PAIIRGraph, input_formats: dict[str, DataFormat] | None = None
) -> None:
    """Infer and fill data format parameters on every :class:`OfflineCoreOp`.

    Runs on a **fused** graph (after :func:`fuse_to_offline_cores`).
    Fills ``input_sign``, ``input_width``, ``output_sign``, ``output_width``,
    ``weight_sign``, and ``weight_width`` on each node's ``core_params``.

    The pass ensures that predecessor output format matches successor input
    format at every edge.

    Args:
        graph: A fused :class:`PAIIRGraph`.
        input_formats: Per-:class:`InputNode` data format override, keyed
            by node name.  Nodes not listed use a default inferred from
            the first successor's ``snn_mode``: SNN -> ``UNSIGNED 1BIT``,
            ANN -> ``SIGNED 8BIT``.
    """
    if input_formats is None:
        input_formats = {}

    # Validate that input_formats keys refer to actual InputNodes
    input_node_names = {
        n for n, node in graph.nodes.items() if isinstance(node, InputNode)
    }
    for name in input_formats:
        if name not in input_node_names:
            warnings.warn(
                f"input_formats key '{name}' does not match any InputNode in the graph"
            )

    # Resolved output format for every node name (including InputNode/OutputNode)
    resolved: dict[str, DataFormat] = {}

    # Phase 1: local inference (output + weight, no cross-node dependencies)
    for name in graph.topo_sort():
        node = graph.nodes[name]

        if isinstance(node, InputNode):
            if name in input_formats:
                resolved[name] = input_formats[name]
            else:
                resolved[name] = _infer_input_node_default(graph, name)
            continue

        if isinstance(node, OutputNode):
            # OutputNode inherits format from its predecessor
            preds = graph.predecessors(name)
            # Will be filled in Phase 2 after all predecessors are resolved
            continue

        if isinstance(node, ConcatOp):
            # ConcatOp is resolved in Phase 2 (needs predecessor formats)
            continue

        if not isinstance(node, OfflineCoreOp):
            continue

        # -- Output format --
        out_fmt = _infer_node_output_format(node)
        node.core_params.set_output_format(out_fmt)
        resolved[name] = out_fmt

        # -- Weight format --
        w_fmt = _infer_node_weight_format(node)
        node.core_params.set_weight_format(w_fmt)

    # Phase 2: propagate input format from predecessors
    for name in graph.topo_sort():
        node = graph.nodes[name]

        if isinstance(node, (InputNode, OutputNode)):
            # OutputNode: resolve from predecessor
            if isinstance(node, OutputNode):
                preds = graph.predecessors(name)
                if preds and preds[0] in resolved:
                    resolved[name] = resolved[preds[0]]
            continue

        if isinstance(node, ConcatOp):
            # Pass-through: output format = merge of all input formats
            pred_formats = [
                resolved[p] for p in graph.predecessors(name) if p in resolved
            ]
            if pred_formats:
                resolved[name] = merge_data_formats(pred_formats)
            continue

        if not isinstance(node, OfflineCoreOp):
            continue

        pred_formats = [resolved[p] for p in graph.predecessors(name) if p in resolved]
        in_fmt = merge_data_formats(pred_formats)
        node.core_params.set_input_format(in_fmt)


def _infer_input_node_default(graph: PAIIRGraph, name: str) -> DataFormat:
    """Infer default data format for an InputNode from its first successor."""
    succs = graph.successors(name)
    for succ_name in succs:
        succ = graph.nodes[succ_name]
        if isinstance(succ, OfflineCoreOp):
            if succ.core_params.snn_mode == SNNMode.SNN:
                return _DEFAULT_SNN_INPUT
            return _DEFAULT_ANN_INPUT
    return _DEFAULT_ANN_INPUT


def _infer_node_output_format(node: OfflineCoreOp) -> DataFormat:
    """Infer output format from an OfflineCoreOp's activation."""
    act = getattr(node, "act", None)
    if act is not None:
        return infer_output_format(act)
    # No activation: potential pass-through
    return DataSign.SIGNED, DataWidth.WIDTH_8BIT


def _infer_node_weight_format(node: OfflineCoreOp) -> DataFormat:
    """Infer weight format from an OfflineCoreOp's weights property."""
    weights = node.weights

    if weights is None:
        return _WEIGHTLESS_WEIGHT_FORMAT

    all_weights = torch.cat([w.flatten() for w in weights])
    w_min = int(all_weights.min().item())
    w_max = int(all_weights.max().item())
    return infer_weight_format(w_min, w_max)


class TickOverride(TypedDict, total=False):
    """Per-node timing override for :func:`assign_tick_params`.

    All fields are optional.  Only specified keys take effect;
    omitted keys fall through to graph-level defaults or auto-inference.

    Keys:
        tick_start: Override the starting sync_all number for this node.
        tick_duration: Override how many time steps this node works.
        auto_reset: Override whether this node auto-resets neuron state.
    """

    tick_start: int
    tick_duration: int
    auto_reset: bool


def _validate_overrides(graph: PAIIRGraph, overrides: dict[str, TickOverride]) -> None:
    """Validate override keys and values before assignment."""
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
    """Assign timing parameters on every :class:`OfflineCoreOp`.

    For each :class:`OfflineCoreOp` whose ``tick_start`` is ``None``,
    computes it from the DAG depth (topological distance from
    :class:`InputNode`).  Nodes with an explicit ``tick_start`` (set by
    the user or via *overrides*) are left unchanged.

    ``tick_duration`` is the graph-level default: every node that still
    has its default value (0) gets the value provided here. Per-node
    overrides take priority.

    ``tick_initial`` assignment depends on the mode:

    - **ANN** (``snn_mode == SNNMode.ANN``): always set to 1, so the
      core clears membrane potential every time step.
    - **SNN**: controlled by ``auto_reset``:

      - ``True`` (default): when ``tick_duration > 0``, set
        ``tick_initial = tick_duration`` so the core automatically resets
        neuron state after each work cycle.  When ``tick_duration == 0``
        (always working), ``tick_initial`` stays 0.
      - ``False``: ``tick_initial`` is always 0 (no automatic reset).

    Per-node overrides can be specified via *overrides*, keyed by
    node name::

        assign_tick_params(graph, tick_duration=100, overrides={
            "seq_op_0": {"tick_start": 5},
            "seq_op_1": {"tick_duration": 200, "auto_reset": False},
        })

    After assignment, :meth:`OfflineCoreParams.validate_tick_params` is
    called on every node.

    Args:
        graph: A fused :class:`PAIIRGraph`.
        tick_duration: Global default for how many time steps each core
            works.  0 = always working (default).
        auto_reset: Global default for automatic neuron state reset.
        overrides: Per-node timing overrides, keyed by node name.
            Each value is a dict with optional keys ``tick_start``,
            ``tick_duration``, and ``auto_reset``.

    Raises:
        ValueError: If any tick parameter is invalid after assignment.
    """
    if tick_duration < 0:
        raise ValueError(f"'tick_duration' must be non-negative, got {tick_duration}")

    if overrides is None:
        overrides = {}

    _validate_overrides(graph, overrides)

    # Compute depth from InputNode for every node.
    # ConcatOp is a routing-only node (no hardware core), so it inherits
    # the max depth of its predecessors WITHOUT incrementing.
    depth: dict[str, int] = {}
    for name in graph.topo_sort():
        node = graph.nodes[name]

        if isinstance(node, InputNode):
            depth[name] = 0
            continue

        preds = graph.predecessors(name)
        pred_depth = max(depth.get(p, 0) for p in preds) if preds else 0

        if isinstance(node, ConcatOp):
            depth[name] = pred_depth
        else:
            depth[name] = pred_depth + 1

    # Assign tick params on OfflineCoreOp nodes
    for name in graph.topo_sort():
        node = graph.nodes[name]
        if not isinstance(node, OfflineCoreOp):
            continue

        cp = node.core_params
        ovr = overrides.get(name, {})

        # --- tick_start ---
        if "tick_start" in ovr:
            cp.tick_start = ovr["tick_start"]
        elif cp.tick_start is None:
            cp.tick_start = depth[name]

        # --- tick_duration ---
        if "tick_duration" in ovr:
            cp.tick_duration = ovr["tick_duration"]
        elif cp.tick_duration == 0 and tick_duration != 0:
            cp.tick_duration = tick_duration

        # --- tick_initial ---
        if cp.snn_mode == SNNMode.ANN:
            # ANN cores always need tick_initial=1 to clear membrane
            # potential every time step.
            cp.tick_initial = 1
        else:
            node_auto_reset = ovr.get("auto_reset", auto_reset)
            if node_auto_reset and cp.tick_duration > 0:
                cp.tick_initial = cp.tick_duration
            else:
                cp.tick_initial = 0

        cp.validate_tick_params()
