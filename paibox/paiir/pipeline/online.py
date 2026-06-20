"""Minimal online-core compile support for work-mode refinement."""

from dataclasses import dataclass

from paicorelib import LCN_EX, OnlineCoreRegLimV2
from torch import nn

from ..exceptions import GraphValidationError
from ..ir.calc_params import (
    OnlineCoreParams,
    OnlineCoreSemanticMode,
    OnlineCoreType,
    OnlineCoreUpdateType,
    OnlineCoreWorkMode,
    OnlineDataWidth,
    OnlineGradientRole,
    OnlineSNNMode,
    OnlineUpdateDirection,
)
from ..ir.graph import PAIIRGraph
from ..ir.ir_base import InputNode, OutputNode
from ..ir.op_node import OnlineCoreOp, TransformOp

__all__ = [
    "OnlineUpdateStagePlan",
    "analyze_online_update_stage_plans",
    "compile_online_graph",
    "has_online_nodes",
    "refine_online_work_modes",
    "validate_online_compiled_graph",
]

_PHASE1_UPDATE_DIRECTION = OnlineUpdateDirection.FORWARD
_PHASE1_UPDATE_WORK_MODE = OnlineCoreWorkMode.FORWARD_WEIGHT_UPDATE
_PHASE1_LATER_UPDATE_TYPES = frozenset(
    (
        OnlineCoreUpdateType.KAHAN_WEIGHT,
        OnlineCoreUpdateType.KAHAN_WEIGHT_BIAS,
    )
)


@dataclass(frozen=True)
class OnlineUpdateStagePlan:
    """Compile-time binding for one logical online update stage.

    The current Phase 1 path still materializes only ``FORWARD_WEIGHT_UPDATE``.
    This plan keeps the corresponding forward peer and backward/gradient peer
    explicit so later backend work can expand the logical update into richer
    hardware roles without re-deriving the layer pairing.
    """

    layer_idx_from_output: int
    forward_name: str
    backward_peer_name: str
    update_name: str
    gradient_role: OnlineGradientRole | None
    phase1_work_mode: OnlineCoreWorkMode = _PHASE1_UPDATE_WORK_MODE
    future_backward_work_mode: OnlineCoreWorkMode = (
        OnlineCoreWorkMode.BACKWARD_WEIGHT_UPDATE
    )

    @property
    def logical_sync_targets(self) -> tuple[str, str]:
        return (self.forward_name, self.backward_peer_name)

    @property
    def logical_sync_target_summary(self) -> str:
        forward_name, backward_peer_name = self.logical_sync_targets
        return (
            f"forward='{forward_name}' and "
            f"backward_peer='{backward_peer_name}'"
        )


def has_online_nodes(graph: PAIIRGraph) -> bool:
    """Return whether *graph* contains any online-core operators."""
    return any(isinstance(node, OnlineCoreOp) for node in graph.nodes.values())


def compile_online_graph(
    graph: PAIIRGraph, timesteps: int = 1, auto_reset: bool = True
) -> PAIIRGraph:
    """Run the current minimal online-core compile path."""
    _assign_online_tick_params(graph, timesteps, auto_reset)
    refine_online_work_modes(graph)
    validate_online_compiled_graph(graph)
    graph.eval()
    return graph


def _assign_online_tick_params(
    graph: PAIIRGraph, timesteps: int = 1, auto_reset: bool = True
) -> None:
    if timesteps <= 0:
        raise ValueError(f"'timesteps' must be positive, got {timesteps}.")
    if not isinstance(auto_reset, bool):
        raise TypeError(f"'auto_reset' must be a bool, got {auto_reset!r}")

    tick_duration = 0 if auto_reset else timesteps
    tick_initial = timesteps if auto_reset else 0

    for name in _online_node_names(graph):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        params = node.core_params
        if params.tick_start is None:
            params.tick_start = 1
        params.tick_duration = tick_duration
        params.tick_initial = tick_initial


def analyze_online_update_stage_plans(graph: PAIIRGraph) -> list[OnlineUpdateStagePlan]:
    """Return logical update-stage bindings in output-to-input order."""
    online_names = _online_node_names(graph)
    forward_names = _online_stage_names(graph, OnlineCoreSemanticMode.FORWARD)
    gradient_names = _online_stage_names(graph, OnlineCoreSemanticMode.GRADIENT)
    update_names = _online_stage_names(graph, OnlineCoreSemanticMode.UPDATE)

    if not forward_names and not gradient_names and not update_names:
        return []
    if len(gradient_names) != len(forward_names) or len(update_names) != len(
        forward_names
    ):
        raise ValueError(
            "online update-stage planning expects one gradient node and one "
            "update node per forward node"
        )
    plan_errors = _validate_online_stage_structure(graph, online_names)
    plan_errors.extend(_validate_online_stage_chain(graph))
    if plan_errors:
        raise ValueError("; ".join(plan_errors))

    plans: list[OnlineUpdateStagePlan] = []
    for idx, (forward_name, gradient_name, update_name) in enumerate(
        zip(reversed(forward_names), gradient_names, update_names)
    ):
        gradient_node = _online_node(graph, gradient_name)
        plans.append(
            OnlineUpdateStagePlan(
                layer_idx_from_output=idx,
                forward_name=forward_name,
                backward_peer_name=gradient_name,
                update_name=update_name,
                gradient_role=gradient_node.core_params.gradient_role,
            )
        )

    return plans


def _expected_gradient_role(layer_idx_from_output: int) -> OnlineGradientRole:
    return (
        OnlineGradientRole.OUTPUT
        if layer_idx_from_output == 0
        else OnlineGradientRole.HIDDEN
    )


def refine_online_work_modes(graph: PAIIRGraph) -> None:
    """Resolve online semantic modes into concrete hardware ``work_mode`` values.

    The current online IR models one logical update stage per layer. In this
    phase that stage is compiled as the layer's forward-weight update, while
    the mirrored backward-weight synchronization remains a later backend step.
    """
    gradient_names = _online_stage_names(
        graph, OnlineCoreSemanticMode.GRADIENT
    )
    for idx, name in enumerate(gradient_names):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        expected = _expected_gradient_role(idx)
        if node.core_params.gradient_role is None:
            node.core_params.gradient_role = expected

    for name in _online_stage_names(graph, OnlineCoreSemanticMode.UPDATE):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        if node.core_params.update_direction is None:
            node.core_params.update_direction = _PHASE1_UPDATE_DIRECTION
        if isinstance(node.comp, nn.Linear) and not isinstance(
            node.core_params.output_width, OnlineCoreUpdateType
        ):
            node.core_params.output_width = _expected_update_type(node)

    for name in _online_node_names(graph):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        node.core_params.refine_work_mode()


def validate_online_compiled_graph(graph: PAIIRGraph) -> None:
    """Validate the current minimal online-core compile result."""
    errors: list[str] = []
    update_plans: dict[str, OnlineUpdateStagePlan] = {}
    stage_errors: list[str] = []
    chain_errors: list[str] = []

    online_names = _online_node_names(graph)
    if not online_names:
        errors.append("online compile path requires at least one OnlineCoreOp node")
    else:
        try:
            graph.lint()
        except GraphValidationError as exc:
            errors.extend(exc.errors)

        errors.extend(_validate_supported_node_kinds(graph, online_names))
        stage_errors = _validate_online_stage_structure(graph, online_names)
        errors.extend(stage_errors)
        chain_errors = _validate_online_stage_chain(graph)
        errors.extend(chain_errors)
        if not stage_errors and not chain_errors:
            try:
                update_plans = {
                    plan.update_name: plan
                    for plan in analyze_online_update_stage_plans(graph)
                }
            except ValueError as exc:
                errors.append(str(exc))
        errors.extend(_validate_online_outputs(graph))
        errors.extend(_validate_online_intercore_contracts(graph))
        errors.extend(
            _validate_online_hardware_fields(graph, online_names, update_plans)
        )

    gradient_names = _online_stage_names(graph, OnlineCoreSemanticMode.GRADIENT)
    for idx, name in enumerate(gradient_names):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        expected = _expected_gradient_role(idx)
        if node.core_params.gradient_role is not expected:
            errors.append(
                f"'{name}': gradient_role must be '{expected.value}' for its "
                "position in the online gradient chain"
            )

    for name in _online_node_names(graph):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        try:
            resolved = node.core_params.resolve_work_mode()
        except ValueError as exc:
            errors.append(f"'{name}': {exc}")
            continue

        if node.core_params.work_mode != resolved:
            errors.append(f"'{name}': work_mode was not materialized in compile pass")

        if node.core_params.semantic_mode is OnlineCoreSemanticMode.UPDATE:
            errors.extend(
                _validate_phase1_update_contract(
                    name, node, resolved, update_plans.get(name)
                )
            )

    if errors:
        raise GraphValidationError(errors)


def _validate_supported_node_kinds(
    graph: PAIIRGraph, online_names: list[str]
) -> list[str]:
    errors: list[str] = []
    positions = {name: idx for idx, name in enumerate(graph.topo_sort())}
    first_online_pos = positions[online_names[0]]

    for name in graph.topo_sort():
        node = graph.nodes[name]
        if isinstance(node, (InputNode, OutputNode, OnlineCoreOp)):
            continue
        if isinstance(node, TransformOp):
            if positions[name] > first_online_pos:
                errors.append(
                    f"'{name}': TransformOp after the first online forward node is "
                    "not supported in the current online compile path"
                )
            continue

        errors.append(
            f"'{name}': current online compile supports only TransformOp as a "
            "non-online interior node"
        )

    return errors


def _validate_online_stage_structure(
    graph: PAIIRGraph, online_names: list[str]
) -> list[str]:
    errors: list[str] = []
    semantic_modes = [
        _online_node(graph, name).core_params.semantic_mode for name in online_names
    ]
    forward_names = _online_stage_names(graph, OnlineCoreSemanticMode.FORWARD)
    loss_names = _online_stage_names(graph, OnlineCoreSemanticMode.LOSS)
    gradient_names = _online_stage_names(graph, OnlineCoreSemanticMode.GRADIENT)
    update_names = _online_stage_names(graph, OnlineCoreSemanticMode.UPDATE)
    pool_gradient_names = _online_stage_names(
        graph, OnlineCoreSemanticMode.POOL_GRADIENT
    )

    if pool_gradient_names:
        errors.append(
            "pool-gradient online compile is not wired in the current Phase 1 path"
        )

    if not forward_names:
        errors.append("online compile requires at least one forward-semantic node")

    if len(loss_names) != 1:
        errors.append(
            f"online compile expects exactly one loss node, got {len(loss_names)}"
        )

    if len(gradient_names) != len(forward_names):
        errors.append(
            "online compile expects one gradient node per forward node, "
            f"got {len(gradient_names)} gradient node(s) for {len(forward_names)} "
            "forward node(s)"
        )

    if len(update_names) != len(forward_names):
        errors.append(
            "online compile expects one update node per forward node, "
            f"got {len(update_names)} update node(s) for {len(forward_names)} "
            "forward node(s)"
        )

    expected_modes = (
        [OnlineCoreSemanticMode.FORWARD] * len(forward_names)
        + [OnlineCoreSemanticMode.LOSS]
        + [OnlineCoreSemanticMode.GRADIENT] * len(forward_names)
        + [OnlineCoreSemanticMode.UPDATE] * len(forward_names)
    )
    if semantic_modes != expected_modes:
        errors.append(
            "online semantic stages must appear in the order "
            "forward* -> loss -> gradient* -> update*"
        )

    for name in forward_names + gradient_names + update_names:
        node = _online_node(graph, name)
        if not isinstance(node.comp, nn.Linear):
            errors.append(
                f"'{name}': current online compile supports nn.Linear only for "
                f"{node.core_params.semantic_mode.value} stages"
            )

    if loss_names:
        loss_node = _online_node(graph, loss_names[0])
        if loss_node.comp is not None:
            errors.append(
                f"'{loss_node.name}': loss stage must not carry a compute module"
            )

    return errors


def _validate_online_stage_chain(graph: PAIIRGraph) -> list[str]:
    errors: list[str] = []
    forward_names = _online_stage_names(graph, OnlineCoreSemanticMode.FORWARD)
    loss_names = _online_stage_names(graph, OnlineCoreSemanticMode.LOSS)
    gradient_names = _online_stage_names(graph, OnlineCoreSemanticMode.GRADIENT)
    update_names = _online_stage_names(graph, OnlineCoreSemanticMode.UPDATE)

    errors.extend(
        _validate_direct_chain(
            graph,
            forward_names,
            "forward chain",
        )
    )
    errors.extend(
        _validate_direct_chain(
            graph,
            gradient_names,
            "gradient chain",
        )
    )
    errors.extend(
        _validate_direct_chain(
            graph,
            update_names,
            "update chain",
        )
    )

    if forward_names and loss_names:
        loss_preds = graph.predecessors(loss_names[0])
        if loss_preds != [forward_names[-1]]:
            errors.append(
                f"'{loss_names[0]}': loss node must consume the last forward node "
                f"directly, got predecessors {loss_preds}"
            )

    if loss_names and gradient_names:
        grad_preds = graph.predecessors(gradient_names[0])
        if grad_preds != [loss_names[0]]:
            errors.append(
                f"'{gradient_names[0]}': first gradient node must consume the loss "
                f"node directly, got predecessors {grad_preds}"
            )

    if gradient_names and update_names:
        update_preds = graph.predecessors(update_names[0])
        if update_preds != [gradient_names[-1]]:
            errors.append(
                f"'{update_names[0]}': first update node must consume the last "
                f"gradient node directly, got predecessors {update_preds}"
            )

    return errors


def _validate_online_outputs(graph: PAIIRGraph) -> list[str]:
    errors: list[str] = []
    outputs = graph.output_nodes()
    forward_names = _online_stage_names(graph, OnlineCoreSemanticMode.FORWARD)
    update_names = _online_stage_names(graph, OnlineCoreSemanticMode.UPDATE)

    if len(outputs) != 2:
        errors.append(
            f"online compile currently expects exactly 2 OutputNode nodes, got {len(outputs)}"
        )
        return errors

    predecessor_lists = [graph.predecessors(node.name) for node in outputs]
    if forward_names and [forward_names[-1]] not in predecessor_lists:
        errors.append(
            "online compile expects one model output fed directly by the last "
            "forward node"
        )

    if update_names and [update_names[-1]] not in predecessor_lists:
        errors.append(
            "online compile expects one training-tail output fed directly by the "
            "last update node"
        )

    return errors


def _validate_direct_chain(
    graph: PAIIRGraph, names: list[str], chain_name: str
) -> list[str]:
    errors: list[str] = []
    for upstream, downstream in zip(names, names[1:]):
        preds = graph.predecessors(downstream)
        if preds != [upstream]:
            errors.append(
                f"'{downstream}': current online compile requires a direct serial "
                f"{chain_name}; got predecessors {preds}"
            )
    return errors


def _online_node(graph: PAIIRGraph, name: str) -> OnlineCoreOp:
    node = graph.nodes[name]
    assert isinstance(node, OnlineCoreOp)
    return node


def _validate_online_hardware_fields(
    graph: PAIIRGraph,
    online_names: list[str],
    update_plans: dict[str, OnlineUpdateStagePlan],
) -> list[str]:
    errors: list[str] = []
    forward_names = _online_stage_names(graph, OnlineCoreSemanticMode.FORWARD)
    first_forward_name = forward_names[0] if forward_names else None
    last_forward_name = forward_names[-1] if forward_names else None

    for name in online_names:
        node = _online_node(graph, name)
        params = node.core_params

        if params.snn_mode is not OnlineSNNMode.ANN_NO_ACT:
            errors.append(
                f"'{name}': current online compile supports ANN_NO_ACT only, "
                f"got {params.snn_mode.name}"
            )

        errors.extend(
            _validate_online_core_type_fields(
                name,
                params,
                first_forward_name=first_forward_name,
                last_forward_name=last_forward_name,
            )
        )

        if params.input_width is not OnlineDataWidth.TYPE_FP16:
            errors.append(
                f"'{name}': current online compile supports fp16 input_width only, "
                f"got {params.input_width.name}"
            )

        errors.extend(_validate_online_lcn_fields(name, params))

        if params.semantic_mode is OnlineCoreSemanticMode.UPDATE:
            if isinstance(node.comp, nn.Linear):
                expected_update_type = _expected_update_type(node)
                if params.output_width in _PHASE1_LATER_UPDATE_TYPES:
                    errors.append(
                        f"'{name}': {params.output_width.name} remains a later "
                        "phase together with Kahan/update-core semantics; "
                        f"Phase 1 requires {expected_update_type.name}"
                    )
                elif params.output_width is not expected_update_type:
                    errors.append(
                        f"'{name}': update output_width must be "
                        f"{expected_update_type.name} for the current Linear config"
                    )
        else:
            if not isinstance(params.output_width, OnlineDataWidth):
                errors.append(
                    f"'{name}': non-update online stages must use OnlineDataWidth "
                    "as output_width"
                )
            elif params.output_width is not OnlineDataWidth.TYPE_FP16:
                errors.append(
                    f"'{name}': current online compile supports fp16 output_width "
                    f"only, got {params.output_width.name}"
                )

        errors.extend(
            _validate_online_numeric_fields(name, params, update_plans.get(name))
        )

    return errors


def _validate_online_core_type_fields(
    name: str,
    params: OnlineCoreParams,
    *,
    first_forward_name: str | None,
    last_forward_name: str | None,
) -> list[str]:
    errors: list[str] = []

    if params.input_core is not OnlineCoreType.ONLINE and not (
        params.input_core is OnlineCoreType.OFFLINE
        and params.semantic_mode is OnlineCoreSemanticMode.FORWARD
        and name == first_forward_name
    ):
        errors.append(
            f"'{name}': current online compile supports OFFLINE -> ONLINE at the "
            "graph entry only"
        )

    if params.output_core is not OnlineCoreType.ONLINE and not (
        params.output_core is OnlineCoreType.OFFLINE
        and params.semantic_mode is OnlineCoreSemanticMode.FORWARD
        and name == last_forward_name
    ):
        detail = (
            "only the last forward node may preserve output_core=OFFLINE as the "
            "external model-output boundary"
            if params.semantic_mode is OnlineCoreSemanticMode.FORWARD
            else "synthetic training stages must remain on the internal "
            "ONLINE -> ONLINE path"
        )
        errors.append(
            f"'{name}': current online compile supports ONLINE -> OFFLINE at the "
            f"graph exit only; {detail}"
        )

    return errors


def _validate_online_numeric_fields(
    name: str,
    params: OnlineCoreParams,
    plan: OnlineUpdateStagePlan | None,
) -> list[str]:
    errors: list[str] = []

    axon_skew = getattr(params, "axon_skew")
    if not (
        OnlineCoreRegLimV2.AXON_SKEW_MIN
        <= axon_skew
        <= OnlineCoreRegLimV2.AXON_SKEW_MAX
    ):
        errors.append(
            f"'{name}': axon_skew must be in "
            f"[{OnlineCoreRegLimV2.AXON_SKEW_MIN}, "
            f"{OnlineCoreRegLimV2.AXON_SKEW_MAX}], got {axon_skew}"
        )

    _validate_non_negative_le(
        errors,
        name,
        "neuron_number",
        getattr(params, "neuron_number"),
        OnlineCoreRegLimV2.NEURON_NUMBER_MAX,
    )
    _validate_non_negative_le(
        errors,
        name,
        "update_number",
        getattr(params, "update_number"),
        OnlineCoreRegLimV2.UPDATE_NUMBER_MAX,
    )
    _validate_non_negative_le(
        errors,
        name,
        "thread_number",
        getattr(params, "thread_number"),
        OnlineCoreRegLimV2.THREAD_NUMBER_MAX,
    )

    _validate_gt_one_le(
        errors,
        name,
        "busy_cycle",
        getattr(params, "busy_cycle"),
        OnlineCoreRegLimV2.BUSY_CYCLE_MAX,
    )
    _validate_gt_one_le(
        errors,
        name,
        "delay_cycle",
        getattr(params, "delay_cycle"),
        OnlineCoreRegLimV2.DELAY_CYCLE_MAX,
    )
    _validate_gt_one_le(
        errors,
        name,
        "width_cycle",
        getattr(params, "width_cycle"),
        OnlineCoreRegLimV2.WIDTH_CYCLE_MAX,
    )

    for field in (
        "update_core_xy",
        "update_core_x",
        "update_core_y",
    ):
        _validate_phase1_zero_only(
            errors,
            name,
            field,
            getattr(params, field),
            OnlineCoreRegLimV2.TEST_CORE_COORD_MIN,
            OnlineCoreRegLimV2.TEST_CORE_COORD_MAX,
            reason=(
                _phase1_update_route_reason(plan)
                if (
                    params.semantic_mode is OnlineCoreSemanticMode.UPDATE
                    and field.startswith("update_core_")
                )
                else None
            ),
        )

    for field in ("test_core_xy", "test_core_x", "test_core_y"):
        _validate_range(
            errors,
            name,
            field,
            getattr(params, field),
            OnlineCoreRegLimV2.TEST_CORE_COORD_MIN,
            OnlineCoreRegLimV2.TEST_CORE_COORD_MAX,
        )

    for field in ("global_send", "global_receive"):
        _validate_phase1_zero_only(
            errors,
            name,
            field,
            getattr(params, field),
            0,
            getattr(OnlineCoreRegLimV2, f"{field.upper()}_MAX"),
            reason=(
                _phase1_update_global_reason(plan)
                if params.semantic_mode is OnlineCoreSemanticMode.UPDATE
                else None
            ),
        )

    tick_start = getattr(params, "tick_start")
    if tick_start is not None:
        try:
            params.validate_tick_params()
        except ValueError as exc:
            errors.append(f"'{name}': {exc}")
    else:
        _validate_non_negative_le(
            errors,
            name,
            "tick_duration",
            getattr(params, "tick_duration"),
            OnlineCoreRegLimV2.TICK_DURATION_MAX,
        )
        _validate_non_negative_le(
            errors,
            name,
            "tick_initial",
            getattr(params, "tick_initial"),
            OnlineCoreRegLimV2.TICK_INITIAL_MAX,
        )

    return errors


def _validate_online_intercore_contracts(graph: PAIIRGraph) -> list[str]:
    errors: list[str] = []

    for edge in graph.edges:
        src = graph.nodes[edge.src]
        dst = graph.nodes[edge.dst]
        if not isinstance(src, OnlineCoreOp) or not isinstance(dst, OnlineCoreOp):
            continue

        src_params = src.core_params
        dst_params = dst.core_params
        if src_params.target_lcn_at is not dst_params.lcn_at:
            errors.append(
                f"'{src.name}' -> '{dst.name}': target_lcn_at must match the "
                f"destination lcn_at ({src_params.target_lcn_at.name} != "
                f"{dst_params.lcn_at.name})"
            )
        if src_params.target_lcn_mp is not dst_params.lcn_mp:
            errors.append(
                f"'{src.name}' -> '{dst.name}': target_lcn_mp must match the "
                f"destination lcn_mp ({src_params.target_lcn_mp.name} != "
                f"{dst_params.lcn_mp.name})"
            )
        if src_params.target_lcn_lg is not dst_params.lcn_lg:
            errors.append(
                f"'{src.name}' -> '{dst.name}': target_lcn_lg must match the "
                f"destination lcn_lg ({src_params.target_lcn_lg.name} != "
                f"{dst_params.lcn_lg.name})"
            )
        if src_params.thread_number != dst_params.thread_number:
            errors.append(
                f"'{src.name}' -> '{dst.name}': thread_number must match across "
                f"the current online serial path ({src_params.thread_number} != "
                f"{dst_params.thread_number})"
            )

    return errors


def _validate_online_lcn_fields(name: str, params: OnlineCoreParams) -> list[str]:
    errors: list[str] = []
    fields = (
        "lcn_at",
        "lcn_mp",
        "lcn_lg",
        "target_lcn_at",
        "target_lcn_mp",
        "target_lcn_lg",
    )
    values: dict[str, LCN_EX] = {}

    for field in fields:
        value = getattr(params, field)
        if not isinstance(value, LCN_EX):
            errors.append(
                f"'{name}': {field} must be an LCN_EX value, got "
                f"{type(value).__name__}"
            )
            continue
        values[field] = value

    if errors:
        return errors

    first = values["lcn_at"]
    if any(value is not first for value in values.values()):
        joined = ", ".join(f"{field}={value.name}" for field, value in values.items())
        errors.append(
            f"'{name}': current online compile supports one explicit unified LCN "
            f"only; lcn_* / target_lcn_* must all match, got {joined}"
        )

    return errors


def _validate_non_negative_le(
    errors: list[str], name: str, field: str, value: int, limit: int
) -> None:
    if value < 0 or value > limit:
        errors.append(f"'{name}': {field} must be in [0, {limit}], got {value}")


def _validate_gt_one_le(
    errors: list[str], name: str, field: str, value: int, limit: int
) -> None:
    if value <= 1 or value > limit:
        errors.append(f"'{name}': {field} must be in [2, {limit}], got {value}")


def _validate_range(
    errors: list[str], name: str, field: str, value: int, minimum: int, maximum: int
) -> None:
    if value < minimum or value > maximum:
        errors.append(
            f"'{name}': {field} must be in [{minimum}, {maximum}], got {value}"
        )


def _validate_phase1_zero_only(
    errors: list[str],
    name: str,
    field: str,
    value: int,
    minimum: int,
    maximum: int,
    reason: str | None = None,
) -> None:
    if value < minimum or value > maximum:
        errors.append(
            f"'{name}': {field} must be in [{minimum}, {maximum}], got {value}"
        )
    elif value != 0:
        if reason is None:
            errors.append(
                f"'{name}': current online compile does not assign {field} yet; "
                "it must remain 0 in Phase 1"
            )
        else:
            errors.append(f"'{name}': {field} {reason}; it must remain 0 in Phase 1")


def _expected_update_type(node: OnlineCoreOp) -> OnlineCoreUpdateType:
    comp = node.comp
    assert isinstance(comp, nn.Linear)
    return (
        OnlineCoreUpdateType.WEIGHT_BIAS
        if comp.bias is not None
        else OnlineCoreUpdateType.WEIGHT
    )


def _phase1_update_route_reason(plan: OnlineUpdateStagePlan | None) -> str:
    target_detail = _phase1_update_target_detail(plan)
    return (
        "remains a later phase because "
        f"{target_detail}, while the current field still looks like a single update-core "
        "route address"
    )


def _phase1_update_global_reason(plan: OnlineUpdateStagePlan | None) -> str:
    target_detail = _phase1_update_target_detail(plan)
    return (
        "remains a later phase because "
        f"{target_detail}, while the current field still looks like a single global "
        "signaling bitmap"
    )


def _phase1_update_target_detail(plan: OnlineUpdateStagePlan | None) -> str:
    if plan is None:
        return (
            "the logical update still needs to synchronize both forward and "
            "backward peers"
        )
    return (
        f"logical update layer {plan.layer_idx_from_output} must synchronize "
        f"{plan.logical_sync_target_summary}"
    )


def _validate_phase1_update_contract(
    name: str,
    node: OnlineCoreOp,
    resolved: OnlineCoreWorkMode,
    plan: OnlineUpdateStagePlan | None,
) -> list[str]:
    params = node.core_params
    if (
        params.update_direction is _PHASE1_UPDATE_DIRECTION
        and resolved is _PHASE1_UPDATE_WORK_MODE
    ):
        return []

    direction = (
        "unset" if params.update_direction is None else params.update_direction.value
    )
    detail = ""
    if plan is not None:
        detail = (
            f" Logical update layer {plan.layer_idx_from_output} binds "
            f"forward='{plan.forward_name}' and "
            f"backward_peer='{plan.backward_peer_name}'."
        )
    return [
        f"'{name}': current online compile requires "
        f"(update_direction='{_PHASE1_UPDATE_DIRECTION.value}', "
        f"work_mode={_PHASE1_UPDATE_WORK_MODE.name}); got "
        f"(update_direction='{direction}', work_mode={resolved.name}). "
        f"BACKWARD_WEIGHT_UPDATE remains a later phase.{detail}"
    ]


def _online_node_names(graph: PAIIRGraph) -> list[str]:
    return [
        name for name in graph.topo_sort() if isinstance(graph.nodes[name], OnlineCoreOp)
    ]


def _online_stage_names(
    graph: PAIIRGraph, semantic_mode: OnlineCoreSemanticMode
) -> list[str]:
    names: list[str] = []
    for name in _online_node_names(graph):
        node = graph.nodes[name]
        assert isinstance(node, OnlineCoreOp)
        if node.core_params.semantic_mode is semantic_mode:
            names.append(name)
    return names
