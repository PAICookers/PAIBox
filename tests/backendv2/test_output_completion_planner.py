import pytest
from paicorelib import CoordXY, CoordZXYOffset

from paibox.backendv2.output_completion_planner import (
    EmptyRelayCore,
    OutputCompletionError,
    OutputCompletionNoFeasiblePlanError,
    OutputCompletionPlan,
    OutputProducer,
    OutputRouteDecision,
    candidate_offsets,
    debug_output_completion_plan,
    render_output_completion_plan_ascii,
    route_coord_path,
    route_len,
    route_stays_in_grid,
    select_output_completion_plan,
    terminal_route_side,
    validate_output_completion_plan,
)
from paibox.backendv2.route_solver import (
    OFFLINE_CORE_COORDS,
    ONLINE_CORE_COORDS,
)


def _all_empty_offline_except(*used: CoordXY) -> set[CoordXY]:
    used_set = set(used)
    return {coord for coord in OFFLINE_CORE_COORDS if coord not in used_set}


def _all_empty_online_except(*used: CoordXY) -> set[CoordXY]:
    used_set = set(used)
    return {coord for coord in ONLINE_CORE_COORDS if coord not in used_set}


def _coord_set(cores: tuple[EmptyRelayCore, ...]) -> set[CoordXY]:
    return {core.coord for core in cores}


def _kind_map(cores: tuple[EmptyRelayCore, ...]) -> dict[CoordXY, str]:
    return {core.coord: core.kind for core in cores}


def _assert_shared_join_suffix(plan: OutputCompletionPlan) -> None:
    complete_path = route_coord_path(plan.global_signal_root, plan.control_offset)
    suffix = complete_path[complete_path.index(plan.completion_join_point) :]
    assert terminal_route_side(plan.control_offset) == plan.ingress_side

    for route in plan.output_routes:
        data_path = route_coord_path(route.producer_coord, route.offset)
        assert data_path[data_path.index(plan.completion_join_point) :] == suffix
        assert terminal_route_side(route.offset) == plan.ingress_side


@pytest.mark.parametrize(
    ("offset", "side"),
    [
        (CoordZXYOffset(2, 0, 0), ("xy", 1)),
        (CoordZXYOffset(2, -1, 0), ("x", -1)),
        (CoordZXYOffset(2, -1, 3), ("y", 1)),
        (CoordZXYOffset(0, 0, 0), ("local", 0)),
    ],
)
def test_terminal_route_side(offset, side):
    assert terminal_route_side(offset) == side


def test_route_coord_path_follows_z_x_y_order():
    assert route_coord_path(CoordXY(3, 4), CoordZXYOffset(-2, 1, -1)) == (
        CoordXY(3, 4),
        CoordXY(2, 3),
        CoordXY(1, 2),
        CoordXY(2, 2),
        CoordXY(2, 1),
    )


def test_candidate_offsets_are_sorted_by_length_then_offset_tuple():
    offsets = candidate_offsets(CoordXY(3, 4), CoordXY(0, 0))

    assert offsets[0] == CoordZXYOffset(-3, 0, -1)
    assert [route_len(offset) for offset in offsets] == sorted(
        route_len(offset) for offset in offsets
    )
    assert all(
        route_stays_in_grid(CoordXY(3, 4), offset, CoordXY(0, 0)) for offset in offsets
    )


def test_single_output_keeps_used_producer_as_join_and_source():
    producer = CoordXY(0, 2)

    plan = select_output_completion_plan(
        [OutputProducer(producer)],
        {producer},
        _all_empty_offline_except(producer),
        _all_empty_online_except(producer),
    )

    assert plan.global_signal_root == producer
    assert plan.completion_join_point == producer
    assert plan.required_route_cores == ()
    assert CoordXY(0, 1) in route_coord_path(producer, plan.output_routes[0].offset)
    _assert_shared_join_suffix(plan)


def test_multi_output_shortest_data_then_used_join_and_source():
    producers = [CoordXY(3, 4), CoordXY(5, 6)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
    )

    assert plan.completion_join_point == CoordXY(3, 4)
    assert plan.global_signal_root == CoordXY(3, 4)
    assert {route.producer_coord: route.offset for route in plan.output_routes} == {
        CoordXY(3, 4): CoordZXYOffset(-3, 0, -1),
        CoordXY(5, 6): CoordZXYOffset(-5, 0, -1),
    }
    assert plan.required_route_cores == ()
    _assert_shared_join_suffix(plan)


def test_data_offsets_adjust_when_needed_to_form_shortest_shared_suffix():
    producers = [CoordXY(4, 2), CoordXY(2, 4)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
    )

    assert plan.global_signal_root == CoordXY(2, 4)
    assert plan.completion_join_point == CoordXY(0, 2)
    assert {route.producer_coord: route.offset for route in plan.output_routes} == {
        CoordXY(4, 2): CoordZXYOffset(0, -4, -2),
        CoordXY(2, 4): CoordZXYOffset(-2, 0, -2),
    }
    assert _kind_map(plan.required_route_cores) == {CoordXY(0, 2): "offline"}
    _assert_shared_join_suffix(plan)


def test_offline_join_beats_empty_online_within_shortest_data_plan():
    producers = [CoordXY(0, 2), CoordXY(2, 3)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
    )

    assert plan.global_signal_root == CoordXY(0, 2)
    assert plan.completion_join_point == CoordXY(0, 2)
    assert plan.required_route_cores == ()
    _assert_shared_join_suffix(plan)


def test_empty_online_source_is_allowed_when_enabled():
    producers = [CoordXY(1, 1), CoordXY(3, 2)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        set(),
        _all_empty_online_except(*producers),
        allow_empty_online=True,
    )

    assert plan.global_signal_root == CoordXY(2, 1)
    assert plan.completion_join_point == CoordXY(1, 1)
    assert _kind_map(plan.required_route_cores) == {CoordXY(2, 1): "online"}
    _assert_shared_join_suffix(plan)


def test_disable_online_relay_rejects_online_only_candidate():
    producers = [CoordXY(1, 1), CoordXY(3, 2)]

    with pytest.raises(OutputCompletionNoFeasiblePlanError):
        select_output_completion_plan(
            [OutputProducer(producer) for producer in producers],
            set(producers),
            set(),
            _all_empty_online_except(*producers),
            allow_empty_online=False,
        )


def test_online_transit_is_allowed_when_empty_online_is_disabled():
    producers = [CoordXY(0, 2), CoordXY(2, 3)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
        allow_empty_online=False,
    )

    data_path = route_coord_path(CoordXY(0, 2), plan.output_routes[0].offset)
    assert CoordXY(0, 1) in data_path
    assert plan.completion_join_point == CoordXY(0, 2)
    assert plan.required_route_cores == ()
    _assert_shared_join_suffix(plan)


def test_blocks01_producers_share_offline_join_with_online_transit_disabled():
    producers = [
        CoordXY(3, 2),
        CoordXY(3, 3),
        CoordXY(3, 4),
        CoordXY(3, 5),
        CoordXY(3, 6),
        CoordXY(5, 2),
        CoordXY(5, 3),
        CoordXY(5, 4),
        CoordXY(5, 5),
        CoordXY(5, 6),
    ]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
        allow_empty_online=False,
    )

    assert plan.global_signal_root == CoordXY(3, 2)
    assert plan.completion_join_point == CoordXY(0, 2)
    assert _kind_map(plan.required_route_cores) == {CoordXY(0, 2): "offline"}
    for route in plan.output_routes:
        assert CoordXY(0, 1) in route_coord_path(route.producer_coord, route.offset)
    _assert_shared_join_suffix(plan)


def test_source_prefers_nearer_used_root_within_used_tier():
    producers = [CoordXY(0, 2), CoordXY(2, 3)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
    )

    assert plan.global_signal_root == CoordXY(0, 2)


def test_unavailable_thread_path_core_rejects_candidate():
    producers = [CoordXY(0, 2), CoordXY(2, 3)]

    with pytest.raises(
        OutputCompletionNoFeasiblePlanError,
        match=r"DATA relay cores are not supported",
    ):
        select_output_completion_plan(
            [OutputProducer(producer) for producer in producers],
            set(producers),
            set(),
            set(),
            allow_empty_online=False,
        )


def test_validate_rejects_complete_path_not_matching_join_suffix():
    producer = CoordXY(0, 2)
    plan = OutputCompletionPlan(
        (OutputRouteDecision(producer, CoordXY(0, 0), CoordZXYOffset(0, 0, -2)),),
        CoordZXYOffset(1, -1, -3),
        ("y", -1),
        producer,
        producer,
        (),
        (),
    )

    with pytest.raises(OutputCompletionError, match="common suffix"):
        validate_output_completion_plan(plan, (OutputProducer(producer),), {producer})


def test_validate_rejects_missing_required_thread_core():
    producers = (OutputProducer(CoordXY(0, 3)), OutputProducer(CoordXY(2, 2)))
    plan = OutputCompletionPlan(
        (
            OutputRouteDecision(CoordXY(0, 3), CoordXY(0, 0), CoordZXYOffset(0, 0, -3)),
            OutputRouteDecision(
                CoordXY(2, 2), CoordXY(0, 0), CoordZXYOffset(0, -2, -2)
            ),
        ),
        CoordZXYOffset(0, 0, -3),
        ("y", -1),
        CoordXY(0, 3),
        CoordXY(0, 2),
        (),
        (),
    )

    with pytest.raises(OutputCompletionError, match="join point.*configured"):
        validate_output_completion_plan(plan, producers, {CoordXY(0, 3), CoordXY(2, 2)})


def test_validate_rejects_wrong_relay_kind_for_region():
    producer = OutputProducer(CoordXY(1, 1))
    plan = OutputCompletionPlan(
        (OutputRouteDecision(CoordXY(1, 1), CoordXY(0, 0), CoordZXYOffset(0, -1, -1)),),
        CoordZXYOffset(0, -1, -1),
        ("y", -1),
        CoordXY(1, 1),
        CoordXY(1, 1),
        (EmptyRelayCore(CoordXY(0, 1), "offline"),),
        (),
    )

    with pytest.raises(OutputCompletionError, match="kind does not match"):
        validate_output_completion_plan(plan, (producer,), {CoordXY(1, 1)})


def test_validate_accepts_online_required_route_core():
    producers = (OutputProducer(CoordXY(1, 1)), OutputProducer(CoordXY(2, 2)))
    plan = OutputCompletionPlan(
        (
            OutputRouteDecision(
                CoordXY(1, 1), CoordXY(0, 0), CoordZXYOffset(0, -1, -1)
            ),
            OutputRouteDecision(
                CoordXY(2, 2), CoordXY(0, 0), CoordZXYOffset(-1, -1, -1)
            ),
        ),
        CoordZXYOffset(0, -1, -1),
        ("y", -1),
        CoordXY(1, 1),
        CoordXY(1, 1),
        (EmptyRelayCore(CoordXY(0, 1), "online"),),
        (),
    )

    validate_output_completion_plan(plan, producers, {CoordXY(1, 1), CoordXY(2, 2)})


def test_validate_rejects_online_required_core_when_disabled():
    producer = OutputProducer(CoordXY(1, 1))
    plan = OutputCompletionPlan(
        (OutputRouteDecision(CoordXY(1, 1), CoordXY(0, 0), CoordZXYOffset(0, -1, -1)),),
        CoordZXYOffset(0, -1, -1),
        ("y", -1),
        CoordXY(1, 1),
        CoordXY(1, 1),
        (EmptyRelayCore(CoordXY(0, 1), "online"),),
        (),
    )

    with pytest.raises(OutputCompletionError, match="Online relay cores are disabled"):
        validate_output_completion_plan(
            plan,
            (producer,),
            {CoordXY(1, 1)},
            allow_empty_online=False,
        )


def test_debug_trace_and_ascii_render_show_key_route_points():
    producers = (OutputProducer(CoordXY(3, 4)), OutputProducer(CoordXY(5, 6)))
    used = {producer.coord for producer in producers}
    plan = select_output_completion_plan(
        list(producers),
        used,
        _all_empty_offline_except(*used),
        _all_empty_online_except(*used),
    )

    trace = "\n".join(debug_output_completion_plan(plan, producers, used))
    rendered = render_output_completion_plan_ascii(plan, producers, used)

    assert "source=(3,4)" in trace
    assert "join=(3,4)" in trace
    assert "selected_empty_completion_cores" in trace
    assert "S" in rendered
    assert "M" in rendered
    assert "C" in rendered
    assert "P0=(3,4)" in rendered
    assert "P1=(5,6)" in rendered
    assert "*" in rendered
