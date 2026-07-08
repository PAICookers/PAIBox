import pytest
from paicorelib import CoordXY, CoordZXYOffset, route_coord_path

from paibox.backendv2 import output_completion_planner as planner_mod
from paibox.backendv2.output_completion_planner import (
    EmptyThreadCore,
    OutputCompletionError,
    OutputCompletionNoFeasiblePlanError,
    OutputCompletionPlan,
    OutputProducer,
    OutputRouteDecision,
    candidate_offsets,
    select_output_completion_plan,
    terminal_route_side,
    validate_output_completion_plan,
)
from paibox.backendv2.route_scope import get_route_scope

SINGLE_SCOPE = get_route_scope("single")


def _all_empty_offline_except(*used: CoordXY) -> set[CoordXY]:
    used_set = set(used)
    return {
        coord for coord in SINGLE_SCOPE.offline_core_coords if coord not in used_set
    }


def _all_empty_online_except(*used: CoordXY) -> set[CoordXY]:
    used_set = set(used)
    return {coord for coord in SINGLE_SCOPE.online_core_coords if coord not in used_set}


def _coord_set(cores: tuple[EmptyThreadCore, ...]) -> set[CoordXY]:
    return {core.coord for core in cores}


def _kind_map(cores: tuple[EmptyThreadCore, ...]) -> dict[CoordXY, str]:
    return {core.coord: core.kind for core in cores}


def _assert_shared_join_suffix(plan: OutputCompletionPlan) -> None:
    assert plan.global_signal_root == plan.completion_join_point
    complete_path = route_coord_path(plan.global_signal_root, plan.control_offset)
    suffix = complete_path[complete_path.index(plan.completion_join_point) :]
    assert terminal_route_side(plan.control_offset) == plan.ingress_side

    for route in plan.output_routes:
        data_path = route_coord_path(route.producer_coord, route.offset)
        assert plan.global_signal_root in data_path
        assert data_path[data_path.index(plan.completion_join_point) :] == suffix
        assert terminal_route_side(route.offset) == plan.ingress_side


def _assert_data_prefix_cores_are_required(
    plan: OutputCompletionPlan, used: set[CoordXY]
) -> None:
    final_thread = (
        used
        | _coord_set(plan.completion_thread_cores)
        | _coord_set(plan.global_signal_relay_cores)
    )
    for route in plan.output_routes:
        path = route_coord_path(route.producer_coord, route.offset)
        prefix = path[: path.index(plan.completion_join_point) + 1]
        for coord in prefix:
            if (
                coord not in SINGLE_SCOPE.cpu_coords
                and SINGLE_SCOPE.is_configurable_thread_core_coord(coord)
            ):
                assert coord in final_thread


def _assert_shared_suffix_transit_is_not_required(plan: OutputCompletionPlan) -> None:
    complete_path = route_coord_path(plan.global_signal_root, plan.control_offset)
    suffix = complete_path[complete_path.index(plan.completion_join_point) :]
    required = _coord_set(plan.completion_thread_cores)
    for coord in suffix[1:-1]:
        assert coord not in required


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


def test_candidate_offsets_are_sorted_by_l1_norm_then_offset_tuple():
    offsets = candidate_offsets(CoordXY(3, 4), CoordXY(0, 0))

    assert offsets[0] == CoordZXYOffset(-3, 0, -1)
    assert [offset.l1_norm() for offset in offsets] == sorted(
        offset.l1_norm() for offset in offsets
    )
    assert all(
        SINGLE_SCOPE.route_path_valid(
            route_coord_path(CoordXY(3, 4), offset), CoordXY(0, 0)
        )
        for offset in offsets
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
    assert plan.completion_thread_cores == ()
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
    assert _kind_map(plan.completion_thread_cores) == {CoordXY(4, 5): "offline"}
    _assert_data_prefix_cores_are_required(plan, set(producers))
    _assert_shared_join_suffix(plan)


def test_data_offsets_adjust_when_needed_to_form_shortest_shared_suffix():
    producers = [CoordXY(4, 2), CoordXY(2, 4)]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
    )

    assert plan.completion_join_point == CoordXY(0, 2)
    assert plan.global_signal_root == CoordXY(0, 2)
    assert {route.producer_coord: route.offset for route in plan.output_routes} == {
        CoordXY(4, 2): CoordZXYOffset(0, -4, -2),
        CoordXY(2, 4): CoordZXYOffset(-2, 0, -2),
    }
    assert _kind_map(plan.completion_thread_cores) == {
        CoordXY(0, 2): "offline",
        CoordXY(1, 2): "offline",
        CoordXY(1, 3): "offline",
        CoordXY(2, 2): "offline",
        CoordXY(3, 2): "offline",
    }
    _assert_data_prefix_cores_are_required(plan, set(producers))
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
    assert _kind_map(plan.completion_thread_cores) == {CoordXY(1, 2): "offline"}
    _assert_data_prefix_cores_are_required(plan, set(producers))
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

    assert plan.global_signal_root == CoordXY(1, 1)
    assert plan.completion_join_point == CoordXY(1, 1)
    assert _kind_map(plan.completion_thread_cores) == {CoordXY(2, 1): "online"}
    _assert_data_prefix_cores_are_required(plan, set(producers))
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
    assert _kind_map(plan.completion_thread_cores) == {CoordXY(1, 2): "offline"}
    _assert_data_prefix_cores_are_required(plan, set(producers))
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

    assert plan.completion_join_point == CoordXY(0, 2)
    assert plan.global_signal_root == CoordXY(0, 2)
    assert CoordXY(0, 2) in _coord_set(plan.completion_thread_cores)
    for route in plan.output_routes:
        assert CoordXY(0, 1) in route_coord_path(route.producer_coord, route.offset)
    assert all(core.kind == "offline" for core in plan.completion_thread_cores)
    _assert_data_prefix_cores_are_required(plan, set(producers))
    _assert_shared_suffix_transit_is_not_required(plan)
    _assert_shared_join_suffix(plan)


def test_reported_blocks01_two_row_producers_pass_through_global_signal_root():
    producers = [
        CoordXY(0, 3),
        CoordXY(1, 3),
        CoordXY(2, 3),
        CoordXY(3, 3),
        CoordXY(4, 3),
        CoordXY(0, 5),
        CoordXY(1, 5),
        CoordXY(2, 5),
        CoordXY(3, 5),
        CoordXY(4, 5),
    ]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
        allow_empty_online=False,
    )

    assert plan.global_signal_root == CoordXY(0, 2)
    assert plan.completion_join_point == CoordXY(0, 2)
    assert {route.producer_coord: route.offset for route in plan.output_routes} == {
        CoordXY(0, 3): CoordZXYOffset(0, 0, -3),
        CoordXY(1, 3): CoordZXYOffset(-1, 0, -2),
        CoordXY(2, 3): CoordZXYOffset(-1, -1, -2),
        CoordXY(3, 3): CoordZXYOffset(-1, -2, -2),
        CoordXY(4, 3): CoordZXYOffset(-1, -3, -2),
        CoordXY(0, 5): CoordZXYOffset(0, 0, -5),
        CoordXY(1, 5): CoordZXYOffset(-1, 0, -4),
        CoordXY(2, 5): CoordZXYOffset(-2, 0, -3),
        CoordXY(3, 5): CoordZXYOffset(-3, 0, -2),
        CoordXY(4, 5): CoordZXYOffset(-3, -1, -2),
    }
    assert _kind_map(plan.completion_thread_cores) == {
        CoordXY(0, 2): "offline",
        CoordXY(0, 4): "offline",
        CoordXY(1, 2): "offline",
        CoordXY(1, 4): "offline",
        CoordXY(2, 2): "offline",
        CoordXY(2, 4): "offline",
        CoordXY(3, 2): "offline",
        CoordXY(3, 4): "offline",
    }
    _assert_data_prefix_cores_are_required(plan, set(producers))
    _assert_shared_suffix_transit_is_not_required(plan)
    _assert_shared_join_suffix(plan)


def test_u_shape_output_producers_require_prefix_route_cores():
    producers = [
        CoordXY(0, 2),
        CoordXY(1, 3),
        CoordXY(2, 4),
        CoordXY(3, 5),
        CoordXY(4, 6),
        CoordXY(5, 6),
        CoordXY(5, 5),
        CoordXY(5, 4),
        CoordXY(5, 3),
        CoordXY(5, 2),
    ]

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
        allow_empty_online=False,
    )

    assert plan.global_signal_root == CoordXY(0, 2)
    assert plan.completion_join_point == CoordXY(0, 2)
    assert _coord_set(plan.completion_thread_cores) == {
        CoordXY(1, 2),
        CoordXY(2, 2),
        CoordXY(2, 3),
        CoordXY(3, 2),
        CoordXY(3, 3),
        CoordXY(3, 4),
        CoordXY(4, 2),
        CoordXY(4, 3),
        CoordXY(4, 4),
        CoordXY(4, 5),
    }
    assert all(core.kind == "offline" for core in plan.completion_thread_cores)
    _assert_data_prefix_cores_are_required(plan, set(producers))
    _assert_shared_suffix_transit_is_not_required(plan)
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
        match=r"DATA computation relay cores are not supported",
    ):
        select_output_completion_plan(
            [OutputProducer(producer) for producer in producers],
            set(producers),
            set(),
            set(),
            allow_empty_online=False,
        )


def test_same_suffix_falls_back_to_longer_data_route_when_prefix_core_unavailable():
    producer = CoordXY(0, 1)
    join_point = CoordXY(0, 3)
    unavailable_prefix_core = CoordXY(1, 3)
    empty_offline = _all_empty_offline_except(producer)
    empty_offline.remove(unavailable_prefix_core)

    planner = planner_mod.OutputCompletionPlanner(
        [OutputProducer(producer)],
        {producer},
        empty_offline,
        _all_empty_online_except(producer),
        False,
    )
    suffix = next(
        suffix
        for suffix in planner_mod._common_suffixes_for_path(
            route_coord_path(producer, CoordZXYOffset(2, -2, -3)),
            SINGLE_SCOPE,
        )
        if suffix.join_point == join_point
    )
    options = planner._enumerate_data_route_options(OutputProducer(producer))
    selected = planner._best_data_route_for_suffix(options[suffix], suffix)
    assert selected is not None

    route, thread_cores = selected
    assert route.offset == CoordZXYOffset(3, -3, -4)
    assert unavailable_prefix_core not in route_coord_path(
        route.producer.coord, route.offset
    )
    assert _coord_set(thread_cores) == {
        CoordXY(0, 3),
        CoordXY(0, 4),
        CoordXY(1, 2),
        CoordXY(1, 4),
        CoordXY(2, 3),
        CoordXY(2, 4),
        CoordXY(3, 4),
    }


def test_completion_candidate_selection_falls_back_when_best_join_has_no_source(
    monkeypatch,
):
    producers = [CoordXY(0, 2), CoordXY(2, 3)]
    blocked_join = CoordXY(0, 2)
    original_build = planner_mod.OutputCompletionPlanner._build_completion_candidates

    def fake_build_completion_candidates(self, join_candidate):
        if join_candidate.suffix.join_point == blocked_join:
            return ()
        return original_build(self, join_candidate)

    monkeypatch.setattr(
        planner_mod.OutputCompletionPlanner,
        "_build_completion_candidates",
        fake_build_completion_candidates,
    )

    plan = select_output_completion_plan(
        [OutputProducer(producer) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
        _all_empty_online_except(*producers),
        allow_empty_online=False,
    )

    assert plan.completion_join_point != blocked_join
    assert plan.completion_join_point == CoordXY(0, 3)
    _assert_data_prefix_cores_are_required(plan, set(producers))
    _assert_shared_join_suffix(plan)


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
        (EmptyThreadCore(CoordXY(0, 1), "offline"),),
        (),
    )

    with pytest.raises(OutputCompletionError, match="kind does not match"):
        validate_output_completion_plan(plan, (producer,), {CoordXY(1, 1)})


def test_validate_accepts_online_completion_thread_core():
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
        (EmptyThreadCore(CoordXY(0, 1), "online"),),
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
        (EmptyThreadCore(CoordXY(0, 1), "online"),),
        (),
    )

    with pytest.raises(
        OutputCompletionError, match="Online empty thread cores are disabled"
    ):
        validate_output_completion_plan(
            plan, (producer,), {CoordXY(1, 1)}, allow_empty_online=False
        )
