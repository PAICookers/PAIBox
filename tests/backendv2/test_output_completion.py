import pytest
from paicorelib import CoordXY, find_coordxy_shortest_path

from paibox.backendv2.global_signal import GlobalSignalTree
from paibox.backendv2.output_completion import (
    OutputCompletionNoFeasiblePlanError,
    OutputProducer,
    select_output_completion_plan,
)
from paibox.backendv2.output_route_offsets import route_coord_path, route_len


def _all_empty_offline_except(*used: CoordXY) -> set[CoordXY]:
    used_set = set(used)
    return {
        CoordXY(x, y)
        for x in range(9)
        for y in range(9)
        if CoordXY(x, y) not in used_set
    }


def _assert_control_loop_cost_fields(plan) -> None:
    assert plan.score.cpu_to_root_path_len == find_coordxy_shortest_path(
        plan.global_signal_root
    )[1]
    assert plan.score.complete_path_len == route_len(plan.control_offset)
    assert (
        plan.score.control_loop_cost
        == plan.score.cpu_to_root_path_len
        + 2 * plan.score.global_signal_tree_max_depth
        + plan.score.complete_path_len
    )


def test_single_output_root_is_on_data_path_and_complete_path_is_suffix():
    producer = CoordXY(3, 4)
    cpu = CoordXY(0, 0)

    plan = select_output_completion_plan(
        [OutputProducer(producer, cpu, 7)],
        {producer},
        _all_empty_offline_except(producer),
    )

    route = plan.output_routes[0]
    data_path = route_coord_path(route.producer_coord, route.offset)
    root_index = data_path.index(plan.global_signal_root)

    assert plan.global_signal_root == producer
    assert plan.root_kind == "used"
    assert plan.score.data_penalty == 0
    assert (
        route_coord_path(plan.global_signal_root, plan.control_offset)
        == data_path[root_index:]
    )
    _assert_control_loop_cost_fields(plan)


def test_multi_output_selects_common_suffix_root():
    producers = [CoordXY(4, 2), CoordXY(2, 4)]
    cpu = CoordXY(0, 0)

    plan = select_output_completion_plan(
        [OutputProducer(producer, cpu, 1) for producer in producers],
        set(producers),
        _all_empty_offline_except(*producers),
    )

    complete_path = route_coord_path(plan.global_signal_root, plan.control_offset)
    assert plan.global_signal_root == CoordXY(0, 1)
    assert plan.root_kind == "empty_offline"
    assert plan.score.data_penalty == 1
    assert "minimum-penalty fallback" in plan.diagnostics[0]
    _assert_control_loop_cost_fields(plan)

    for route in plan.output_routes:
        data_path = route_coord_path(route.producer_coord, route.offset)
        root_index = data_path.index(plan.global_signal_root)
        assert data_path[root_index:] == complete_path


def test_control_loop_cost_ranks_tree_and_complete_path_together(monkeypatch):
    producer = CoordXY(1, 4)
    far_shallow_root = CoordXY(0, 3)
    near_deeper_root = CoordXY(0, 1)

    def fake_global_signal_tree(raw_points, root=None, verbose=False):
        del raw_points, verbose
        assert root is not None
        depth_by_root = {
            producer: 5,
            far_shallow_root: 1,
            near_deeper_root: 2,
        }
        return GlobalSignalTree(
            root=root,
            order=[root],
            added=[],
            send_directions={},
            max_depth=depth_by_root.get(root, 99),
            total_edges=4,
        )

    monkeypatch.setattr(
        "paibox.backendv2.output_completion.solve_global_signal_tree",
        fake_global_signal_tree,
    )

    plan = select_output_completion_plan(
        [OutputProducer(producer, CoordXY(0, 0), 1)],
        {producer, far_shallow_root, near_deeper_root},
        _all_empty_offline_except(producer, far_shallow_root, near_deeper_root),
    )

    assert plan.global_signal_root == near_deeper_root
    assert plan.root_kind == "used"
    assert plan.score.data_penalty == 0
    assert plan.score.global_signal_tree_max_depth == 2
    assert plan.score.global_signal_tree_total_edges == 4
    assert plan.score.cpu_to_root_path_len == 1
    assert plan.score.complete_path_len == 1
    assert plan.score.control_loop_cost == 6


@pytest.mark.parametrize(
    (
        "available_empty_offline",
        "available_empty_online",
        "allow_empty_online",
        "empty_online_frame_supported",
    ),
    [
        ({CoordXY(2, 3)}, set(), False, False),
        ({CoordXY(2, 3)}, {CoordXY(2, 3)}, True, True),
    ],
    ids=["used_beats_empty", "used_beats_online_empty"],
)
def test_used_root_priority_for_equal_data_penalty(
    available_empty_offline,
    available_empty_online,
    allow_empty_online,
    empty_online_frame_supported,
):
    producer = CoordXY(3, 4)
    cpu = CoordXY(0, 0)

    plan = select_output_completion_plan(
        [OutputProducer(producer, cpu, 1)],
        {producer},
        available_empty_offline,
        available_empty_online,
        allow_empty_online=allow_empty_online,
        empty_online_frame_supported=empty_online_frame_supported,
    )

    assert plan.global_signal_root == producer
    assert plan.root_kind == "used"
    assert plan.score.data_penalty == 0
    assert not plan.root_is_online_empty


def test_online_empty_is_not_selected_when_frame_export_is_unavailable():
    producer = CoordXY(3, 4)
    online_only_root = CoordXY(2, 3)
    cpu = CoordXY(0, 0)

    plan = select_output_completion_plan(
        [OutputProducer(producer, cpu, 1)],
        {producer},
        set(),
        {online_only_root},
        allow_empty_online=True,
        empty_online_frame_supported=False,
    )

    assert plan.global_signal_root != online_only_root
    assert plan.root_kind == "used"
    assert "online frame1" in plan.diagnostics[0]


def test_no_feasible_root_raises_with_context():
    producer = CoordXY(3, 4)

    with pytest.raises(
        OutputCompletionNoFeasiblePlanError,
        match=r"Cannot find a global signal root.*producers=.*used=",
    ):
        select_output_completion_plan(
            [OutputProducer(producer, CoordXY(0, 0), 1)], set(), set(), set()
        )
