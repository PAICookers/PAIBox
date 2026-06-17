import pytest
from paicorelib import CoordXY

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
    assert plan.data_penalty == 0
    assert (
        route_coord_path(plan.global_signal_root, plan.control_offset)
        == data_path[root_index:]
    )
    assert plan.complete_path_len == route_len(plan.control_offset)


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
    assert plan.data_penalty == 1
    assert "minimum-penalty fallback" in plan.diagnostics[0]

    for route in plan.output_routes:
        data_path = route_coord_path(route.producer_coord, route.offset)
        root_index = data_path.index(plan.global_signal_root)
        assert data_path[root_index:] == complete_path


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
    assert plan.data_penalty == 0
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
