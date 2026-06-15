import pytest
from paicorelib import CoordXY, CoordZXYOffset, find_coordxy_shortest_path

from paibox.backendv2.output_cpu_ingress import (
    CpuIngressNoFeasiblePlanError,
    OutputRouteEndpoint,
    select_output_cpu_ingress_plan,
    terminal_cpu_ingress_side,
)


def test_terminal_cpu_ingress_side_uses_last_nonzero_zxy_leg():
    assert terminal_cpu_ingress_side(CoordZXYOffset(2, 0, 0)) == ("xy", 1)
    assert terminal_cpu_ingress_side(CoordZXYOffset(2, -1, 0)) == ("x", -1)
    assert terminal_cpu_ingress_side(CoordZXYOffset(2, -1, 3)) == ("y", 1)
    assert terminal_cpu_ingress_side(CoordZXYOffset(0, 0, 0)) == ("local", 0)


def test_select_output_cpu_ingress_plan_keeps_output_target_when_side_is_shared():
    root = CoordXY(5, 5)
    producer = CoordXY(6, 5)
    plan = select_output_cpu_ingress_plan(
        root, [OutputRouteEndpoint(producer, CoordXY(0, 0))]
    )

    assert plan.output_target is None
    assert plan.control_target == CoordXY(0, 1)
    assert plan.ingress_side == ("x", -1)

    data_offset, _ = find_coordxy_shortest_path(CoordXY(0, 0), producer)
    root_offset, _ = find_coordxy_shortest_path(plan.control_target, root)
    assert terminal_cpu_ingress_side(data_offset) == terminal_cpu_ingress_side(
        root_offset
    )


def test_select_output_cpu_ingress_plan_keeps_output_target_for_out_34_layout():
    root = CoordXY(2, 2)
    producer = CoordXY(3, 4)
    plan = select_output_cpu_ingress_plan(
        root, [OutputRouteEndpoint(producer, CoordXY(0, 0))]
    )

    assert plan.output_target is None
    assert plan.control_target == CoordXY(1, 0)
    assert plan.ingress_side == ("y", -1)


def test_select_output_cpu_ingress_plan_retargets_when_data_sides_differ():
    root = CoordXY(2, 2)
    producers = [CoordXY(0, 2), CoordXY(5, 2)]
    endpoints = [
        OutputRouteEndpoint(producer, CoordXY(0, 0)) for producer in producers
    ]

    assert {
        terminal_cpu_ingress_side(find_coordxy_shortest_path(CoordXY(0, 0), p)[0])
        for p in producers
    } == {("y", -1), ("x", -1)}

    plan = select_output_cpu_ingress_plan(root, endpoints)

    assert plan.output_target == CoordXY(4, 0)
    assert plan.control_target == CoordXY(4, 0)
    assert {
        terminal_cpu_ingress_side(find_coordxy_shortest_path(plan.output_target, p)[0])
        for p in producers
    } == {plan.ingress_side}


def test_select_output_cpu_ingress_plan_error_includes_route_details(monkeypatch):
    root = CoordXY(2, 2)
    endpoint = OutputRouteEndpoint(CoordXY(6, 5), CoordXY(0, 0))

    monkeypatch.setattr(
        "paibox.backendv2.output_cpu_ingress._cpu_io_target_candidates",
        lambda preferred: [],
    )

    with pytest.raises(
        CpuIngressNoFeasiblePlanError,
        match=r"producer=\(6,5\).*target=\(0,0\).*offset=.*ingress_side=",
    ):
        select_output_cpu_ingress_plan(root, [endpoint])
