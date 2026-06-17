import pytest
from paicorelib import CoordXY, CoordZXYOffset, find_coordxy_shortest_path

from paibox.backendv2.output_cpu_ingress import (
    CpuIngressNoFeasiblePlanError,
    select_output_cpu_ingress_plan,
    terminal_control_ingress_side,
    terminal_data_ingress_side,
)
from paibox.backendv2.output_routes import OutputRouteEndpoint


@pytest.mark.parametrize(
    "offset",
    [
        CoordZXYOffset(2, 0, 0),
        CoordZXYOffset(2, -1, 0),
        CoordZXYOffset(2, -1, 3),
        CoordZXYOffset(0, 0, 0),
    ],
)
def test_cpu_ingress_side_wrappers_share_the_same_physical_side(offset):
    assert terminal_data_ingress_side(offset) == terminal_control_ingress_side(offset)


@pytest.mark.parametrize(
    ("root", "producer", "output_offset", "control_offset"),
    [
        (
            CoordXY(5, 5),
            CoordXY(6, 5),
            CoordZXYOffset(-4, -2, -1),
            CoordZXYOffset(-4, -1, -1),
        ),
        (
            CoordXY(2, 2),
            CoordXY(3, 4),
            CoordZXYOffset(-3, 0, -1),
            CoordZXYOffset(-1, -1, -1),
        ),
    ],
    ids=["failing_65", "out_34"],
)
def test_single_output_plan_keeps_cpu_target(
    root, producer, output_offset, control_offset
):
    cpu = CoordXY(0, 0)
    plan = select_output_cpu_ingress_plan(root, [OutputRouteEndpoint(producer, cpu)])

    assert plan.ingress_side == ("y", -1)
    assert plan.control_offset == control_offset
    assert plan.output_route_offsets() == {(producer, cpu): output_offset}
    assert terminal_data_ingress_side(output_offset) == terminal_control_ingress_side(
        control_offset
    )


def test_multi_output_plan_aligns_sides_without_retargeting_cpu():
    root = CoordXY(2, 2)
    producers = [CoordXY(0, 2), CoordXY(5, 2)]
    cpu = CoordXY(0, 0)
    endpoints = [OutputRouteEndpoint(producer, cpu) for producer in producers]

    assert {
        terminal_data_ingress_side(find_coordxy_shortest_path(cpu, p)[0])
        for p in producers
    } == {("y", -1), ("x", -1)}

    plan = select_output_cpu_ingress_plan(root, endpoints)

    assert {route.target_coord for route in plan.output_routes} == {cpu}
    assert {
        terminal_data_ingress_side(route.offset) for route in plan.output_routes
    } == {plan.ingress_side}
    assert terminal_control_ingress_side(plan.control_offset) == plan.ingress_side


def test_no_plan_error_includes_route_details(monkeypatch):
    root = CoordXY(2, 2)
    endpoint = OutputRouteEndpoint(CoordXY(6, 5), CoordXY(0, 0))

    def fake_offsets_by_ingress(start_coord, target_coord):
        if start_coord == root:
            return {("x", -1): CoordZXYOffset(0, -1, 0)}
        return {("y", -1): CoordZXYOffset(0, 0, -1)}

    monkeypatch.setattr(
        "paibox.backendv2.output_cpu_ingress._offsets_by_ingress",
        fake_offsets_by_ingress,
    )

    with pytest.raises(
        CpuIngressNoFeasiblePlanError,
        match=r"producer=\(6,5\).*target=\(0,0\).*offset=.*ingress_side=",
    ):
        select_output_cpu_ingress_plan(root, [endpoint])
