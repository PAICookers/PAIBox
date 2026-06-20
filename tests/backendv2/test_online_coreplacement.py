import numpy as np
from paicorelib import (
    CoordXY,
    CoordZXYOffset,
    OnlineCoreUpdateType,
    OnlineFrameGenV2,
    find_coordxy_shortest_path,
)

from paibox.backendv2.core_config import to_online_core_reg
from paibox.backendv2.coreplacement import OnlineCorePlacementV2
from paibox.paiir.ir.calc_params import (
    OnlineCoreParams,
    OnlineCoreSemanticMode,
    OnlineCoreWorkMode,
    OnlineGradientRole,
    OnlineUpdateDirection,
)


def _make_online_params() -> OnlineCoreParams:
    return OnlineCoreParams(
        semantic_mode=OnlineCoreSemanticMode.GRADIENT,
        gradient_role=OnlineGradientRole.OUTPUT,
        update_direction=OnlineUpdateDirection.FORWARD,
        output_width=OnlineCoreUpdateType.WEIGHT,
        neuron_number=8,
        thread_number=3,
        tick_start=5,
    )


def test_to_online_core_reg_resolves_work_mode_and_maps_fields():
    coord = CoordXY(2, 3)
    params = _make_online_params()

    core_reg = to_online_core_reg(params, coord)

    assert core_reg.name == "online_core_reg_at_(2,3)"
    assert core_reg.work_mode == OnlineCoreWorkMode.OUTPUT_LAYER_GRADIENT
    assert core_reg.neuron_number == 8
    assert core_reg.thread_number == 3
    assert core_reg.tick_start == 5


def test_to_online_core_reg_uses_auto_test_route_when_unset():
    coord = CoordXY(2, 3)
    params = _make_online_params()
    auto_conf = OnlineCorePlacementV2(params)
    auto_conf._coord = coord
    auto_conf.set_auto_core_config(CoordZXYOffset(-2, 0, -1))

    core_reg = to_online_core_reg(params, coord, auto_conf=auto_conf.auto_core_config)

    assert (core_reg.test_core_xy, core_reg.test_core_x, core_reg.test_core_y) == (
        -2,
        0,
        -1,
    )


def test_to_online_core_reg_preserves_explicit_test_route_over_auto():
    coord = CoordXY(2, 3)
    params = _make_online_params()
    params.test_core_xy = 1
    params.test_core_x = -1
    params.test_core_y = 0
    auto_conf = OnlineCorePlacementV2(params)
    auto_conf._coord = coord
    auto_conf.set_auto_core_config(CoordZXYOffset(-2, 0, -1))

    core_reg = to_online_core_reg(params, coord, auto_conf=auto_conf.auto_core_config)

    assert (core_reg.test_core_xy, core_reg.test_core_x, core_reg.test_core_y) == (
        1,
        -1,
        0,
    )


def test_online_core_placement_to_frame_uses_online_generators():
    placement = OnlineCorePlacementV2(_make_online_params(), n_timestep=7)
    placement._coord = CoordXY(4, 1)
    placement.set_auto_core_config()

    frame1, frame2, frame3 = placement.to_frame()
    pkt_offset, _ = find_coordxy_shortest_path(placement.coord)
    core_reg = to_online_core_reg(
        placement.core_params, placement.coord, auto_conf=placement.auto_core_config
    )
    expected_frame2 = np.concatenate(
        [
            OnlineFrameGenV2.gen_control_frame1(pkt_offset, n_timestep=7),
            OnlineFrameGenV2.gen_control_frame2(pkt_offset),
            OnlineFrameGenV2.gen_control_frame3(
                pkt_offset, thread_id=core_reg.thread_number
            ),
            OnlineFrameGenV2.gen_control_frame4(pkt_offset),
        ],
        axis=0,
    )

    assert np.array_equal(
        frame1, OnlineFrameGenV2.gen_config_frame1(pkt_offset, core_reg)
    )
    assert np.array_equal(frame2, expected_frame2)
    assert frame3 is None
