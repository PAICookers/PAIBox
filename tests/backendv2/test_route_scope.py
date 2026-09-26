import pytest
from paicorelib import AERPacketZXYCopy, CoordXY, CoordZXYOffset

from paibox.backendv2 import BoardName
from paibox.backendv2.board import (
    ChipCoord,
    LocalCoreCoord,
    get_board_profile,
)
from paibox.backendv2.route_scope import TargetBoard, get_route_scope


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        (BoardName.SINGLE, "single"),
        ("single", "single"),
        (BoardName.ARRAY_2X2, "array_2x2"),
        ("array_2x2", "array_2x2"),
        ("array2x2", "array_2x2"),
    ],
)
def test_board_names_resolve_to_one_canonical_profile(
    board: TargetBoard, expected: str
):
    profile = get_board_profile(board)
    scope = get_route_scope(board)

    assert profile.name.value == expected
    assert scope.name == expected
    assert scope.board is profile
    assert scope is get_route_scope(expected)


@pytest.mark.parametrize("board", ["array_3x3", "ARRAY_2X2", "", 3])
def test_unknown_board_name_is_rejected(board):
    with pytest.raises(ValueError, match="unsupported board profile"):
        get_route_scope(board)


def test_single_scope_excludes_cpu_from_route_and_core_pools():
    scope = get_route_scope("single")

    assert scope.default_cpu.coord == CoordXY(0, 0)
    assert scope.default_cpu.chip_id == 0
    assert scope.chip_id(0, 0) == 0
    assert scope.chip_xy(0) == (0, 0)
    assert scope.default_cpu.coord not in scope.offline_core_coords
    assert scope.default_cpu.coord not in scope.online_core_coords
    assert scope.default_cpu.coord in scope.global_route_coords
    assert scope.copy_limits == (6, 8, 6)


def test_board_profiles_encode_xy_and_same_order_data_lanes():
    profile = get_board_profile("array_2x2")
    assert profile.core_grid_size == (9, 9)
    assert profile.core_grid_width == 9
    assert profile.core_grid_height == 9
    assert [chip.xy for chip in profile.chips] == ["00", "10", "01", "11"]
    assert [port.chip_xy for port in profile.cpu_ports] == ["00", "10", "01", "11"]
    assert LocalCoreCoord(8, 8).xy == "88"


def test_array_2x2_scope_uses_row_major_chip_ids_and_stride():
    scope = get_route_scope("array_2x2")

    assert [
        (cpu.chip_id, cpu.chip_x, cpu.chip_y, cpu.coord) for cpu in scope.cpu_endpoints
    ] == [
        (0, 0, 0, CoordXY(0, 0)),
        (1, 1, 0, CoordXY(9, 0)),
        (2, 0, 1, CoordXY(0, 9)),
        (3, 1, 1, CoordXY(9, 9)),
    ]
    assert scope.chip_id(1, 0) == 1
    assert scope.chip_id(0, 1) == 2
    assert scope.chip_xy(3) == (1, 1)
    assert len(scope.offline_core_coords) == 4 * len(
        get_route_scope("single").offline_core_coords
    )
    assert all(cpu.coord in scope.global_route_coords for cpu in scope.cpu_endpoints)
    assert scope.copy_limits == (6, 17, 6)


def test_array_cpu_coordinate_can_be_noc_transit_but_only_control_cpu_is_terminal():
    scope = get_route_scope("array_2x2")
    transit = scope.audit_aer_packet(
        CoordXY(0, 2), CoordZXYOffset(0, 9, 0), AERPacketZXYCopy()
    )
    assert transit.path[1] == CoordXY(1, 2)
    assert transit.valid

    non_control = scope.audit_aer_packet(
        CoordXY(0, 2), CoordZXYOffset(0, 9, -2), AERPacketZXYCopy()
    )
    assert non_control.failure is not None
    assert "control CPU" in non_control.failure.message


def test_array_data_copy_rejects_cross_chip_xy_but_allows_cardinal_lane():
    scope = get_route_scope("array_2x2")
    diagonal = scope.audit_aer_packet(
        CoordXY(8, 7), CoordZXYOffset(), AERPacketZXYCopy(1, 0, 0)
    )
    assert diagonal.failure is not None
    assert diagonal.failure.code == "cross_chip_xy"

    cardinal = scope.audit_aer_packet(
        CoordXY(8, 7), CoordZXYOffset(), AERPacketZXYCopy(0, 1, 0)
    )
    assert cardinal.valid

    claims = scope.data_edge_claims(
        CoordXY(8, 7), CoordZXYOffset(0, 1, 0), AERPacketZXYCopy()
    )
    assert claims == {(ChipCoord(0, 0), ChipCoord(1, 0))}


def test_aer_audit_rejects_online_local_delivery():
    scope = get_route_scope("single")

    result = scope.audit_aer_packet(
        CoordXY(0, 0),
        CoordZXYOffset(1, 0, 1),
        AERPacketZXYCopy(1, 6, 3),
    )

    assert len(result.actual_local) == 44
    assert len(result.expected_local) == 38
    assert result.failure is not None
    assert result.failure.code == "online_local_delivery"
    assert result.failure.coord == CoordXY(1, 1)
    assert not result.valid


def test_aer_audit_allows_online_transit_without_local_delivery():
    scope = get_route_scope("single")

    result = scope.audit_aer_packet(
        CoordXY(0, 0), CoordZXYOffset(0, 1, 2), AERPacketZXYCopy()
    )

    assert result.path == (
        CoordXY(0, 0),
        CoordXY(1, 0),
        CoordXY(1, 1),
        CoordXY(1, 2),
    )
    assert result.actual_local == (CoordXY(1, 2),)
    assert result.valid
