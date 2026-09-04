from paicorelib import AERPacketZXYCopy, CoordXY, CoordZXYOffset

from paibox.backendv2.route_scope import get_route_scope


def test_single_scope_excludes_cpu_from_route_and_core_pools():
    scope = get_route_scope("single")

    assert scope.default_cpu.coord == CoordXY(0, 0)
    assert scope.default_cpu.chip_id == 0
    assert scope.chip_id(0, 0) == 0
    assert scope.chip_xy(0) == (0, 0)
    assert scope.default_cpu.coord not in scope.offline_core_coords
    assert scope.default_cpu.coord not in scope.online_core_coords
    assert scope.default_cpu.coord not in scope.global_route_coords
    assert scope.copy_limits == (6, 8, 6)


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
    assert all(
        cpu.coord not in scope.global_route_coords for cpu in scope.cpu_endpoints
    )
    assert scope.copy_limits == (6, 17, 6)


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
