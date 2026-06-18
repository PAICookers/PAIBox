import pytest
from paicorelib import CoordXY, CoordZXYOffset

from paibox.backendv2.output_route_offsets import (
    candidate_offsets,
    route_coord_path,
    route_len,
    route_stays_in_grid,
    terminal_route_side,
)


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
