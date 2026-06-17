import pytest
from paicorelib import CoordXY

from paibox.backendv2.coreplacement import (
    EmptyOfflineCorePlacementV2,
    EmptyOnlineCorePlacementV2,
    OfflineCorePlacementV2,
)
from paibox.backendv2.global_signal import set_global_signal


def _offline_core(coord: CoordXY) -> OfflineCorePlacementV2:
    core = OfflineCorePlacementV2()
    core._coord = coord
    return core


@pytest.mark.parametrize(
    ("relay_kind", "expected_type"),
    [
        ("offline", EmptyOfflineCorePlacementV2),
        ("online", EmptyOnlineCorePlacementV2),
    ],
)
def test_set_global_signal_uses_planner_selected_empty_root(relay_kind, expected_type):
    root = CoordXY(1, 1)
    cores = [_offline_core(CoordXY(3, 3))]

    placements, global_starts = set_global_signal(cores, root, {root: relay_kind})

    root_placements = [placement for placement in placements if placement.coord == root]
    assert len(root_placements) == 1
    assert isinstance(root_placements[0], expected_type)
    assert CoordXY(0, 0) + global_starts[0].to_xy() == root


def test_set_global_signal_is_quiet_by_default(capsys):
    root = CoordXY(1, 1)
    cores = [_offline_core(CoordXY(3, 3))]

    set_global_signal(cores, root, {root: "offline"})

    assert capsys.readouterr().out == ""
