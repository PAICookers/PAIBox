import pytest
from paicorelib import CoordXY, global_signal_direction_names

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
    # Per chip design, every core (empty relay cores included) sets the local bit.
    assert "local" in global_signal_direction_names(
        root_placements[0].auto_core_config.global_send, include_local=True
    )
    assert CoordXY(0, 0) + global_starts[0].to_xy() == root


def test_set_global_signal_is_quiet_by_default(capsys):
    root = CoordXY(1, 1)
    cores = [_offline_core(CoordXY(3, 3))]

    set_global_signal(cores, root, {root: "offline"})

    assert capsys.readouterr().out == ""


def test_set_global_signal_local_bit_set_for_added_relay_core():
    root = CoordXY(1, 1)
    cores = [_offline_core(CoordXY(3, 3))]

    placements, _ = set_global_signal(cores, root, {root: "offline"})

    by_coord = {placement.coord: placement for placement in placements}
    # Per chip design, the local bit is set on every core, including the empty
    # relay core added at (2, 2) purely to carry the global signal.
    assert "local" in global_signal_direction_names(
        by_coord[CoordXY(3, 3)].auto_core_config.global_send, include_local=True
    )
    assert "local" in global_signal_direction_names(
        by_coord[CoordXY(2, 2)].auto_core_config.global_send, include_local=True
    )


def test_set_global_signal_local_bit_set_for_existing_empty_root():
    root = CoordXY(1, 1)
    empty_root = EmptyOfflineCorePlacementV2()
    empty_root._coord = root
    cores = [empty_root, _offline_core(CoordXY(3, 3))]

    placements, _ = set_global_signal(cores, root, {root: "offline"})

    by_coord = {placement.coord: placement for placement in placements}
    # Per chip design, the local bit is set on every core, empty root included.
    assert "local" in global_signal_direction_names(
        by_coord[root].auto_core_config.global_send, include_local=True
    )
    assert "local" in global_signal_direction_names(
        by_coord[CoordXY(3, 3)].auto_core_config.global_send, include_local=True
    )
