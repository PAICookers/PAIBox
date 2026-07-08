from paicorelib import (
    global_signal_direction_from_name,
    global_signal_direction_name,
    global_signal_direction_names,
    global_signal_directions,
    global_signal_opposite_direction,
)

from ...model import CoreRole

GRID_WIDTH = 9
GRID_HEIGHT = 9
CHIP_ID = 0

GLOBAL_SIGNAL_DIRECTIONS = {
    global_signal_direction_name(direction): (direction.value.x, direction.value.y)
    for direction in global_signal_directions()
}
GLOBAL_SIGNAL_REVERSE_DIRECTIONS = {
    name: global_signal_direction_name(
        global_signal_opposite_direction(global_signal_direction_from_name(name))
    )
    for name in GLOBAL_SIGNAL_DIRECTIONS
}


def global_signal_dirs(bits: int, *, include_local: bool = False) -> list[str]:
    return list(global_signal_direction_names(bits, include_local=include_local))


def reverse_global_signal_dir(direction: str) -> str:
    return GLOBAL_SIGNAL_REVERSE_DIRECTIONS[direction]


def chip_core_role(x: int, y: int) -> CoreRole:
    if x == 0 and y == 0:
        return "cpu"
    if y in (0, 1):
        return "online"
    return "offline"
