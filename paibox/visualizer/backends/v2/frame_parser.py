from ...model import CoreRole

GRID_WIDTH = 9
GRID_HEIGHT = 9
CHIP_ID = 0

_DIRS_BY_BIT = {
    5: ("+xy", (1, 1)),
    4: ("-xy", (-1, -1)),
    3: ("+x", (1, 0)),
    2: ("-x", (-1, 0)),
    1: ("+y", (0, 1)),
    0: ("-y", (0, -1)),
}
GLOBAL_SIGNAL_DIRECTIONS = {name: delta for _, (name, delta) in _DIRS_BY_BIT.items()}
GLOBAL_SIGNAL_REVERSE_DIRECTIONS = {
    "+xy": "-xy",
    "-xy": "+xy",
    "+x": "-x",
    "-x": "+x",
    "+y": "-y",
    "-y": "+y",
}


def global_signal_dirs(bits: int, *, include_local: bool = False) -> list[str]:
    dirs = [
        name
        for bit, (name, _) in sorted(_DIRS_BY_BIT.items(), reverse=True)
        if bits & (1 << bit)
    ]
    if include_local and bits & (1 << 6):
        dirs.insert(0, "local")
    return dirs


def reverse_global_signal_dir(direction: str) -> str:
    return GLOBAL_SIGNAL_REVERSE_DIRECTIONS[direction]


def chip_core_role(x: int, y: int) -> CoreRole:
    if x == 0 and y == 0:
        return "cpu"
    if y in (0, 1):
        return "online"
    return "offline"
