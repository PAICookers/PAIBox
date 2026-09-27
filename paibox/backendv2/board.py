"""Fixed PAICORE 2.5 board profiles used by backendv2 deployment."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class BoardName(StrEnum):
    SINGLE = "single"
    ARRAY_2X2 = "array_2x2"


TargetBoard = BoardName | Literal["single", "array2x2", "array_2x2"]
PAICORE_2p5_CORE_GRID_SIZE = (9, 9)


@dataclass(frozen=True, slots=True, order=True)
class ChipCoord:
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("chip coordinates must be non-negative")

    @property
    def xy(self) -> str:
        return f"{self.x}{self.y}"


@dataclass(frozen=True, slots=True, order=True)
class LocalCoreCoord:
    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("core coordinates must be non-negative")

    @property
    def xy(self) -> str:
        return f"{self.x}{self.y}"


@dataclass(frozen=True, slots=True, order=True)
class BoardCoreAddr:
    chip: ChipCoord
    core: LocalCoreCoord

    @property
    def chip_xy(self) -> str:
        return self.chip.xy

    @property
    def core_xy(self) -> str:
        return self.core.xy


@dataclass(frozen=True, slots=True)
class BoardEdge:
    src: ChipCoord
    dst: ChipCoord


@dataclass(frozen=True, slots=True)
class BoardProfile:
    """Authoritative hardware facts for one backendv2 board target.

    ``copy_limits`` is the maximum absolute AER multicast copy count for the
    ``(Z, X, Y)`` copy fields.  It bounds route-shape enumeration; it is not a
    chip count or a coordinate extent.
    """

    name: BoardName
    version: int
    chips: tuple[ChipCoord, ...]
    data_edges: tuple[BoardEdge, ...]
    control_cpu: BoardCoreAddr
    cpu_ports: tuple[BoardCoreAddr, ...]
    online_row_count: int
    copy_limits: tuple[int, int, int]
    core_grid_size: tuple[int, int] = PAICORE_2p5_CORE_GRID_SIZE

    def __post_init__(self) -> None:
        width, height = self.core_grid_size
        if width <= 0 or height <= 0:
            raise ValueError("core grid dimensions must be positive")
        if not 0 <= self.online_row_count < height:
            raise ValueError("online row count must be within the core grid")
        for address in (self.control_cpu, *self.cpu_ports):
            if address.chip not in self.chips:
                raise ValueError(f"CPU chip {address.chip.xy} is outside the board")
            if address.core.x >= width or address.core.y >= height:
                raise ValueError(
                    f"CPU core {address.core_xy} is outside the "
                    f"{width}x{height} core grid"
                )

    @property
    def core_grid_width(self) -> int:
        return self.core_grid_size[0]

    @property
    def core_grid_height(self) -> int:
        return self.core_grid_size[1]

    def chip_at(self, x: int, y: int) -> ChipCoord:
        chip = ChipCoord(x, y)
        if chip not in self.chips:
            raise ValueError(f"chip {chip.xy} is not in board profile {self.name!r}")
        return chip

    def has_edge(self, src: ChipCoord, dst: ChipCoord) -> bool:
        return any(edge.src == src and edge.dst == dst for edge in self.data_edges)

def _edges(chips: tuple[ChipCoord, ...]) -> tuple[BoardEdge, ...]:
    result: list[BoardEdge] = []
    for chip in chips:
        for dx, dy in ((1, 0), (0, 1)):
            other = ChipCoord(chip.x + dx, chip.y + dy)
            if other in chips:
                result.extend((BoardEdge(chip, other), BoardEdge(other, chip)))
    return tuple(result)


_SINGLE_CHIPS = (ChipCoord(0, 0),)
_ARRAY_CHIPS = tuple(ChipCoord(x, y) for y in range(2) for x in range(2))

BOARD_PROFILES: dict[BoardName, BoardProfile] = {
    BoardName.SINGLE: BoardProfile(
        name=BoardName.SINGLE,
        version=1,
        chips=_SINGLE_CHIPS,
        data_edges=(),
        control_cpu=BoardCoreAddr(ChipCoord(0, 0), LocalCoreCoord(0, 0)),
        cpu_ports=(BoardCoreAddr(ChipCoord(0, 0), LocalCoreCoord(0, 0)),),
        online_row_count=2,
        copy_limits=(6, 8, 6),
    ),
    BoardName.ARRAY_2X2: BoardProfile(
        name=BoardName.ARRAY_2X2,
        version=1,
        chips=_ARRAY_CHIPS,
        data_edges=_edges(_ARRAY_CHIPS),
        control_cpu=BoardCoreAddr(ChipCoord(0, 0), LocalCoreCoord(0, 0)),
        cpu_ports=tuple(
            BoardCoreAddr(chip, LocalCoreCoord(0, 0)) for chip in _ARRAY_CHIPS
        ),
        online_row_count=2,
        copy_limits=(6, 17, 6),
    ),
}


def get_board_profile(name: TargetBoard) -> BoardProfile:
    """Resolve a board enum or accepted string alias to its canonical profile."""
    if name == "array2x2":
        name = BoardName.ARRAY_2X2
    try:
        return BOARD_PROFILES[BoardName(name)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported board profile {name!r}") from exc
