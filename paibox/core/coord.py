from enum import Enum
from typing import Tuple, Union
from math import sqrt
from pydantic.dataclasses import dataclass
from pydantic import field_validator


@dataclass
class Coord:
    """Coordinates of the cores.

    NOTE: Set coordinates (x, y) for every cores. Left to right, +X, up to down, +Y.
    """

    _COORD_MAX_LIMIT = 32
    _COORD_LOW_LIMIT = 0

    x: int
    y: int

    @field_validator("x", "y", mode="before")
    def _coord_range_limit(cls, v: int) -> int:
        if v < cls._COORD_LOW_LIMIT or v >= cls._COORD_MAX_LIMIT:
            raise ValueError(
                f"Out of range: [{cls._COORD_LOW_LIMIT}, {cls._COORD_MAX_LIMIT}), but got {v}"
            )

        return v

    @classmethod
    def from_tuple(cls, pos):
        return cls(*pos)

    def __add__(self, other) -> "Coord":
        """
        Examples:

        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + CoordOffset(1, 1)

        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + (1, 2)

        NOTE: Coord + Coord is meaningless.
        """
        if isinstance(other, Tuple):
            return Coord(self.x + other[0], self.y + other[1])

        return Coord(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "CoordOffset") -> "Coord":
        """
        Example:

        >>> c1 = Coord(1, 1)
        >>> c1 += CoordOffset(1, 1)
        """
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other) -> "CoordOffset":
        if isinstance(other, Tuple):
            return CoordOffset(self.x - other[0], self.y - other[1])

        return CoordOffset(self.x - other.x, self.y - other.y)

    def __eq__(self, other) -> bool:
        """
        Examples:
        1.
        >>> Coord(4, 5) == (4, 5)
        True

        2.
        >>> Coord(4, 5) == Coord(4, 6)
        False
        """
        if isinstance(other, Tuple):
            return self.to_tuple() == other

        if isinstance(other, CoordOffset):
            raise TypeError("A CoordOffset cannot be compared with a Coord!")

        return self.x == other.x and self.y == other.y

    def __ne__(self, other) -> bool:
        return self.x != other.x or self.y != other.y

    def __lt__(self, other) -> bool:
        """Whether on the left or below"""
        return self.x < other.x or self.y < other.y

    def __gt__(self, other) -> bool:
        """Whether on the right and above"""
        return (
            (self.x > other.x and self.y > other.y)
            or (self.x == other.x and self.y > other.y)
            or (self.x > other.x and self.y == other.y)
        )

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    __repr__ = __str__

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple"""
        return (self.x, self.y)


class CoordOffset(Coord):
    """Offset of coordinates"""

    _COORDOFFSET_MAX_LIMIT = 32
    _COORDOFFSET_LOW_LIMIT = -32

    def __init__(self, _x: int, _y: int) -> None:
        if not (
            self._COORDOFFSET_LOW_LIMIT < _x < self._COORDOFFSET_MAX_LIMIT
            and self._COORDOFFSET_LOW_LIMIT < _y < self._COORDOFFSET_MAX_LIMIT
        ):
            raise ValueError(
                f"({_x}, {_y}) is out of range: ({self._COORDOFFSET_LOW_LIMIT}, {self._COORDOFFSET_MAX_LIMIT})"
            )

        self.x: int = _x
        self.y: int = _y

    def __add__(self, other) -> Union["CoordOffset", Coord]:
        """
        NOTE:
        1. CoordOffset = CoordOffset + CoordOffset
        2. Coord = CoordOffset + Coord
        3. CoordOffset = CoordOffset + Tuple
        """
        if isinstance(other, CoordOffset):
            return CoordOffset(self.x + other.x, self.y + other.y)
        elif isinstance(other, Tuple):
            return CoordOffset(self.x + other[0], self.y + other[1])

        return Coord(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "CoordOffset") -> "CoordOffset":
        """Attention: CoordOffset += Coord is illegal."""
        self.x += other.x
        self.y += other.y

        return self

    def __sub__(self, other) -> "CoordOffset":
        if isinstance(other, Coord):
            raise TypeError("A CoordOffset cannot substract a Coord")

        return CoordOffset(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "CoordOffset") -> "CoordOffset":
        """
        Attention: CoordOffset -= Coord is illegal.
        """
        self.x -= other.x
        self.y -= other.y

        return self

    # Functions below defines types of the distance
    def get_euclidean_distance(self) -> float:
        """Euclidean distance"""
        return sqrt((self.x**2 + self.y**2))

    def get_manhattan_distance(self) -> int:
        """Manhattan distance"""
        return abs(self.x) + abs(self.y)

    def get_chebyshev_distance(self) -> int:
        """Chebyshev distance"""
        return max(abs(self.x), abs(self.y))


class BitStatusType(Enum):
    ZERO = 0
    ONE = 1
    ANY = 2


class ReplicationID(Coord):
    status: BitStatusType

    # TODO


if __name__ == "__main__":
    c = Coord.from_tuple((2, 31))

    print(c)
