from typing import overload, Tuple, Union, final
from enum import Enum
from math import sqrt
from pydantic.dataclasses import dataclass
from pydantic import Field


__all__ = ["Coord", "CoordOffset", "DistanceType"]


@dataclass
@final
class Coord:
    """Coordinates of the cores. Set coordinates (x, y) for every cores.

    Left to right, +X, up to down, +Y.
    """

    _COORD_MAX_LIMIT = (1 << 5) - 1
    _COORD_LOW_LIMIT = 0

    x: int = Field(..., ge=_COORD_LOW_LIMIT, le=_COORD_MAX_LIMIT)
    y: int = Field(..., ge=_COORD_LOW_LIMIT, le=_COORD_MAX_LIMIT)

    @classmethod
    def from_tuple(cls, pos) -> "Coord":
        return cls(*pos)

    @overload
    def __sub__(self, other: "Coord") -> "CoordOffset":
        ...

    @overload
    def __sub__(self, other: "CoordOffset") -> "Coord":
        ...

    def __add__(self, other: "CoordOffset") -> "Coord":
        """
        Examples:

        Coord = Coord + CoordOffset
        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + CoordOffset(1, 1)
        c1: Coord(2, 2)

        NOTE: Coord + Coord is meaningless.
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        return Coord(self.x + other.delta_x, self.y + other.delta_y)

    def __iadd__(self, other: "CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c1 += CoordOffset(1, 1)
        Coord(2, 2)
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.x += other.delta_x
        self.y += other.delta_y

        return self

    def __sub__(
        self, other: Union["Coord", "CoordOffset"]
    ) -> Union["Coord", "CoordOffset"]:
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = Coord(2, 2) - c1
        c2: CoordOffset(1, 1)
        """
        if isinstance(other, Coord):
            return CoordOffset(self.x - other.x, self.y - other.y)

        if isinstance(other, CoordOffset):
            return Coord(self.x - other.delta_x, self.y - other.delta_y)

        raise TypeError(f"Unsupported type: {type(other)}")

    def __isub__(self, other: "CoordOffset") -> "Coord":
        """
        Example:
        >>> c1 = Coord(2, 2)
        >>> c1 -= CoordOffset(1, 1)
        Coord(1, 1)
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.x -= other.delta_x
        self.y -= other.delta_y

        return self

    """Operations below are used only when comparing with a Cooord."""

    def __eq__(self, other: "Coord") -> bool:
        """
        Example:
        >>> Coord(4, 5) == Coord(4, 6)
        False
        """
        if not isinstance(other, Coord):
            raise TypeError(f"Unsupported type: {type(other)}")

        return self.x == other.x and self.y == other.y

    def __ne__(self, other: "Coord") -> bool:
        """
        Examples:
        >>> Coord(4, 5) != Coord(4, 6)
        True

        >>> Coord(4, 5) != Coord(5, 5)
        True
        """
        if not isinstance(other, Coord):
            raise TypeError(f"Unsupported type: {type(other)}")

        return self.x != other.x or self.y != other.y

    def __lt__(self, other: "Coord") -> bool:
        """Whether the coord is on the left OR below of other.

        Examples:
        >>> Coord(4, 5) < Coord(4, 6)
        True

        >>> Coord(4, 5) < Coord(5, 5)
        True

        >>> Coord(4, 5) < Coord(5, 3)
        True
        """
        if not isinstance(other, Coord):
            raise TypeError(f"Unsupported type: {type(other)}")

        return self.x < other.x or self.y < other.y

    def __gt__(self, other: "Coord") -> bool:
        """Whether the coord is on the right AND above of other.

        Examples:
        >>> Coord(5, 5) > Coord(4, 5)
        True

        >>> Coord(4, 6) > Coord(4, 5)
        True

        >>> Coord(5, 4) > Coord(4, 5)
        False
        """
        if not isinstance(other, Coord):
            raise TypeError(f"Unsupported type: {type(other)}")

        # Except the `__eq__`
        return (
            (self.x > other.x and self.y > other.y)
            or (self.x == other.x and self.y > other.y)
            or (self.x > other.x and self.y == other.y)
        )

    def __le__(self, other: "Coord") -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: "Coord") -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coord({self.x}, {self.y})"

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple"""
        return (self.x, self.y)

    def _to_address(self) -> int:
        """Convert to address, 10 bits"""
        return (self.x << 5) | self.y

    @property
    def address(self) -> int:
        return self._to_address()


class DistanceType(Enum):
    DISTANCE_ENCLIDEAN = 0
    DISTANCE_MANHATTAN = 1
    DISTANCE_CHEBYSHEV = 2


@dataclass
class CoordOffset:
    """Offset of coordinates"""

    _COORDOFFSET_MAX_LIMIT = (1 << 5) - 1
    _COORDOFFSET_LOW_LIMIT = -(1 << 5)

    delta_x: int = Field(..., ge=_COORDOFFSET_LOW_LIMIT, le=_COORDOFFSET_MAX_LIMIT)
    delta_y: int = Field(..., ge=_COORDOFFSET_LOW_LIMIT, le=_COORDOFFSET_MAX_LIMIT)

    @overload
    def __add__(self, other: Coord) -> Coord:
        ...

    @overload
    def __add__(self, other: "CoordOffset") -> "CoordOffset":
        ...

    def __add__(
        self, other: Union["CoordOffset", Coord]
    ) -> Union["CoordOffset", Coord]:
        """
        Examples:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = delta_c1 + CoordOffset(1, 1)
        delta_c2: CoordOffset(2, 2)

        Coord = CoordOffset + Coord
        >>> delta_c = CoordOffset(1, 1)
        >>> c1 = Coord(2, 3)
        >>> c2 = delta_c + c1
        c2: Coord(3, 4)
        """
        if isinstance(other, CoordOffset):
            return CoordOffset(
                self.delta_x + other.delta_x, self.delta_y + other.delta_y
            )

        if isinstance(other, Coord):
            return Coord(self.delta_x + other.x, self.delta_y + other.y)

        raise TypeError(f"Unsupported type: {type(other)}")

    def __iadd__(self, other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c += CoordOffset(1, 1)
        delta_c: CoordOffset(2, 2)
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.delta_x += other.delta_x
        self.delta_y += other.delta_y

        return self

    def __sub__(self, other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = CoordOffset(2, 2)
        >>> delta_c = delta_c1 - delta_c2
        delta_c: CoordOffset(-1, -1)
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        return CoordOffset(self.delta_x - other.delta_x, self.delta_y - other.delta_y)

    def __isub__(self, other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c -= CoordOffset(1, 1)
        delta_c: CoordOffset(0, 0)
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.delta_x -= other.delta_x
        self.delta_y -= other.delta_y

        return self

    def __eq__(self, other: "CoordOffset") -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) == CoordOffset(4, 6)
        False
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        return self.delta_x == other.delta_x and self.delta_y == other.delta_y

    def __ne__(self, other: "CoordOffset") -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) != CoordOffset(4, 6)
        True
        """
        if not isinstance(other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(other)}")

        return self.delta_x != other.delta_x or self.delta_y != other.delta_y

    def distance(
        self, distance_type: DistanceType = DistanceType.DISTANCE_ENCLIDEAN
    ) -> Union[float, int]:
        """Distance between two coordinates."""
        if distance_type is DistanceType.DISTANCE_ENCLIDEAN:
            return self._euclidean_distance()
        elif distance_type is DistanceType.DISTANCE_MANHATTAN:
            return self._manhattan_distance()
        else:
            return self._chebyshev_distance()

    def _euclidean_distance(self) -> float:
        """Euclidean distance"""
        return sqrt((self.delta_x**2 + self.delta_y**2))

    def _manhattan_distance(self) -> int:
        """Manhattan distance"""
        return abs(self.delta_x) + abs(self.delta_y)

    def _chebyshev_distance(self) -> int:
        """Chebyshev distance"""
        return max(abs(self.delta_x), abs(self.delta_y))
