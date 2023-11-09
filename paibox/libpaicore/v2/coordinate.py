from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Sequence, Tuple, TypeVar, Union, final, overload

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass


class Identifier(ABC):
    """Identifier. At least the subclasses of identifier can `__eq__` and `__ne__`."""

    @abstractmethod
    def __eq__(self, __other) -> ...:
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, __other) -> ...:
        raise NotImplementedError


_COORD_MAX_LIMIT = (1 << 5) - 1
_COORD_LOW_LIMIT = 0


@dataclass
class Coord(Identifier):
    """Coordinates of the cores. Set coordinates (x, y) for every cores.

    Left to right, +X, up to down, +Y.
    """

    x: int = Field(..., ge=_COORD_LOW_LIMIT, le=_COORD_MAX_LIMIT)
    y: int = Field(..., ge=_COORD_LOW_LIMIT, le=_COORD_MAX_LIMIT)

    @classmethod
    def from_tuple(cls, pos):
        return cls(*pos)

    @classmethod
    def from_addr(cls, addr: int):
        return cls(addr >> 5, addr & _COORD_MAX_LIMIT)

    @classmethod
    def default(cls):
        return cls(0, 0)

    def __add__(self, __other: "CoordOffset") -> "Coord":
        """
        Examples:

        Coord = Coord + CoordOffset
        >>> c1 = Coord(1, 1)
        >>> c2 = c1 + CoordOffset(1, 1)
        c1: Coord(2, 2)

        NOTE: Coord + Coord is meaningless.
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return Coord(self.x + __other.delta_x, self.y + __other.delta_y)

    def __iadd__(self, __other: Union[int, "CoordOffset"]) -> "Coord":
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c1 += CoordOffset(1, 1)
        Coord(2, 2)
        """
        if not isinstance(__other, (int, CoordOffset)):
            raise TypeError(f"Unsupported type: {type(__other)}")

        _dx = __other if isinstance(__other, int) else __other.delta_x
        _dy = __other if isinstance(__other, int) else __other.delta_y

        self.x += _dx
        self.y += _dy

        return self

    @overload
    def __sub__(self, __other: "Coord") -> "CoordOffset":
        ...

    @overload
    def __sub__(self, __other: "CoordOffset") -> "Coord":
        ...

    def __sub__(
        self, __other: Union["Coord", "CoordOffset"]
    ) -> Union["Coord", "CoordOffset"]:
        """
        Example:
        >>> c1 = Coord(1, 1)
        >>> c2 = Coord(2, 2) - c1
        c2: CoordOffset(1, 1)
        """
        if isinstance(__other, Coord):
            return CoordOffset(self.x - __other.x, self.y - __other.y)

        if isinstance(__other, CoordOffset):
            return Coord(self.x - __other.delta_x, self.y - __other.delta_y)

        raise TypeError(f"Unsupported type: {type(__other)}")

    def __isub__(self, __other: Union[int, "CoordOffset"]) -> "Coord":
        """
        Example:
        >>> c1 = Coord(2, 2)
        >>> c1 -= CoordOffset(1, 1)
        Coord(1, 1)
        """
        if not isinstance(__other, (int, CoordOffset)):
            raise TypeError(f"Unsupported type: {type(__other)}")

        _dx = __other if isinstance(__other, int) else __other.delta_x
        _dy = __other if isinstance(__other, int) else __other.delta_y

        self.x -= _dx
        self.y -= _dy

        return self

    """Operations below are used only when comparing with a Cooord."""

    def __eq__(self, __other: "Coord") -> bool:
        """
        Example:
        >>> Coord(4, 5) == Coord(4, 6)
        False
        """
        if not isinstance(__other, Coord):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return self.x == __other.x and self.y == __other.y

    def __ne__(self, __other: "Coord") -> bool:
        """
        Examples:
        >>> Coord(4, 5) != Coord(4, 6)
        True

        >>> Coord(4, 5) != Coord(5, 5)
        True
        """
        if not isinstance(__other, Coord):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return self.x != __other.x or self.y != __other.y

    def __lt__(self, __other: "Coord") -> bool:
        """Whether the coord is on the left OR below of __other.

        Examples:
        >>> Coord(4, 5) < Coord(4, 6)
        True

        >>> Coord(4, 5) < Coord(5, 5)
        True

        >>> Coord(4, 5) < Coord(5, 3)
        True
        """
        if not isinstance(__other, Coord):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return self.x < __other.x or self.y < __other.y

    def __gt__(self, __other: "Coord") -> bool:
        """Whether the coord is on the right AND above of __other.

        Examples:
        >>> Coord(5, 5) > Coord(4, 5)
        True

        >>> Coord(4, 6) > Coord(4, 5)
        True

        >>> Coord(5, 4) > Coord(4, 5)
        False
        """
        if not isinstance(__other, Coord):
            raise TypeError(f"Unsupported type: {type(__other)}")

        # Except the `__eq__`
        return (
            (self.x > __other.x and self.y > __other.y)
            or (self.x == __other.x and self.y > __other.y)
            or (self.x > __other.x and self.y == __other.y)
        )

    def __le__(self, __other: "Coord") -> bool:
        return self.__lt__(__other) or self.__eq__(__other)

    def __ge__(self, __other: "Coord") -> bool:
        return self.__gt__(__other) or self.__eq__(__other)

    def __xor__(self, __other: "Coord") -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    def __hash__(self) -> int:
        return hash(self.address)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coord({self.x}, {self.y})"

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple"""
        return (self.x, self.y)

    @property
    def address(self) -> int:
        """Convert to address, 10 bits"""
        return (self.x << 5) | self.y


@final
class ReplicationId(Coord):
    def __and__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x & __other.x, self.y & __other.y)

    def __or__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x | __other.x, self.y | __other.y)

    def __xor__(self, __other: Union[Coord, "ReplicationId"]) -> "ReplicationId":
        return ReplicationId(self.x ^ __other.x, self.y ^ __other.y)

    # def __lshift__(self, __bit: int) -> int:
    #     return self.address << __bit

    # def __rshift__(self, __bit: int) -> int:
    #     return self.address >> __bit


class DistanceType(Enum):
    DISTANCE_ENCLIDEAN = 0
    DISTANCE_MANHATTAN = 1
    DISTANCE_CHEBYSHEV = 2


_COORDOFFSET_MAX_LIMIT = (1 << 5) - 1
_COORDOFFSET_LOW_LIMIT = -(1 << 5)


@dataclass
class CoordOffset:
    """Offset of coordinates"""

    delta_x: int = Field(..., ge=_COORDOFFSET_LOW_LIMIT, le=_COORDOFFSET_MAX_LIMIT)
    delta_y: int = Field(..., ge=_COORDOFFSET_LOW_LIMIT, le=_COORDOFFSET_MAX_LIMIT)

    @classmethod
    def from_tuple(cls, pos) -> "CoordOffset":
        return cls(*pos)

    @classmethod
    def default(cls) -> "CoordOffset":
        return cls(0, 0)

    @overload
    def __add__(self, __other: Coord) -> Coord:
        ...

    @overload
    def __add__(self, __other: "CoordOffset") -> "CoordOffset":
        ...

    def __add__(
        self, __other: Union["CoordOffset", Coord]
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
        if isinstance(__other, CoordOffset):
            return CoordOffset(
                self.delta_x + __other.delta_x, self.delta_y + __other.delta_y
            )

        if isinstance(__other, Coord):
            return Coord(self.delta_x + __other.x, self.delta_y + __other.y)

        raise TypeError(f"Unsupported type: {type(__other)}")

    def __iadd__(self, __other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c += CoordOffset(1, 1)
        delta_c: CoordOffset(2, 2)
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        self.delta_x += __other.delta_x
        self.delta_y += __other.delta_y

        return self

    def __sub__(self, __other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c1 = CoordOffset(1, 1)
        >>> delta_c2 = CoordOffset(2, 2)
        >>> delta_c = delta_c1 - delta_c2
        delta_c: CoordOffset(-1, -1)
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return CoordOffset(
            self.delta_x - __other.delta_x, self.delta_y - __other.delta_y
        )

    def __isub__(self, __other: "CoordOffset") -> "CoordOffset":
        """
        Example:
        >>> delta_c = CoordOffset(1, 1)
        >>> delta_c -= CoordOffset(1, 1)
        delta_c: CoordOffset(0, 0)
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        self.delta_x -= __other.delta_x
        self.delta_y -= __other.delta_y

        return self

    def __eq__(self, __other: "CoordOffset") -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) == CoordOffset(4, 6)
        False
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return self.delta_x == __other.delta_x and self.delta_y == __other.delta_y

    def __ne__(self, __other: "CoordOffset") -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) != CoordOffset(4, 6)
        True
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return self.delta_x != __other.delta_x or self.delta_y != __other.delta_y

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
        return np.sqrt(self.delta_x**2 + self.delta_y**2)

    def _manhattan_distance(self) -> int:
        """Manhattan distance"""
        return np.abs(self.delta_x) + np.abs(self.delta_y)

    def _chebyshev_distance(self) -> int:
        """Chebyshev distance"""
        return np.maximum(np.abs(self.delta_x), np.abs(self.delta_y))


CoordLike = TypeVar("CoordLike", Coord, int, List[int], Tuple[int, int])


def to_coord(coordlike: CoordLike) -> Coord:
    if isinstance(coordlike, int):
        return Coord.from_addr(coordlike)

    if isinstance(coordlike, (list, tuple)):
        if len(coordlike) != 2:
            raise ValueError(
                f"Must be a tuple or list of 2 elements to represent a coordinate: {len(coordlike)}"
            )

        return Coord(*coordlike)

    return coordlike


def to_coords(coordlikes: Sequence[CoordLike]) -> List[Coord]:
    return [to_coord(coordlike) for coordlike in coordlikes]
