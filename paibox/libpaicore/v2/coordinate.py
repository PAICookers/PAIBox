from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Sequence, Tuple, TypeVar, Union, final, overload

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .hw_defs import HwConfig

__all__ = [
    "Coord",
    "CoordOffset",
    "ReplicationId",
    "CoordLike",
    "RIdLike",
    "to_coord",
    "to_coordoffset",
    "to_rid",
]


class Identifier(ABC):
    """Identifier. The subclass of identifier must implement `__eq__` & `__ne__`."""

    @abstractmethod
    def __eq__(self, __other) -> ...:
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, __other) -> ...:
        raise NotImplementedError


@dataclass
class Coord(Identifier):
    """Coordinates of the cores. Set coordinates (x, y) for every core.

    Left to right, +X, up to down, +Y.
    """

    x: int = Field(ge=HwConfig.CORE_X_MIN, le=HwConfig.CORE_X_MAX, frozen=True)
    y: int = Field(ge=HwConfig.CORE_Y_MIN, le=HwConfig.CORE_Y_MAX, frozen=True)

    @classmethod
    def from_tuple(cls, pos):
        return cls(*pos)

    @classmethod
    def from_addr(cls, addr: int):
        return cls(addr >> HwConfig.N_BIT_CORE_X, addr & HwConfig.CORE_Y_MAX)

    @classmethod
    def default(cls):
        return cls(HwConfig.CORE_X_MIN, HwConfig.CORE_Y_MIN)

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

        return self.to_tuple() == __other.to_tuple()

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

        return self.to_tuple() != __other.to_tuple()

    # def __lt__(self, __other: "Coord") -> bool:
    #     """Whether the coord is on the left OR below of __other.

    #     Examples:
    #     >>> Coord(4, 5) < Coord(4, 6)
    #     True

    #     >>> Coord(4, 5) < Coord(5, 5)
    #     True

    #     >>> Coord(4, 5) < Coord(5, 3)
    #     True
    #     """
    #     if not isinstance(__other, Coord):
    #         raise TypeError(f"Unsupported type: {type(__other)}")

    #     return self.x < __other.x or self.y < __other.y

    # def __gt__(self, __other: "Coord") -> bool:
    #     """Whether the coord is on the right AND above of __other.

    #     Examples:
    #     >>> Coord(5, 5) > Coord(4, 5)
    #     True

    #     >>> Coord(4, 6) > Coord(4, 5)
    #     True

    #     >>> Coord(5, 4) > Coord(4, 5)
    #     False
    #     """
    #     if not isinstance(__other, Coord):
    #         raise TypeError(f"Unsupported type: {type(__other)}")

    #     # Except the `__eq__`
    #     return (
    #         (self.x > __other.x and self.y > __other.y)
    #         or (self.x == __other.x and self.y > __other.y)
    #         or (self.x > __other.x and self.y == __other.y)
    #     )

    # def __le__(self, __other: "Coord") -> bool:
    #     return self.__lt__(__other) or self.__eq__(__other)

    # def __ge__(self, __other: "Coord") -> bool:
    #     return self.__gt__(__other) or self.__eq__(__other)

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
    DISTANCE_ENCLIDEAN = auto()
    DISTANCE_MANHATTAN = auto()
    DISTANCE_CHEBYSHEV = auto()


@dataclass
class CoordOffset:
    """Offset of coordinates"""

    delta_x: int = Field(
        ge=-HwConfig.CORE_X_MAX - 1, le=HwConfig.CORE_X_MAX, frozen=True
    )
    delta_y: int = Field(
        ge=-HwConfig.CORE_Y_MAX - 1, le=HwConfig.CORE_Y_MAX, frozen=True
    )

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

        return (self.delta_x, self.delta_y) == (__other.delta_x, __other.delta_y)

    def __ne__(self, __other: "CoordOffset") -> bool:
        """
        Example:
        >>> CoordOffset(4, 5) != CoordOffset(4, 6)
        True
        """
        if not isinstance(__other, CoordOffset):
            raise TypeError(f"Unsupported type: {type(__other)}")

        return (self.delta_x, self.delta_y) != (__other.delta_x, __other.delta_y)

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
RIdLike = TypeVar("RIdLike", ReplicationId, int, List[int], Tuple[int, int])


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


def to_coordoffset(offset: int) -> CoordOffset:
    return CoordOffset(
        offset % (HwConfig.CORE_X_MAX + 1), offset // (HwConfig.CORE_Y_MAX + 1)
    )


def to_rid(ridlike: RIdLike) -> ReplicationId:
    if isinstance(ridlike, int):
        return ReplicationId.from_addr(ridlike)

    if isinstance(ridlike, (list, tuple)):
        if len(ridlike) != 2:
            raise ValueError(
                f"Must be a tuple or list of 2 elements to represent a replication ID: {len(ridlike)}"
            )

        return ReplicationId(*ridlike)

    return ridlike
