from typing import Optional, Tuple, Union


class Coord:
    """Coordinate class"""

    def __init__(
        self, _x: Union[Tuple[int, int], int], _y: Optional[int] = None
    ) -> None:
        if isinstance(_x, Tuple):
            x, y = _x[0], _x[1]
            if isinstance(_y, int):
                raise ValueError(f"Wrong Argument: {_y}")
        elif isinstance(_x, int):
            if isinstance(_y, int):
                x, y = _x, _y
            else:
                raise ValueError("Missing Argument: y")
        else:
            raise ValueError("Wrong Argument")

        if not (0 <= x < 32 and 0 <= y < 32):
            raise ValueError(f"0 <= x < 32, 0 <= y < 32: ({x}, {y})")

        self.x, self.y = x, y

    def __add__(self, other) -> "Coord":
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "CoordOffset":
        return CoordOffset(self.x - other.x, self.y - other.y)

    def __eq__(self, other) -> bool:
        if isinstance(other, Tuple):
            return (self.x, self.y) == other

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


class CoordOffset(Coord):
    """Coordinate offset class"""

    def __init__(self, _x: int, _y: int) -> None:
        if not (-32 < _x < 32 and -32 < _y < 32):
            raise ValueError(f"-32 < x < 32, -32 < y < 32: ({_x}, {_y})")

        self.x, self.y = _x, _y

    def __add__(self, other):
        if isinstance(other, CoordOffset):
            return CoordOffset(self.x + other.x, self.y + other.y)
        else:
            return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "CoordOffset":
        if isinstance(other, Coord):
            raise TypeError("A CoordOffset cannot substract a Coord")

        return CoordOffset(self.x - other.x, self.y - other.y)

