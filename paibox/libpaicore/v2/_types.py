from enum import Enum, Flag, IntEnum, unique
from typing import Literal


@unique
class RouterOp(Enum):
    UP = 0
    DOWN_UNICAST = 1
    DOWN_MULTICAST = 2


@unique
class RouterLevel(IntEnum):
    L0 = 0  # Leaves for storing the data. A L0-layer is a core.
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


@unique
class ReplicationFlag(Flag):
    NONE = 0
    L1 = 1
    L2 = 2
    L3 = 4
    L4 = 8
    L5 = 16


@unique
class RouterDirection(Enum):
    X0Y0 = (0, 0)
    X0Y1 = (0, 1)
    X1Y0 = (1, 0)
    X1Y1 = (1, 1)

    def to_index(self, method: Literal["X", "Y"] = "Y") -> int:
        x, y = self.value

        if method == "Y":
            return x << 1 + y
        else:
            return y << 1 + x
