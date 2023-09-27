from enum import Enum, Flag, IntEnum, unique


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
