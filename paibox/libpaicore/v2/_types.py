from enum import auto, Enum, Flag, IntEnum, unique


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
    """Indicate the 4 children of a node.
    
    NOTE: There is an X/Y coordinate priority method \
        to specify the order of the 4 children.
    """

    X0Y0 = (0, 0)
    X0Y1 = (0, 1)
    X1Y0 = (1, 0)
    X1Y1 = (1, 1)
    ANY = (-1, -1)  # Don't care when a level direction is `ANY`.

    def to_index(self) -> int:
        """Convert the direction to index in children list."""
        if self is RouterDirection.ANY:
            # TODO
            raise ValueError

        x, y = self.value

        return (x << 1) + y
