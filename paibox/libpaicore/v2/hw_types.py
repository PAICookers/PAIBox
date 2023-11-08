from typing import NamedTuple


class NeuronSegment(NamedTuple):
    index: slice
    """The original index of this segment of neurons."""
    addr_offset: int
    """The offset of the RAM address."""

    @property
    def addr_ram(self) -> slice:
        return slice(
            self.addr_offset, self.addr_offset + self.index.stop - self.index.start
        )


class AxonCoord(NamedTuple):
    tick_relative: int
    addr_axon: int


class AxonSegment(NamedTuple):
    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset+addr_width)."""

    # index: slice
    """The original index of this segment of axons."""

    addr_offset: int
    """The offset of the assigned address."""
