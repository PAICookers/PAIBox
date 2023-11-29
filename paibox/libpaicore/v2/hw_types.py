from abc import ABC, abstractmethod
from typing import ClassVar, List, NamedTuple, Tuple

from paibox.exceptions import ResourceError

from .hw_defs import HwConfig
from .reg_types import CoreMode

__all__ = ["NeuronSegment", "AxonCoord", "AxonSegment", "HwCore"]


class NeuronSegment(NamedTuple):
    """Segment of neuron.

    Example:
        index = (0, 100, 1)
        addr_offset = 10
        interval = 2
        addr_ram = (10, 11, ..., 210)

        for addr in addr_ram:
            write in addr with tick_relative and addr_axon at 'addr/interval'.
    """

    index: slice
    """The original index of this segment of neurons."""
    addr_offset: int
    """The offset of the RAM address."""
    interval: int = 1
    """The interval of address when mapping neuron attributes on the RAM."""
    weight_steps: int = 1
    """Reserved."""

    @property
    def n_neuron(self) -> int:
        return self.index.stop - self.index.start

    @property
    def addr_ram(self) -> List[int]:
        """Convert index of neuron into address of RAM."""
        if (
            _addr_max := self.addr_offset
            + self.interval * (self.index.stop - self.index.start)
        ) > HwConfig.ADDR_RAM_MAX:
            raise ResourceError(
                f"Address of RAM out of {HwConfig.ADDR_RAM_MAX}: {_addr_max}"
            )

        return list(range(self.addr_offset, _addr_max))

    @property
    def addr_slice(self) -> slice:
        """Display the address of RAM in slice format."""
        return slice(
            self.addr_offset + self.index.start,
            self.addr_offset + self.index.stop,
            self.interval,
        )


class AxonCoord(NamedTuple):
    tick_relative: int
    addr_axon: int


class AxonSegment(NamedTuple):
    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset + addr_width)."""
    addr_offset: int
    """The offset of the assigned address."""


class HwCore(ABC):
    """Hardware core abstraction."""

    mode: ClassVar[CoreMode]

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of the core."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_dendrite(self) -> int:
        """#N of valid dendrites."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_core_required(self) -> int:
        """#N of cores required to accommodate neurons inside self."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls):
        ...
