from abc import ABC, abstractmethod
from typing import ClassVar, List, NamedTuple

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
        """Convert index of neuron into address RAM."""
        if (
            self.addr_offset + self.interval * (self.index.stop - self.index.start)
            > HwConfig.ADDR_RAM_MAX
        ):
            # TODO
            raise AttributeError

        return list(
            range(
                self.addr_offset,
                self.addr_offset + self.interval * (self.index.stop - self.index.start),
            )
        )


class AxonCoord(NamedTuple):
    tick_relative: int
    addr_axon: int


class AxonSegment(NamedTuple):
    n_axon: int
    """#N of axons."""
    addr_width: int
    """The range of axon address is [addr_offset, addr_offset+addr_width)."""
    addr_offset: int
    """The offset of the assigned address."""


class HwCore(ABC):
    """Hardware core abstraction."""

    mode: ClassVar[CoreMode]

    @property
    @abstractmethod
    def n_dendrite(self) -> int:
        """#N of valid dendrites"""
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
