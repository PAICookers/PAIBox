from typing import List, NamedTuple

from paibox.exceptions import PAICoreError

from .hw_defs import HwConfig

__all__ = ["NeuronSegment", "AxonCoord", "AxonSegment"]


class NeuronSegment(NamedTuple):
    """Segment of neuron.

    index =(0, 100, 1)
    addr_offset = 0
    interval = 2

    The address of RAM will be: slice(0, 200, 2)
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
    def nrange(self) -> slice:
        if (
            self.addr_offset + self.interval * (self.index.stop - self.index.start)
            > HwConfig.N_NEURON_ONE_CORE_MAX
        ):
            # TODO
            raise PAICoreError(
                f"Out of address, address limit is 512, "
                f"but we got {self.addr_offset + self.interval * (self.index.stop - self.index.start)}"
            )

        return slice(
            self.addr_offset,
            self.addr_offset + (self.index.stop - self.index.start),
            1,
        )

    @property
    def addr_ram(self) -> List[int]:
        """Convert index of neuron into address RAM."""
        addr_ram = []

        for addr in range(HwConfig.N_NEURON_ONE_CORE_MAX)[self.nrange]:
            addr_ram.extend([addr] * self.interval)

        return addr_ram


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
