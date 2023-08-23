from typing import Optional

import numpy as np

from paibox._types import Shape
from paibox.base import PAIBoxObject
from paibox.utils import shape2num

from .neurons import MetaNeuron


class Group(PAIBoxObject):
    """A groups of neurons with the same parameters."""

    detectable = ("input", "output")

    def __init__(
        self,
        neurons_num: Shape,
        neuron_type: type[MetaNeuron],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:

        """
        super().__init__(name)
        self.shape = np.shape(neurons_num)
        self.neurons_num = shape2num(neurons_num)
        self.neuron_type = neuron_type

    @property
    def neurons(self):
        return Neurons(self)

    @property
    def shape_in(self):
        return self.shape

    @property
    def shape_out(self):
        return self.shape

    @property
    def num(self) -> int:
        return self.neurons_num

    def __len__(self) -> int:
        return self.neurons_num


class Neurons:
    def __init__(self, group: Group) -> None:
        self._group = group

    def __len__(self) -> int:
        return self.group.neurons_num

    def __str__(self) -> str:
        return f"<Neurons of {self.group}>"

    def __eq__(self, other: "Neurons") -> bool:
        return type(self) == type(other) and self.group is other.group

    @property
    def group(self):
        return self._group

    @property
    def shape_in(self):
        return self.group.neurons_num

    @property
    def shape_out(self):
        return self.group.neurons_num
