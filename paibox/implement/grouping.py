from typing import List, Optional

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.libpaicore.v2 import LCN_EX, Coord
from paibox.synapses import SynSys


def axon2lcn_ex(n_axon: int) -> LCN_EX:
    if n_axon < 1:
        # TODO
        raise ValueError

    lcn_ex = LCN_EX((n_axon - 1) // 1152)
    if lcn_ex > LCN_EX.LCN_64X:
        # TODO
        raise ValueError

    return lcn_ex


class GroupedObj(PAIBoxObject):
    @classmethod
    def build(cls):
        raise NotImplementedError

    @property
    def obj(self):
        raise NotImplementedError


class GroupedSyn(GroupedObj):
    """Grouped synapse. A synapse will be divided into core(s).

    All the synapses will be grouped first. Then we get a list of `GroupedSyn`.
    """

    coords: List[Coord] = []

    def __init__(
        self,
        parent: List[SynSys],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent synapses.
        """
        super().__init__(name)
        self._parent = parent

        self._lcn_ex = axon2lcn_ex(self.n_axons)
        self._get_limit_info()

    @classmethod
    def build(cls, synapses: List[SynSys], name: Optional[str] = None) -> "GroupedSyn":
        """Build the `GroupedSyn` for a node.

        Use LCN extension optimization in grouping a synapse.

        Description: always find the minimum LCN extension \
            that ALL the axons in this synapse satisfies.

            For two pre-synapses, S1 [A1*M] & S2 [A2*M], combine then split.

            The total number of axons = A1+A2 -> LCN -> n_neuron.

            Now consider S1 & S2 are 1-bit weights.
            TODO If weights precision of S1 & S2 differ? Place on different cores.
        """
        if not cls.all_dtype_equal(synapses):
            raise NotImplementedError

        assert synapses[0].connectivity.dtype == np.bool_

        return cls(synapses, name)

    @staticmethod
    def all_dtype_equal(syns: List[SynSys]) -> bool:
        _dtype = syns[0].connectivity.dtype

        for syn in syns:
            if _dtype != syn.connectivity.dtype:
                return False

        return True

    def set_lcn_ex(self, lcn_ex: LCN_EX) -> None:
        if lcn_ex < self.lcn_ex or lcn_ex > LCN_EX.LCN_64X:
            # TODO
            raise ValueError

        self.lcn_ex = lcn_ex
        self._get_limit_info()

    def _get_limit_info(self) -> None:
        """Limit the grouped synapses given a new LCN extension.

        Re limit `lcn_ex`, `_n_core`, & `n_neuron_each`.
        """
        n_neuron_per_core = (512 // self.n_dendrite_per_neuron) >> self.lcn_ex

        n_core = (self.n_neuron - 1) // n_neuron_per_core + 1
        n_neuron_each = []

        for _ in range(n_core):
            n_neuron_each.append(n_neuron_per_core)

        n_neuron_each[-1] = self.n_neuron % n_neuron_per_core

        self._n_core = n_core
        self._n_neuron_each = n_neuron_each

    @property
    def obj(self) -> List[SynSys]:
        return self._parent

    @property
    def source(self):
        return [parent.source for parent in self.obj]

    @property
    def dest(self) -> List[NeuDyn]:
        return [parent.dest for parent in self.obj]

    @property
    def n_axons(self) -> int:
        return sum([parent.num_axon for parent in self.obj])

    @property
    def n_neuron(self) -> int:
        return self.obj[0].num_out

    @property
    def n_core(self) -> int:
        return self._n_core

    @property
    def n_neuron_each(self) -> List[int]:
        return self._n_neuron_each

    @property
    def n_dendrite_per_neuron(self) -> int:
        # TODO Now consider all the pre-synapses are 1-bit weights.
        return 1

    @property
    def lcn_ex(self) -> LCN_EX:
        return self._lcn_ex

    @lcn_ex.setter
    def lcn_ex(self, lcn_ex: LCN_EX) -> None:
        if lcn_ex >= LCN_EX.LCN_MAX:
            raise ValueError

        print(f"LCN of {self.name} is been updated: {self.lcn_ex} -> {lcn_ex}")
        self._lcn_ex = lcn_ex

    @property
    def weight_combined(self) -> np.ndarray:
        """Combine all the matrices in one piece."""
        return np.concatenate([syn.connectivity for syn in self.obj], axis=0)

    @property
    def weight_divided(self) -> List[np.ndarray]:
        """Divide the connectivity matrix based on `n_neuron_each`.
        Return a list of weights divided in cores.
        """
        pos = []
        for i in range(1, self.n_core):
            pos.append(sum(self.n_neuron_each[:i]))

        return np.split(self.weight_combined, pos, axis=1)

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"


def _syns_max_lcn_ex(syns: List[GroupedSyn]) -> LCN_EX:
    """Find the max LCN extenion of grouped post-synapses"""
    return max(syns, key=lambda syn: syn.lcn_ex).lcn_ex


class GroupedSynOnCore(GroupedObj):
    """The divided synapse placed on a single CORE.

    (Parent, Axons, Neurons, binary_conn):
        (S1, start-to-end <1>, start-to-end <1>, binary_conn <1>)
    e.g.:
    D1: (SynSys1, 0-500,     0-100, [500*100, np.bool_])
    D2: (SynSys2, 500-1000,100-500, [500*400, np.bool_])

    A = 1152*LCN_EX(1)
     _________     _________
    |         |   |         |
    |   S1-1  |   |   S1-2  |
    |_________|   |_________|
    |         | + |         |
    |   S2-1  |   |   S2-2  |
    |_________|   |_________|
    |_________|   |_________|
        100           400
    """

    n_core: int = 1

    def __init__(
        self,
        parent: SynSys,
        lcn_ex: LCN_EX,
        n_neuron: int,
        n_axon: int,
        weights: np.ndarray,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent synapse(complete).
            - lcn_ex: the LCN extension type.
            - n_neuron: the number of neurons used in the CORE.
            - n_axon: the number of axons used in the CORE.
            - weights: the weights divided into the CORE.
            - coord: the coordinate assigning to the CORE.
        """
        super().__init__(name)

        self._parent = parent
        self._lcn_ex = lcn_ex
        self._n_neuron = n_neuron
        self._n_axon = n_axon
        self._binary_conn = self._get_binary_conn(weights)

    def _get_binary_conn(self, weights) -> np.ndarray:
        """Get the binary connection based on the divided weights matrix.

        TODO ONLY for `np.bool_` now.
        """
        assert self._n_neuron * self._lcn_ex * 1 <= 512

        o_matrix = np.zeros((1152, 512), dtype=np.bool_)

        for i in range(self._n_neuron):
            o_matrix[:, 2 * i] = weights[:1152, i]
            o_matrix[:, 2 * i + 1] = np.pad(
                weights[1152:, i],
                (0, 2 * 1152 - self._n_axon),
                "constant",
                constant_values=0,
            )

        return o_matrix.astype(np.bool_)

    @property
    def obj(self) -> SynSys:
        return self._parent

    @property
    def source(self):
        return self.obj.source

    @property
    def dest(self) -> NeuDyn:
        return self.obj.dest

    @property
    def crossbar(self) -> np.ndarray:
        return self._binary_conn
