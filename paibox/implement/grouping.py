from typing import List, Optional, Tuple, Type

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.libpaicore.v2 import LCN_EX
from paibox.synapses import SynSys


class GroupedObj(PAIBoxObject):
    @classmethod
    def build(cls):
        raise NotImplementedError

    @property
    def obj(self):
        raise NotImplementedError


class GroupedSyn(GroupedObj):
    """Grouped synapse. A synapse will be divided into core(s)."""

    def __init__(
        self,
        myself: SynSys,
        n_core: int,
        lcn_ex: LCN_EX,
        n_neuron_each: List[int],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - myself: the synapse itself.
            - n_core: the number of cores.
            - lcn_ex: the LCN extension type.
            - n_neuron_each: a list of number of neurons in each core. \
                `len(n_neuron_each)` == `n_core`.
        """
        super().__init__(name)
        self.myself = myself
        self.n_core = n_core
        self._lcn_ex = lcn_ex
        self.n_neuron_each = n_neuron_each

    @classmethod
    def build(cls, synapse: SynSys, name: Optional[str] = None) -> "GroupedSyn":
        """
        Description: always find the minimum LCN extension \
            that ALL the axons in this synapse satisfies.
        """
        n_axon_max = synapse.n_axon_each.max(axis=0)
        n_neuron = synapse.num_dentrite

        assert n_neuron > 0

        # Use the `n_axon` to limit the LCN extension
        l_limit, n_limit = cls._get_core_limit(n_axon_max, synapse.connectivity.dtype)

        # At this point, the LCN extension is satisfied. Calculate the #N cores needed.
        n_core = (n_neuron - 1) // n_limit + 1
        # assert n_core == 1

        n_neuron_each = []
        for i in range(n_core):
            n_neuron_each.append(n_limit)

        n_neuron_each[-1] = n_neuron % n_limit

        return cls(synapse, n_core, l_limit, n_neuron_each, name)

    @staticmethod
    def _get_core_limit(n_axon: int, _dt: Type[np.dtype]) -> Tuple[LCN_EX, int]:
        """Get the LCN extension limit & maximum number of neurons.
        Argument:
            - n_axon: the number of axons of the synapse.
        """
        if n_axon < 1:
            # TODO
            raise ValueError

        lcn_ex_max_in_core = LCN_EX((n_axon - 1) // 1152)
        if lcn_ex_max_in_core > LCN_EX.LCN_64X:
            # TODO
            raise ValueError

        n_core_combined = 1 if _dt == np.bool_ else 8

        n_max_in_core = (512 // n_core_combined) >> lcn_ex_max_in_core

        return lcn_ex_max_in_core, n_max_in_core

    @property
    def obj(self) -> SynSys:
        return self.myself

    @property
    def dest(self) -> NeuDyn:
        return self.obj.dest

    @property
    def source(self):
        return self.obj.source

    @property
    def lcn_ex(self) -> LCN_EX:
        return self._lcn_ex

    @lcn_ex.setter
    def lcn_ex(self, value: LCN_EX) -> None:
        if value >= LCN_EX.LCN_MAX:
            raise ValueError

        print(f"LCN of {self.name} is been updated: {self.lcn_ex} -> {value}")
        self._lcn_ex = value

    @property
    def weight_divided(self) -> List[np.ndarray]:
        """Divide the connectivity matrix of `myself` based on `n_neuron_each`.

        For a `LCN_2X` grouped synapse, the matrix [N*M] will be divided like:
            [N*(M1+M2)]
        where 1152 < N <= 1152*LCN_2X in total and M1, M2 <= 256.

        TODO However, the weight ram in CORE is 1152*512. So we need a further transform.
        """
        pos = []
        for i in range(1, self.n_core):
            pos.append(sum(self.n_neuron_each[:i]))

        return np.split(self.obj.connectivity, pos, axis=1)


class GroupedLayer(GroupedObj):
    """Grouped layer. A layer will be grouped and mapped to core(s)."""

    def __init__(
        self,
        myself: NeuDyn,
        pre_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - myself: the layer itself.
            - pre_syns: the grouped pre-synapses.
            - name: the name of the grouped layer. Optional.
        """
        super().__init__(name)
        self.myself = myself
        self._pre_syns = pre_syns

    @classmethod
    def build(
        cls,
        neurons: NeuDyn,
        grouped_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> "GroupedLayer":
        pre = cls._find_pre_syns(neurons, grouped_syns)

        return cls(neurons, pre, name)

    @staticmethod
    def _find_pre_syns(
        node: NeuDyn, grouped_syns: List[GroupedSyn]
    ) -> List[GroupedSyn]:
        """Find the previous grouped synapses of the given node."""
        pre = []

        for syn in grouped_syns:
            if node.name == syn.dest.name:
                # S -> N
                pre.append(syn)

        return pre

    @staticmethod
    def _find_bi_syns(
        node: NeuDyn, grouped_syns: List[GroupedSyn]
    ) -> Tuple[List[GroupedSyn], List[GroupedSyn]]:
        """Find bidirectional grouped synapses."""
        pre = []
        post = []

        for syn in grouped_syns:
            if node.name == syn.dest.name:
                # S -> N
                pre.append(syn)

            if node.name == syn.source.name:
                # N -> S
                post.append(syn)

        return pre, post

    @property
    def obj(self) -> NeuDyn:
        return self.myself

    @property
    def pre(self) -> List[GroupedSyn]:
        return self._pre_syns

    @property
    def source(self) -> List[GroupedSyn]:
        return self.pre

    @property
    def n_axon(self) -> int:
        # Get a list of axons in the previous synapses.
        axons = [syn.myself.num_axon for syn in self.pre]
        return sum(axons)

    @property
    def n_neuron(self) -> int:
        return self.myself.num_in
