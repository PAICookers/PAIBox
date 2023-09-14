from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from paibox.base import NeuDyn, PAIBoxObject
from paibox.core.reg_types import LCNExtensionType as LCN_EX
from paibox.synapses import SynSys


def _get_core_limit(n_axon: int) -> Tuple[LCN_EX, int]:
    """Get the LCN extension limit & maximum number of neurons.
    Argument:
        - n_axon: the number of axons of the synapse.
    """
    assert n_axon > 0
    lcn_ex_max_in_core = LCN_EX(int((n_axon - 1) / 1152))
    if lcn_ex_max_in_core > LCN_EX.LCN_64X:
        # TODO
        raise ValueError
    n_max_in_core = 512 >> lcn_ex_max_in_core
    return lcn_ex_max_in_core, n_max_in_core


class GroupedSyn(PAIBoxObject):
    """Grouped synapse. A synapse will be split into core(s)."""

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
        self.lcn_ex = lcn_ex
        self.n_neuron_each = n_neuron_each

    @classmethod
    def build(cls, synapse: SynSys, name: Optional[str] = None) -> "GroupedSyn":
        n_axon = synapse.num_axon
        n_neuron = synapse.num_dentrite

        # Use the `n_axon` to limit the LCN extension
        l_limit, n_limit = _get_core_limit(n_axon)

        # At this point, the LCN extension is satisfied. Calculate the #N cores needed.
        n_core = int(n_neuron / n_limit) + 1
        # assert n_core == 1

        n_neuron_each = []
        for i in range(n_core):
            n_neuron_each.append(n_limit)

        n_neuron_each[-1] = n_neuron % n_limit

        return cls(synapse, n_core, l_limit, n_neuron_each, name)

    @property
    def obj(self) -> SynSys:
        return self.myself

    @property
    def dest(self) -> NeuDyn:
        return self.obj.dest

    @property
    def source(self):
        return self.obj.source


@dataclass
class GroupedLayer(PAIBoxObject):
    """Grouped layer. A layer will be grouped and mapped to core(s)."""

    def __init__(
        self,
        myself: NeuDyn,
        master_syns: List[GroupedSyn],
        target_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - myself: the layer itself.
            - master_syns: the grouped pre-synapses.
            - target_syns: the grouped post-synapses.
        """
        super().__init__(name)
        self.myself = myself
        self._master_syns = master_syns
        self._target_syns = target_syns

    @classmethod
    def build(
        cls,
        neurons: NeuDyn,
        grouped_syns: List[GroupedSyn],
        name: Optional[str] = None,
    ) -> "GroupedLayer":
        m, t = cls._find_grouped_syn(neurons, grouped_syns)

        return cls(neurons, m, t, name)

    @staticmethod
    def _find_grouped_syn(
        node: NeuDyn, grouped_syns: List[GroupedSyn]
    ) -> Tuple[List[GroupedSyn], List[GroupedSyn]]:
        """Find the master & target grouped synapses of the given node."""
        master = []
        target = []

        for syn in grouped_syns:
            if node.name == syn.dest.name:
                master.append(syn)

            if node.name == syn.source.name:
                target.append(syn)

        return master, target

    @property
    def obj(self) -> NeuDyn:
        return self.myself

    @property
    def source(self) -> List[GroupedSyn]:
        return self._master_syns

    @property
    def dest(self) -> List[GroupedSyn]:
        return self._target_syns
