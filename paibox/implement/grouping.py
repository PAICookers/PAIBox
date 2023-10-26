from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.libpaicore.v2 import LCN_EX, Coord, HwConfig, RoutingNodeCoord
from paibox.projection import InputProj
from paibox.synapses import SynSys

SourceNodeType = Union[NeuDyn, InputProj]
DestNodeType = NeuDyn


def n_axon2lcn_ex(n_axon: int) -> LCN_EX:
    """Convert #N(of axons) to LCN_EX.

    Description:
        LCN_EX = log2[ceil(#N/1152)]

    where, LCN_EX = 0 is `LCN_1X`.
    """
    if n_axon < 1:
        # TODO
        raise ValueError

    lcn_ex = LCN_EX(((n_axon - 1) // HwConfig.N_AXON_DEFAULT).bit_length())
    if lcn_ex >= LCN_EX.LCN_MAX:
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

    def __init__(
        self,
        *parent: SynSys,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent synapses.
        """
        super().__init__(name)
        self._parent = parent

        self._n_axon_each = self._get_n_axon_each()
        self._lcn_ex = n_axon2lcn_ex(self.n_axon)
        self._resource_consumption()

    def _get_n_axon_each(self) -> List[int]:
        """Get the axons of each unique parents."""
        seen = set()
        n_axon_unique = []

        for src_node in self.source:
            if src_node not in seen:
                seen.add(src_node)
                n_axon_unique.append(src_node.num_out)

        return n_axon_unique

    @classmethod
    def build(cls, *synapses: SynSys, name: Optional[str] = None):
        """Build the `GroupedSyn` for a node.

        Use LCN extension optimization in grouping a synapse.

        Description: always find the minimum LCN extension \
            that ALL the axons in this synapse satisfies.

            For two pre-synapses, S1 [A1*M] & S2 [A2*M], combine then split.

            The total number of axons = A1+A2 -> LCN -> n_neuron.

            Now consider S1 & S2 are 1-bit weights.
            TODO If weights precision of S1 & S2 differ? Place on different cores.
        """
        if not cls.all_dtype_equal(*synapses):
            raise NotImplementedError

        assert synapses[0].connectivity.dtype == np.bool_

        return cls(*synapses, name=name)

    @staticmethod
    def all_dtype_equal(*syns: SynSys) -> bool:
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
        self._resource_consumption()

    def _resource_consumption(self) -> None:
        """Limit the grouped synapses by giving a new LCN extension.

        Re limit `lcn_ex`, `_n_core`, & `n_neuron_each`.
        """
        n_neuron_per_core = (
            HwConfig.N_NEURON_DEFAULT // self.n_dendrite_per_neuron
        ) >> self.lcn_ex

        n_core = (self.n_neuron - 1) // n_neuron_per_core + 1
        n_neuron_each = []

        for _ in range(n_core):
            n_neuron_each.append(n_neuron_per_core)

        n_neuron_each[-1] = self.n_neuron % n_neuron_per_core

        self._n_core = n_core
        self._n_neuron_each = n_neuron_each

    def build_syn_on_core(self) -> List["GroupedSynOnCore"]:
        syn_on_core = []

        for i in range(self.n_core):
            syn_on_core.append(GroupedSynOnCore.build(self, i))

        return syn_on_core

    @property
    def obj(self) -> Tuple[SynSys, ...]:
        return self._parent

    @property
    def source(self) -> List[SourceNodeType]:
        """Ordered unique source nodes.

        TODO Maybe consider to return `OrderedSet`.
        """
        # return OrderedSet([parent.source for parent in self.obj])
        return list(set([parent.source for parent in self.obj]))

    @property
    def dest(self) -> List[DestNodeType]:
        """Ordered unique destination nodes."""
        # return OrderedSet(set([parent.dest for parent in self.obj]))
        return list(set([parent.dest for parent in self.obj]))

    @property
    def n_axon_each(self) -> List[int]:
        return self._n_axon_each

    @property
    def n_axon(self) -> int:
        return sum(self.n_axon_each)

    @property
    def n_neuron(self) -> int:
        """Accumulate the `num_in` for all unique destination nodes."""
        return sum([node.num_in for node in self.dest])

    @property
    def n_core(self) -> int:
        return self._n_core

    @property
    def n_neuron_each(self) -> List[int]:
        return self._n_neuron_each

    @property
    def n_dendrite_per_neuron(self) -> int:
        # TODO Now consider all the pre-synapses are 1-bit weights.
        return 1 << self.lcn_ex

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
    def weight_divided(self) -> List[np.ndarray]:
        """Divide the combined weights based on `n_neuron_each`.

        Return:
            - a list of weights divided in cores.
        """
        pos = []
        for i in range(1, self.n_core):
            pos.append(sum(self.n_neuron_each[:i]))

        return np.split(self._get_weight_combined(), pos, axis=1)

    def _get_syn(self, src: SourceNodeType, dest: DestNodeType) -> Optional[SynSys]:
        for syn in self.obj:
            if syn.source == src and syn.dest == dest:
                return syn

        return None

    def _get_weight_combined(self) -> np.ndarray:
        """Combine all the matrices in one piece.

        Combined weight:
            [s1, d1], ..., [s1, dn]
            ...
            [sn, d1], ..., [sn, dn]
        """
        w = []

        for d in self.dest:
            ws = []
            for s in self.source:
                if syn := self._get_syn(s, d):
                    ws.append(syn.connectivity)
                else:
                    # Fill with 0.
                    ws.append(np.zeros((s.num_out, d.num_in), dtype=np.bool_))

            w.append(np.concatenate(ws, axis=0))

        wc = np.concatenate(w, axis=1)
        wc.setflags(write=False)

        # Reserved for debug.
        self.weight_combined = wc

        return wc

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
    D1: (SynSys1, 0-500,     0-400, [500*400, np.bool_])
    D2: (SynSys2, 500-1000,400-500, [500*100, np.bool_])

    A = 1152*LCN_EX(1)
     _________     _________
    |         |   |         |
    |   S1-1  |   |   S1-2  |
    |_________|   |_________|
    |         | + |         |
    |   S2-1  |   |   S2-2  |
    |_________|   |_________|
    |_________|   |_________|
        400           100
    """

    n_core: ClassVar[int] = 1

    def __init__(
        self,
        parent: GroupedSyn,
        position: int,
        n_neuron: int,
        weights: np.ndarray,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent grouped synapse(complete).
            - position: the position of the targeted `GroupedSynOnCore` \
                in the parent.
            - n_neuron: the number of neurons used in the CORE.
            - weights: the weights divided into the single CORE.
        """
        super().__init__(name)

        self._parent = parent
        self._pos = position
        self._n_neuron = n_neuron
        self._routing_coord = RoutingNodeCoord()

        self._check()

        self._binary_conn = self._get_binary_conn(weights)
        # self._need_broadcast = need_broadcast

    def _check(self) -> None:
        assert (
            self.n_neuron * self.obj.n_dendrite_per_neuron * 1
            <= HwConfig.N_NEURON_DEFAULT
        )

    def _get_binary_conn(self, weights: np.ndarray) -> np.ndarray:
        """Reshape the divided weight into the binary connection."""
        bc_shape = (HwConfig.N_AXON_DEFAULT, HwConfig.N_NEURON_DEFAULT)
        bc = np.zeros(bc_shape, dtype=np.bool_)
        n_col_groups = self.obj.n_dendrite_per_neuron

        if self.lcn_ex > LCN_EX.LCN_1X:
            # TODO Verifiy the algorithm.
            for i in range(self.n_neuron):
                w_col = weights[:, i]

                # Traverse every column.
                col_group = 0
                # Rest of axons >= 1152? Here cannot be >=!
                while (r_axon := self.n_axon - bc_shape[0] * col_group) > bc_shape[0]:
                    bc[:, n_col_groups * i + col_group] = w_col[
                        bc_shape[0] * col_group : bc_shape[0] * (col_group + 1)
                    ]
                    col_group += 1

                # Pad for the rest of axons.
                bc[:, n_col_groups * i + col_group] = np.pad(
                    w_col[bc_shape[0] * col_group],
                    pad_width=(0, r_axon),
                    mode="constant",
                    constant_values=0,
                )
        else:
            # Other is filled with 0.
            bc[: self.n_axon, : self.n_neuron] = weights

        return bc.astype(np.bool_)

    @classmethod
    def build(cls, parent: GroupedSyn, position: int):
        """Build the divided synapse placing on a single CORE."""
        n_neuron = parent.n_neuron_each[position]
        weights = parent.weight_divided[position]

        return cls(parent, position, n_neuron, weights)

    @property
    def obj(self) -> GroupedSyn:
        return self._parent

    @property
    def position(self) -> int:
        return self._pos

    @property
    def n_axon(self) -> int:
        return self.obj.n_axon

    @property
    def lcn_ex(self) -> LCN_EX:
        return self.obj.lcn_ex

    @property
    def n_neuron(self) -> int:
        return self._n_neuron

    @property
    def source(self) -> List[SourceNodeType]:
        return self.obj.source

    @property
    def dest(self) -> List[DestNodeType]:
        return self.obj.dest

    @property
    def crossbar(self) -> np.ndarray:
        return self._binary_conn

    @property
    def coordinate(self) -> Coord:
        return self._routing_coord.coordinate

    @coordinate.setter
    def coordinate(self, coord: RoutingNodeCoord) -> None:
        self._routing_coord = coord
