from collections import defaultdict
from typing import ClassVar, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    AxonSegment,
    Coord,
    HwConfig,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronSegment,
)
from paibox.libpaicore import ReplicationId as RId
from paibox.libpaicore import (
    SNNModeEnable,
    SpikeWidthFormat,
    WeightPrecision,
    get_replication_id,
)
from paibox.projection import InputProj
from paibox.synapses import SynSys

from .config_template import CoreConfigDict, GlobalConfig, NeuronConfig

SourceNodeType = Union[NeuDyn, InputProj]
DestNodeType = NeuDyn
NeuSeg = NamedTuple("NeuSeg", [("parent", NeuDyn), ("segment", NeuronSegment)])


class PlacementObj(PAIBoxObject):
    pass


class CoreBlock(PlacementObj):
    """We gather some synapses as a block.

    All the synapses will be grouped first. Then we get a list of `CoreBlock`.
    """

    supported_weight_precision: ClassVar[Tuple[WeightPrecision, ...]] = (
        WeightPrecision.WEIGHT_WIDTH_1BIT,
    )

    def __init__(
        self,
        *parent: SynSys,
        weight_precision: WeightPrecision,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent synapses.
            - weight_precision: the precision of weight matrix.

        Axons ->                LCN    -> neuron_capacity -> n_core -> n_neuron_each
        tweight_precision -> n_dendrite -------> |
        """

        super().__init__(name)
        self._parent = parent
        self._lcn_ex = n_axon2lcn_ex(self.n_axon)
        self.weight_precision = weight_precision

        self.seed = np.uint64(seed)
        """Random seed, uint64."""

        self.target_lcn = LCN_EX.LCN_1X
        """The target(destination) LCN."""

        self.lcn_locked = False
        """Used to indicate whether `lcn_ex` has been adjusted."""

        self.axon_segments: Dict[SourceNodeType, AxonSegment] = defaultdict()

        self.core_coords: List[Coord] = []
        """Core coordinates.

        TODO Record the occuppied but unused coordinates here?
        """

        self.core_placements: List[CorePlacement] = []
        """Core placements."""

        # Segment the group of axons.
        self._get_axon_segments()

    def _get_axon_segments(self) -> None:
        pos = 0

        for axon in self.axons:
            segments, pos = get_axon_segments(axon, self.timeslot, pos)
            self.axon_segments[axon] = segments

    @classmethod
    def build(cls, *synapses: SynSys, name: Optional[str] = None):
        """Combine the SAME weight precision synapses and build the `CoreBlock`.

        Use LCN extension optimization in grouping a synapse.

        Description: always find the minimum LCN extension \
            that ALL the axons in this synapse satisfies.

            For two pre-synapses, S1 [A1*M] & S2 [A2*M], combine then split.

            The total number of axons = A1+A2 -> LCN -> n_neuron.

            Now consider S1 & S2 are 1-bit weights.
        """
        if not cls.all_dtype_equal(*synapses):
            raise NotImplementedError

        assert synapses[0].connectivity.dtype == np.bool_

        if synapses[0].connectivity.dtype == np.bool_:
            wp = WeightPrecision.WEIGHT_WIDTH_1BIT
        elif synapses[0].connectivity.dtype == np.int8:
            wp = WeightPrecision.WEIGHT_WIDTH_8BIT
        else:
            raise NotImplementedError

        if wp not in cls.supported_weight_precision:
            raise ValueError(f"{wp} is not supported by {cls.__class__}")

        return cls(*synapses, weight_precision=wp, name=name)

    @staticmethod
    def all_dtype_equal(*syns: SynSys) -> bool:
        _dtype = syns[0].connectivity.dtype

        for syn in syns:
            if _dtype != syn.connectivity.dtype:
                return False

        return True

    def set_lcn_ex(self, lcn_ex: LCN_EX) -> None:
        self.lcn_ex = lcn_ex
        self.lcn_locked = True

    def core_alloc(self) -> None:
        """Allocate the `CoreBlock` to the cores.

        NOTE: Do it after `lcn_ex_adjustment`.
        """
        if not self.lcn_locked:
            # TODO
            raise Exception

        # First, get the placement of all gsyn_on_cores.
        self.neuron_segs_of_cb = get_neu_segments(self.dest, self.neuron_capacity)

        for i in range(self.n_core):
            n_neuron = self.n_neuron_of_cb[i]
            nstart = sum(self.n_neuron_of_cb[:i])
            nstop = nstart + n_neuron

            self.core_placements.append(
                CorePlacement(
                    self,
                    self.core_coords[i],
                    n_neuron,
                    weights=self.weight_concat[:, nstart:nstop],  # divide at axis=1
                    neu_segs=self.neuron_segs_of_cb[i],
                )
            )

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
    def axons(self) -> List[SourceNodeType]:
        return self.source

    @property
    def dest(self) -> List[DestNodeType]:
        """Ordered unique destination nodes."""
        # return OrderedSet(set([parent.dest for parent in self.obj]))
        return list(set([parent.dest for parent in self.obj]))

    @property
    def n_axon_of_source(self) -> List[int]:
        """The #N of axons of each source neurons."""
        return [s.num_out for s in self.source]

    @property
    def n_axon(self) -> int:
        return sum(self.n_axon_of_source)

    @property
    def n_neuron_of_dest(self) -> List[int]:
        """The #N of neurons of each destination neurons."""
        return [d.num_in for d in self.dest]

    @property
    def n_neuron(self) -> int:
        return sum(self.n_neuron_of_dest)

    @property
    def neuron_capacity(self) -> int:
        return (
            HwConfig.N_NEURON_ONE_CORE_MAX // self.n_dendrite_combined
        ) >> self.lcn_ex

    @property
    def n_core(self) -> int:
        """The #N of cores required for placement."""
        return (self.n_neuron - 1) // self.neuron_capacity + 1

    @property
    def n_neuron_of_cb(self) -> List[int]:
        """A list of the #N of neurons on each `CorePlacement`."""
        n = [0] * (self.n_core)

        for i in range(self.n_core - 1):
            n[i] = self.neuron_capacity

        n[-1] = self.n_neuron % self.neuron_capacity

        return n

    @property
    def weight_concat(self) -> np.ndarray:
        """The weight matrix concatenated by the synapses \
            involved in the object.

        Concatenated weight for each destination node:
             For N1,  ...,  for Nn
            [s1, d1], ..., [s1, dn]
            ...
            [sn, d1], ..., [sn, dn]

        Then concatenate them in one piece.
        """

        def get_syn(src: SourceNodeType, dest: DestNodeType) -> Optional[SynSys]:
            for syn in self.obj:
                if syn.source == src and syn.dest == dest:
                    return syn

            return None

        w_of_neurons: List[
            np.ndarray
        ] = []  # The concatenated weight for each destination node.

        for d in self.dest:
            w_of_dest = []  # The weights for each destination node.

            for s in self.source:
                if syn := get_syn(s, d):
                    w_of_dest.append(syn.connectivity)
                else:
                    # Fill with 0.
                    w_of_dest.append(np.zeros((s.num_out, d.num_in), dtype=np.bool_))

            w_dest = np.concatenate(w_of_dest, axis=0)
            w_of_neurons.append(w_dest)

        w_in_gsyn = np.concatenate(w_of_neurons, axis=1)
        w_in_gsyn.setflags(write=False)

        return w_in_gsyn

    @property
    def n_dendrite_combined(self) -> int:
        """Multiple dendrites will be combined to \
            achieve higher precision weights.
        """
        # TODO Now consider all the synapses are 1-bit weights.
        return 1 << self.weight_precision

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
    def timeslot(self) -> int:
        return 1 << self.lcn_ex

    @property
    def replicationId(self) -> RId:
        rid = get_replication_id(self.core_coords)

        # Check
        # if len(get_multicast_cores(self.core_coords[0], rid)) > self.n_core:
        #     raise ValueError

        return rid

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"

    def copy(self):
        raise NotImplementedError

    def export_core_to_dict(self) -> Dict[Coord, CoreConfigDict]:
        """Export the parameters of the core into a dictionary."""
        cb_config: Dict[Coord, CoreConfigDict] = {}

        for i in range(0, self.n_core):
            # fmt: off
            cb_config[self.core_coords[i]] = CoreConfigDict(
                self.seed,                              # random_seed
                self.core_placements[i].crossbar,       # weight_ram
                self.weight_precision,                  # weight_precision
                self.lcn_ex,                            # lcn_extension
                InputWidthFormat.WIDTH_1BIT,            # input_width_format
                SpikeWidthFormat.WIDTH_1BIT,            # spike_width_format
                self.n_neuron_of_cb[i],                 # neuron_num
                MaxPoolingEnable.DISABLE,               # max_pooling_en
                0,                                      # tick_wait_start
                0,                                      # tick_wait_end
                SNNModeEnable.ENABLE,                   # snn_mode_en
                self.target_lcn,                        # target_lcn
                GlobalConfig.TEST_CHIP_ADDR.address,    # test_chip_addr
            )
            # fmt: on

        return cb_config


class CorePlacement(PlacementObj):
    """The divided synapse placed on a single CORE."""

    _excluded_vars = ()

    n_core: ClassVar[int] = 1
    binary_conn_shape: ClassVar[Tuple[int, int]] = (
        HwConfig.N_AXON_DEFAULT,
        HwConfig.N_NEURON_ONE_CORE_MAX,
    )

    def __init__(
        self,
        parent: CoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        *,
        weights: np.ndarray,
        neu_segs: List[NeuSeg],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent grouped synapse(complete).
            - idx: The index number where this object is located.
            - n_neuron: the number of neurons used in the CORE.
            - weights: the weights in this single CORE.
            - neuron_segment: The placement of the destination neurons.
        """
        super().__init__(name)

        self.parent = parent
        self.coord = routing_coord
        """Routing coordinate"""

        self.n_neuron = n_neuron
        self.weights = weights
        self.neu_segs = neu_segs

        self.neuron_config: Dict[NeuDyn, Dict] = defaultdict()

    def _get_binary_conn(self, weights: np.ndarray) -> np.ndarray:
        """Reshape the divided weight matrix into the shape 1152*512."""
        binary_conn = np.zeros(self.binary_conn_shape, dtype=np.bool_)
        row, _ = self.binary_conn_shape

        if self.lcn_ex > LCN_EX.LCN_1X:
            n_col_groups = self.parent.n_dendrite_combined

            for i in range(self.n_neuron):
                w_col = weights[:, i]

                # Traverse every column.
                col_group = 0
                # Rest of axons >= 1152? Here cannot be >=!
                while (n_rest_axon := self.n_axon - row * col_group) > row:
                    binary_conn[:, n_col_groups * i + col_group] = w_col[
                        row * col_group : row * (col_group + 1)
                    ]
                    col_group += 1

                # Fill the remaining positions with 0.
                binary_conn[:, n_col_groups * i + col_group] = np.pad(
                    w_col[row * col_group],
                    pad_width=(0, row - n_rest_axon),
                    mode="constant",
                    constant_values=0,
                )
        else:
            # Other places are filled with 0.
            binary_conn[: self.n_axon, : self.n_neuron] = weights

        return binary_conn.astype(np.bool_)

    def export_neu_config(self, neu_seg: NeuSeg, axon_dests: List[CoreBlock]):
        for axon_dest in axon_dests:
            axon_coords = aligned_coords(
                neu_seg.segment.index, axon_dest.axon_segments[neu_seg.parent]
            )

            config = NeuronConfig.build(
                neu_seg.parent,
                neu_seg.segment.addr_ram,
                axon_coords,
                axon_dest.core_coords,
            )

            self.neuron_config[neu_seg.parent] = config

    @property
    def weight_precision(self) -> WeightPrecision:
        return self.parent.weight_precision

    @property
    def n_axon(self) -> int:
        return self.parent.n_axon

    @property
    def lcn_ex(self) -> LCN_EX:
        return self.parent.lcn_ex

    @property
    def target_lcn(self) -> LCN_EX:
        return self.parent.target_lcn

    @property
    def timeslot(self) -> int:
        return self.parent.timeslot

    @property
    def source(self) -> List[SourceNodeType]:
        return self.parent.source

    @property
    def dest(self):
        """The destination nodes within it.

        NOTE: This attribute is different from the one of its parent.
        """
        return [p.parent for p in self.neu_segs]

    # @property
    # def n_neuron_of_dest(self) -> List[int]:
    #     return [p.addr.stop - p.addr.start for p in self.neuron_slices]

    @property
    def crossbar(self) -> np.ndarray:
        return self._get_binary_conn(self.weights)


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


def max_lcn_of_cb(cb: List[CoreBlock]) -> LCN_EX:
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex


# def _get_neuron_placements(
#     nodes: Union[List[DestNodeType], List[SourceNodeType]],
#     n_each: List[int],
#     capacity: int,
#     method: Literal["dense", "class"] = "class",
# ) -> List[NeuPlacement]:
#     """Get the arrangement of each input axons/output neurons \
#         group on the core.

#     Args:
#         - nodes: the input or output nodes.
#         - n_each: list of #N input axons or #N output neurons.
#         - capacity: the group capacity, 1152/or #N neurons in one core.
#         - method: `dense` is densely arranging all groups of neurons. \
#             `class` is arranging by category.

#     Return:
#         - A list, where each element represents how many groups \
#             the input axons or output neurons are divided into, \
#             and the slices of each node in each group.
#     """
#     plm: List[NeuPlacement] = []
#     temp_plm: NeuPlacement = []

#     rest_plm = 0

#     for i, node in enumerate(nodes):
#         n = n_each[i]

#         if method == "dense":
#             if rest_plm == 0:
#                 # Go to place the #i node.
#                 n_left = n
#             elif rest_plm < n:
#                 # If there is already a group that is not filled,
#                 # continue to allocate axons/neurons to it.
#                 temp_plm.append(NeuronSlice(node, slice(0, rest_plm, 1)))
#                 plm.append(temp_plm)

#                 n_left = n - rest_plm
#                 temp_plm = []
#             else:
#                 # The #i node can all be placed in it.
#                 temp_plm.append(NeuronSlice(node, slice(0, n, 1)))
#                 # Maybe some rest_plm of axons/neurons can be placed in this group.
#                 rest_plm -= n
#                 continue
#         else:
#             n_left = n

#         pos_start = rest_plm  # The start position to place `n_left`.
#         n_full_group = n_left // capacity  # The #N of full groups required.
#         n_in_last_group = (
#             n_left % capacity
#         )  # Rest of axons/neurons are left, and will place in the next group.

#         if n_full_group > 0:
#             for j in range(n_full_group):
#                 p = NeuronSlice(
#                     node,
#                     slice(
#                         pos_start + j * capacity,
#                         pos_start + (j + 1) * capacity,
#                         1,
#                     ),
#                 )
#                 # Place it directly.
#                 plm.append([p])

#         if n_in_last_group > 0:
#             # Place the rest_plm of axons/neurons in the next group.
#             p = NeuronSlice(
#                 node,
#                 slice(pos_start + n_full_group * capacity, n, 1),
#             )

#             if method == "dense":
#                 temp_plm.append(p)
#                 rest_plm = capacity - n_in_last_group
#             else:
#                 plm.append([p])
#                 rest_plm = 0

#     if temp_plm:
#         plm.append(temp_plm)

#     return plm


def get_neu_segments(
    neurons: Sequence[NeuDyn], capacity: int, method="class"
) -> List[List[NeuSeg]]:
    """Slice the neuron by class."""
    neu_segs = []

    for neuron in neurons:
        segs: List[NeuSeg] = []
        n_core = neuron.num_in // capacity
        n_neuron_rest = neuron.num_in % capacity

        if n_core > 0:
            for i in range(n_core):
                pos = i * capacity
                segs.append(
                    NeuSeg(neuron, NeuronSegment(slice(pos, pos + capacity, 1), pos))
                )

        if n_neuron_rest > 0:
            segs.append(
                NeuSeg(
                    neuron, NeuronSegment(slice(n_core * capacity, neuron.num_in, 1), 0)
                )
            )

        neu_segs.append(segs)

    return neu_segs


def get_axon_segments(
    axon: Union[NeuDyn, InputProj], tr_max: int, pos: int, method="class"
) -> tuple[AxonSegment, int]:
    # The width of assigned address
    if axon.num_out % tr_max > 0:
        addr_width = axon.num_out // tr_max + 1
        # n_axon_rest = axon.num_out % addr_width
    else:
        addr_width = axon.num_out // tr_max
        # n_axon_rest = 0

    return AxonSegment(axon.num_out, addr_width, pos), pos + addr_width


def aligned_coords(neu_index: slice, axon_seg: AxonSegment) -> List[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment."""
    axon_coords = []

    tr_start, tr_stop = (
        neu_index.start // axon_seg.addr_width,
        neu_index.stop // axon_seg.addr_width,
    )
    addr_start, addr_stop = (
        neu_index.start % axon_seg.addr_width,
        neu_index.stop % axon_seg.addr_width,
    )

    assert tr_stop >= tr_start

    if tr_stop == tr_start:
        for addr in range(addr_start, addr_stop):
            axon_coords.append(AxonCoord(tr_start, axon_seg.addr_offset + addr))
    else:
        for addr in range(addr_start, axon_seg.addr_width):
            axon_coords.append(AxonCoord(tr_start, axon_seg.addr_offset + addr))

        if tr_stop - tr_start > 1:
            for tr in range(tr_start + 1, tr_stop):
                for addr in range(0, axon_seg.addr_width):
                    axon_coords.append(AxonCoord(tr, axon_seg.addr_offset + addr))

        for addr in range(0, addr_stop):
            axon_coords.append(AxonCoord(tr_stop, axon_seg.addr_offset + addr))

    return axon_coords


# def _get_axon_slice(pos_start: int, pos_end: int) -> List[AxonCoord]:
#     """Get the arrangement of each input axons/output neurons \
#         group on the core.

#     Args:
#         - nodes: the input or output nodes.
#         - n_each: list of #N input axons or #N output neurons.
#         - 1152: the group 1152, 1152/or #N neurons in one core.
#         - method: `dense` is densely arranging all groups of neurons. \
#             `class` is arranging by category.

#     Return:
#         - A list, where each element represents how many groups \
#             the input axons or output neurons are divided into, \
#             and the slices of each node in each group.

#     """
#     slices = []

#     # Start at `tr_now`
#     tr_now = pos_start // HwConfig.N_AXON_DEFAULT
#     # Already place `a_now` axons
#     a_now = pos_start % HwConfig.N_AXON_DEFAULT

#     if a_now > 0:
#         if pos_end <= tr_now * HwConfig.N_AXON_DEFAULT:
#             # Place in this tick_relative slot
#             return [AxonCoord(tr_now, slice(a_now, pos_end, 1))]

#         else:
#             slices.append(AxonCoord(tr_now, slice(a_now, HwConfig.N_AXON_DEFAULT, 1)))

#     tr_need = pos_end // HwConfig.N_AXON_DEFAULT - (tr_now + 1)  # 0
#     n_axon_rest = pos_end % HwConfig.N_AXON_DEFAULT  # 700

#     if tr_need > 0:
#         for tr in range(tr_need):
#             slices.append(AxonCoord(tr, slice(0, HwConfig.N_AXON_DEFAULT, 1)))

#     tr_now += tr_need
#     if n_axon_rest > 0:
#         _end = pos_end - (tr_now + 1) * HwConfig.N_AXON_DEFAULT
#         slices.append(AxonCoord(tr_now + 1, slice(0, _end, 1)))

#     return slices
