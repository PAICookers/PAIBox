from collections import defaultdict
from typing import (
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from paibox.base import NeuDyn, PAIBoxObject
from paibox.exceptions import BuildError, NotSupportedError, ResourceError
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    AxonSegment,
    Coord,
    CoreMode,
    HwConfig,
    HwCore,
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

from .config_template import CoreConfigDict, CorePlacementConfig, NeuronConfig
from .context import _BACKEND_CONTEXT

SourceNodeType = Union[NeuDyn, InputProj]
DestNodeType = NeuDyn
NeuSeg = NamedTuple("NeuSeg", [("parent", DestNodeType), ("segment", NeuronSegment)])


class CoreAbstract(HwCore, PAIBoxObject):
    supported_weight_precision: ClassVar[Tuple[WeightPrecision, ...]] = (
        WeightPrecision.WEIGHT_WIDTH_1BIT,
    )
    supported_mode: ClassVar[Tuple[CoreMode, ...]] = (CoreMode.MODE_SNN,)


class CoreBlock(CoreAbstract):
    """Core Block for `MODE_SNN` ONLY."""

    mode: ClassVar[CoreMode] = CoreMode.MODE_SNN

    # InputWidthFormat.WIDTH_1BIT
    # SpikeWidthFormat.WIDTH_1BIT
    # SNNModeEnable.ENABLE

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

        Axons ->                LCN    -> dendrite_capacity -> n_core_required -> n_neuron_each
        weight_precision -> n_dendrite -------> |
        """

        super().__init__(name)
        self._parent = parent
        self._lcn_ex = n_axon2lcn_ex(self.n_axon, self.n_fanin_max)
        self.weight_precision = weight_precision

        self.seed = np.uint64(seed)
        """Random seed, uint64."""

        self.target_lcn = LCN_EX.LCN_1X
        """The target(destination) LCN."""

        self.lcn_locked = False
        """Used to indicate whether `lcn_ex` has been adjusted."""

        self.core_coords: List[Coord] = []
        """Core coordinates.

        TODO Record the occuppied but unused coordinates here?
        """

        self.core_placements: Dict[Coord, CorePlacement] = dict()
        """Core placements."""

        # Segment the group of axons.
        self.axon_segments: Dict[SourceNodeType, AxonSegment] = get_axon_segments(
            self.axons, self.timeslot, self.n_fanin_max
        )
        """A dictionary of segments of each axon(source node)."""

        self.neuron_segs_of_cb: Dict[Coord, List[NeuSeg]] = defaultdict(list)
        """"""

    @classmethod
    def build(cls, *synapses: SynSys):
        """Combine the SAME weight precision synapses and build the `CoreBlock`.

        Use LCN extension optimization in grouping a synapse.

        Description: always find the minimum LCN extension \
            that ALL the axons in this synapse satisfies.

            For two pre-synapses, S1 [A1*M] & S2 [A2*M], combine then split.

            The total number of axons = A1+A2 -> LCN -> n_neuron.

            Now consider S1 & S2 are 1-bit weights.
        """
        if not cls.all_dtype_equal(*synapses):
            raise NotSupportedError(f"Mixed weight precision is not supported yet")

        if synapses[0].connectivity.dtype == np.bool_:
            wp = WeightPrecision.WEIGHT_WIDTH_1BIT
        elif synapses[0].connectivity.dtype == np.int8:
            wp = WeightPrecision.WEIGHT_WIDTH_8BIT
        else:
            raise NotImplementedError

        if wp not in cls.supported_weight_precision:
            raise NotSupportedError(f"{wp} is not supported yet by {cls.__class__}")

        return cls(*synapses, weight_precision=wp)

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

    def core_plm_alloc(self) -> None:
        """Allocate the `CoreBlock` to the cores.

        NOTE: Do it after `lcn_ex_adjustment`.
        """
        if not self.lcn_locked:
            raise BuildError(f"Allocate the core placements after lcn_ex is locked.")

        # First, get the neuron segments.
        neu_segs_of_cb = get_neu_segments(
            self.dest,
            self.neuron_capacity,
            weight_precision=self.weight_precision,
            lcn_ex=self.lcn_ex,
            method="catagory",
        )

        assert len(neu_segs_of_cb) == len(self.core_coords)

        for coord, segs in zip(self.core_coords, neu_segs_of_cb):
            self.neuron_segs_of_cb[coord] = segs

        for i, coord in enumerate(self.core_coords):
            n_neuron = self.n_neuron_of_cb[i]
            nstart = sum(self.n_neuron_of_cb[:i])
            nstop = nstart + n_neuron

            # divide at axis=1
            self.core_placements[coord] = CorePlacement.build(
                self, coord, i, self.weight_concat[:, nstart:nstop]
            )

    """Interfaces"""

    @property
    def obj(self) -> Tuple[SynSys, ...]:
        return self._parent

    @property
    def source(self) -> List[SourceNodeType]:
        """Ordered unique source nodes."""
        return list(set([parent.source for parent in self.obj]))

    @property
    def axons(self) -> List[SourceNodeType]:
        return self.source

    @property
    def dest(self) -> List[DestNodeType]:
        """Ordered unique destination nodes."""
        return list(set([parent.dest for parent in self.obj]))

    @property
    def n_axon_of_source(self) -> List[int]:
        """The #N of axons of each source neurons."""
        return [s.num_out for s in self.source]

    """Boundary limitations"""

    @property
    def neuron_capacity(self) -> int:
        """Neuron capacity. #N of valid dendrites/#N of dendrites required per neuron.

        FIXME This method only works in SNN mode. For ANN mode, use table lookup.
        """
        return (self.n_dendrite >> self.lcn_ex) // self.n_dendrite_per_neuron

    @property
    def n_fanin_max(self) -> int:
        """Maximum #N of fan-in per dendrite."""
        return (
            HwConfig.N_FANIN_PER_DENDRITE_ANN
            if self.mode is CoreMode.MODE_ANN
            else HwConfig.N_FANIN_PER_DENDRITE_SNN
        )

    @property
    def n_core_required(self) -> int:
        return (self.n_neuron - 1) // self.neuron_capacity + 1

    @property
    def n_dendrite_per_neuron(self) -> int:
        """Multiple dendrites will be combined to achieve higher    \
            precision weights.

        FIXME The limit on the number of dendrites in SNN/ANN modes \
            is different, which affects the capacity of neurons in  \
            the physical core.
        """
        return 1 << self.weight_precision

    @property
    def lcn_ex(self) -> LCN_EX:
        return self._lcn_ex

    @lcn_ex.setter
    def lcn_ex(self, lcn_ex: LCN_EX) -> None:
        if lcn_ex > LCN_EX.LCN_64X:
            raise ResourceError(
                f"LCN extension required out of {LCN_EX.LCN_64X}: {lcn_ex}"
            )

        self._lcn_ex = lcn_ex

    @property
    def timeslot(self) -> int:
        return 1 << self.lcn_ex

    """Resource attributes."""

    @property
    def n_axon(self) -> int:
        return sum(self.n_axon_of_source)

    @property
    def n_dendrite(self) -> int:
        return (
            HwConfig.N_DENDRITE_MAX_ANN
            if self.mode is CoreMode.MODE_ANN
            else HwConfig.N_DENDRITE_MAX_SNN
        )

    @property
    def n_neuron_of_dest(self) -> List[int]:
        """The #N of neurons of each destination neurons."""
        return [d.num_in for d in self.dest]

    @property
    def n_neuron(self) -> int:
        return sum(self.n_neuron_of_dest)

    @property
    def n_neuron_of_cb(self) -> List[int]:
        """A list of the #N of neurons on each `CorePlacement` \
            in descending order.

        FIXME Different in SNN/ANN mode.
        """
        if len(self.core_coords) == 0:
            raise BuildError(f"Do this after coordinates assignment.")

        n = [0] * (self.n_core_required)

        for i in range(self.n_core_required - 1):
            n[i] = self.neuron_capacity

        n[-1] = self.n_neuron % self.neuron_capacity

        # nd = defaultdict(int)
        # for i in range(self.n_core_required - 1):
        #     nd[self.core_coords[i]] = self.neuron_capacity

        # nd[self.core_coords[-1]] = self.n_neuron % self.neuron_capacity

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

        # The concatenated weight for each destination node.
        w_of_neurons: List[np.ndarray] = []

        for d in self.dest:
            # The weights for each destination node.
            w_of_dest = []

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
    def replicationId(self) -> RId:
        rid = get_replication_id(self.core_coords)

        # Check
        # if len(get_multicast_cores(self.core_coords[0], rid)) > self.n_core_required:
        #     raise ValueError

        return rid

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"

    def copy(self):
        raise NotImplementedError

    @classmethod
    def export_core_plm_config(cls, cb: "CoreBlock") -> Dict[Coord, CoreConfigDict]:
        """Export the parameters of the core into a dictionary."""
        cb_config = dict()

        for coord, core_plm in cb.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config


class CorePlacement(CoreAbstract):
    """The divided synapse placed on a single CORE."""

    _excluded_vars = ()

    n_core_required: ClassVar[int] = 1
    binary_conn_shape: ClassVar[Tuple[int, int]] = (
        HwConfig.N_FANIN_PER_DENDRITE_SNN,
        HwConfig.N_DENDRITE_MAX_SNN,
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
        self.neu_config: Dict[NeuDyn, NeuronConfig] = defaultdict()

    @classmethod
    def build(cls, parent: CoreBlock, coord: Coord, idx: int, weights: np.ndarray):
        n_neuron = parent.n_neuron_of_cb[idx]
        neu_segs = parent.neuron_segs_of_cb[coord]

        return cls(parent, coord, n_neuron, weights=weights, neu_segs=neu_segs)

    def _get_binary_conn(self, weights: np.ndarray) -> np.ndarray:
        """Reshape the divided weight matrix into the shape 1152*512."""
        binary_conn = np.zeros(self.binary_conn_shape, dtype=np.bool_)
        row, _ = self.binary_conn_shape

        if self.lcn_ex > LCN_EX.LCN_1X:
            n_col_groups = self.parent.n_dendrite_per_neuron

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

    def export_param_config(self) -> CoreConfigDict:
        # fmt: off
        cb_config = CoreConfigDict(
            self.weight_precision,              # weight_precision
            self.lcn_ex,                        # lcn_extension
            InputWidthFormat.WIDTH_1BIT,        # input_width_format
            SpikeWidthFormat.WIDTH_1BIT,        # spike_width_format
            self.n_dendrite,                    # num_dendrite
            MaxPoolingEnable.DISABLE,           # max_pooling_en
            0,                                  # tick_wait_start
            0,                                  # tick_wait_end
            SNNModeEnable.ENABLE,               # snn_mode_en
            self.target_lcn,                    # target_lcn
            _BACKEND_CONTEXT["test_chip_addr"], # test_chip_addr
        )
        # fmt: on
        return cb_config

    def export_neu_config(self, neu_seg: NeuSeg, axon_dests: List[CoreBlock]) -> None:
        """
        TODO Change the name of the method.
        """
        for axon_dest in axon_dests:
            axon_coords = aligned_coords(
                neu_seg.segment.index, axon_dest.axon_segments[neu_seg.parent]
            )

            config = NeuronConfig.encapsulate(
                neu_seg.parent,
                neu_seg.segment.addr_ram,
                neu_seg.segment.addr_offset,
                axon_coords,
                axon_dest.core_coords,
            )

            self.neu_config[neu_seg.parent] = config

    def export_core_config(self) -> CorePlacementConfig:
        return CorePlacementConfig.encapsulate(
            self.coord,
            self.parent.seed,  # random_seed
            self.crossbar,  # weight_ram
            self.export_param_config(),
            self.neu_config,
        )

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
    def n_dendrite(self) -> int:
        return self.n_neuron * self.parent.n_dendrite_per_neuron

    @property
    def source(self) -> List[SourceNodeType]:
        return self.parent.source

    @property
    def dest(self):
        """The destination nodes within it.

        NOTE: This attribute is different from the one of its parent.
        """
        return [p.parent for p in self.neu_segs]

    @property
    def crossbar(self) -> np.ndarray:
        return self._get_binary_conn(self.weights)


def n_axon2lcn_ex(n_axon: int, fan_in_max: int) -> LCN_EX:
    """Convert #N(of axons) to `LCN_EX`.

    Description:
        LCN_EX = log2[ceil(#N/fan-in per dendrite)]

    where, LCN_EX = 0 is `LCN_1X`.
    """
    if n_axon < 1:
        # TODO
        raise ValueError(f"The #N of axons > 0, but got {n_axon}")

    lcn_ex = LCN_EX(((n_axon - 1) // fan_in_max).bit_length())

    if lcn_ex > LCN_EX.LCN_64X:
        raise ResourceError(
            f"LCN extension required out of {LCN_EX.LCN_64X}: {lcn_ex}"
        )

    return lcn_ex


def max_lcn_of_cb(cb: List[CoreBlock]) -> LCN_EX:
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex


def get_neu_segments(
    neu_groups: Sequence[NeuDyn],
    capacity: int,
    *,
    weight_precision: WeightPrecision,
    lcn_ex: LCN_EX,
    method: Literal["catagory", "dense"] = "catagory",
) -> List[List[NeuSeg]]:
    """
    TODO Add description.
    """
    interval = (1 << weight_precision) * (1 << lcn_ex)

    if method == "catagory":
        return _get_neu_segments_catagory(neu_groups, capacity, interval)
    elif method == "dense":
        return _get_neu_segments_dense(neu_groups, capacity, interval)

    raise NotSupportedError(f"Method {method} is not supported yet")


def _get_neu_segments_catagory(
    neu_groups: Sequence[NeuDyn], capacity: int, interval: int = 1
) -> List[List[NeuSeg]]:
    """Group by category of neuron groups."""
    neu_segs: List[List[NeuSeg]] = []  # The final result

    for neuron in neu_groups:
        i = 0
        num = neuron.num_out

        while i < (num - 1) // capacity:
            seg = NeuronSegment(slice(i * capacity, capacity * (i + 1), 1), 0, interval)
            neu_segs.append([NeuSeg(neuron, seg)])

            i += 1

        seg = NeuronSegment(slice(i * capacity, num, 1), 0, interval)
        neu_segs.append([NeuSeg(neuron, seg)])

    return neu_segs


def _get_neu_segments_dense(
    neu_groups: Sequence[NeuDyn], capacity: int, interval: int = 1
) -> List[List[NeuSeg]]:
    """Dense grouping. Based on method `catagory`, use the greedy algorithm to \
        group the remaining neuron groups.
    """
    neu_segs: List[List[NeuSeg]] = []  # The final result
    rest_segs: List[NeuSeg] = []

    for neuron in neu_groups:
        i = 0
        num = neuron.num_out

        while i < (num - 1) // capacity:
            seg = NeuronSegment(slice(i * capacity, capacity * (i + 1), 1), 0, interval)
            neu_segs.append([NeuSeg(neuron, seg)])
            i += 1

        seg = NeuronSegment(slice(i * capacity, num, 1), 0, interval)
        rest_segs.append(NeuSeg(neuron, seg))

    # Sort the rest of segments in descending order
    rest_segs.sort(key=lambda neu_seg: neu_seg.segment.n_neuron, reverse=True)

    # The remaining neuron groups can then be grouped up to N cores
    n_core_max = len(rest_segs)
    n_cur_reg = 0

    def backtrack(i: int, cur_addr_offset: int, taken: List[NeuSeg]) -> None:
        nonlocal n_cur_reg

        if i == n_core_max or n_cur_reg == n_core_max:
            return

        if cur_addr_offset + rest_segs[n_cur_reg].segment.n_neuron > capacity:
            neu_segs.append(taken)
            return
        else:
            taken.append(
                NeuSeg(
                    rest_segs[n_cur_reg].parent,
                    NeuronSegment(
                        rest_segs[n_cur_reg].segment.index, cur_addr_offset, interval
                    ),
                )
            )
            cur_addr_offset += rest_segs[n_cur_reg].segment.n_neuron
            n_cur_reg += 1

        if n_cur_reg == n_core_max:
            neu_segs.append(taken)
            return

        # Continue to place
        backtrack(i, cur_addr_offset, taken)
        # Place to next physical core
        backtrack(i + 1, 0, [])

    backtrack(0, 0, [])

    return neu_segs


def get_axon_segments(
    axons: Sequence[SourceNodeType], tr_max: int, fan_in_max: int, method="class"
) -> Dict[SourceNodeType, AxonSegment]:
    """Divide axons into segments by group to fit the hardware constraints.

    Args:
        - axons: The axons to be segmented.
        - tr_max: The maximum value of the time slot(=timeslot).

    TODO Provide an alternative when failed using this method.
    """

    def _seg_alloc(axon: SourceNodeType, offset: int) -> Tuple[AxonSegment, int]:
        """Allocate an axon segment, return the next offset of axon address."""
        # The width of assigned address
        if axon.num_out % tr_max > 0:
            addr_width = axon.num_out // tr_max + 1
            # n_axon_rest = axon.num_out % addr_width
        else:
            addr_width = axon.num_out // tr_max
            # n_axon_rest = 0

        if offset + addr_width > fan_in_max:
            raise ResourceError(
                f"Address of axons out of range{fan_in_max}: {offset + addr_width}"
            )

        return AxonSegment(axon.num_out, addr_width, offset), offset + addr_width

    offset = 0
    axon_segments = dict()

    for axon in axons:
        segment, offset = _seg_alloc(axon, offset)
        axon_segments[axon] = segment

    return axon_segments


def aligned_coords(neu_index: slice, axon_seg: AxonSegment) -> List[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment.

    The length of axon coordinates is the same as `neu_index`.
    """
    axon_coords = []

    tr_start, tr_stop = (
        neu_index.start // axon_seg.addr_width,
        neu_index.stop // axon_seg.addr_width,
    )
    addr_start, addr_stop = (
        neu_index.start % axon_seg.addr_width,
        neu_index.stop % axon_seg.addr_width,
    )

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
