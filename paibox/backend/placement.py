import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass, field
from typing import ClassVar, Literal, NamedTuple, Optional, overload

import numpy as np
from paicorelib import LCN_EX, ChipCoord, Coord, CoreMode, HwConfig, MaxPoolingEnable
from paicorelib import ReplicationId as RId
from paicorelib import WeightWidth as WW
from paicorelib.framelib import OfflineFrameGen
from paicorelib.routing_defs import get_replication_id

from paibox import _logging
from paibox.base import PAIBoxObject
from paibox.components import Neuron
from paibox.exceptions import (
    GraphBuildError,
    LockedAttrOverwriteError,
    ResourceError,
    TruncationWarning,
)
from paibox.types import WEIGHT_DTYPE, WeightType
from paibox.utils import check_attr_same

from ._slice import (
    DestSliceType,
    EdgeSlice,
    PrttnSliceType,
    SourceSliceType,
    covered_by,
)
from .conf_types import CoreConfig, CoreConfInChip, CorePlmConfig, NeuronConfig
from .context import _BACKEND_CONTEXT
from .segment_utils import aligned_coords, get_axon_segments, get_neu_segments
from .types import (
    _COORD_UNSET,
    _RID_UNSET,
    N_BIT_PACKED_WEIGHT,
    WRAM_PACKED_DTYPE,
    WRAM_UNPACKED_DTYPE,
    AxonCoord,
    AxonSegment,
    DestNodeType,
    NeuSegment,
    NeuSegOfCoreBlock,
    NeuSegOfCorePlm,
    WRAMPackedType,
    WRAMUnpackedType,
    _1st_core_coord_repr,
    _coord_to_bin_str,
    is_iw8,
)

cb_log = _logging.get_artifact_logger(__name__, "core_block_info")

# Get the fan-out by the combination rate of dendrites
if hasattr(HwConfig, "FANOUT_IW8"):
    FANOUT_IW8 = HwConfig.FANOUT_IW8
else:
    FANOUT_IW8 = [HwConfig.N_NEURON_MAX_ANN, 1364, 876, 512, 256, 128, 64, 32, 16, 8]


NEURON_PARAMS_BIT_LENGTH = 214  # A constant of frame definition


class CoreAbstract(PAIBoxObject, ABC):
    """Abstract core class."""

    rt_mode: CoreMode

    @property
    @abstractmethod
    def n_core_required(self) -> int:
        """#N of cores required to accommodate neurons inside self."""
        ...

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs): ...


class CoreBlock(CoreAbstract):

    _parents: tuple[EdgeSlice, ...]
    seed: int
    """Random seed, legal integer, no more than uint64."""
    _lcn_ex: LCN_EX
    _lcn_locked: bool
    """Indicate whether `lcn_ex` has been adjusted & locked."""
    target_lcn: LCN_EX
    """The target(destination core block) LCN."""
    chip_coord: ChipCoord
    """A core block must be placed on a chip."""
    core_coords: list[Coord]
    """Assigned core coordinates."""
    core_placements: dict[Coord, "CorePlacement"]
    """Core placements."""
    axon_segments: dict[SourceSliceType, AxonSegment] = dict()
    """A dictionary of segments of each axon(source node)."""
    neuron_segs_of_cb: NeuSegOfCoreBlock = []
    """Neuron segments in the core block. Each element in the list represents the neuron    \
        segments in core placement.
    """

    def __init__(
        self,
        *parents: EdgeSlice,
        seed: int,
        mode: CoreMode,
        name: Optional[str] = None,
    ) -> None:
        """Core blocks in SNN mode.

        Args:
            parents: the parent synapses.
            seed: random seed. Default value is 0.
            mode: runtime mode of the core block.
        """
        super().__init__(name)
        self._parents = parents
        self.rt_mode = mode
        self.seed = seed
        self._lcn_ex = LCN_EX.LCN_1X

        self.target_lcn = LCN_EX.LCN_1X
        self.core_coords = []
        self.chip_coord = _COORD_UNSET
        self.core_placements = dict()
        self.axon_segments = dict()
        self.neuron_segs_of_cb = []
        self._ordered_axons: list[SourceSliceType] = []
        """Axons in global + private types order."""

        self._lcn_locked = False
        self._neurons_grouped = False

    def group_neurons(
        self, optim_target: Literal["latency", "core", "both"] = "both"
    ) -> None:
        """Group the neurons to determine the #N of cores required."""
        if not self._lcn_locked:
            raise GraphBuildError("group the neurons after 'lcn_ex' is locked.")

        self.neuron_segs_of_cb = get_neu_segments(
            self.dest, self.n_fanout, self.n_neuron_repl, optim_target
        )

        self._neurons_grouped = True

    def core_plm_alloc(self) -> None:
        """Allocate `CoreBlock` to physical cores."""
        if not self._lcn_locked:
            raise GraphBuildError("allocate core placements after 'lcn_ex' is locked.")

        for i, coord in enumerate(self.core_coords):
            self.core_placements[coord] = CorePlacement.build(self, i)

    def _get_syn_of(
        self, src: SourceSliceType, dest: DestSliceType
    ) -> Optional[EdgeSlice]:
        for syn in self.obj:
            if syn.source is src and syn.dest is dest:
                return syn

        return None

    def _n_axon2lcn_ex(self) -> LCN_EX:
        """Convert #N(of axons) to `LCN_EX` & check.

        NOTE: LCN_EX = log2[ceil(#N/fan-in per dendrite)], where `LCN_1X` = 0.
        """
        if self.n_axon < 1:
            raise ValueError(
                f"the number of axons must be positive, but got {self.n_axon}."
            )

        if (
            lcn := ((self.n_axon - 1) // self.n_fanin_base).bit_length()
        ) > LCN_EX.LCN_64X:
            _max_n_axons = self.n_fanin_base << LCN_EX.LCN_64X
            raise ResourceError(
                f"required LCN out of range {LCN_EX.LCN_64X} ({lcn}). The number of axons "
                f"must be <= {_max_n_axons}, but synapses {self._obj_repr} have a total of "
                f"{self.n_axon} axons."
            )

        return LCN_EX(lcn)

    def assign_coord(
        self, chip_coord: Coord, allocated: list[Coord]
    ) -> tuple[list[Coord], list[Coord]]:
        self.chip_coord = chip_coord
        self.core_coords = allocated
        return allocated, []

    def copy(self):
        raise NotImplementedError

    @property
    def obj(self) -> tuple[EdgeSlice, ...]:
        return self._parents

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ordered_axons), len(self.dest))

    @property
    def source(self) -> list[SourceSliceType]:
        """Ordered unique source nodes."""
        return list(set([parent.source for parent in self.obj]))

    @property
    def axons(self) -> list[SourceSliceType]:
        return self.source

    @property
    def dest(self) -> list[DestSliceType]:
        """Ordered unique destination nodes."""
        return list(set([parent.dest for parent in self.obj]))

    def n_axon_of(self, index: int) -> int:
        """Get the #N of axons of `index`-th source neuron."""
        return self.ordered_axons[index].num_out

    """Boundary limitations"""

    @property
    def n_fanin_base(self) -> int:
        """The fan-in of cores."""
        return (
            HwConfig.N_FANIN_PER_DENDRITE_SNN
            if self.rt_mode.is_snn
            else HwConfig.N_FANIN_PER_DENDRITE_ANN
        )

    @property
    def n_core_required(self) -> int:
        return len(self.neuron_segs_of_cb)

    @property
    def weight_width(self) -> WW:
        # `weight_width` is optimized in FullConnectedSyn.
        return max(s.weight_width for s in self.obj)

    @property
    def n_weight_bits(self) -> int:
        """Multiple dendrites will be combined to achieve higher precision weights."""
        return 1 << self.weight_width

    @property
    def lcn_ex(self) -> LCN_EX:
        return self._lcn_ex

    @lcn_ex.setter
    def lcn_ex(self, lcn_ex: LCN_EX) -> None:
        # if self._lcn_locked:
        #     raise LockedAttrOverwriteError("`lcn_ex` has been locked.")

        if lcn_ex > LCN_EX.LCN_64X:
            raise ValueError(f"required LCN out of range {LCN_EX.LCN_64X} ({lcn_ex}).")

        self._lcn_ex = lcn_ex
        self._lcn_locked = True

    @property
    def n_timeslot(self) -> int:
        return 1 << self.lcn_ex

    @property
    def dendrite_comb_rate(self) -> int:
        """#N of dendrites will be combined."""
        return self.lcn_ex + self.weight_width

    @property
    def tws(self) -> int:
        """Attribute `tick_wait_start`."""
        _check_attr = "tick_wait_start"
        if not check_attr_same(self.dest, _check_attr):
            raise AttributeError(
                f"attribute '{_check_attr}' of the core block are not equal."
            )

        return self.dest[0].tick_wait_start

    @property
    def twe(self) -> int:
        """Attribute `tick_wait_end`."""
        _check_attr = "tick_wait_end"
        if not check_attr_same(self.dest, _check_attr):
            raise AttributeError(
                f"attribute '{_check_attr}' of the core block are not equal."
            )

        return self.dest[0].tick_wait_end

    @property
    def pool_max(self) -> MaxPoolingEnable:
        """Attribute `pool_max`."""
        _check_attr = "pool_max"
        if not check_attr_same(self.dest, _check_attr):
            raise AttributeError(
                f"attribute '{_check_attr}' of the core block are not equal."
            )

        return self.dest[0].pool_max

    @property
    def n_axon(self) -> int:
        return sum(s.num_out for s in self.ordered_axons)

    @property
    def n_fanout(self) -> int:
        """The fan-out of cores."""
        return (
            HwConfig.N_DENDRITE_MAX_SNN >> self.dendrite_comb_rate
            if self.rt_mode.is_snn
            else FANOUT_IW8[self.dendrite_comb_rate]
        )

    @property
    def n_neuron(self) -> int:
        return sum(d.num_in for d in self.dest)

    @property
    def unrolling_factor(self) -> list[int]:
        return [d.unrolling_factor for d in self.dest]

    @property
    def n_neuron_of_plm(self) -> list[int]:
        """A list of the #N of neurons on each `CorePlacement`."""
        if len(self.core_coords) == 0:
            raise GraphBuildError("do this after coordinates assignment.")

        # Get #N of neurons on each `CorePlacement` according to the
        # maximum address required of neuron segments on each `CorePlacement`.
        return [
            sum(seg.n_neuron for seg in neuron_segs)
            for neuron_segs in self.neuron_segs_of_cb
        ]

    @property
    def ordered_axons(self) -> list[SourceSliceType]:
        return self._ordered_axons

    @ordered_axons.setter
    def ordered_axons(self, axons: list[SourceSliceType]) -> None:
        self._ordered_axons = axons
        self._lcn_ex = self._n_axon2lcn_ex()  # not use `@lcn_ex.setter` here

    def group_axons(self) -> None:
        """Group the axons, including the global & private parts. Sort the axons in order."""
        # The ordered axon has global(multicast) & private axons arranged in order.
        self.axon_segments = get_axon_segments(
            self.ordered_axons, self.n_timeslot, self.n_fanin_base
        )

    @property
    def raw_weight_of_dest(self) -> list[WeightType]:
        """Merge and then split the weight matrix according to the grouping of neurons."""
        # The concatenated weight for each destination node.
        w_of_neurons: list[WeightType] = []

        for d in self.dest:
            # The weights for each destination node.
            w_of_dest = []

            for s in self.ordered_axons:
                if syn := self._get_syn_of(s, d):
                    w_of_dest.append(syn.connectivity)
                else:
                    # Fill with 0.
                    w_of_dest.append(
                        np.zeros((s.num_out, d.num_in), dtype=WEIGHT_DTYPE)
                    )

            w_dest = np.vstack(w_of_dest)
            w_of_neurons.append(w_dest)

        # Check
        assert all(
            w_of_neurons[0].shape[0] == w_of_neuron.shape[0]
            for w_of_neuron in w_of_neurons
        )

        return w_of_neurons

    def get_raw_weight_of_coord(self, idx: int) -> list[WeightType]:
        """Get the corresponding part of the original weight matrix corresponding to each CP."""
        w_of_neu_segs: list[WeightType] = []
        _idx = 0
        sub_slice = slice(0, 0)

        for neu_seg in self.neuron_segs_of_cb[idx]:
            _not_covered = True

            for i, dest_sl in enumerate(self.dest):
                # Get the corresponding part of `neu_seg` on the slice of `dest_sl`.
                if covered_by(neu_seg, dest_sl):
                    _not_covered = False
                    _idx = i
                    sub_slice = slice(
                        neu_seg.index.start - dest_sl.index.start,
                        neu_seg.index.stop - dest_sl.index.start,
                    )
                    break

            assert (
                not _not_covered
            ), f"neuron segment {neu_seg} is not covered by any dest: {self.dest}."

            w_of_dest = self.raw_weight_of_dest[_idx]
            w_of_neu_seg = w_of_dest[:, sub_slice].copy()
            w_of_neu_seg.setflags(write=False)
            w_of_neu_segs.append(w_of_neu_seg)

        return w_of_neu_segs

    @property
    def n_neuron_repl(self) -> int:
        """The number of neurons that need to be repeatedly placed into NRAM.

        For example, in SNN mode, N[0:3] with LCN_2X & WW8:
            NRAM [0]  [1]  ... [15] [16] [17] ... [31] ...
                 N[0] N[0] ... N[0] N[1] N[1] ... N[1] ...

        But at 8-bit input width, neurons don't need to be replicated.
            NRAM [0]  [1]  ... [15]  [16]  ...
                 N[0] N[1] ... N[15] N[16] ...
        """
        return 1 << self.dendrite_comb_rate if self.rt_mode.is_snn else 1

    def __len__(self) -> int:
        return self.n_core_required

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self._obj_repr}'>"

    def __str__(self) -> str:
        ind1 = "\t"
        ind2 = "\t\t"

        _repr = self.name + "\n"
        _repr += ind1 + f"lcn_ex: {self.lcn_ex}\n"
        _repr += ind1 + f"weight_width: {self.weight_width}\n"
        _repr += ind1 + f"fan_out: {self.n_fanout}\n"

        _repr += ind1 + "dests:\n"
        for dest in self.dest:
            _repr += ind2 + str(dest) + "\n"

        _repr += ind1 + "cores "
        for i, neu_seg in enumerate(self.neuron_segs_of_cb):
            _repr += f"#{i}:\n"
            for seg in neu_seg:
                _repr += ind2 + f"{seg.target.name}[{seg.index}]\n"

        return _repr

    @property
    def _obj_repr(self) -> str:
        """The representation of the names of target objects."""
        return ", ".join(str(n) for n in self.obj)

    @classmethod
    def build(cls, *synapses: EdgeSlice, rt_mode: CoreMode, seed: int = 0):
        """Group synapses & build `CoreBlock`."""
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )

        return cls(*synapses, mode=rt_mode, seed=seed)

    def export_core_plm_config(self) -> CoreConfInChip:
        """Export the parameters of the core into a dictionary."""
        cb_config: CoreConfInChip = dict()

        for coord, core_plm in self.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config

    def dump(
        self, indents: int = 0, father_logger: Optional[logging.Logger] = None
    ) -> None:
        _logger = cb_log if father_logger is None else father_logger

        tabs = "\t" * indents
        ind1 = tabs + "\t"
        ind2 = tabs + "\t\t"

        _logger.debug(tabs + f"{self.name} ({self.n_core_required} cores):")
        _logger.debug(ind1 + f"LCN: {self.lcn_ex}")
        _logger.debug(ind1 + f"Weight width: {self.weight_width}")

        _logger.debug(ind1 + "Axons:")
        for axon in self.ordered_axons:
            _logger.debug(ind2 + str(axon))

        _logger.debug(ind1 + "Dests:")
        for dest in self.dest:
            _logger.debug(ind2 + str(dest))

        _logger.debug(ind1 + "Edges:")
        for edge in self.obj:
            _logger.debug(ind2 + str(edge))

    def _start_core_coord_repr(self) -> str:
        return _1st_core_coord_repr(self.core_coords)


@dataclass
class SliceDest:
    """Used to represent the destination details of a `NeuronSlice`."""

    dest_chip_coord: ChipCoord
    dest_axon: AxonSegment
    timeslot: int
    rt_mode: CoreMode
    dest_coords: list[Coord] = field(default_factory=list)
    rid: RId = field(init=False, repr=False)
    base_coord: Coord = field(init=False, repr=False)

    def set_rid(self) -> None:
        if len(self.dest_coords) == 0:
            raise ValueError("No destination coordinates.")

        self.rid = get_replication_id(self.dest_coords)
        # TODO will add a direction method to calculate this
        x = self.dest_coords[0].x & (~self.rid.x)
        y = self.dest_coords[0].y & (~self.rid.y)
        self.base_coord = Coord(x, y)

    def __str__(self) -> str:
        _repr = f"chip addr: {self.dest_chip_coord}\n"
        # fmt: off
        if hasattr(self, "base_coord") and hasattr(self, "rid"):
            _repr += f"core addr: {_coord_to_bin_str(self.base_coord)}\n" + \
                     f"multicast: {_coord_to_bin_str(self.rid)}\n"

        _repr +=     f"axon addr: {self.dest_axon}\n"
        # fmt: on
        return _repr


def _check_dest_attrs_same(sl_dest: SliceDest, *args) -> None:
    """Check if the attributes of the destination details are the same as the given arguments."""
    recorded = (
        sl_dest.dest_chip_coord,
        sl_dest.dest_axon,
        sl_dest.timeslot,
        sl_dest.rt_mode,
    )

    for r, arg in zip(recorded, args):
        if r != arg:
            raise ValueError(
                f"The attributes of the destination are not equal: {r} != {arg}."
            )


class NeuSegDestPair(NamedTuple):
    """Used to record the neuron segment & corresponding slice destination details for the original     \
        `NeuSegment`. Read only.
    """

    neu_seg: NeuSegment
    dest: SliceDest


@dataclass
class SliceDestPair:
    """Slice-destination pair."""

    slice_: PrttnSliceType
    dest: SliceDest


class SourceDest(UserList[SliceDestPair]):
    """Used to represent the destination details of an entire neuron node. Since a neuron node may be   \
        divided into multiple `NeuronSlice`, it contains a list consisting of slice-destination pairs.

        It provides a method to obtain the destination details of the specified `NeuSegment`.
    """

    def add_dest(
        self, dest_slice: SourceSliceType, dest_ax_seg: AxonSegment, cb: CoreBlock
    ) -> None:
        """Using the information of core block `cb` where the axon segment is, record the slice info & destination details."""
        dest_coords = cb.core_coords.copy()
        dest_chip_coord = cb.chip_coord
        timeslot = cb.n_timeslot
        mode = cb.rt_mode

        if dest_slice.index not in self.slices:
            # Add the destination slice in record.
            d = SliceDest(dest_chip_coord, dest_ax_seg, timeslot, mode, dest_coords)
            self.append(SliceDestPair(dest_slice.index, d))
        else:
            # When the destination slice has been recorded, the info of the destination axon segment &
            # the core block where it's located also needs to be the same as the recorded info.
            idx = self.slices.index(dest_slice.index)
            d = self.dests[idx]
            _check_dest_attrs_same(d, dest_chip_coord, dest_ax_seg, timeslot, mode)
            # In this case, only the core coordinates of the core blocks where the destination slice
            # is located are append to the recorded list.
            d.dest_coords.extend(dest_coords)

    def set_slice_dest_rid(self) -> None:
        for d in self.dests:
            d.set_rid()

    def sort_slice_dest_pairs(self) -> None:
        """Sort the slice-destination pairs by the start position of the slice."""
        self.sort(key=lambda p: p.slice_.start)

    def is_undivided_dest(self) -> SliceDest:
        if len(self) > 1:
            raise ValueError("Multiple destinations")

        return self[0].dest

    def get_slice_dest_pairs(self, neu_seg: NeuSegment) -> list[NeuSegDestPair]:
        """According to the given neuron segment, find the corresponding neuron segments & slice destination details.

        Returns:
            A list of `NeuSegDestPair` containing the neuron segments and their corresponding destination details.
        """
        pairs: list[NeuSegDestPair] = []
        start = neu_seg.index.start
        stop = neu_seg.index.stop
        cur_start = start

        if len(self) == 0:
            raise ValueError(f"No destination information for {neu_seg}.")

        for i, pos in enumerate(s.stop for s in self.slices):
            if pos <= start:
                continue

            elif start < pos < stop:
                pairs.append(
                    NeuSegDestPair(
                        neu_seg[cur_start - start : pos - start], self.dests[i]
                    )
                )
                cur_start = pos

            elif pos >= stop:
                pairs.append(
                    NeuSegDestPair(
                        neu_seg[cur_start - start : stop - start], self.dests[i]
                    )
                )
                break  # No need to traverse the rest.

        return pairs

    @property
    def slices(self) -> list[PrttnSliceType]:
        """Return all the slices in the slice-destination pair."""
        return [pair.slice_ for pair in self]

    @property
    def dests(self) -> list[SliceDest]:
        """Return all the destination details in the slice-destination pair."""
        return [pair.dest for pair in self]

    def __str__(self) -> str:
        _repr = ""
        for pairs in self:
            # Align with the content of the destination details
            _repr += f"slice    : ({pairs.slice_.start},{pairs.slice_.stop})\n"
            _repr += str(pairs.dest)

        return _repr


class CorePlacement(CoreAbstract):
    parent: CoreBlock
    coord: Coord
    """Routing coordinate"""
    n_neuron: int
    raw_weights: list[WeightType]
    """The folded weights."""
    neu_segs_of_cplm: NeuSegOfCorePlm
    neu_configs: dict[Neuron, NeuronConfig]

    WRAM_BASE_SHAPE: ClassVar[tuple[int, int]] = (
        HwConfig.ADDR_AXON_MAX + 1,
        HwConfig.ADDR_RAM_MAX + 1,
    )
    """The base shape of weight RAM."""

    N_U64_ON_WRAM_ADDR: ClassVar[int] = WRAM_BASE_SHAPE[0] // N_BIT_PACKED_WEIGHT
    """The number of u64 at each address of weight RAM."""

    def __init__(
        self,
        parent: CoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        raw_weights: list[WeightType],
        neu_segs_of_cplm: NeuSegOfCorePlm,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent core block.
            - idx: The index number where this object is located.
            - n_neuron: the number of neurons used in the physical core.
            - raw_weights: the raw weights in the physical core.
            - neu_segs_of_cplm: The segment of the neurons in the physical core.
        """
        super().__init__(name)
        self.parent = parent
        self.rt_mode = parent.rt_mode
        self.coord = routing_coord
        self.n_neuron = n_neuron
        self.raw_weights = raw_weights
        self.neu_segs_of_cplm = neu_segs_of_cplm
        self.neu_configs = dict()

    @classmethod
    def build(cls, parent: CoreBlock, idx: int):
        coord = parent.core_coords[idx]
        n_neuron = parent.n_neuron_of_plm[idx]
        neu_segs_of_cplm = parent.neuron_segs_of_cb[idx]
        raw_weights = parent.get_raw_weight_of_coord(idx)

        return cls(parent, coord, n_neuron, raw_weights, neu_segs_of_cplm)

    def _fold_raw_weights(self, raw_weights: list[WeightType]) -> WeightType:
        """Fold the weights into LCN-sized blocks."""
        w_folded_list = []
        w_folded_of_axon_segs = []
        n_fold = self.n_timeslot

        if self.lcn_ex == LCN_EX.LCN_1X:
            return np.hstack(raw_weights)

        # LCN_EX > LCN_1X
        for raw_weight in raw_weights:
            w_folded_of_axon_segs.clear()
            _n_axon_last = 0

            for s in self.source:
                axon_seg = self.parent.axon_segments[s]

                # Retrive the weight of the axon segment
                w_of_axon_seg = raw_weight[
                    _n_axon_last : _n_axon_last + axon_seg.n_axon, :
                ]
                _n_axon_last += axon_seg.n_axon

                # Fold the weight of axon segment
                w_folded_of_axon_seg = self._nfold_weight(
                    w_of_axon_seg, axon_seg.addr_width, n_fold
                )
                w_folded_of_axon_segs.append(w_folded_of_axon_seg)

            w_folded = np.vstack(w_folded_of_axon_segs)
            w_folded_list.append(w_folded)

        return np.hstack(w_folded_list)

    def _weight_ram_mapping(self) -> WRAMPackedType:
        """Map the raw weights to the weight RAM(WRAM). The mapping is different for 1 & 8-bit input widths.

        NOTE: When the input width is 1-bit, no neurons need to be mapped to the WRAM. When the input width is 8-bit,   \
            some neurons may be mapped to the WRAM when the #N of neurons inside the core placement > 512.

            This function was tested using only the prototype functions. For test items, please refer to                \
            tests/backend/test_placement.py::TestWeightRamMapping for details.

        Returns:
            The packed matrix of weights mapped to the WRAM, with shape (x, N_U64_ON_WRAM_ADDR) uint64 (x <= 512). The  \
            entire WRAM contains up to 4 parts: the mapped & unallocated part for weights & neuron parameters.          \
            For example,

            W1 = W[:x1  ,:]: the mapped part for weights.
            W2 = W[x1:x2,:]: the unallocated part for weights(0).
            W3 = W[x2:x3,:]: the mapped part for neurons parameters.
            W4 = W[x3:  ,:]: the unallocated part for neurons parameters(0). Since it is at the end of WRAM, we don't   \
                care about it.

            0 < x1 < x2 < x3 <= 512.

            This function only processes the weight part, that is, returns W1+W2 = W[:x2,:].
        """
        w_folded = self._fold_raw_weights(self.raw_weights)
        folded_row, _ = w_folded.shape

        iw = 8 if is_iw8(self.rt_mode) else 1
        n_dendrite_comb = 1 << self.dendrite_comb_rate
        # oc * e / (8/w) = oc * d / 8
        orig_col = self.n_neuron
        result_col = math.ceil(orig_col * n_dendrite_comb / iw)
        # Units are divided into small blocks of columns, fan-in extension
        cew_block = np.zeros(
            (orig_col, self.n_timeslot, self.n_weight_bits, self.parent.n_fanin_base),
            dtype=WRAM_UNPACKED_DTYPE,
        )

        # (N, M)(int8) -> (M, N, 1)(uint8)
        w_folded_3d = np.expand_dims(w_folded.T, axis=2).view(WRAM_UNPACKED_DTYPE)
        for c in range(orig_col):
            for lcn in range(self.n_timeslot):
                # For every column, unpack the array (N, 1) -> (N, n_weight_bits)
                unpacked = np.unpackbits(
                    w_folded_3d[c * self.n_timeslot + lcn, :, :],
                    axis=1,
                    count=self.n_weight_bits,
                    bitorder=HwConfig.WEIGHT_BITORDER,
                )

                for bit in range(self.n_weight_bits):
                    cew_block[c, lcn, bit, :folded_row] = unpacked[:, bit].squeeze()

        if n_dendrite_comb >= iw:  # For 1-bit input width, it must go into this case
            # At least 1 fan-in is required to be combined in one column
            w_mapped = cew_block.reshape((result_col, -1)).T
        else:
            # 2/4/8 original columns are combined in one column
            n_col_comb_in_col = iw // n_dendrite_comb
            cew_block = cew_block.reshape((orig_col, -1))

            if (r := orig_col % n_col_comb_in_col) > 0:
                cew_block = np.pad(cew_block, ((0, n_col_comb_in_col - r), (0, 0)))

            # Now, length of padded columns is a multiple of 'n_col_comb_in_col'
            w_mapped = cew_block.reshape(
                (cew_block.shape[0] // n_col_comb_in_col, -1)
            ).T

        wram_packed = self._weight_pack(w_mapped)

        # Available columns for weight mapping to the WRAM.
        if iw == 1:
            n_col_weight_on_wram = CorePlacement.WRAM_BASE_SHAPE[1]
        else:
            n_144b_dendrites = (
                FANOUT_IW8[self.dendrite_comb_rate] << self.dendrite_comb_rate
            )
            n_col_weight_on_wram = n_144b_dendrites // iw

        # The mapped & unallocated part for weights, W1+W2
        wram_weight_packed = np.zeros(
            (n_col_weight_on_wram, CorePlacement.N_U64_ON_WRAM_ADDR),
            dtype=WRAM_PACKED_DTYPE,
        )
        wram_weight_packed[: wram_packed.shape[0], :] = wram_packed
        wram_weight_packed.setflags(write=False)

        return wram_weight_packed

    @staticmethod
    def _nfold_weight(
        raw_weight: WeightType, expected_row: int, n_fold: int
    ) -> WeightType:
        """Fold the weight matrix according to the folding ratio.

        Args:
            raw_weight: the raw weight matrix.
            expected_row: the expected #N of row.
            n_fold: the folding ratio (1 << LCN).
        """
        raw_row, raw_col = raw_weight.shape
        n_row_folded, r = divmod(raw_row, n_fold)  # #N of rows after folding

        if r > 0:
            n_row_folded += 1
            _raw_weight = np.pad(raw_weight, ((0, n_fold - r), (0, 0)))
        else:
            _raw_weight = raw_weight

        w_splited = np.vsplit(_raw_weight, n_fold)
        w_folded = np.zeros((expected_row, raw_col * n_fold), dtype=WEIGHT_DTYPE)

        for i, j in np.ndindex((n_fold, raw_col)):
            w_col = w_splited[i][:, j]
            w_folded[:n_row_folded, j * n_fold + i] = w_col

        return w_folded

    @staticmethod
    def _weight_pack(w_unpacked: WRAMUnpackedType) -> WRAMPackedType:
        """Convert the unpacked weights into a mapping form, corresponding to the WRAM address. Each address contains   \
            uint64.
            (1152, x) -> (x, 1152) -> (x*18, 64) -> (x*18, 8) uint8 -> (x*18, 1) uint64 -> (x, 18) uint64.

            TODO simpler (1152, x) -> (x, 1152) -> pack -> (x, 144) uint8 -> (x, 18) uint64.

        Returns:
            The packed matrix of weights with shape (x, 18) where x <= 512.
        """
        # Reshape to 64 columns to avoid contiguous problem.
        w_unpacked_aligned = w_unpacked.T.reshape((-1, N_BIT_PACKED_WEIGHT))
        # (x*18, 64) uint8 -> (x*18, 8) uint8
        w_packed_u8 = np.packbits(
            w_unpacked_aligned, axis=1, bitorder=HwConfig.WEIGHT_BITORDER
        )
        # (x*18, 8) uint8 -> (x*18, 1) uint64 -> (x, 18) uint64
        w_packed_u64 = w_packed_u8.view(WRAM_PACKED_DTYPE).reshape(
            (w_unpacked.shape[1], -1)
        )
        # TODO If everything is okay, use the simpler method as follows:
        # w_packed_u8 = np.packbits(
        #     w_unpacked.T, axis=1, bitorder=HwConfig.WEIGHT_BITORDER
        # )
        # w_packed_u64 = np.ascontiguousarray(w_packed_u8).view(WRAM_PACKED_DTYPE)
        w_packed_u64.setflags(write=False)

        # TODO If the assertion is useless, remove it.
        assert w_packed_u64.shape[1] == CorePlacement.N_U64_ON_WRAM_ADDR
        return w_packed_u64

    @staticmethod
    def neu_params_mapping(neu_confs: list[NeuronConfig]) -> WRAMPackedType:
        """Map the extra neurons parameters to the WRAM. This only happens when the input width is 8 bits.

        NOTE: This function was tested using only the prototype functions. For test items, please refer to              \
            `tests/backend/test_placement.py::TestWeightRamMapping` for details.

        Returns:
            The packed matrix W3 with shape (L, 18) where L is the used columns for mapping neurons parameters. See     \
            details in function `_weight_ram_mapping`.
        """
        neu_conf_params_list: list[WRAMUnpackedType] = []

        for neu_conf in neu_confs:
            neu_conf_params = np.zeros(
                (neu_conf.neu_seg.n_neuron, NEURON_PARAMS_BIT_LENGTH),
                dtype=WRAM_UNPACKED_DTYPE,
            )

            # Only the packges will be used.
            frame3 = OfflineFrameGen.gen_config_frame3(
                _COORD_UNSET,
                _COORD_UNSET,
                _RID_UNSET,
                0,
                neu_conf.neu_seg.n_neuron,
                neu_conf.neuron_attrs,
                neu_conf.neuron_dest_info,
                1,
            )

            for i in range(neu_conf.neu_seg.n_neuron):
                params = frame3.packages[i * 4 : (i + 1) * 4]
                neu_conf_params[i, :] = np.unpackbits(
                    params.view(WRAM_UNPACKED_DTYPE),
                    axis=0,
                    bitorder=HwConfig.WEIGHT_BITORDER,
                )[:NEURON_PARAMS_BIT_LENGTH]

            neu_conf_params_list.append(neu_conf_params)

        neu_params = np.vstack(neu_conf_params_list)

        N_NEURON_PARAM_IN_COL = (
            CorePlacement.WRAM_BASE_SHAPE[0] // NEURON_PARAMS_BIT_LENGTH
        )
        n_col_occupied, r = divmod(neu_params.shape[0], N_NEURON_PARAM_IN_COL)
        if r > 0:
            n_col_occupied += 1
            neu_params = np.pad(neu_params, ((0, N_NEURON_PARAM_IN_COL - r), (0, 0)))

        neu_params = neu_params.reshape((n_col_occupied, -1))

        # (1152, y)
        result = np.zeros(
            (CorePlacement.WRAM_BASE_SHAPE[0], n_col_occupied),
            dtype=WRAM_UNPACKED_DTYPE,
        )
        _n_bit_nparams = NEURON_PARAMS_BIT_LENGTH * N_NEURON_PARAM_IN_COL
        result[:_n_bit_nparams, :] = neu_params.T

        # (1152, y) -> (y, 18)
        return CorePlacement._weight_pack(result)

    def export_param_config(self) -> CoreConfig:
        _mode_params = self.rt_mode.conf

        # fmt: off
        cb_config = CoreConfig(
            self.name,                          # name of the core
            self.weight_width,                  # weight_precision
            self.lcn_ex,                        # lcn_extension
            _mode_params[0],                    # input_width_format
            _mode_params[1],                    # spike_width_format
            self.n_working_dendrite,            # num_dendrite
            self.pool_max,                      # max_pooling_en
            self.tws,                           # tick_wait_start
            self.twe,                           # tick_wait_end
            _mode_params[2],                    # snn_mode_en
            self.target_lcn,                    # target_lcn
            _BACKEND_CONTEXT.test_chip_addr,    # test_chip_addr
        )
        # fmt: on
        return cb_config

    @overload
    def export_neu_config(
        self, neu_seg: NeuSegment, source_dest: SourceDest
    ) -> None: ...

    @overload
    def export_neu_config(
        self, neu_seg: NeuSegment, *, output_core_coord: Coord
    ) -> None: ...

    def export_neu_config(
        self,
        neu_seg: NeuSegment,
        source_dest: Optional[SourceDest] = None,
        output_core_coord: Optional[Coord] = None,
    ) -> None:
        """Export the neuron configuration."""
        if isinstance(source_dest, SourceDest):
            neu_seg_dest_pairs = source_dest.get_slice_dest_pairs(neu_seg)
            for seg, dest in neu_seg_dest_pairs:
                axon_coords = aligned_coords(
                    seg.index,
                    dest.dest_axon,
                    seg.target.delay_relative,
                    dest.timeslot,
                    is_iw8(dest.rt_mode),
                )
                config = NeuronConfig(
                    seg, axon_coords, dest.dest_coords, dest.dest_chip_coord
                )
                self.neu_configs[seg.target] = config
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)
            # TODO Only leverage the axon coordinate attributes in `AxonCoord` and do not use the
            # `tick_relative` attribute, which causes the number of an output node cannot be
            # greater than `N_FANIN_PER_DENDRITE_MAX`(=1152).
            axon_coords = [
                AxonCoord.build(0, i)
                for i in range(neu_seg.index.start, neu_seg.index.stop)
            ]

            config = NeuronConfig(
                neu_seg,
                axon_coords,
                [output_core_coord],
                # output chip coordinate for output node
                _BACKEND_CONTEXT["output_chip_addr"],
            )

            self.neu_configs[neu_seg.target] = config

    def export_core_plm_config(self) -> CorePlmConfig:
        core_param = self.export_param_config()
        return CorePlmConfig.encapsulate(
            self.parent.seed, self.weight_ram, core_param, self.neu_configs
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.source), len(self.dest))

    @property
    def weight_width(self) -> WW:
        return self.parent.weight_width

    @property
    def n_weight_bits(self) -> int:
        return self.parent.n_weight_bits

    @property
    def n_timeslot(self) -> int:
        return self.parent.n_timeslot

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
    def dendrite_comb_rate(self) -> int:
        return self.parent.dendrite_comb_rate

    @property
    def tws(self) -> int:
        return self.parent.tws

    @property
    def twe(self) -> int:
        return self.parent.twe

    @property
    def pool_max(self) -> MaxPoolingEnable:
        return self.parent.pool_max

    @property
    def n_working_dendrite(self) -> int:
        """The number of actual working dendrites.

        NOTE: n_neuron * (2^comb_rate) = n_neuron << comb_rate
        """
        return self.n_neuron << self.dendrite_comb_rate

    @property
    def source(self) -> list[SourceSliceType]:
        return self.parent.ordered_axons

    @property
    def dest(self) -> list[DestNodeType]:
        """The destination nodes within it.

        NOTE: This attribute is different from the one of its parent.
        """
        return [p.target for p in self.neu_segs_of_cplm]

    @property
    def weight_ram(self) -> WRAMPackedType:
        return self._weight_ram_mapping()

    @property
    def n_core_required(self):
        return 1

    def __len__(self) -> int:
        return self.n_core_required


class EmptyCorePlacement(CoreAbstract):
    """Empty core placement."""

    _EMPTY_WRAM: int = 0

    def __init__(self, coord: Coord, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.coord = coord

    def export_param_config(self) -> CoreConfig:
        _mode_params = CoreMode.MODE_SNN.conf

        # fmt: off
        cb_config = CoreConfig(
            self.name,                          # name of the core
            WW.WEIGHT_WIDTH_1BIT,               # weight_precision
            LCN_EX.LCN_1X,                      # lcn_extension
            _mode_params[0],                    # input_width_format
            _mode_params[1],                    # spike_width_format
            0,                                  # num_dendrite
            MaxPoolingEnable.DISABLE,           # max_pooling_en
            0,                                  # tick_wait_start
            0,                                  # tick_wait_end
            _mode_params[2],                    # snn_mode_en
            LCN_EX.LCN_1X,                      # target_lcn
            _BACKEND_CONTEXT.test_chip_addr,    # test_chip_addr
        )
        # fmt: on
        return cb_config

    def export_core_plm_config(self) -> CorePlmConfig:
        core_param = self.export_param_config()
        # For empty core placements, we don't care random seed, WRAM & neurons cfg.
        return CorePlmConfig.encapsulate(0, self._EMPTY_WRAM, core_param, {})  # type: ignore

    @classmethod
    def build(cls, coord: Coord):
        return cls(coord)

    @property
    def n_core_required(self) -> int:
        return 1


def max_lcn_of_cb(cb: list[CoreBlock]) -> LCN_EX:
    """Get the max LCN extenion of given core blocks."""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex
