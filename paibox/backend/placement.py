import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast, overload

import numpy as np
from numpy.typing import NDArray
from paicorelib import (
    LCN_EX,
    ChipCoord,
    Coord,
    CoreMode,
    DecayRandomEnable,
    HwConfig,
    LeakOrder,
    MaxPoolingEnable,
    OffCoreCfg,
    OnCoreCfg,
    OnlineModeEnable,
)
from paicorelib import ReplicationId as RId
from paicorelib import WeightWidth as WW
from paicorelib.framelib import OfflineFrameGen
from paicorelib.framelib.types import LUT_DTYPE, LUTDataType
from paicorelib.routing_defs import get_replication_id

from paibox import _logging
from paibox.base import PAIBoxObject
from paibox.components import Neuron, OnlineNeuron
from paibox.components.synapses.lut import LUT_LEN
from paibox.exceptions import GraphBuildError, ResourceError, TruncationWarning
from paibox.types import WEIGHT_DTYPE, WeightType
from paibox.utils import check_attr_same

from .conf_types import (
    CoreConfig,
    CoreConfInChip,
    CorePlmConfig,
    NeuConfig,
    OfflineCoreConfig,
    OfflineCorePlmConfig,
    OfflineNeuConfig,
    OnlineCoreConfig,
    OnlineCorePlmConfig,
    OnlineNeuConfig,
)
from .context import _BACKEND_CONTEXT
from .segment_utils import get_axon_segments, get_dendrite_segments
from .sub_utils import SubDestType, SubEdge, SubSourceType, list_to_str
from .types import (
    _COORD_UNSET,
    _RID_UNSET,
    N_BIT_PACKED_WEIGHT,
    WRAM_PACKED_DTYPE,
    WRAM_UNPACKED_DTYPE,
    AxonCoord,
    AxonSegment,
    CoreAllocationOfCoreBlock,
    CustomIndex,
    DendriteSegment,
    DestNodeType,
    SubNeuOfCorePlm,
    WRAMPackedType,
    WRAMUnpackedType,
    _1st_core_coord_repr,
    _coord_to_bin_str,
    is_iw8,
)

cb_log = _logging.get_artifact_logger(__name__, "core_block_info")

# Get the fan-out by the combination rate of dendrites
if hasattr(OffCoreCfg, "FANOUT_IW8"):
    FANOUT_IW8 = OffCoreCfg.FANOUT_IW8
else:
    FANOUT_IW8 = [OffCoreCfg.N_NEURON_MAX_ANN, 1364, 876, 512, 256, 128, 64, 32, 16, 8]


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
    _parents: tuple[SubEdge, ...]
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
    axon_segments: dict[SubSourceType, AxonSegment] = dict()
    """A dictionary of segments of each axon(source node)."""
    core_allocation_of_cb: CoreAllocationOfCoreBlock = []
    """Neuron segments in the core block. Each element in the list represents the neuron    \
        segments in core placement.
    """
    online: bool = False

    def __init__(self, *parents: SubEdge, seed: int, name: str | None = None) -> None:
        """Core blocks in SNN mode.

        Args:
            parents: the parent synapses.
            seed: random seed. Default value is 0.
            mode: runtime mode of the core block.
        """
        super().__init__(name)
        self._parents = parents
        self.rt_mode = CoreMode.MODE_SNN
        self.seed = seed
        self._lcn_ex = LCN_EX.LCN_1X

        self.target_lcn = LCN_EX.LCN_1X
        self.core_coords = []
        self.chip_coord = _COORD_UNSET
        self.core_placements = dict()
        self.axon_segments = dict()
        self.core_allocation_of_cb = []
        self._ordered_axons: list[SubSourceType] = []
        """Axons in global + private types order."""

        self._lcn_locked = False
        self._neurons_grouped = False

    @abstractmethod
    def core_plm_alloc(self) -> None:
        pass

    @property
    @abstractmethod
    def n_fanin_base(self) -> int:
        pass

    @property
    @abstractmethod
    def n_fanout(self) -> int:
        pass

    @property
    @abstractmethod
    def n_neuron_repl(self) -> int:
        pass

    @classmethod
    def build(
        cls,
        *synapses: SubEdge,
        online: bool = False,
        seed: int = 0,
        rt_mode: CoreMode = CoreMode.MODE_SNN,
    ):
        """Single CoreBlock should not contain different SubNeurons from the same Neuron.
        if there are different SubNeurons should be merged into one SubNeuron before building CoreBlock.
        """
        sub_neus: dict[Neuron, set[CustomIndex]] = dict()
        for syn in synapses:
            sub_neu = syn.dest
            if sub_neu.target in sub_neus:
                if not sub_neus[sub_neu.target] == sub_neu.custom_index_set:
                    raise ValueError(
                        f"Single CoreBlock cannot contain different SubNeurons from the same Neuron.\n"
                        f"but got {sub_neu.target.name}[{list_to_str(list(sub_neu.custom_index_set))}]\n"
                        f"and {sub_neu.target.name}[{list_to_str(list(sub_neus[sub_neu.target]))}]"
                    )
            else:
                sub_neus[sub_neu.target] = sub_neu.custom_index_set

        """Group synapses & build `CoreBlock`."""
        if online:
            return OnlineCoreBlock.build(*synapses, seed=seed)
        else:
            return OfflineCoreBlock.build(*synapses, rt_mode=rt_mode, seed=seed)

    def group_neurons(
        self, optim_target: Literal["latency", "core", "both"] = "both"
    ) -> None:
        """Group the neurons to determine the #N of cores required."""
        if not self._lcn_locked:
            raise GraphBuildError("group the neurons after 'lcn_ex' is locked.")

        self.core_allocation_of_cb = get_dendrite_segments(
            self.dest, self.n_fanout, self.n_neuron_repl, optim_target
        )

        self._neurons_grouped = True

    def _get_syn_of(self, src: SubSourceType, dest: SubDestType) -> SubEdge | None:
        for syn in self.obj:
            if syn.source == src and syn.dest == dest:
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
        ) > self.max_lcn_ex:
            _max_n_axons = self.n_fanin_base << self.max_lcn_ex
            raise ResourceError(
                f"required LCN out of range {self.max_lcn_ex} ({lcn}). The number of axons "
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
    def obj(self) -> tuple[SubEdge, ...]:
        return self._parents

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ordered_axons), len(self.dest))

    @property
    def source(self) -> list[SubSourceType]:
        """Ordered unique source nodes."""
        return list(set([parent.source for parent in self.obj]))

    @property
    def axons(self) -> list[SubSourceType]:
        return self.source

    @property
    def dest(self) -> list[SubDestType]:
        """Ordered unique destination nodes."""
        return list(set([parent.dest for parent in self.obj]))

    def n_axon_of(self, index: int) -> int:
        """Get the #N of axons of `index`-th source neuron."""
        return self.ordered_axons[index].num_out

    """Boundary limitations"""

    @property
    def n_core_required(self) -> int:
        return len(self.core_allocation_of_cb)

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

    @property
    def max_lcn_ex(self) -> LCN_EX:
        return LCN_EX.LCN_64X if not self.online else LCN_EX.LCN_8X

    @lcn_ex.setter
    def lcn_ex(self, lcn_ex: LCN_EX) -> None:
        # if self._lcn_locked:
        #     raise LockedAttrOverwriteError("`lcn_ex` has been locked.")

        if lcn_ex > self.max_lcn_ex:
            raise ValueError(f"required LCN out of range {self.max_lcn_ex} ({lcn_ex}).")

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
    def n_axon(self) -> int:
        return sum(s.num_in for s in self.ordered_axons)

    @property
    def n_neuron(self) -> int:
        return sum(d.num_out for d in self.dest)

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
            for neuron_segs in self.core_allocation_of_cb
        ]

    @property
    def ordered_axons(self) -> list[SubSourceType]:
        return self._ordered_axons

    @ordered_axons.setter
    def ordered_axons(self, axons: list[SubSourceType]) -> None:
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

    def get_raw_weight_of_coord(self, idx: int) -> WeightType:
        """Get the corresponding part of the original weight matrix corresponding to each CP."""
        w_of_sub_neus: list[WeightType] = []
        _idx = 0
        sub_slice = slice(0, 0)

        for sub_neu in self.core_allocation_of_cb[idx]:
            _not_covered = True
            sub_neu_index_set = set(sub_neu.index)

            for i, dest_sub_neu in enumerate(self.dest):
                # Get the corresponding part of `neu_seg` on the slice of `dest_sl`.
                if sub_neu_index_set.issubset(dest_sub_neu.custom_index_set):
                    _not_covered = False
                    _idx = i
                    # the sub_neu is a continuous slice of dest_sub_neu, dicided by _coarse_group()
                    sub_slice = slice(
                        dest_sub_neu.index.index(sub_neu.index[0]),
                        dest_sub_neu.index.index(sub_neu.index[-1]) + 1,
                    )

                    if dest_sub_neu.index[sub_slice] != sub_neu.index:
                        raise ValueError(
                            f"neuron segment index mismatch: {dest_sub_neu.index[sub_slice]} != {sub_neu.index}"
                        )
                    break

            assert (
                not _not_covered
            ), f"neuron segment {sub_neu} is not covered by any dest: {self.dest}."

            w_of_dest = self.raw_weight_of_dest[_idx]
            w_of_sub_neu = w_of_dest[:, sub_slice].copy()
            w_of_sub_neu.setflags(write=False)
            w_of_sub_neus.append(w_of_sub_neu)

        raw_weight = np.hstack(w_of_sub_neus)
        raw_weight = np.pad(
            raw_weight,
            (
                (0, self.n_fanin_base * self.n_timeslot - raw_weight.shape[0]),
                (0, self.n_fanout - raw_weight.shape[1]),
            ),
            "constant",
            constant_values=0,
        )

        return raw_weight

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
        _repr += ind1 + f"mode: {self.rt_mode.name}\n"

        _repr += ind1 + "dests:\n"
        for dest in self.dest:
            _repr += ind2 + str(dest) + "\n"

        _repr += ind1 + "cores:\n"
        for i, neu_seg in enumerate(self.core_allocation_of_cb):
            _repr += ind2 + f"#{i}:\n"
            for seg in neu_seg:
                _repr += ind2 + f"\t{seg.target.name}{list_to_str(seg.index)}\n"

        return _repr

    @property
    def _obj_repr(self) -> str:
        """The representation of the names of target objects."""
        return ", ".join(str(n) for n in self.obj)

    def export_core_plm_config(self) -> CoreConfInChip:
        """Export the parameters of the core into a dictionary."""
        cb_config: CoreConfInChip = dict()

        for coord, core_plm in self.core_placements.items():
            cb_config[coord] = core_plm.export_core_config()

        return cb_config

    def dump(
        self, indents: int = 0, father_logger: logging.Logger | None = None
    ) -> None:
        _logger = cb_log if father_logger is None else father_logger

        tabs = "\t" * indents
        ind1 = tabs + "\t"
        ind2 = tabs + "\t\t"

        _logger.debug(tabs + f"{self.name} ({self.n_core_required} cores):")
        _logger.debug(ind1 + f"LCN: {self.lcn_ex}")
        _logger.debug(ind1 + f"Weight width: {self.weight_width}")
        _logger.debug(ind1 + f"fan_out: {self.n_fanout}")
        _logger.debug(ind1 + f"Online: {self.online}")
        _logger.debug(ind1 + f"Mode: {self.rt_mode.name}")

        _logger.debug(ind1 + "Axons:")
        for axon in self.ordered_axons:
            _logger.debug(ind2 + str(axon))

        _logger.debug(ind1 + "Dests:")
        for dest in self.dest:
            _logger.debug(ind2 + str(dest))

        _logger.debug(ind1 + "Edges:")
        for edge in self.obj:
            _logger.debug(ind2 + str(edge))

        if indents == 0:
            _logger.debug("")

    def _start_core_coord_repr(self) -> str:
        return _1st_core_coord_repr(self.core_coords)


class OfflineCoreBlock(CoreBlock):
    def __init__(
        self, *parents: SubEdge, seed: int, mode: CoreMode, name: str | None = None
    ) -> None:
        super().__init__(*parents, seed=seed, name=name)
        self.rt_mode = mode
        self.online = False

    def core_plm_alloc(self) -> None:
        """Allocate `CoreBlock` to physical cores."""
        if not self._lcn_locked:
            raise GraphBuildError("allocate core placements after 'lcn_ex' is locked.")

        for i, coord in enumerate(self.core_coords):
            self.core_placements[coord] = OfflineCorePlacement.build(self, i)

    @property
    def n_fanin_base(self) -> int:
        """The fan-in of cores."""
        return (
            OffCoreCfg.N_FANIN_PER_DENDRITE_SNN
            if self.rt_mode.is_snn
            else OffCoreCfg.N_FANIN_PER_DENDRITE_ANN
        )

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
    def n_fanout(self) -> int:
        """The fan-out of cores."""
        return (
            OffCoreCfg.N_DENDRITE_MAX_SNN >> self.dendrite_comb_rate
            if self.rt_mode.is_snn
            else FANOUT_IW8[self.dendrite_comb_rate]
        )

    @property
    def n_neuron_repl(self) -> int:
        """The number of neurons that need to be repeatedly placed on the neuron address space.

        For example, in SNN mode, N[0:99] with LCN_2X & WW8:
            Neuron address:  [0]     [1]    ...     [15]    [16]    [17]    ...      [31] ...
                            N[0]    N[0]    ...     N[0]    N[1]    N[1]    ...      N[1] ...
                             |<- repeatedly placed ->|       |<-- repeatedly placed -->|

        But in ANN mode(8-bit input width), neurons don't need to be placed repeatedly.
            Neuron address:   [0]    [1]    ...      [15]    [16]   ...
                             N[0]   N[1]    ...     N[15]   N[16]   ...
        """
        return 1 << self.dendrite_comb_rate if self.rt_mode.is_snn else 1

    @classmethod
    def build(cls, *synapses: SubEdge, rt_mode: CoreMode, seed: int = 0):
        """Group synapses & build `CoreBlock`."""
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )
        return cls(*synapses, mode=rt_mode, seed=seed)


class OnlineCoreBlock(CoreBlock):
    """A core block in online mode."""

    """The input width and spike width are both 1-bit fixed."""

    def __init__(self, *parents: SubEdge, seed: int, name: str | None = None) -> None:
        super().__init__(*parents, seed=seed, name=name)
        self.online = True
        self.inhi_rid = Coord(0, 0)

    def core_plm_alloc(self) -> None:
        """Allocate `CoreBlock` to physical cores."""
        if not self._lcn_locked:
            raise GraphBuildError("allocate core placements after 'lcn_ex' is locked.")

        for i, coord in enumerate(self.core_coords):
            self.core_placements[coord] = OnlineCorePlacement.build(self, i)

    @property
    def n_fanin_base(self) -> int:
        """The fan-in of cores."""

        """online mode is not supported in CoreMode yet"""
        return OnCoreCfg.N_FANIN_PER_DENDRITE_MAX

    @property
    def n_fanout(self) -> int:
        """The fan-out of cores."""
        return OnCoreCfg.N_DENDRITE_MAX >> self.dendrite_comb_rate

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
        return 1 << self.dendrite_comb_rate

    @classmethod
    def build(cls, *synapses: SubEdge, seed: int = 1):
        """Group synapses & build `CoreBlock`."""
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )
        elif seed == 0:
            seed = 1

        return cls(*synapses, seed=seed)

    @property
    def first_neuron(self) -> OnlineNeuron:
        online_neu = cast(OnlineNeuron, self.dest[0].target)
        return online_neu

    @property
    def laterl_inhi_target(self) -> set[OnlineNeuron]:
        return self.first_neuron.lateral_inhi_target

    @property
    def laterl_inhi_source(self) -> set[OnlineNeuron]:
        return self.first_neuron.lateral_inhi_source

    @property
    def lateral_inhi_value(self) -> int:
        return self.first_neuron.lateral_inhi_value

    @property
    def weight_decay_value(self) -> int:
        return int(self.first_neuron.weight_decay_value)

    @property
    def upper_weight(self) -> int:
        return self.first_neuron.upper_weight

    @property
    def lower_weight(self) -> int:
        return self.first_neuron.lower_weight

    @property
    def inhi_core_x_ex(self) -> int:
        return self.inhi_rid.x

    @property
    def inhi_core_y_ex(self) -> int:
        return self.inhi_rid.y

    @property
    def lut_random_en(self) -> NDArray[np.uint8]:
        return self.first_neuron.lut_random_en

    @property
    def decay_random_en(self) -> DecayRandomEnable:
        return self.first_neuron.decay_random_en

    @property
    def leak_order(self) -> LeakOrder:
        return self.first_neuron.leak_comparison

    @property
    def online_mode_en(self) -> OnlineModeEnable:
        return self.first_neuron.online_mode_en

    @property
    def lut(self) -> LUTDataType:
        return self.first_neuron.lut

    @property
    def random_seed(self) -> int:
        """Random seed, legal integer, no more than uint64."""
        return self.seed


@dataclass
class DestCoreInfo:
    dest_chip_coord: ChipCoord
    timeslot: int
    dest_coords: list[Coord] = field(default_factory=list)
    rid: RId = field(init=False, repr=False)
    base_coord: Coord = field(init=False, repr=False)

    def set_rid(self) -> None:
        if len(self.dest_coords) == 0:
            raise ValueError("No destination coordinates.")

        self.base_coord, self.rid = get_replication_id(self.dest_coords)
        self.dest_coords = []  # Free memory

    def __hash__(self) -> int:
        """hash according to the attributes except `dest_axon`."""
        """so the DestInfo with same chip/core addr, timeslot and rid will have the same hash value."""
        return hash((self.dest_chip_coord, self.base_coord, self.timeslot, self.rid))

    def __eq__(self, other: "DestCoreInfo") -> bool:
        if not isinstance(other, DestCoreInfo):
            return NotImplemented

        return (
            self.dest_chip_coord == other.dest_chip_coord
            and self.base_coord == other.base_coord
            and self.timeslot == other.timeslot
            and self.rid == other.rid
        )


class DestInfo:
    """Used to represent the destination details of a single dendrite."""

    def __init__(
        self,
        dest_chip_coord: ChipCoord,
        dest_axon: AxonCoord,
        timeslot: int,
        dest_coords: list[Coord],
    ) -> None:
        self.dest_core_info = DestCoreInfo(
            dest_chip_coord=dest_chip_coord, timeslot=timeslot, dest_coords=dest_coords
        )
        self.dest_axon = dest_axon

    def set_rid(self) -> None:
        self.dest_core_info.set_rid()

    def __str__(self) -> str:
        _repr = f"chip addr: {self.dest_chip_coord}\n"
        # fmt: off
        if hasattr(self, "base_coord") and hasattr(self, "rid"):
            _repr += f"core addr: {_coord_to_bin_str(self.base_coord)}\n" + \
                     f"multicast: {_coord_to_bin_str(self.rid)}\n"

        _repr +=     f"axon addr: {self.dest_axon}\n"
        # fmt: on
        return _repr

    def info_str(self, line_prefix="") -> str:
        _repr = f"{line_prefix}chip addr: {self.dest_chip_coord}\n"
        if hasattr(self, "base_coord") and hasattr(self, "rid"):
            _repr += (
                f"{line_prefix}core addr: {_coord_to_bin_str(self.base_coord)}\n"
                + f"{line_prefix}multicast: {_coord_to_bin_str(self.rid)}\n"
            )
        _repr += f"{line_prefix}axon addr: {self.dest_axon}\n"
        return _repr

    @property
    def dest_chip_coord(self) -> ChipCoord:
        return self.dest_core_info.dest_chip_coord

    @property
    def timeslot(self) -> int:
        return self.dest_core_info.timeslot

    @property
    def base_coord(self) -> Coord:
        return self.dest_core_info.base_coord

    @property
    def rid(self) -> RId:
        return self.dest_core_info.rid

    @property
    def dest_coords(self) -> list[Coord]:
        return self.dest_core_info.dest_coords


def get_axon_coords(
    sub_source: SubSourceType,
    dest_ax_seg: AxonSegment,
    source_delay: int,
    dest_n_timeslot: int,
    is_iw8: bool,
) -> list[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment.

    NOTE: Axons are described in a tuple (tick_relative, axon_addr). Axis 'tr' is used as the row   \
        coordinates while axis 'axon' is used as the column coordinates.

    AxonSegment with `n_axon`, `addr_offset` represents a segment of axons address A[offset:offset+n_axon].

    tr=0                A[0]            A[1]                ... A[FAN_IN_BASE-1]
    tr=1                A[FAN_IN_BASE]  A[FAN_IN_BASE + 1]  ... A[2*FAN_IN_BASE-1]
    ...
    tr=MAX_TIMESLOT-1   A[(MAX_TIMESLOT-1)*FAN_IN_BASE]     ... A[MAX_TIMESLOT*FAN_IN_BASE-1]

    When the input width is 8 bits, each A[x] occupies 8 bits. The interval of axons is 8.
    """
    axon_coords: list[AxonCoord] = []
    tr_base = dest_n_timeslot * (source_delay - 1)

    _addr_interval = 8 if is_iw8 else 1
    fanin_base = dest_ax_seg.fanin_base
    axon_addr_start = dest_ax_seg.addr_offset

    if not sub_source.num_out == dest_ax_seg.n_axon:
        raise ValueError(
            f"The #N of axons in sub_source {sub_source.num_out} is not equal to that in dest_ax_seg {dest_ax_seg.n_axon}."
        )

    axon_addr_end = dest_ax_seg.addr_offset + sub_source.num_out
    for axon_addr in range(axon_addr_start, axon_addr_end):
        tick_relative = axon_addr // fanin_base + tr_base
        addr_axon = (axon_addr % fanin_base) * _addr_interval
        coord = AxonCoord.build(tick_relative, addr_axon)
        axon_coords.append(coord)

    return axon_coords


def _check_dest_attrs_same(
    dest_info: DestInfo, dest_chip_coord: ChipCoord, dest_axon: AxonCoord, timeslot: int
) -> None:
    """Check if the attributes of the destination details are the same as the given arguments."""
    recorded = (
        dest_info.dest_chip_coord,
        dest_info.dest_axon,
        dest_info.timeslot,
    )

    for r, arg in zip(recorded, (dest_chip_coord, dest_axon, timeslot)):
        if r != arg:
            raise ValueError(
                f"The attributes of the destination are not equal: {r} != {arg}."
            )


class SourceDest:
    """Used to represent the destination details of an entire neuron node. Since a neuron node may be   \
        divided into multiple `NeuronSlice`, it contains a list consisting of slice-destination pairs.

        It provides a method to obtain the destination details of the specified `NeuSegment`.
    """

    def __init__(self) -> None:
        self.dest_info: dict[CustomIndex, DestInfo] = dict()

    def add_dest(
        self, sub_source: SubSourceType, dest_ax_seg: AxonSegment, cb_of_dest: CoreBlock
    ) -> None:
        """Using the information of core block `cb` where the axon segment is, record the slice info & destination details."""
        dest_coords = cb_of_dest.core_coords.copy()
        dest_chip_coord = cb_of_dest.chip_coord
        dest_timeslot = cb_of_dest.n_timeslot
        dest_mode = cb_of_dest.rt_mode
        dest_axon_coords: list[AxonCoord] = get_axon_coords(
            sub_source,
            dest_ax_seg,
            source_delay=sub_source.target.delay_relative,
            dest_n_timeslot=dest_timeslot,
            is_iw8=is_iw8(dest_mode),
        )

        for custom_index, dest_axon_coord in zip(sub_source.index, dest_axon_coords):
            if custom_index in self.dest_info:
                # When the destination slice has been recorded, the info of the destination axon segment &
                # the core block where it's located also needs to be the same as the recorded info.
                d = self.dest_info[custom_index]
                _check_dest_attrs_same(
                    d, dest_chip_coord, dest_axon_coord, dest_timeslot
                )
                # In this case, only the core coordinates of the core blocks where the destination slice
                # is located are append to the recorded list.
                d.dest_coords.extend(dest_coords)
            else:
                # Add the destination slice in record.
                d = DestInfo(
                    dest_chip_coord, dest_axon_coord, dest_timeslot, dest_coords
                )
                self.dest_info[custom_index] = d

    def set_dest_rid(self) -> None:
        for dest_info in self.dest_info.values():
            dest_info.set_rid()

    def __str__(self) -> str:
        length = len(self.dest_info)
        _repr = f"{length} dest_infos:\n"

        if length <= 5:
            for custom_index, dest_info in self.dest_info.items():
                # Align with the content of the destination details
                _repr += f"custom index: {custom_index}\n"
                _repr += dest_info.info_str("\t")
        else:
            items = list(self.dest_info.items())
            for custom_index, dest_info in items[:3]:
                # Align with the content of the destination details
                _repr += f"custom index: {custom_index}\n"
                _repr += dest_info.info_str("\t")
            _repr += "\t...\n"
            for custom_index, dest_info in items[-2:]:
                _repr += f"custom index: {custom_index}\n"
                _repr += dest_info.info_str("\t")

        return _repr

    def sort_dest_info(self) -> None:
        """Sort the dest infos by the custom index."""
        self.dest_info = dict(sorted(self.dest_info.items(), key=lambda item: item[0]))

    def get_undivided_dest(self) -> tuple[DestCoreInfo, list[AxonCoord]]:
        """check if the destination is undivided."""
        dest_axon_coods: list[AxonCoord] = list()

        for i, (custom_index, dest_info) in enumerate(self.dest_info.items()):
            if CustomIndex(i, 0) != custom_index:
                raise ValueError(
                    "The custom index is not continuous, divided destination."
                )
            if i == 0:
                dest_core_info = dest_info.dest_core_info
            else:
                if dest_core_info != dest_info.dest_core_info:
                    raise ValueError(
                        "The destination core is not the same, divided destination."
                    )
            dest_axon_coods.append(dest_info.dest_axon)

        return dest_core_info, dest_axon_coods

    def devide_dest_info(self, neu_seg: DendriteSegment):
        """According to the given neuron segment, find the corresponding destination details."""
        # devide the dest_info according to the given indexs, if two dest_info have the same DestCoreInfo,
        # they will be add to the same group and their custom_index will be merged.
        dest_info_groups: dict[DestCoreInfo, list[CustomIndex]] = dict()
        for custom_index in neu_seg.index:
            if custom_index not in self.dest_info:
                raise ValueError(f"custom index {custom_index} not in dest_info.")
            dest_info = self.dest_info[custom_index]
            if dest_info.dest_core_info in dest_info_groups:
                dest_info_groups[dest_info.dest_core_info].append(custom_index)
            else:
                dest_info_groups[dest_info.dest_core_info] = [custom_index]

        pairs: list[tuple[DendriteSegment, DestCoreInfo, list[AxonCoord]]] = []
        base_offset = neu_seg.offset
        for dest_core_info, custom_indexs in dest_info_groups.items():
            sub_seg = DendriteSegment(
                neu_seg.target, custom_indexs, base_offset, neu_seg.repeat
            )
            dest_axon_coords = [self.dest_info[ci].dest_axon for ci in custom_indexs]
            pairs.append((sub_seg, dest_core_info, dest_axon_coords))
            base_offset += sub_seg.n_neuron
        return pairs


class CorePlacement(CoreAbstract):
    _parent: CoreBlock
    coord: Coord
    """Routing coordinate"""
    n_neuron: int
    raw_weight: WeightType
    """The folded weights."""
    neu_segs_of_cplm: SubNeuOfCorePlm

    def __init__(
        self,
        parent: CoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        raw_weight: WeightType,
        neu_segs_of_cplm: SubNeuOfCorePlm,
        name: str | None = None,
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
        self._parent = parent
        self.rt_mode = parent.rt_mode
        self.coord = routing_coord
        self.n_neuron = n_neuron
        self.raw_weight = raw_weight
        self.neu_segs_of_cplm = neu_segs_of_cplm

    @classmethod
    @abstractmethod
    def build(cls, parent: CoreBlock, idx: int) -> "CorePlacement":
        pass

    @abstractmethod
    def export_core_config(self) -> CoreConfig:
        pass

    @overload
    @abstractmethod
    def export_neu_config(
        self, neu_seg: DendriteSegment, source_dest: SourceDest
    ) -> None:
        pass

    @overload
    @abstractmethod
    def export_neu_config(
        self, neu_seg: DendriteSegment, *, output_core_coord: Coord
    ) -> None:
        pass

    @abstractmethod
    def export_neu_config(
        self,
        neu_seg: DendriteSegment,
        source_dest: SourceDest | None = None,
        output_core_coord: Coord | None = None,
    ) -> None:
        pass

    @abstractmethod
    def export_core_plm_config(self) -> CorePlmConfig:
        pass

    @property
    @abstractmethod
    def neu_configs(self) -> dict[Neuron, NeuConfig]:
        pass

    @property
    @abstractmethod
    def parent(self) -> CoreBlock:
        pass

    def _fold_raw_weight(self, raw_weight: WeightType) -> WeightType:
        """Fold the weights into LCN-sized blocks."""
        return self._nfold_weight(raw_weight, self.n_timeslot)

    @staticmethod
    def _nfold_weight(raw_weight: np.ndarray, n_fold: int) -> np.ndarray:
        raw_row, raw_col = raw_weight.shape
        n_row_folded, r = divmod(raw_row, n_fold)

        if r > 0:
            n_row_folded += 1
            pad = n_fold - r
            raw_weight = np.pad(raw_weight, ((0, pad), (0, 0)))

        # reshape 成 (n_fold, n_row_folded, raw_col)
        w = raw_weight.reshape(n_fold, n_row_folded, raw_col)

        # 转置并 reshape 成 (n_row_folded, raw_col * n_fold)
        w_folded = w.transpose(1, 2, 0).reshape(n_row_folded, raw_col * n_fold)

        return w_folded.astype(WEIGHT_DTYPE, copy=False)

    @staticmethod
    def _weight_ram_mapping(
        raw_weight: np.ndarray,
        weight_width: WW,
        n_timeslot: int,
        n_u64_per_wram_addr: int,
        online: bool,
    ) -> WRAMPackedType:
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

        weight_bit_num = 1 << weight_width

        w_folded = CorePlacement._nfold_weight(raw_weight, n_timeslot)

        # 转成 uint8 保留补码
        arr = w_folded.astype(np.uint8)

        # 展开为 bit (低位优先)
        bits = np.unpackbits(arr[:, :, None], axis=2, bitorder=HwConfig.WEIGHT_BITORDER)

        # 只保留低 weight_bit_num 位
        bits = bits[:, :, :weight_bit_num]

        # reshape 成 (M, N * weight_bit_num)
        M, N = arr.shape
        w_unpacked = bits.reshape(M, N * weight_bit_num)

        # shape = (N * weight_bit_num, M)
        unpacked_T = w_unpacked.T

        # 扁平化 + 补 0 (按 64bit 对齐)
        flat = unpacked_T.ravel()
        pad_len = (-len(flat)) % 64
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), constant_values=0)

        # packbits -> uint8，再视图转 uint64
        packed = np.packbits(flat, bitorder=HwConfig.WEIGHT_BITORDER)
        packed64 = packed.view(WRAM_PACKED_DTYPE)

        # pad packed64 to shape (x * N_U64_ON_WRAM_ADDR,)
        pad_len = (-len(packed64)) % n_u64_per_wram_addr
        if pad_len > 0:
            packed64 = np.pad(packed64, (0, pad_len), constant_values=0)

        # reshape to (x, N_U64_ON_WRAM_ADDR)
        w_packed = packed64.reshape(-1, n_u64_per_wram_addr)
        if online:
            w_packed = w_packed[:, ::-1]  # Online mode WRAM frame in MSB order

        w_packed.setflags(write=False)
        return w_packed

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
        assert w_packed_u64.shape[1] == OfflineCorePlacement.N_U64_ON_WRAM_ADDR
        return w_packed_u64

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
    def n_working_dendrite(self) -> int:
        """The number of actual working dendrites.

        NOTE: n_neuron * (2^comb_rate) = n_neuron << comb_rate
        """
        return self.n_neuron << self.dendrite_comb_rate

    @property
    def source(self) -> list[SubSourceType]:
        return self.parent.ordered_axons

    @property
    def dest(self) -> list[DestNodeType]:
        """The destination nodes within it.

        NOTE: This attribute is different from the one of its parent.
        """
        return [p.target for p in self.neu_segs_of_cplm]

    @property
    def weight_ram(self) -> WRAMPackedType:
        N_U64_ON_WRAM_ADDR = (
            OnlineCorePlacement.N_U64_ON_WRAM_ADDR
            if self.online
            else OfflineCorePlacement.N_U64_ON_WRAM_ADDR
        )
        return CorePlacement._weight_ram_mapping(
            self.raw_weight,
            self.weight_width,
            self.n_timeslot,
            N_U64_ON_WRAM_ADDR,
            self.online,
        )

    @property
    def n_core_required(self):
        return 1

    @property
    def online(self) -> bool:
        return self.parent.online

    def __len__(self) -> int:
        return self.n_core_required


class OfflineCorePlacement(CorePlacement):
    """each coreplacement only won't contain dendrite segments from the same Neuron,
    so the neu_configs won't have same key."""

    _neu_configs: dict[Neuron, OfflineNeuConfig] = dict()

    N_U64_ON_WRAM_ADDR: ClassVar[int] = (
        OffCoreCfg.WEIGHT_RAM_SHAPE[1] // N_BIT_PACKED_WEIGHT
    )
    """The number of u64 at each address of weight RAM."""

    def __init__(
        self,
        parent: OfflineCoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        raw_weight: WeightType,
        neu_segs_of_cplm: SubNeuOfCorePlm,
        name: str | None = None,
    ) -> None:
        self._neu_configs = dict()
        super().__init__(
            parent, routing_coord, n_neuron, raw_weight, neu_segs_of_cplm, name
        )

    @classmethod
    def build(cls, parent: OfflineCoreBlock, idx: int):
        coord = parent.core_coords[idx]
        n_neuron = parent.n_neuron_of_plm[idx]
        sub_neus_of_cplm = parent.core_allocation_of_cb[idx]
        raw_weight = parent.get_raw_weight_of_coord(idx)

        return cls(parent, coord, n_neuron, raw_weight, sub_neus_of_cplm)

    @staticmethod
    def neu_params_mapping(neu_confs: list[OfflineNeuConfig]) -> WRAMPackedType:
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
            OffCoreCfg.WEIGHT_RAM_SHAPE[1] // NEURON_PARAMS_BIT_LENGTH
        )
        n_col_occupied, r = divmod(neu_params.shape[0], N_NEURON_PARAM_IN_COL)
        if r > 0:
            n_col_occupied += 1
            neu_params = np.pad(neu_params, ((0, N_NEURON_PARAM_IN_COL - r), (0, 0)))

        neu_params = neu_params.reshape((n_col_occupied, -1))

        # (1152, y)
        result = np.zeros(
            (OffCoreCfg.WEIGHT_RAM_SHAPE[1], n_col_occupied),
            dtype=WRAM_UNPACKED_DTYPE,
        )
        _n_bit_nparams = NEURON_PARAMS_BIT_LENGTH * N_NEURON_PARAM_IN_COL
        result[:_n_bit_nparams, :] = neu_params.T

        # (1152, y) -> (y, 18)
        return CorePlacement._weight_pack(result)

    def export_core_config(self) -> OfflineCoreConfig:
        _mode_params = self.rt_mode.conf

        # fmt: off
        cb_config = OfflineCoreConfig(
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
        self, neu_seg: DendriteSegment, source_dest: SourceDest
    ) -> None: ...

    @overload
    def export_neu_config(
        self, neu_seg: DendriteSegment, *, output_core_coord: Coord
    ) -> None: ...

    def export_neu_config(
        self,
        neu_seg: DendriteSegment,
        source_dest: SourceDest | None = None,
        output_core_coord: Coord | None = None,
    ) -> None:
        if neu_seg not in self.neu_segs_of_cplm:
            raise ValueError(
                f"The given neu_seg {neu_seg} is not in the 'neu_segs_of_cplm' of this core placement."
            )
        """Export the neuron configuration."""
        if isinstance(source_dest, SourceDest):
            neu_seg_dest_pairs = source_dest.devide_dest_info(neu_seg)
            for seg, dest, axon_coords in neu_seg_dest_pairs:
                config = OfflineNeuConfig(
                    seg, axon_coords, dest.base_coord, dest.rid, dest.dest_chip_coord
                )
                self.neu_configs[seg.target] = config
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)
            # TODO Only leverage the axon coordinate attributes in `AxonCoord` and do not use the
            # `tick_relative` attribute, which causes the number of an output node cannot be
            # greater than `N_FANIN_PER_DENDRITE_MAX`(=1152).

            # there are not copy neurons in output node, so no need to check the copy_id
            axon_coords = [
                (
                    AxonCoord.build(0, i.index)
                    if i.index < 1152
                    else AxonCoord.build(i.index // 1152, i.index % 1152)
                )
                for i in neu_seg.index
            ]

            config = OfflineNeuConfig(
                neu_seg,
                axon_coords,
                output_core_coord,
                _RID_UNSET,
                # output chip coordinate for output node
                _BACKEND_CONTEXT.output_chip_addr,
            )

            self.neu_configs[neu_seg.target] = config

    def export_core_plm_config(self) -> OfflineCorePlmConfig:
        core_param = self.export_core_config()
        return OfflineCorePlmConfig.encapsulate(
            self.parent.seed, self.weight_ram, core_param, self.neu_configs
        )

    @property
    def pool_max(self) -> MaxPoolingEnable:
        return self.parent.pool_max

    @property
    def neu_configs(self) -> dict[Neuron, OfflineNeuConfig]:
        """The neuron configurations of the core placement."""
        return self._neu_configs

    @property
    def parent(self) -> OfflineCoreBlock:
        """The parent core block."""
        if isinstance(self._parent, OfflineCoreBlock):
            return self._parent
        else:
            raise TypeError(
                f"Parent must be an instance of {OfflineCoreBlock.__name__}, but got {type(self._parent).__name__}."
            )


class OnlineCorePlacement(CorePlacement):
    """each coreplacement only won't contain dendrite segments from the same Neuron,
    so the neu_configs won't have same key."""

    _neu_configs: dict[Neuron, OnlineNeuConfig] = dict()

    N_U64_ON_WRAM_ADDR: ClassVar[int] = (
        OnCoreCfg.WEIGHT_RAM_SHAPE[1] // N_BIT_PACKED_WEIGHT
    )
    """The number of u64 at each address of weight RAM."""

    def __init__(
        self,
        parent: OnlineCoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        raw_weight: WeightType,
        neu_segs_of_cplm: SubNeuOfCorePlm,
        name: str | None = None,
    ) -> None:
        self._neu_configs = dict()
        super().__init__(
            parent, routing_coord, n_neuron, raw_weight, neu_segs_of_cplm, name
        )

    @classmethod
    def build(cls, parent: OnlineCoreBlock, idx: int):
        coord = parent.core_coords[idx]
        n_neuron = parent.n_neuron_of_plm[idx]
        raw_weight = parent.get_raw_weight_of_coord(idx)
        neu_segs_of_cplm = parent.core_allocation_of_cb[idx]

        return cls(parent, coord, n_neuron, raw_weight, neu_segs_of_cplm)

    def export_core_config(self) -> OnlineCoreConfig:
        cb_config = OnlineCoreConfig(
            self.name,
            self.weight_width,
            self.lcn_ex,
            self.lateral_inhi_value,
            self.weight_decay_value,
            self.upper_weight,
            self.lower_weight,
            self.neuron_start,
            self.neuron_end,
            self.inhi_core_x_ex,
            self.inhi_core_y_ex,
            self.tws,
            self.twe,
            self.lut_random_en,
            self.decay_random_en,
            self.leak_order,
            self.online_mode_en,
            _BACKEND_CONTEXT.test_chip_addr,
            self.random_seed,
        )
        return cb_config

    @overload
    def export_neu_config(
        self, neu_seg: DendriteSegment, source_dest: SourceDest
    ) -> None: ...

    @overload
    def export_neu_config(
        self, neu_seg: DendriteSegment, *, output_core_coord: Coord
    ) -> None: ...

    def export_neu_config(
        self,
        neu_seg: DendriteSegment,
        source_dest: SourceDest | None = None,
        output_core_coord: Coord | None = None,
    ) -> None:
        if neu_seg not in self.neu_segs_of_cplm:
            raise ValueError(
                f"The given neu_seg {neu_seg} is not in the 'neu_segs_of_cplm' of this core placement."
            )

        """Export the neuron configuration."""
        if isinstance(source_dest, SourceDest):
            neu_seg_dest_pairs = source_dest.devide_dest_info(neu_seg)
            for seg, dest, axon_coords in neu_seg_dest_pairs:
                config = OnlineNeuConfig(
                    seg,
                    axon_coords,
                    dest.base_coord,
                    dest.rid,
                    dest.dest_chip_coord,
                    self.weight_width,
                )
                self.neu_configs[seg.target] = config
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)

            # online core as a output node, the target is offline core
            # but online core's target lcn should below LCN_8X,
            # so the max index should be 1152 * 8 = 9216

            max_index = max(i.index for i in neu_seg.index)
            assert max_index < (OffCoreCfg.ADDR_AXON_MAX + 1) * 8

            axon_coords = [
                (
                    AxonCoord.build(0, i.index)
                    if i.index < 1152
                    else AxonCoord.build(i.index // 1152, i.index % 1152)
                )
                for i in neu_seg.index
            ]

            config = OnlineNeuConfig(
                neu_seg,
                axon_coords,
                output_core_coord,
                _RID_UNSET,
                # output chip coordinate for output node
                _BACKEND_CONTEXT.output_chip_addr,
                self.weight_width,
            )

            self.neu_configs[neu_seg.target] = config

    def export_core_plm_config(self) -> OnlineCorePlmConfig:
        core_param = self.export_core_config()
        return OnlineCorePlmConfig.encapsulate(
            self.weight_ram, core_param, self.lut, self.neu_configs
        )

    @property
    def lateral_inhi_value(self) -> int:
        return self.parent.lateral_inhi_value

    @property
    def weight_decay_value(self) -> int:
        return self.parent.weight_decay_value

    @property
    def upper_weight(self) -> int:
        return self.parent.upper_weight

    @property
    def lower_weight(self) -> int:
        return self.parent.lower_weight

    @property
    def neuron_start(self) -> int:
        return 0

    @property
    def neuron_end(self) -> int:
        return self.n_neuron - 1

    @property
    def inhi_core_x_ex(self) -> int:
        return self.parent.inhi_core_x_ex

    @property
    def inhi_core_y_ex(self) -> int:
        return self.parent.inhi_core_y_ex

    @property
    def lut_random_en(self) -> NDArray[np.uint8]:
        return self.parent.lut_random_en

    @property
    def decay_random_en(self) -> DecayRandomEnable:
        return self.parent.decay_random_en

    @property
    def leak_order(self) -> LeakOrder:
        return self.parent.leak_order

    @property
    def online_mode_en(self) -> OnlineModeEnable:
        return self.parent.online_mode_en

    @property
    def lut(self) -> LUTDataType:
        return self.parent.lut

    @property
    def random_seed(self) -> int:
        return self.parent.random_seed

    @property
    def neu_configs(self) -> dict[Neuron, OnlineNeuConfig]:
        """The neuron configurations of the core placement."""
        return self._neu_configs

    @property
    def parent(self) -> OnlineCoreBlock:
        """The parent core block."""
        if isinstance(self._parent, OnlineCoreBlock):
            return self._parent
        else:
            raise TypeError(
                f"Parent must be an instance of {OnlineCoreBlock.__name__}, but got {type(self._parent).__name__}."
            )


class EmptyCorePlacement(CoreAbstract):
    def __init__(self, coord: Coord, name: str | None = None) -> None:
        super().__init__(name)
        self.coord = coord

    @abstractmethod
    def export_core_config(self) -> CoreConfig:
        pass

    @abstractmethod
    def export_core_plm_config(self) -> CorePlmConfig:
        pass

    @classmethod
    def build(cls, coord: Coord, online: bool):
        if online:
            return EmptyOnlineCorePlacement(coord)
        else:
            return EmptyOfflineCorePlacement(coord)


class EmptyOfflineCorePlacement(EmptyCorePlacement):
    """Empty offline core placement."""

    _EMPTY_WRAM: int = 0

    def __init__(self, coord: Coord, name: str | None = None) -> None:
        super().__init__(coord, name)

    def export_core_config(self) -> OfflineCoreConfig:
        _mode_params = CoreMode.MODE_SNN.conf

        # fmt: off
        cb_config = OfflineCoreConfig(
            self.name,                          # name of the core
            WW.WEIGHT_WIDTH_1BIT,               # weight_width
            LCN_EX.LCN_1X,                      # lcn
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

    def export_core_plm_config(self) -> OfflineCorePlmConfig:
        core_param = self.export_core_config()
        # For empty core placements, we don't care random seed, WRAM & neurons cfg.
        return OfflineCorePlmConfig.encapsulate(0, self._EMPTY_WRAM, core_param, {})  # type: ignore

    @classmethod
    def build(cls, coord: Coord):
        return cls(coord)

    @property
    def n_core_required(self) -> int:
        return 1


class EmptyOnlineCorePlacement(EmptyCorePlacement):
    """Empty online core placement."""

    _EMPTY_WRAM: int = 0

    def __init__(self, coord: Coord, name: str | None = None) -> None:
        super().__init__(coord, name)

    def export_core_config(self) -> OnlineCoreConfig:
        # fmt: off
        cb_config = OnlineCoreConfig(
            self.name,
            WW.WEIGHT_WIDTH_1BIT,               # weight_width
            LCN_EX.LCN_1X,                      # lcn
            0,                                  # lateral_inhi_value
            0,                                  # weight_decay_value
            0,                                  # upper_weight
            0,                                  # lower_weight
            0,                                  # neuron_start
            0,                                  # neuron_end
            0,                                  # inhi_core_x_ex
            0,                                  # inhi_core_y_ex
            0,                                  # tick_wait_start
            0,                                  # tick_wait_end
            np.zeros(LUT_LEN, dtype=np.uint8),  # lut_random_en
            DecayRandomEnable.DISABLE,          # decay_random_en
            LeakOrder.LEAK_BEFORE_COMP,         # leak_order
            OnlineModeEnable.DISABLE,           # online_mode_en
            _BACKEND_CONTEXT.test_chip_addr,    # test_chip_addr
            1,                                  # random_seed
        )
        # fmt: on
        return cb_config

    def export_core_plm_config(self) -> OnlineCorePlmConfig:
        core_param = self.export_core_config()
        # For empty core placements, we don't care WRAM & neurons cfg.
        return OnlineCorePlmConfig.encapsulate(
            self._EMPTY_WRAM,
            core_param,
            np.zeros(LUT_LEN, dtype=LUT_DTYPE),
            {},  # type: ignore
        )

    @classmethod
    def build(cls, coord: Coord):
        return cls(coord)

    @property
    def n_core_required(self) -> int:
        return 1


def max_lcn_of_cb(cb: list[CoreBlock]) -> LCN_EX:
    """Get the max LCN extenion of given core blocks."""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex
