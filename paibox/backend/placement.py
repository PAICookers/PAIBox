import sys
import warnings
from dataclasses import field
from functools import cached_property
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import (
    LCN_EX,
    AxonCoord,
    AxonSegment,
    ChipCoord,
    Coord,
    CoreMode,
    CoreModeDict,
    HwConfig,
    HwCore,
    MaxPoolingEnable,
)
from paicorelib import WeightPrecision as WP

from paibox.base import NeuDyn, PAIBoxObject, SynSys
from paibox.exceptions import GraphBuildError, ResourceError, TruncationWarning
from paibox.types import WeightType
from paibox.utils import check_attr_same, count_unique_elem

from .conf_template import CoreConfig, CorePlmConfig, EmptyCorePlmConfig, NeuronConfig
from .context import _BACKEND_CONTEXT
from .graphs_types import DestNodeType, SourceNodeType
from .segment_utils import (
    NeuSeg,
    NeuSegOfCoreBlock,
    NeuSegOfCorePlm,
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
)

WeightRamType: TypeAlias = NDArray[np.uint64]  # uint64 weights mapped in weight RAM


class CoreAbstract(HwCore, PAIBoxObject):
    SUPPORTED_MODE: ClassVar[Tuple[CoreMode, ...]] = (CoreMode.MODE_SNN,)
    """Supported core modes."""


class CoreBlock(CoreAbstract):
    """Core Block for `MODE_SNN` ONLY."""

    RUNTIME_MODE: ClassVar[CoreMode] = CoreMode.MODE_SNN

    def __init__(
        self, *parents: SynSys, routing_id: int, seed: int, name: Optional[str] = None
    ) -> None:
        """Core blocks in SNN mode.

        Args:
            - parents: the parent synapses.
            - routing_id: id of routing group.
            - seed: random seed. Default value is 0.
            - name: name of the core block. Optional.
        """
        super().__init__(name)
        self._parents = parents
        self._lcn_ex = self._n_axon2lcn_ex()
        self._wp = WP.WEIGHT_WIDTH_8BIT  # default value
        self._routing_id = routing_id

        self.seed = seed
        """Random seed, legal integer, no more than uint64."""

        self.target_lcn = LCN_EX.LCN_1X
        """The target(destination core block) LCN."""

        self.lcn_locked = False
        """Used to indicate whether `lcn_ex` has been adjusted."""

        self.core_coords: List[Coord] = list()
        """Assigned core coordinates."""

        self.chip_coord: ChipCoord = Coord(0, 0)  # default
        """A core block must be placed on a chip."""

        self.core_placements: Dict[Coord, CorePlacement] = dict()
        """Core placements."""

        # Segment the group of axons.
        self.axon_segments: Dict[SourceNodeType, AxonSegment] = get_axon_segments(
            self.axons, self.n_timeslot, self.n_fanin_max
        )
        """A dictionary of segments of each axon(source node)."""

        self.neuron_segs_of_cb: NeuSegOfCoreBlock = []
        """Neuron segments in the core block. Each element in the list \
            represents the neuron segments in core placement(physical core).
        """

    def group_neurons(
        self, optim_target: Literal["latency", "core", "both"] = "both"
    ) -> None:
        """Group the neurons to determine the #N of cores required."""
        if not self.lcn_locked:
            raise GraphBuildError("Group the neurons after lcn_ex is locked.")

        self.neuron_segs_of_cb = get_neu_segments(
            self.dest,
            self.neuron_capacity,
            _neuron_repl_prop(self.n_weight_bits, self.n_timeslot),
            optim_target,
        )

    def core_plm_alloc(self) -> None:
        """Allocate `CoreBlock` to physical cores."""
        if not self.lcn_locked:
            raise GraphBuildError("Allocate core placements after lcn_ex is locked.")

        for i, coord in enumerate(self.core_coords):
            # assert self.get_raw_weight_of_coord(i)[0].shape[0] == self.n_axon
            self.core_placements[coord] = CorePlacement.build(self, i)

    def _get_syn_of(self, src: SourceNodeType, dest: DestNodeType) -> Optional[SynSys]:
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
            lcn := int((self.n_axon - 1) // self.n_fanin_max).bit_length()
        ) > LCN_EX.LCN_64X:
            _max_n_axons = self.n_fanin_max * (1 << LCN_EX.LCN_64X)
            raise ResourceError(
                f"required LCN extension out of range {LCN_EX.LCN_64X} ({lcn}). "
                f"The number of axons must be <= {_max_n_axons}. "
                f"But synapses {self._obj_repr()} have a total of {self.n_axon} axons."
            )

        return LCN_EX(lcn)

    def copy(self):
        raise NotImplementedError

    """Interfaces"""

    @property
    def obj(self) -> Tuple[SynSys, ...]:
        return self._parents

    @property
    def shape(self) -> Tuple[int, int]:
        return (count_unique_elem(self.source), count_unique_elem(self.dest))

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

    def n_axon_of(self, index: int) -> int:
        """Get the #N of axons of `index`-th source neuron."""
        return self.axons[index].num_out

    """Boundary limitations"""

    @property
    def neuron_capacity(self) -> int:
        """Neuron capacity. #N of valid dendrites/#N of dendrites required per neuron.

        FIXME This method ONLY works in SNN RUNTIME_MODE. For ANN RUNTIME_MODE, use table lookup?
        """
        return (self.n_dendrite_max >> self.lcn_ex) // self.n_dendrite_per_neuron

    @property
    def n_fanin_max(self) -> int:
        """Maximum #N of fan-in per dendrite."""
        return (
            HwConfig.N_FANIN_PER_DENDRITE_ANN
            if self.RUNTIME_MODE is CoreMode.MODE_ANN
            else HwConfig.N_FANIN_PER_DENDRITE_SNN
        )

    @property
    def n_core_required(self) -> int:
        return len(self.neuron_segs_of_cb)

    @property
    def weight_precision(self) -> WP:
        # Optimized in `s.weight_precision`.
        return max(s.weight_precision for s in self.obj)

    @property
    def n_dendrite_per_neuron(self) -> int:
        """Multiple dendrites will be combined to achieve higher precision weights.

        FIXME The limit on the number of dendrites in SNN/ANN modes is different, which affects \
            the capacity of neurons in physical core.
        """
        return 1 << self.weight_precision

    @property
    def n_weight_bits(self) -> int:
        return self.n_dendrite_per_neuron

    @property
    def lcn_ex(self) -> LCN_EX:
        return self._lcn_ex

    @lcn_ex.setter
    def lcn_ex(self, lcn_ex: LCN_EX) -> None:
        """Set or adjust the `lcn_ex` & lock."""
        if lcn_ex > LCN_EX.LCN_64X:
            raise ResourceError(
                f"required LCN extension out of range {LCN_EX.LCN_64X} ({lcn_ex})."
            )

        self._lcn_ex = lcn_ex
        self.lcn_locked = True

    @property
    def n_timeslot(self) -> int:
        return 1 << self.lcn_ex

    @property
    def tws(self) -> int:
        """Attribute `tick_wait_start`."""
        if not check_attr_same(self.dest, "tick_wait_start"):
            raise AttributeError(
                "Attribute 'tick_wait_start' of the core block are not equal."
            )

        return self.dest[0].tick_wait_start

    @property
    def twe(self) -> int:
        """Attribute `tick_wait_end.`"""
        if not check_attr_same(self.dest, "tick_wait_end"):
            raise AttributeError(
                "Attribute 'tick_wait_end' of the core block are not equal."
            )

        return self.dest[0].tick_wait_end

    @property
    def n_axon(self) -> int:
        return sum(s.num_out for s in self.axons)

    @property
    def n_dendrite_max(self) -> int:
        return (
            HwConfig.N_DENDRITE_MAX_ANN
            if self.RUNTIME_MODE is CoreMode.MODE_ANN
            else HwConfig.N_DENDRITE_MAX_SNN
        )

    @property
    def n_neuron(self) -> int:
        return sum(d.num_in for d in self.dest)

    @property
    def unrolling_factor(self) -> List[int]:
        return [d.unrolling_factor for d in self.dest]

    @property
    def n_neuron_of_plm(self) -> List[int]:
        """A list of the #N of neurons on each `CorePlacement`.

        FIXME Different in SNN/ANN RUNTIME_MODE.
        """
        if len(self.core_coords) == 0:
            raise GraphBuildError(f"do this after coordinates assignment.")

        # Get #N of neurons on each `CorePlacement` according to the
        # maximum address required of neuron segments on each `CorePlacement`.
        assert [] not in self.neuron_segs_of_cb  # TODO if it never happens, remove it.

        return [
            sum(seg.n_neuron for seg in neuron_segs)
            for neuron_segs in self.neuron_segs_of_cb
        ]

    @cached_property
    def raw_weight_of_dest(self) -> List[WeightType]:
        """Merge and then split the weight matrix according to the grouping of neurons."""
        # The concatenated weight for each destination node.
        w_of_neurons: List[WeightType] = []

        for d in self.dest:
            # The weights for each destination node.
            w_of_dest = []

            for s in self.source:
                if syn := self._get_syn_of(s, d):
                    w_of_dest.append(syn.connectivity)
                else:
                    # Fill with 0.
                    w_of_dest.append(np.zeros((s.num_out, d.num_in), dtype=np.int8))

            w_dest = np.vstack(w_of_dest, dtype=np.int8)
            w_of_neurons.append(w_dest)

        # Check
        assert all(
            w_of_neurons[0].shape[0] == w_of_neuron.shape[0]
            for w_of_neuron in w_of_neurons
        )

        return w_of_neurons

    def get_raw_weight_of_coord(self, idx: int) -> List[WeightType]:
        """Get the raw weight of a coordinate(on each `CorePlacement`)."""
        w_of_neu_segs: List[WeightType] = []

        for neu_seg in self.neuron_segs_of_cb[idx]:
            w_of_dest = self.raw_weight_of_dest[self.dest.index(neu_seg.parent)]
            w_of_neu_seg = w_of_dest[:, neu_seg.segment.index].copy()
            w_of_neu_seg.setflags(write=False)
            w_of_neu_segs.append(w_of_neu_seg)

        return w_of_neu_segs

    def __len__(self) -> int:
        return self.n_core_required

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"

    def _obj_repr(self) -> str:
        """The representation of the names of target objects."""
        return ", ".join(n.name for n in self.obj)

    @classmethod
    def build(cls, *synapses: SynSys, routing_id: int, seed: int = 0):
        """Group synapses & build `CoreBlock`."""
        # FIXME where does the parameter check do?
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )

        return cls(*synapses, routing_id=routing_id, seed=seed)

    @classmethod
    def export_core_plm_config(cls, cb: "CoreBlock") -> Dict[Coord, CoreConfig]:
        """Export the parameters of the core into a dictionary."""
        cb_config = dict()

        for coord, core_plm in cb.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config


class CorePlacement(CoreAbstract):
    """The divided synapse placed on a single CORE."""

    WEIGHT_RAM_SHAPE: ClassVar[Tuple[int, int]] = (
        HwConfig.N_FANIN_PER_DENDRITE_SNN,
        HwConfig.N_DENDRITE_MAX_SNN,
    )
    """SNN mode ONLY."""

    def __init__(
        self,
        parent: CoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        raw_weights: List[WeightType],
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
        self.coord = routing_coord
        """Routing coordinate"""

        self.n_neuron = n_neuron

        self._weights_folded = self._fold_raw_weights(raw_weights)
        """The folded weights."""

        self.neu_segs_of_cplm = neu_segs_of_cplm
        self.neu_configs: Dict[NeuDyn, NeuronConfig] = dict()

    @classmethod
    def build(cls, parent: CoreBlock, idx: int):
        coord = parent.core_coords[idx]
        n_neuron = parent.n_neuron_of_plm[idx]
        neu_segs_of_cplm = parent.neuron_segs_of_cb[idx]
        raw_weights = parent.get_raw_weight_of_coord(idx)

        return cls(parent, coord, n_neuron, raw_weights, neu_segs_of_cplm)

    def _fold_raw_weights(self, raw_weights: List[WeightType]) -> WeightType:
        """Fold the weights into LCN-sized blocks."""
        w_folded_list = []
        w_folded_of_axon_segs = []
        n_fold = self.n_timeslot

        if self.lcn_ex == LCN_EX.LCN_1X:
            w_folded = np.hstack(raw_weights, dtype=np.int8)
            w_folded.setflags(write=False)

            return w_folded

        # LCN_EX > LCN_1X
        for raw_weight in raw_weights:
            w_folded_of_axon_segs.clear()

            for s in self.source:
                axon_seg = self.parent.axon_segments[s]

                # Retrive the weight of the axon segment
                w_of_axon_seg = raw_weight[: axon_seg.n_axon, :]

                # Fold the weight of axon segment
                w_folded_of_axon_seg = self._nfold_weight(
                    w_of_axon_seg, axon_seg.addr_width, n_fold
                )
                w_folded_of_axon_segs.append(w_folded_of_axon_seg)

            w_folded = np.vstack(w_folded_of_axon_segs, dtype=np.int8)
            w_folded_list.append(w_folded)

        w_folded = np.hstack(w_folded_list, dtype=np.int8)
        w_folded.setflags(write=False)

        return w_folded

    def _weight_ram_mapping(self) -> WeightRamType:
        row, col = self._weights_folded.shape
        w_unpacked = np.zeros(self.WEIGHT_RAM_SHAPE, dtype=np.uint8)

        if self.n_weight_bits == 1:
            w_unpacked[:row, :col] = self._weights_folded
        else:
            # (N, M) -> (M*N, 1)
            w_folded_3d = np.expand_dims(self._weights_folded.T, axis=2).astype(
                np.uint8
            )

            for i in range(col):
                # For every column, unpack the array [N*1] -> [N*n_weight_bits]
                unpacked = np.unpackbits(
                    w_folded_3d[i],
                    axis=1,
                    count=self.n_weight_bits,
                    bitorder=HwConfig.WEIGHT_BITORDER,
                )

                w_unpacked[
                    :row, self.n_weight_bits * i : self.n_weight_bits * (i + 1)
                ] = unpacked

        assert np.max(w_unpacked, axis=None) <= np.uint8(1)
        assert np.min(w_unpacked, axis=None) >= np.uint8(0)

        # Convert the unpacked weights into a mapping format,
        # corresponding to the RAM address, each address contains 18 uint64.
        # (1152, 512) -> (512, 1152) -> (512*18, 64)(uint8).
        # Reshape to 64 columns to avoid contiguous problem.
        w_unpacked_T_rehaped = w_unpacked.T.reshape(-1, 64)

        # (512*18, 64)(uint8) -> (512*18, 8)(uint8)
        w_packed_u8 = np.packbits(
            w_unpacked_T_rehaped, axis=1, bitorder=HwConfig.WEIGHT_BITORDER
        )
        # (512*18, 8)(uint8) -> (512*18, 1)(uint64) -> (512, 18)(uint64)
        w_packed_u64 = w_packed_u8.view(np.uint64).reshape(-1, 18)
        w_packed_u64.setflags(write=False)

        return w_packed_u64

    @staticmethod
    def _nfold_weight(
        raw_weight: WeightType, expected_row: int, n_fold: int
    ) -> WeightType:
        """According to the folding ratio `n_fold`, fold the weight matrix.

        Args:
            - raw_weight: the raw weight matrix.
            - expected_row: expected #N of row.
            - n_fold: the folding ratio.
        """
        raw_row, raw_col = raw_weight.shape

        if raw_row % n_fold > 0:
            n_row_padding = n_fold - raw_row % n_fold

            # Check #1
            # assert expected_row * n_fold == raw_row + n_row_padding

            _raw_weight = np.append(
                raw_weight,
                np.zeros((n_row_padding, raw_col), dtype=np.int8),
                axis=0,
            )
        else:
            _raw_weight = raw_weight.copy()

        w_splited = np.vsplit(_raw_weight, n_fold)

        # Check #2
        # assert _raw_weight.shape[0] == expected_row * n_fold

        w_folded = np.zeros((expected_row, raw_col * n_fold), dtype=np.int8)

        for i, j in np.ndindex((n_fold, raw_col)):
            w_col = w_splited[i][:, j]
            w_folded[:, n_fold * j + i] = w_col

        return w_folded

    def export_param_config(self) -> CoreConfig:
        _mode_params = CoreModeDict[self.mode]
        # fmt: off
        cb_config = CoreConfig(
            self.name,                          # name of the core
            self.weight_precision,              # weight_precision
            self.lcn_ex,                        # lcn_extension
            _mode_params[0],                    # input_width_format
            _mode_params[1],                    # spike_width_format
            self.n_dendrite,                    # num_dendrite
            MaxPoolingEnable.DISABLE,           # max_pooling_en
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
        self, neu_seg: NeuSeg, axon_dests: List[CoreBlock]
    ) -> None: ...

    @overload
    def export_neu_config(
        self,
        neu_seg: NeuSeg,
        *,
        output_core_coord: Coord,
        axon_addr_offset: int,
    ) -> int: ...

    def export_neu_config(
        self,
        neu_seg: NeuSeg,
        axon_dests: Optional[List[CoreBlock]] = None,
        output_core_coord: Optional[Coord] = None,
        axon_addr_offset: Optional[int] = None,
    ) -> Optional[int]:
        """Export the neuron configuration."""
        if isinstance(axon_dests, list):
            axon_coords = aligned_coords(
                neu_seg.segment.index,
                axon_dests[0].axon_segments[neu_seg.parent],
                neu_seg.parent.delay_relative,
                axon_dests[0].n_timeslot,
            )

            # Get all core coordinates and replication ids.
            assert all(axon_dests[0].chip_coord == ad.chip_coord for ad in axon_dests)

            dest_core_coords = []
            for ad in axon_dests:
                dest_core_coords.extend(ad.core_coords)

            config = NeuronConfig.encapsulate(
                neu_seg.parent,
                neu_seg.n_neuron,
                neu_seg.segment.addr_ram,
                neu_seg.segment.addr_offset,
                axon_coords,
                dest_core_coords,
                axon_dests[0].chip_coord,
            )

            self.neu_configs[neu_seg.parent] = config
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)
            assert isinstance(axon_addr_offset, int)

            axon_coords = [
                AxonCoord(0, i)
                for i in range(axon_addr_offset, axon_addr_offset + neu_seg.n_neuron)
            ]

            config = NeuronConfig.encapsulate(
                neu_seg.parent,
                neu_seg.n_neuron,
                neu_seg.segment.addr_ram,
                neu_seg.segment.addr_offset,
                axon_coords,
                [output_core_coord],
                # output chip coordinate for output node
                _BACKEND_CONTEXT["output_chip_addr"],
            )

            self.neu_configs[neu_seg.parent] = config

            return axon_addr_offset + neu_seg.n_neuron

    def export_core_plm_config(self) -> CorePlmConfig:
        core_param = self.export_param_config()

        return CorePlmConfig.encapsulate(
            self.parent.seed,
            self.weight_ram,
            core_param,
            self.neu_configs,
        )

    @property
    def mode(self) -> CoreMode:
        return self.parent.RUNTIME_MODE

    @property
    def shape(self) -> Tuple[int, int]:
        return (count_unique_elem(self.source), count_unique_elem(self.dest))

    @property
    def weight_precision(self) -> WP:
        return self.parent.weight_precision

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
    def tws(self) -> int:
        return self.parent.tws

    @property
    def twe(self) -> int:
        return self.parent.twe

    @property
    def n_dendrite(self) -> int:
        return self.n_neuron * _neuron_repl_prop(self.n_weight_bits, self.n_timeslot)

    @property
    def source(self) -> List[SourceNodeType]:
        return self.parent.source

    @property
    def dest(self):
        """The destination nodes within it.

        NOTE: This attribute is different from the one of its parent.
        """
        return [p.parent for p in self.neu_segs_of_cplm]

    @property
    def weight_ram(self) -> WeightRamType:
        return self._weight_ram_mapping()

    @property
    def n_core_required(self):
        return 1

    def __len__(self) -> int:
        return self.n_core_required


class EmptyCorePlacement(CoreAbstract):
    """Empty core placement."""

    _default_wp: ClassVar[WP] = WP.WEIGHT_WIDTH_1BIT
    _default_lcn_ex: ClassVar[LCN_EX] = LCN_EX.LCN_1X
    _default_n_dendrite: ClassVar[int] = 0
    _default_tws: ClassVar[int] = 0
    _default_twe: ClassVar[int] = 0
    _default_target_lcn: ClassVar[LCN_EX] = LCN_EX.LCN_1X

    def __init__(self, coord: Coord, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.coord = coord

    def export_param_config(self) -> CoreConfig:
        _mode_params = CoreModeDict[CoreMode.MODE_SNN]
        # fmt: off
        cb_config = CoreConfig(
            self.name,                          # name of the core
            self._default_wp,                   # weight_precision
            self._default_lcn_ex,               # lcn_extension
            _mode_params[0],                    # input_width_format
            _mode_params[1],                    # spike_width_format
            self._default_n_dendrite,           # num_dendrite
            MaxPoolingEnable.DISABLE,           # max_pooling_en
            self._default_tws,                  # tick_wait_start
            self._default_twe,                  # tick_wait_end
            _mode_params[2],                    # snn_mode_en
            self._default_target_lcn,           # target_lcn
            _BACKEND_CONTEXT.test_chip_addr,    # test_chip_addr
        )
        # fmt: on
        return cb_config

    def export_core_plm_config(self) -> EmptyCorePlmConfig:
        core_param = self.export_param_config()
        return EmptyCorePlmConfig.encapsulate(core_param)

    @classmethod
    def build(cls, coord: Coord):
        return cls(coord)

    @property
    def n_core_required(self):
        return 1


def max_lcn_of_cb(cb: List[CoreBlock]) -> LCN_EX:
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex


def _neuron_repl_prop(nbits: int, ntimeslot: int) -> int:
    """Get the proportion of neuron replication.

    scale = nbits(1 << wp) * n_timeslot(1 << lcn_ex)
    """
    return nbits * ntimeslot


class CoreMapper:
    """Manage to group, combine & place the network into the chip.

    TODO Integrate all the info of building the map.
    """

    core_blocks: List[CoreBlock] = field(default_factory=list)
