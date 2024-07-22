import warnings
from functools import cached_property
from typing import ClassVar, Literal, Optional, overload

import numpy as np
from paicorelib import LCN_EX, ChipCoord, Coord, CoreMode, HwConfig, MaxPoolingEnable
from paicorelib import WeightWidth as WW

from paibox.components import FullConnectedSyn, Neuron
from paibox.exceptions import (
    GraphBuildError,
    NotSupportedError,
    ResourceError,
    TruncationWarning,
)
from paibox.types import WEIGHT_DTYPE, WeightType
from paibox.utils import check_attr_same

from .conf_template import CoreConfig, CoreConfInChip, CorePlmConfig, NeuronConfig
from .context import _BACKEND_CONTEXT
from .segment_utils import aligned_coords, get_axon_segments, get_neu_segments
from .types import (
    _COORD_UNSET,
    WRAM_PACKED_DTYPE,
    WRAM_UNPACKED_DTYPE,
    AxonCoord,
    AxonSegment,
    CoreAbstract,
    DestNodeType,
    NeuSegment,
    NeuSegOfCoreBlock,
    NeuSegOfCorePlm,
    SourceNodeType,
    WRAMPackedType,
    WRAMUnpackedType,
    is_iw8,
)


class CoreBlock(CoreAbstract):

    _parents: tuple[FullConnectedSyn, ...]
    _routing_id: int
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
    axon_segments: dict[SourceNodeType, AxonSegment] = dict()
    """A dictionary of segments of each axon(source node)."""
    neuron_segs_of_cb: NeuSegOfCoreBlock = []
    """Neuron segments in the core block. Each element in the list represents the neuron    \
        segments in core placement.
    """

    def __init__(
        self,
        *parents: FullConnectedSyn,
        routing_id: int,
        seed: int,
        mode: CoreMode = CoreMode.MODE_SNN,
        name: Optional[str] = None,
    ) -> None:
        """Core blocks in SNN mode.

        Args:
            - parents: the parent synapses.
            - routing_id: id of routing group.
            - seed: random seed. Default value is 0.
            - mode: runtime mode of the core block. Default value is `MODE_SNN`.
            - name: name of the core block. Optional.
        """
        super().__init__(name)
        self._parents = parents
        self._routing_id = routing_id
        self.rt_mode = mode
        self.seed = seed
        self._lcn_ex = self._n_axon2lcn_ex()

        self.target_lcn = LCN_EX.LCN_1X
        self._lcn_locked = False
        self.core_coords = []
        self.chip_coord = Coord(_COORD_UNSET, _COORD_UNSET)
        self.core_placements = dict()
        self.axon_segments = dict()
        self.neuron_segs_of_cb = []

    def group_neurons(
        self, optim_target: Literal["latency", "core", "both"] = "both"
    ) -> None:
        """Group the neurons to determine the #N of cores required."""
        if not self._lcn_locked:
            raise GraphBuildError("group the neurons after 'lcn_ex' is locked.")

        self.neuron_segs_of_cb = get_neu_segments(
            self.dest, self.n_fanout, self.n_neuron_repl, optim_target
        )

    def core_plm_alloc(self) -> None:
        """Allocate `CoreBlock` to physical cores."""
        if not self._lcn_locked:
            raise GraphBuildError("allocate core placements after 'lcn_ex' is locked.")

        for i, coord in enumerate(self.core_coords):
            # assert self.get_raw_weight_of_coord(i)[0].shape[0] == self.n_axon
            self.core_placements[coord] = CorePlacement.build(self, i)

    def _get_syn_of(
        self, src: SourceNodeType, dest: DestNodeType
    ) -> Optional[FullConnectedSyn]:
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
        ) > LCN_EX.LCN_64X:
            _max_n_axons = self.n_fanin_base << LCN_EX.LCN_64X
            raise ResourceError(
                f"required LCN out of range {LCN_EX.LCN_64X} ({lcn}). The number of axons "
                f"must be <= {_max_n_axons}, but synapses {self._obj_repr} have a total of "
                f"{self.n_axon} axons."
            )

        return LCN_EX(lcn)

    def copy(self):
        raise NotImplementedError

    @property
    def obj(self) -> tuple[FullConnectedSyn, ...]:
        return self._parents

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.source), len(self.dest))

    @property
    def source(self) -> list[SourceNodeType]:
        """Ordered unique source nodes."""
        return list(set([parent.source for parent in self.obj]))

    @property
    def axons(self) -> list[SourceNodeType]:
        return self.source

    @property
    def dest(self) -> list[DestNodeType]:
        """Ordered unique destination nodes."""
        return list(set([parent.dest for parent in self.obj]))

    def n_axon_of(self, index: int) -> int:
        """Get the #N of axons of `index`-th source neuron."""
        return self.axons[index].num_out

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
        if lcn_ex > LCN_EX.LCN_64X:
            raise ResourceError(
                f"required LCN out of range {LCN_EX.LCN_64X} ({lcn_ex})."
            )

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
        if not check_attr_same(self.dest, "tick_wait_start"):
            raise AttributeError(
                "attribute 'tick_wait_start' of the core block are not equal."
            )

        return self.dest[0].tick_wait_start

    @property
    def twe(self) -> int:
        """Attribute `tick_wait_end.`"""
        if not check_attr_same(self.dest, "tick_wait_end"):
            raise AttributeError(
                "attribute 'tick_wait_end' of the core block are not equal."
            )

        return self.dest[0].tick_wait_end

    @property
    def n_axon(self) -> int:
        return sum(s.num_out for s in self.axons)

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
        assert [] not in self.neuron_segs_of_cb  # TODO if it never happens, remove it.

        return [
            sum(seg.n_neuron for seg in neuron_segs)
            for neuron_segs in self.neuron_segs_of_cb
        ]

    def group_axons(self) -> None:
        if not self._lcn_locked:
            raise GraphBuildError("get axon segments after 'lcn_ex' is locked.")

        self.axon_segments = get_axon_segments(
            self.axons, self.n_timeslot, self.n_fanin_base
        )

    @cached_property
    def raw_weight_of_dest(self) -> list[WeightType]:
        """Merge and then split the weight matrix according to the grouping of neurons."""
        # The concatenated weight for each destination node.
        w_of_neurons: list[WeightType] = []

        for d in self.dest:
            # The weights for each destination node.
            w_of_dest = []

            for s in self.source:
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
        """Get the raw weight of a coordinate(on each `CorePlacement`)."""
        w_of_neu_segs: list[WeightType] = []

        for neu_seg in self.neuron_segs_of_cb[idx]:
            w_of_dest = self.raw_weight_of_dest[self.dest.index(neu_seg.target)]
            w_of_neu_seg = w_of_dest[:, neu_seg.index].copy()
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
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"

    @property
    def _obj_repr(self) -> str:
        """The representation of the names of target objects."""
        return ", ".join(n.name for n in self.obj)

    @classmethod
    def build(
        cls,
        *synapses: FullConnectedSyn,
        routing_id: int,
        rt_mode: CoreMode,
        seed: int = 0,
    ):
        """Group synapses & build `CoreBlock`."""
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )

        return cls(*synapses, routing_id=routing_id, mode=rt_mode, seed=seed)

    @classmethod
    def export_core_plm_config(cls, cb: "CoreBlock") -> CoreConfInChip:
        """Export the parameters of the core into a dictionary."""
        cb_config = dict()

        for coord, core_plm in cb.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config


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
        # See the note of function `_weight_ram_mapping` below.
        n_fold = (
            self.n_timeslot
            if self.rt_mode.is_snn
            else 1 << (self.dendrite_comb_rate - 3)
        )

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
        """Map the raw weights to the weight RAM(WRAM). The mapping is different for both input widths.

        NOTE: When the input width is 8 bits, no neurons need to be mapped to the WRAM when the combination rate of \
            dentrites >= 8, while some neurons need to be mapped to the WRAM when < 8.

            When the input width is 8 bits and with the combination rate of dentrites > 3, the mapping of weights   \
            becomes the key to limiting neuron capacity. In this case, if the weight accuracy is less than 8 bits   \
            (which may also occur when the weight accuracy is optimized), the weight cannot be folded directly in   \
            the fan-in expansion direction, otherwise the column of the WRAM will exceed the upper limit(512).      \

            A portion of the fan-in needs to be expanded to an unfilled portion in the direction of the weight      \
            accuracy. At this point, n_fold=n_timeslot/(8/n_weight_bits)=2^(dendrite_comb_rate - 3). For example,   \
            for LCN_8X & WW8, the n_fold is 3. For LCN_32X & WW4, the n_fold is 4 (instead of 5).

        TODO Now, in ANN mode, only the mapping of 8-bit weights is supported. The weight accuracy optimization is  \
            supposed to disable manually for now.
        """
        if not self.rt_mode.is_snn and self.weight_width < WW.WEIGHT_WIDTH_8BIT:
            raise NotSupportedError("only support 8-bit weights in ANN mode.")

        _weights_folded = self._fold_raw_weights(self.raw_weights)
        row, col = _weights_folded.shape
        # The 1152*512 unpacked weight
        w_unpacked = np.zeros(self.WRAM_BASE_SHAPE, dtype=WRAM_UNPACKED_DTYPE)

        if self.n_weight_bits == 1:
            w_unpacked[:row, :col] = _weights_folded
        else:
            # (N, M)(int8) -> (M, N, 1)(uint8)
            w_folded_3d = np.expand_dims(_weights_folded.T, axis=2).astype(
                WRAM_UNPACKED_DTYPE
            )

            _n_group_bit = HwConfig.N_FANIN_PER_DENDRITE_ANN

            for i in range(col):
                # For every column, unpack the array (N, 1) -> (N, n_weight_bits)
                unpacked = np.unpackbits(
                    w_folded_3d[i],
                    axis=1,
                    count=self.n_weight_bits,
                    bitorder=HwConfig.WEIGHT_BITORDER,
                )

                if self.rt_mode.is_snn:
                    w_unpacked[
                        :row, self.n_weight_bits * i : self.n_weight_bits * (i + 1)
                    ] = unpacked
                else:
                    # In the case of 8-bit input width, the weights are mapped differently
                    for bit in range(self.n_weight_bits):
                        w_unpacked[bit * _n_group_bit : bit * _n_group_bit + row, i] = (
                            unpacked[:, bit]
                        )

        return self._weight_pack(w_unpacked)

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
            _raw_weight = np.append(
                raw_weight,
                np.zeros((n_row_padding, raw_col), dtype=WEIGHT_DTYPE),
                axis=0,
            )
        else:
            _raw_weight = raw_weight

        w_splited = np.vsplit(_raw_weight, n_fold)
        w_folded = np.zeros((expected_row, raw_col * n_fold), dtype=WEIGHT_DTYPE)

        for i, j in np.ndindex((n_fold, raw_col)):
            w_col = w_splited[i][:, j]
            w_folded[:, n_fold * j + i] = w_col

        return w_folded

    @staticmethod
    def _weight_pack(w_unpacked: WRAMUnpackedType) -> WRAMPackedType:
        """Convert the unpacked weights into a mapping format, corresponding to the WRAM address, each address      \
            contains 18 uint64.
            (1152, 512) -> T -> (512*18, 64) -> (512*18, 8) uint8 -> (512*18, 1) uint64 -> (512, 18) uint64.
        """
        _n_bit_packed = WRAM_PACKED_DTYPE(1).nbytes * 8  # #N bit of packed dtype
        # #N of u64 on each NRAM address
        _n_u64_naddr = CorePlacement.WRAM_BASE_SHAPE[0] // _n_bit_packed

        # Reshape to 64 columns to avoid contiguous problem.
        w_unpacked_aligned = w_unpacked.T.reshape(-1, _n_bit_packed)
        # (512*18, 64) uint8 -> (512*18, 8) uint8
        w_packed_u8 = np.packbits(
            w_unpacked_aligned, axis=1, bitorder=HwConfig.WEIGHT_BITORDER
        )
        # (512*18, 8) uint8 -> (512*18, 1) uint64 -> (512, 18) uint64
        w_packed_u64 = w_packed_u8.view(WRAM_PACKED_DTYPE).reshape(-1, _n_u64_naddr)
        w_packed_u64.setflags(write=False)

        return w_packed_u64

    def export_param_config(self) -> CoreConfig:
        _mode_params = self.rt_mode.conf

        # fmt: off
        cb_config = CoreConfig(
            self.name,                          # name of the core
            self.weight_width,             # weight_precision
            self.lcn_ex,                        # lcn_extension
            _mode_params[0],                    # input_width_format
            _mode_params[1],                    # spike_width_format
            self.n_working_dendrite,            # num_dendrite
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
        self, neu_seg: NeuSegment, axon_dests: list[CoreBlock]
    ) -> None: ...

    @overload
    def export_neu_config(
        self,
        neu_seg: NeuSegment,
        *,
        output_core_coord: Coord,
        axon_addr_offset: int,
    ) -> int: ...

    def export_neu_config(
        self,
        neu_seg: NeuSegment,
        axon_dests: Optional[list[CoreBlock]] = None,
        output_core_coord: Optional[Coord] = None,
        axon_addr_offset: Optional[int] = None,
    ) -> Optional[int]:
        """Export the neuron configuration."""
        if isinstance(axon_dests, list):
            axon_coords = aligned_coords(
                neu_seg.index,
                axon_dests[0].axon_segments[neu_seg.target],
                neu_seg.target.delay_relative,
                axon_dests[0].n_timeslot,
                is_iw8(axon_dests[0].rt_mode),
            )

            # Get all core coordinates and replication ids.
            assert all(axon_dests[0].chip_coord == ad.chip_coord for ad in axon_dests)

            dest_core_coords = []
            for ad in axon_dests:
                dest_core_coords.extend(ad.core_coords)

            config = NeuronConfig.encapsulate(
                neu_seg, axon_coords, dest_core_coords, axon_dests[0].chip_coord
            )

            self.neu_configs[neu_seg.target] = config
            return None
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)
            assert isinstance(axon_addr_offset, int)

            axon_coords = [
                AxonCoord(0, i)
                for i in range(axon_addr_offset, axon_addr_offset + neu_seg.n_neuron)
            ]

            config = NeuronConfig.encapsulate(
                neu_seg,
                axon_coords,
                [output_core_coord],
                # output chip coordinate for output node
                _BACKEND_CONTEXT["output_chip_addr"],
            )

            self.neu_configs[neu_seg.target] = config

            return axon_addr_offset + neu_seg.n_neuron

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
    def n_working_dendrite(self) -> int:
        """The number of actual working dendrites. IN ANN mode, the number of working   \
            dendrites N <= 4096. In SNN mode, N <= 512.

        NOTE: n_neuron * (2^comb_rate) = n_neuron << comb_rate
        """
        return self.n_neuron << self.dendrite_comb_rate

    @property
    def source(self) -> list[SourceNodeType]:
        return self.parent.source

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
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex


# Get the fan-out by the combination rate of dendrites
if hasattr(HwConfig, "FANOUT_IW8"):
    FANOUT_IW8 = HwConfig.FANOUT_IW8  # type: ignore
else:
    FANOUT_IW8 = [HwConfig.N_NEURON_MAX_ANN, 1364, 876, 512, 256, 128, 64, 32, 16, 8]
