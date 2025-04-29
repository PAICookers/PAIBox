import math
import warnings
from typing import ClassVar, Literal, Optional, cast, overload

import numpy as np
from paicorelib import LCN_EX, ChipCoord, Coord, CoreMode, HwConfig, MaxPoolingEnable
from paicorelib import WeightWidth as WW
from paicorelib.framelib import OfflineFrameGen

from paibox.components import FullConnectedSyn, Neuron
from paibox.exceptions import (
    GraphBuildError,
    NotSupportedError,
    ResourceError,
    TruncationWarning,
)
from paibox.types import WEIGHT_DTYPE, WeightType
from paibox.utils import check_attr_same

from .conf_types import CoreConfig, CoreConfInChip, CorePlmConfig, NeuronConfig
from .constrs import GraphNodeConstrs
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
    CoreAbstract,
    DestNodeType,
    EdgeType,
    MergedSuccGroup,
    NeuSegment,
    NeuSegOfCoreBlock,
    NeuSegOfCorePlm,
    SourceNodeType,
    WRAMPackedType,
    WRAMUnpackedType,
    is_iw8,
)

# Get the fan-out by the combination rate of dendrites
if hasattr(HwConfig, "FANOUT_IW8"):
    FANOUT_IW8 = HwConfig.FANOUT_IW8
else:
    FANOUT_IW8 = [HwConfig.N_NEURON_MAX_ANN, 1364, 876, 512, 256, 128, 64, 32, 16, 8]


NEURON_PARAMS_BIT_LENGTH = 214  # A constant of frame definition


class CoreBlock(CoreAbstract):

    _parents: tuple[FullConnectedSyn, ...]
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
        self._lcn_locked = False
        self.core_coords = []
        self.chip_coord = _COORD_UNSET
        self.core_placements = dict()
        self.axon_segments = dict()
        self.neuron_segs_of_cb = []
        self._ordered_axons: list[SourceNodeType] = []
        """Axons in private + multicast order."""

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

    def assign_coord(
        self, chip_coord: Coord, allocated: list[Coord]
    ) -> tuple[list[Coord], list[Coord]]:
        self.core_coords = allocated
        self.chip_coord = chip_coord
        return allocated, []

    def copy(self):
        raise NotImplementedError

    @property
    def obj(self) -> tuple[FullConnectedSyn, ...]:
        return self._parents

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.ordered_axons), len(self.dest))

    @property
    def source(self) -> list[SourceNodeType]:
        """Ordered unique source nodes."""
        return cast(
            list[SourceNodeType], list(set([parent.source for parent in self.obj]))
        )

    @property
    def axons(self) -> list[SourceNodeType]:
        return self.source

    @property
    def dest(self) -> list[DestNodeType]:
        """Ordered unique destination nodes."""
        return cast(list[DestNodeType], list(set([parent.dest for parent in self.obj])))

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
    def ordered_axons(self) -> list[SourceNodeType]:
        return self._ordered_axons

    @ordered_axons.setter
    def ordered_axons(self, axons: list[SourceNodeType]):
        self._ordered_axons = axons
        self._lcn_ex = self._n_axon2lcn_ex()

    def group_axons(self) -> None:
        """Group the axons, including the private & the multicast parts.

        NOTE: Take the union of the private axons & the multicast axons, but sort the multicast axons first, then the \
            axons that are in the private part and not in the multicast part.
        """
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
    def build(cls, *synapses: FullConnectedSyn, rt_mode: CoreMode, seed: int = 0):
        """Group synapses & build `CoreBlock`."""
        if seed > (1 << 64) - 1:
            warnings.warn(
                f"random seed {seed} is too large, truncated into 64 bits.",
                TruncationWarning,
            )

        return cls(*synapses, mode=rt_mode, seed=seed)

    @classmethod
    def build_core_blocks(cls, msgrp: MergedSuccGroup) -> list["CoreBlock"]:
        core_blocks: list[CoreBlock] = []
        succ_nodes = list(msgrp.nodes)
        # TODO Currently the runtime mode is not taken into account for grouping constraints,
        # because in general, a network can only have one mode.
        mode = succ_nodes[0].mode
        if any(node.mode != mode for node in succ_nodes):
            raise NotSupportedError("mixed mode is not supported.")

        idx_of_sg = GraphNodeConstrs.apply_constrs(succ_nodes)

        for idx_lst in idx_of_sg:
            succ_edges: set[EdgeType] = set()
            for i in idx_lst:
                succ_edges.update(msgrp.outputs[succ_nodes[i]])

            core_block = CoreBlock.build(*succ_edges, rt_mode=mode)
            core_blocks.append(core_block)

        return core_blocks

    @classmethod
    def export_core_plm_config(cls, cb: "CoreBlock") -> CoreConfInChip:
        """Export the parameters of the core into a dictionary."""
        cb_config = dict()

        for coord, core_plm in cb.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config

    def dump(self, i: int = 0) -> None:
        tabs = "\t" * i
        print(f"{tabs}{self.name} with {self.n_core_required} cores:")
        print(f"{tabs}\tLCN: {self.lcn_ex}")
        for edge in self._parents:
            print(f"{tabs}\t{edge.name}: {edge.source.name} -> {edge.target.name}")


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

            # Get all core coordinates & replication ids.
            assert all(axon_dests[0].chip_coord == ad.chip_coord for ad in axon_dests)

            dest_core_coords = []
            for ad in axon_dests:
                dest_core_coords.extend(ad.core_coords)

            config = NeuronConfig(
                neu_seg, axon_coords, dest_core_coords, axon_dests[0].chip_coord
            )

            self.neu_configs[neu_seg.target] = config
            return None
        else:
            # neu_seg is a part of an output node
            assert isinstance(output_core_coord, Coord)
            assert isinstance(axon_addr_offset, int)

            axon_coords: list[AxonCoord] = []
            for i in range(axon_addr_offset, axon_addr_offset + neu_seg.n_neuron):
                # NOTE: The runtime already supports cases where the #N of output nodes > the base fan-in of axons,
                # so this form of output node allocation can be implemented.
                # See class `PAIBoxRuntime` in runtime.py for details.
                # May be changed at any time without notice.
                axon_coords.append(
                    AxonCoord(*divmod(i, HwConfig.N_FANIN_PER_DENDRITE_MAX))
                )

            config = NeuronConfig(
                neu_seg,
                axon_coords,
                [output_core_coord],
                # output chip coordinate for output node
                _BACKEND_CONTEXT.output_chip_addr,
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
    def pool_max(self) -> MaxPoolingEnable:
        return self.parent.pool_max

    @property
    def n_working_dendrite(self) -> int:
        """The number of actual working dendrites.

        NOTE: n_neuron * (2^comb_rate) = n_neuron << comb_rate
        """
        return self.n_neuron << self.dendrite_comb_rate

    @property
    def source(self) -> list[SourceNodeType]:
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
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex
