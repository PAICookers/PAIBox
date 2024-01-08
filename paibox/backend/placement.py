import sys
import warnings
from dataclasses import field
from functools import cached_property
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
    overload,
)

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
    Coord,
    CoreMode,
    CoreModeDict,
    HwConfig,
    HwCore,
    MaxPoolingEnable,
    NeuronSegment,
)
from paicorelib import WeightPrecision as WP

from paibox.base import NeuDyn, PAIBoxObject
from paibox.exceptions import BuildError, NotSupportedError, ResourceError
from paibox.projection import InputProj
from paibox.synapses import SynSys
from paibox.utils import check_attr_same, count_unique_elem

from .conf_template import CoreConfig, CorePlacementConfig, NeuronConfig
from .context import _BACKEND_CONTEXT

SourceNodeType: TypeAlias = Union[InputProj, NeuDyn]
DestNodeType: TypeAlias = NeuDyn
WeightType: TypeAlias = NDArray[np.int8]  # raw int8 weights
WeightRamType: TypeAlias = NDArray[np.uint64]  # uint64 weights mapped in weight RAM
NeuSeg = NamedTuple("NeuSeg", [("parent", DestNodeType), ("segment", NeuronSegment)])


class CoreAbstract(HwCore, PAIBoxObject):
    supported_wp: ClassVar[Tuple[WP, ...]] = (
        WP.WEIGHT_WIDTH_1BIT,
        # WP.WEIGHT_WIDTH_2BIT,  # Not verified
        # WP.WEIGHT_WIDTH_4BIT,  # Not verified
        WP.WEIGHT_WIDTH_8BIT,  # Not verified
    )
    """Supported weight precision."""

    supported_mode: ClassVar[Tuple[CoreMode, ...]] = (CoreMode.MODE_SNN,)
    """Supported core modes."""


class CoreBlock(CoreAbstract):
    """Core Block for `MODE_SNN` ONLY."""

    mode: ClassVar[CoreMode] = CoreMode.MODE_SNN

    def __init__(
        self,
        *parents: SynSys,
        weight_precision: WP,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parents: the parent synapses.
            - weight_precision: the precision of weight matrix.
            - seed: the random seed.
            - name: the name of the core block. Optional.
        """

        super().__init__(name)
        self._parents = parents
        self._lcn_ex = n_axon2lcn_ex(self.n_axon, self.n_fanin_max)
        self.weight_precision = weight_precision

        self.seed = seed
        """Random seed, legal integer, no more than uint64."""

        self.target_lcn = LCN_EX.LCN_1X
        """The target(destination core block) LCN."""

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
            self.axons, self.n_timeslot, self.n_fanin_max
        )
        """A dictionary of segments of each axon(source node)."""

        self.neuron_segs_of_cb: List[List[NeuSeg]] = []
        """Neuron segments in the core block. Each element in the list \
            represents the Neuron segments in core placement(physical core).
        """

    def group_neurons(self, method: Literal["catagory", "dense"] = "catagory") -> None:
        """Group the neurons to determine the #N of cores required."""
        if not self.lcn_locked:
            raise BuildError("Group the neurons after lcn_ex is locked.")

        # First, get the neuron segments.
        self.neuron_segs_of_cb = get_neu_segments(
            self.dest,
            self.neuron_capacity,
            _addr_ram_interval(self.n_weight_bits, self.n_timeslot),
            method=method,
        )

    def core_plm_alloc(self) -> None:
        """Allocate the `CoreBlock` to the physical cores."""
        if not self.lcn_locked:
            raise BuildError("Allocate the core placements after lcn_ex is locked.")

        for i, coord in enumerate(self.core_coords):
            # assert self.get_raw_weight_of_coord(i)[0].shape[0] == self.n_axon
            self.core_placements[coord] = CorePlacement.build(
                self, i, self.get_raw_weight_of_coord(i)
            )

    def _get_syn_of(self, src: SourceNodeType, dest: DestNodeType) -> Optional[SynSys]:
        for syn in self.obj:
            if syn.source == src and syn.dest == dest:
                return syn

        return None

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

        FIXME This method ONLY works in SNN mode. For ANN mode, use table lookup?
        """
        return (self.n_dendrite_max >> self.lcn_ex) // self.n_dendrite_per_neuron

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
        return len(self.neuron_segs_of_cb)

    @property
    def n_dendrite_per_neuron(self) -> int:
        """Multiple dendrites will be combined to achieve higher    \
            precision weights.

        FIXME The limit on the number of dendrites in SNN/ANN modes \
            is different, which affects the capacity of neurons in  \
            physical core.
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
        """Set or adjust the lcn_ex & lock."""
        if lcn_ex > LCN_EX.LCN_64X:
            raise ResourceError(
                f"LCN extension required out of {LCN_EX.LCN_64X}: {lcn_ex}"
            )

        self._lcn_ex = lcn_ex
        self.lcn_locked = True

    @property
    def n_timeslot(self) -> int:
        return 1 << self.lcn_ex

    @property
    def tick_wait_start(self) -> int:
        if not check_attr_same(self.dest, "tick_wait_start"):
            raise AttributeError(
                "Attribute 'tick_wait_start' of the core block are not equal."
            )

        return self.dest[0].tick_wait_start

    @property
    def tick_wait_end(self) -> int:
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
            if self.mode is CoreMode.MODE_ANN
            else HwConfig.N_DENDRITE_MAX_SNN
        )

    @property
    def n_neuron(self) -> int:
        return sum(d.num_in for d in self.dest)

    def n_neuron_of(self, idx: int) -> int:
        """Get the #N of neurons of `idx`-th dest neurons."""
        return self.dest[idx].num_in

    @property
    def n_neuron_of_plm(self) -> List[int]:
        """A list of the #N of neurons on each `CorePlacement`.

        FIXME Different in SNN/ANN mode.
        """
        if len(self.core_coords) == 0:
            raise BuildError(f"Do this after coordinates assignment.")

        # Get #N of neurons on each `CorePlacement` according to the
        # maximum address required of neuron segments on each core placement.
        n = [
            self.neuron_segs_of_cb[i][-1].segment.addr_max
            for i in range(self.n_core_required)
        ]
        return n

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

    def __repr__(self) -> str:
        return f"<{self.name} at 0x{id(self):x} of target '{self.obj}'>"

    def __str__(self) -> str:
        return f"<{self.name} of target '{self.obj}'>"

    @classmethod
    def build(cls, *synapses: SynSys, seed: int = 0):
        """Combine the SAME weight precision synapses and build the `CoreBlock`."""
        # Check wether weight precision of all synapses equal.
        if not check_attr_same(synapses, "weight_precision"):
            raise NotSupportedError("Mixed weight precision is not supported yet")

        if (wp := synapses[0].weight_precision) > max(cls.supported_wp):
            raise NotSupportedError(f"{wp.name} is not supported yet.")

        elif wp not in cls.supported_wp:
            # Treat 4-bit weights as 8-bit weights.
            wp = cls.supported_wp[-1]

        if seed > (1 << 64) - 1:
            warnings.warn(
                f"Random seed {seed} is too large, truncated into 64 bits!", UserWarning
            )

        return cls(*synapses, weight_precision=wp, seed=seed)

    @classmethod
    def export_core_plm_config(cls, cb: "CoreBlock") -> Dict[Coord, CoreConfig]:
        """Export the parameters of the core into a dictionary."""
        cb_config = dict()

        for coord, core_plm in cb.core_placements.items():
            cb_config[coord] = CorePlacement.export_param_config(core_plm)

        return cb_config


class CorePlacement(CoreAbstract):
    """The divided synapse placed on a single CORE."""

    _excluded_vars = ()

    n_core_required: ClassVar[int] = 1
    weight_ram_shape: ClassVar[Tuple[int, int]] = (
        HwConfig.N_FANIN_PER_DENDRITE_SNN,
        HwConfig.N_DENDRITE_MAX_SNN,
    )
    """SNN mode ONLY."""

    def __init__(
        self,
        parent: CoreBlock,
        routing_coord: Coord,
        n_neuron: int,
        *,
        raw_weights: List[WeightType],
        neu_segs: List[NeuSeg],
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - parent: the parent core block.
            - idx: The index number where this object is located.
            - n_neuron: the number of neurons used in the physical core.
            - raw_weights: the raw weights in the physical core.
            - neu_segs: The segment of the neurons in the physical core.
        """
        super().__init__(name)

        self.parent = parent
        self.coord = routing_coord
        """Routing coordinate"""

        self.n_neuron = n_neuron

        self._weights_folded = self._fold_raw_weights(raw_weights)
        """The folded weights."""

        self.neu_segs = neu_segs
        self.neu_configs: Dict[NeuDyn, NeuronConfig] = dict()

    @classmethod
    def build(cls, parent: CoreBlock, idx: int, raw_weights: List[WeightType]):
        coord = parent.core_coords[idx]
        n_neuron = parent.n_neuron_of_plm[idx]
        neu_segs = parent.neuron_segs_of_cb[idx]

        return cls(
            parent,
            coord,
            n_neuron,
            raw_weights=raw_weights,
            neu_segs=neu_segs,
        )

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
        w_unpacked = np.zeros(self.weight_ram_shape, dtype=np.uint8)

        if self.n_weight_bits == 1:
            w_unpacked[:row, :col] = self._weights_folded
        else:
            # [N*M] -> [M*N*1]
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

        assert np.max(w_unpacked, axis=None) <= 1
        assert np.min(w_unpacked, axis=None) >= 0

        # Convert the unpacked weights into a mapping format,
        # corresponding to the RAM address, each address contains 18 uint64.
        # [512 * 1152] -> [(512*18) * 64](uint8). Reshape to 64 columns to avoid contiguous problem.
        w_unpacked_T_rehaped = w_unpacked.T.reshape(-1, 64)

        # [(512*18) * 64](uint8) -> [(512*18) * 8](uint8)
        w_packed_u8 = np.packbits(
            w_unpacked_T_rehaped, axis=1, bitorder=HwConfig.WEIGHT_BITORDER
        )
        # [(512*18) * 8](uint8) -> [(512*18) * 1](uint64) -> [512 * 18](uint64)
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
            self.tick_wait_start,               # tick_wait_start
            self.tick_wait_end,                 # tick_wait_end
            _mode_params[2],                    # snn_mode_en
            self.target_lcn,                    # target_lcn
            _BACKEND_CONTEXT.test_chip_addr,    # test_chip_addr
        )
        # fmt: on
        return cb_config

    @overload
    def export_neu_config(self, neu_seg: NeuSeg, axon_dests: List[CoreBlock]) -> None:
        ...

    @overload
    def export_neu_config(
        self,
        neu_seg: NeuSeg,
        *,
        output_core_coord: Coord,
        axon_addr_offset: int,
    ) -> int:
        ...

    def export_neu_config(
        self,
        neu_seg: NeuSeg,
        axon_dests: Optional[List[CoreBlock]] = None,
        *,
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

            # Get all core coordinates then get the RID.
            dest_core_coords = []
            for axon_dest in axon_dests:
                dest_core_coords.extend(axon_dest.core_coords)

            config = NeuronConfig.encapsulate(
                neu_seg.parent,
                neu_seg.segment.n_neuron,
                neu_seg.segment.addr_ram,
                neu_seg.segment.addr_offset,
                axon_coords,
                dest_core_coords,
                _BACKEND_CONTEXT["local_chip_addr"],  # Here is local chip coordinate
            )

            self.neu_configs[neu_seg.parent] = config
        else:
            assert isinstance(output_core_coord, Coord)
            assert isinstance(axon_addr_offset, int)

            axon_coords = [
                AxonCoord(0, i)
                for i in range(
                    axon_addr_offset, axon_addr_offset + neu_seg.segment.n_neuron
                )
            ]

            config = NeuronConfig.encapsulate(
                neu_seg.parent,
                neu_seg.segment.n_neuron,
                neu_seg.segment.addr_ram,
                neu_seg.segment.addr_offset,
                axon_coords,
                [output_core_coord],
            )

            self.neu_configs[neu_seg.parent] = config

            return axon_addr_offset + neu_seg.segment.n_neuron

    def export_core_plm_config(self) -> CorePlacementConfig:
        core_param = self.export_param_config()

        return CorePlacementConfig.encapsulate(
            self.parent.seed,
            self.weight_ram,
            core_param,
            self.neu_configs,
        )

    @property
    def mode(self) -> CoreMode:
        return self.parent.mode

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
    def tick_wait_start(self) -> int:
        return self.parent.tick_wait_start

    @property
    def tick_wait_end(self) -> int:
        return self.parent.tick_wait_end

    @property
    def n_dendrite(self) -> int:
        return self.n_neuron * self.n_weight_bits * self.n_timeslot

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
    def weight_ram(self) -> WeightRamType:
        return self._weight_ram_mapping()

    def __len__(self) -> int:
        return self.n_core_required


def n_axon2lcn_ex(n_axon: int, fan_in_max: int) -> LCN_EX:
    """Convert #N(of axons) to `LCN_EX`.

    Description:
        LCN_EX = log2[ceil(#N/fan-in per dendrite)], where LCN_EX = 0 is `LCN_1X`.
    """
    if n_axon < 1:
        raise ValueError(f"The #N of axons must be positive, but got {n_axon}")

    if (lcn_bit := ((n_axon - 1) // fan_in_max).bit_length()) > LCN_EX.LCN_64X:
        raise ResourceError(
            f"LCN extension required out of {LCN_EX.LCN_64X}: {lcn_bit}"
        )

    return LCN_EX(lcn_bit)


def max_lcn_of_cb(cb: List[CoreBlock]) -> LCN_EX:
    """Find the max LCN extenion of previous grouped synapses"""
    return max(cb, key=lambda cb: cb.lcn_ex).lcn_ex


def _addr_ram_interval(nbits: int, ntimeslot: int) -> int:
    """Get the interval of RAM address.

    interval = nbits(1 << wp) * n_timeslot(1 << lcn_ex)
    """
    return nbits * ntimeslot


def get_neu_segments(
    neu_groups: List[NeuDyn],
    capacity: int,
    interval: int,
    *,
    method: Literal["catagory", "dense"],
) -> List[List[NeuSeg]]:
    """Get the neuron segments by given a method, "catagory" or "dense"."""

    if method == "catagory":
        return _get_neu_segments_catagory(neu_groups, capacity, interval)
    elif method == "dense":
        return _get_neu_segments_dense(neu_groups, capacity, interval)
    else:
        raise ValueError(f"Method '{method}' not defined!")


def _get_neu_segments_catagory(
    neu_groups: List[NeuDyn], capacity: int, interval: int = 1
) -> List[List[NeuSeg]]:
    """Group the neuron groups by category."""
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
    neu_groups: List[NeuDyn], capacity: int, interval: int = 1
) -> List[List[NeuSeg]]:
    """Dense grouping. Based on method `catagory`, use the greedy algorithm to \
        group the remaining neuron groups.

    FIXME Not fully verified.
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
    n_cur_neuron = 0
    n_cur_reg = 0

    def backtrack(i: int, cur_addr_offset: int, taken: List[NeuSeg]) -> None:
        nonlocal n_cur_reg
        nonlocal n_cur_neuron

        if i == n_core_max or n_cur_reg == n_core_max:
            return

        if n_cur_neuron + rest_segs[n_cur_reg].segment.n_neuron > capacity:
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
            # Offset = n_neuron * interval
            cur_addr_offset += rest_segs[n_cur_reg].segment.n_neuron * interval
            n_cur_neuron += rest_segs[n_cur_reg].segment.n_neuron
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
        - tr_max: The maximum value of the time slot(=n_timeslot).

    TODO Provide an alternative when failed using this method.
    """

    def _seg_alloc(axon: SourceNodeType) -> AxonSegment:
        """Allocate an axon segment, return the next offset of axon address."""
        nonlocal offset

        # The width of assigned address
        if axon.num_out % tr_max > 0:
            addr_width = axon.num_out // tr_max + 1
            # n_axon_rest = axon.num_out % addr_width
        else:
            addr_width = axon.num_out // tr_max
            # n_axon_rest = 0

        if offset + addr_width > fan_in_max:
            raise ResourceError(
                f"Axons address out of range {fan_in_max}: {offset + addr_width}"
            )

        cur_offset = offset
        offset += addr_width

        return AxonSegment(axon.num_out, addr_width, cur_offset)

    offset = 0
    axon_segments = dict()

    for axon in axons:
        segment = _seg_alloc(axon)
        axon_segments[axon] = segment

    return axon_segments


def aligned_coords(
    neu_index: slice, axon_seg: AxonSegment, delay: int, dest_n_timeslot: int
) -> List[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment.

    The length of axon coordinates is the same as `neu_index`.
    """
    axon_coords = []
    addr_width = axon_seg.addr_width
    addr_offset = axon_seg.addr_offset

    """
        tick_relative = n_timeslot * (delay - 1) + tr_offset (start & end)
    """
    tr_base = dest_n_timeslot * (delay - 1)

    tr_offset_start, tr_offset_stop = (
        neu_index.start // addr_width,
        neu_index.stop // addr_width,
    )
    addr_start, addr_stop = (neu_index.start % addr_width, neu_index.stop % addr_width)

    if tr_offset_stop == tr_offset_start:
        for addr in range(addr_start, addr_stop):
            axon_coords.append(AxonCoord(tr_base + tr_offset_start, addr_offset + addr))
    else:
        for addr in range(addr_start, addr_width):
            axon_coords.append(AxonCoord(tr_base + tr_offset_start, addr_offset + addr))

        for tr in range(tr_offset_start + 1, tr_offset_stop):
            for addr in range(addr_width):
                axon_coords.append(AxonCoord(tr_base + tr, addr_offset + addr))

        for addr in range(addr_stop):
            axon_coords.append(AxonCoord(tr_base + tr_offset_stop, addr_offset + addr))

    return axon_coords


class CoreMapper:
    """Manage to group, combine & place the network into the chip.

    TODO Integrate all the info of building the map.
    """

    core_blocks: List[CoreBlock] = field(default_factory=list)
