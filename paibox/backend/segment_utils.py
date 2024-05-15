import sys
import warnings
from functools import partial
from math import ceil
from typing import Dict, List, Literal, NamedTuple, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from paicorelib import AxonCoord, AxonSegment, NeuronSegment

from paibox.base import NeuDyn
from paibox.exceptions import ParameterInvalidWarning, ResourceError

from .graphs_types import DestNodeType, SourceNodeType


class NeuSeg(NamedTuple):
    parent: DestNodeType
    segment: NeuronSegment

    @property
    def n_neuron(self) -> int:
        return self.segment.n_neuron

    @property
    def n_addr(self) -> int:
        return self.segment.n_addr

    def __str__(self) -> str:
        return f"NeuSeg {self.parent.name} at offset {self.segment.addr_offset}"


NeuSlice: TypeAlias = slice
NeuSegOfCorePlm: TypeAlias = List[NeuSeg]
NeuSegOfCoreBlock: TypeAlias = List[NeuSegOfCorePlm]


def _place_seperately(
    seg_slices_dict: Dict[NeuDyn, List[NeuSlice]], repl_prop: int
) -> NeuSegOfCoreBlock:
    neu_segs_of_cb = []

    for neu, seg_slices in seg_slices_dict.items():
        neu_segs_of_cb.extend(
            [
                [NeuSeg(neu, NeuronSegment(seg_slice, 0, repl_prop))]
                for seg_slice in seg_slices
            ]
        )

    return neu_segs_of_cb


def _coarse_group(
    neu: NeuDyn,
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> List[NeuSlice]:
    """Group neurons with 'average' or 'maximum capacity' load type.

    NOTE: Group neuron seperately, like [N1], [N2], ..., [Nn]. For each neuron, \
        take unrolling factor into consideration, then distribute the neuron to \
        the cores evently.

        #N of cores required of nx = ceil(Ni / capacity) * uf
        Average load of nx = ceil(nx / (#N of cores required of nx))
    """

    def _average_load(n: int, n_part: int) -> List[int]:
        """Distribute #num into #n_part parts evently."""
        quotient = ceil(n / n_part)
        rest = n - (n_part - 1) * quotient

        return [quotient] * (n_part - 1) + [rest]

    def _max_capacity_load(n: int) -> List[int]:
        nonlocal capacity
        n_part = ceil(n / capacity)
        rest = n - (n_part - 1) * capacity

        return [capacity] * (n_part - 1) + [rest]

    neu_seg_slices: List[NeuSlice] = []
    n_neuron = neu.num_out

    if load_type == "average":
        n_core_required = ceil(n_neuron / capacity) * neu.unrolling_factor
        dist = _average_load(n_neuron, n_core_required)
    else:
        dist = _max_capacity_load(n_neuron)

    _sum = 0
    for d in dist:
        neu_seg_slices.append(NeuSlice(_sum, _sum + d, 1))
        _sum += d

    return neu_seg_slices


def _get_nsg_opt_core(
    seg_slices_dict: Dict[NeuDyn, List[NeuSlice]], capacity: int, repl_prop: int
) -> NeuSegOfCoreBlock:
    neu_segs_of_cb: NeuSegOfCoreBlock = []  # The final result
    raise_warning = False

    for neu in seg_slices_dict:
        if neu.unrolling_factor > 1:
            neu.unrolling_factor = 1
            raise_warning = True

    if raise_warning:
        warnings.warn(
            "when grouping neurons with 'core' optimization, unrolling "
            "factor greater than 1 is invalid. Modified to 1.",
            ParameterInvalidWarning,
        )

    neu_segs_basic = _place_seperately(seg_slices_dict, repl_prop)

    # Retrive the list of `NeuSeg`
    neu_segs_max_capa = [
        neu_segs for neu_segs in neu_segs_basic if neu_segs[0].n_neuron == capacity
    ]
    neu_segs_of_cb.extend(neu_segs_max_capa)

    # The remaining segments will be reorganized. Only retrive the `NeuSeg`
    neu_segs_not_full = [
        neu_segs[0] for neu_segs in neu_segs_basic if neu_segs[0].n_neuron < capacity
    ]
    neu_segs_not_full.sort(key=lambda neu_seg: neu_seg.n_neuron, reverse=True)

    # The remaining neuron segments are placed into at most `n_core_req_max` cores.
    n_core_req_max = len(neu_segs_not_full)
    cur_n_neuron = 0
    n_cur_reg = 0

    def backtrack(i: int, cur_addr_offset: int, taken: NeuSegOfCorePlm) -> None:
        nonlocal n_core_req_max
        nonlocal cur_n_neuron
        nonlocal n_cur_reg

        if i == n_core_req_max or n_cur_reg == n_core_req_max:
            return

        if cur_n_neuron + neu_segs_not_full[n_cur_reg].n_neuron > capacity:
            neu_segs_of_cb.append(taken)
            cur_n_neuron = 0
            return
        else:
            taken.append(
                NeuSeg(
                    neu_segs_not_full[n_cur_reg].parent,
                    NeuronSegment(
                        neu_segs_not_full[n_cur_reg].segment.index,
                        cur_addr_offset,
                        repl_prop,
                    ),
                )
            )
            cur_addr_offset += neu_segs_not_full[n_cur_reg].n_addr
            cur_n_neuron += neu_segs_not_full[n_cur_reg].n_neuron
            n_cur_reg += 1

        if n_cur_reg == n_core_req_max:
            neu_segs_of_cb.append(taken)
            cur_n_neuron = 0
            return

        backtrack(i, cur_addr_offset, taken)  # Continue to place
        backtrack(i + 1, 0, [])  # Place to next physical core

    backtrack(0, 0, [])

    return neu_segs_of_cb


def _get_neu_slices(
    neu_groups: List[NeuDyn],
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> Dict[NeuDyn, List[NeuSlice]]:
    """Group the neuron groups by category with load balancing optimization.

    NOTE: Use load balancing optimization automatically.
    """
    seg_slices_dict: Dict[NeuDyn, List[NeuSlice]] = dict()

    for neu in neu_groups:
        seg_slices_dict[neu] = _coarse_group(neu, capacity, load_type)

    return seg_slices_dict


_get_neu_slices_opt_core = partial(_get_neu_slices, load_type="max_capacity")
_get_neu_slices_opt_latency = partial(_get_neu_slices, load_type="average")


def _dense_reorganized(
    seg_slices_dict: Dict[NeuDyn, List[NeuSlice]], capacity: int, repl_prop: int
) -> NeuSegOfCoreBlock:
    """Reorganize densely. Based on the result of 'latency' method, use greedy algorithm to \
        reorganize the incomplete neuron segments for saving cores.
    """

    def _find_neu_in_segs_of_cplm(neu: NeuDyn, seg_of_cplm: NeuSegOfCorePlm) -> bool:
        return any(neu == s.parent for s in seg_of_cplm)

    # If there is only one type of neuron segment slices, place seperately.
    if len(seg_slices_dict) == 1:
        return _place_seperately(seg_slices_dict, repl_prop)

    neu_segs_of_cb: NeuSegOfCoreBlock = []  # The final result
    _seg_slices_sorted_list = sorted(
        seg_slices_dict.items(), key=lambda items: len(items[1]), reverse=True
    )
    # Neuron slices on index 0 requires the most cores
    _max_core_req_neu, _max_core_req_seg_slices = _seg_slices_sorted_list[0]

    _max_seg_slices_of_cplm = [
        [NeuSeg(_max_core_req_neu, NeuronSegment(seg_slice, 0, repl_prop))]
        for seg_slice in _max_core_req_seg_slices
    ]
    neu_segs_of_cb.extend(_max_seg_slices_of_cplm)

    seg_slices_sorted = dict(_seg_slices_sorted_list[1:])
    for neu, seg_slices in seg_slices_sorted.items():
        for seg_slice in seg_slices:
            require_new_cplm = True

            for seg_of_cplm in neu_segs_of_cb:
                cur_addr_offset = sum([seg.n_addr for seg in seg_of_cplm])
                cur_n_neuron = sum([seg.n_neuron for seg in seg_of_cplm])

                # Available to place & insert for the first time
                if (
                    cur_n_neuron + seg_slice.stop - seg_slice.start
                ) <= capacity and not _find_neu_in_segs_of_cplm(neu, seg_of_cplm):
                    # FIXME Necessary check not _find_neu_in_segs_of_cplm?
                    neu_seg = NeuSeg(
                        neu, NeuronSegment(seg_slice, cur_addr_offset, repl_prop)
                    )
                    seg_of_cplm.append(neu_seg)

                    require_new_cplm = False
                    break

            if require_new_cplm:
                neu_seg = NeuSeg(neu, NeuronSegment(seg_slice, 0, repl_prop))
                neu_segs_of_cb.append([neu_seg])

    return neu_segs_of_cb


def get_neu_segments(
    neu_groups: List[NeuDyn],
    capacity: int,
    repl_prop: int,
    optim_target: Literal["latency", "core", "both"],
) -> NeuSegOfCoreBlock:
    """Get the neuron segments with a optimization strategy.

    Args:
        - neu_groups: group of neurons in the core block.
        - capacity: #N of neurons that can be accommodated in a core.
        - repl_prop: the proportion of neuron replication.
        - optim_target: optimization target. 'latency' strategy intends to optimize the latency of nodes. \
            'core' strategy intends to optimize the consumption of cores.
    """
    if optim_target == "core":
        seg_slices_dict = _get_neu_slices_opt_core(neu_groups, capacity)
        return _get_nsg_opt_core(seg_slices_dict, capacity, repl_prop)

    else:
        seg_slices_dict = _get_neu_slices_opt_latency(neu_groups, capacity)

        if optim_target == "latency":
            return _place_seperately(seg_slices_dict, repl_prop)
        else:
            return _dense_reorganized(seg_slices_dict, capacity, repl_prop)


def get_axon_segments(
    axons: Sequence[SourceNodeType], tr_max: int, fan_in_max: int
) -> Dict[SourceNodeType, AxonSegment]:
    """Divide axons into segments by group to fit the hardware constraints.

    Args:
        - axons: The axons to be segmented.
        - tr_max: The maximum value of the time slot(=n_timeslot).
        - fan_in_max: The value of fan-in per dendrite(=N_FANIN_PER_DENDRITE_XNN).

    TODO Provide an alternative when failed.
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
                f"axons address out of range [0, {fan_in_max}) ({offset + addr_width})."
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
    neu_index: NeuSlice, axon_seg: AxonSegment, delay: int, dest_n_timeslot: int
) -> List[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment.

    The length of axon coordinates is the same as `neu_index`.
    """
    axon_coords = []
    addr_width = axon_seg.addr_width
    addr_offset = axon_seg.addr_offset

    # tick_relative = n_timeslot * (delay - 1) + tr_offset (start & end)
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
