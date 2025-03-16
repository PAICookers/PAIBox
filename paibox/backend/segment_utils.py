import warnings
from functools import partial
from math import ceil
from typing import Literal

from paibox.components import Neuron
from paibox.exceptions import ParameterInvalidWarning, ResourceError

from .types import (
    AxonCoord,
    AxonSegment,
    NeuSegment,
    NeuSegOfCoreBlock,
    NeuSegOfCorePlm,
    NeuSlice,
    SourceNodeType,
)


def _place_seperately(
    seg_slices_dict: dict[Neuron, list[NeuSlice]], repl_prop: int
) -> NeuSegOfCoreBlock:
    neu_segs_of_cb = []

    for neu, seg_slices in seg_slices_dict.items():
        neu_segs_of_cb.extend(
            [[NeuSegment(neu, seg_slice, 0, repl_prop)] for seg_slice in seg_slices]
        )

    return neu_segs_of_cb


def _coarse_group(
    neu: Neuron,
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> list[NeuSlice]:
    """Group neurons with 'average' or 'maximum capacity' load type.

    NOTE: Group neuron seperately, like [N1], [N2], ..., [Nn]. For each neuron, \
        take unrolling factor into consideration, then distribute the neuron to \
        the cores evently.

        #N of cores required of nx = ceil(Ni / capacity) * uf
        Average load of nx = ceil(nx / (#N of cores required of nx))
    """

    def _average_load(n: int, n_part: int) -> list[int]:
        """Distribute #num into #n_part parts evently."""
        quotient = ceil(n / n_part)
        rest = n - (n_part - 1) * quotient

        return [quotient] * (n_part - 1) + [rest]

    def _max_capacity_load(n: int) -> list[int]:
        n_part = ceil(n / capacity)
        rest = n - (n_part - 1) * capacity

        return [capacity] * (n_part - 1) + [rest]

    neu_seg_slices: list[NeuSlice] = []
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
    seg_slices_dict: dict[Neuron, list[NeuSlice]], capacity: int, repl_prop: int
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
                NeuSegment(
                    neu_segs_not_full[n_cur_reg].target,
                    neu_segs_not_full[n_cur_reg].index,
                    cur_addr_offset,
                    repl_prop,
                )
            )
            cur_addr_offset += neu_segs_not_full[n_cur_reg].n_occupied_addr
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
    neu_groups: list[Neuron],
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> dict[Neuron, list[NeuSlice]]:
    """Group the neuron groups by category with load balancing optimization.

    NOTE: Use load balancing optimization automatically.
    """
    seg_slices_dict: dict[Neuron, list[NeuSlice]] = dict()

    for neu in neu_groups:
        seg_slices_dict[neu] = _coarse_group(neu, capacity, load_type)

    return seg_slices_dict


_get_neu_slices_opt_core = partial(_get_neu_slices, load_type="max_capacity")
_get_neu_slices_opt_latency = partial(_get_neu_slices, load_type="average")


def _dense_reorganized(
    seg_slices_dict: dict[Neuron, list[NeuSlice]], capacity: int, repl_prop: int
) -> NeuSegOfCoreBlock:
    """Reorganize densely. Based on the result of 'latency' method, use greedy algorithm to \
        reorganize the incomplete neuron segments for saving cores.
    """

    def _find_neu_in_segs_of_cplm(neu: Neuron, seg_of_cplm: NeuSegOfCorePlm) -> bool:
        return any(neu == s.target for s in seg_of_cplm)

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
        [NeuSegment(_max_core_req_neu, seg_slice, 0, repl_prop)]
        for seg_slice in _max_core_req_seg_slices
    ]
    neu_segs_of_cb.extend(_max_seg_slices_of_cplm)

    seg_slices_sorted = dict(_seg_slices_sorted_list[1:])
    for neu, seg_slices in seg_slices_sorted.items():
        for seg_slice in seg_slices:
            require_new_cplm = True

            for seg_of_cplm in neu_segs_of_cb:
                cur_addr_offset = sum([seg.n_occupied_addr for seg in seg_of_cplm])
                cur_n_neuron = sum([seg.n_neuron for seg in seg_of_cplm])

                # Available to place & insert for the first time
                if (
                    cur_n_neuron + seg_slice.stop - seg_slice.start
                ) <= capacity and not _find_neu_in_segs_of_cplm(neu, seg_of_cplm):
                    # FIXME Necessary check not _find_neu_in_segs_of_cplm?
                    neu_seg = NeuSegment(neu, seg_slice, cur_addr_offset, repl_prop)
                    seg_of_cplm.append(neu_seg)

                    require_new_cplm = False
                    break

            if require_new_cplm:
                neu_seg = NeuSegment(neu, seg_slice, 0, repl_prop)
                neu_segs_of_cb.append([neu_seg])

    return neu_segs_of_cb


def get_neu_segments(
    neu_groups: list[Neuron],
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
    axons: list[SourceNodeType], tr_max: int, n_fanin: int
) -> dict[SourceNodeType, AxonSegment]:
    """Divide axons into segments by group to fit the hardware constraints.

    Args:
        - axons: the axons to be segmented.
        - tr_max: the maximum value of the time slot(n_timeslot).
        - n_fanin: the fan-in of cores.
    """

    def _seg_alloc(axon: SourceNodeType, offset: int) -> tuple[AxonSegment, int]:
        """Allocate an axon segment, return the next offset of axon address."""
        # The width of assigned address
        if axon.num_out % tr_max > 0:
            addr_width = axon.num_out // tr_max + 1
            # n_axon_rest = axon.num_out % addr_width
        else:
            addr_width = axon.num_out // tr_max
            # n_axon_rest = 0

        if offset + addr_width > n_fanin:
            raise ResourceError(
                f"axons address out of range [0, {n_fanin}) ({offset + addr_width})."
            )

        return AxonSegment(axon.num_out, addr_width, offset), offset + addr_width

    offset = 0
    axon_segments = dict()

    for ax in axons:
        segment, offset = _seg_alloc(ax, offset)
        axon_segments[ax] = segment

    return axon_segments


def aligned_coords(
    neu_index: NeuSlice,
    axon_seg: AxonSegment,
    delay: int,
    dest_n_timeslot: int,
    is_iw8: bool,
) -> list[AxonCoord]:
    """Find the axon segments aligned with the index of neuron segment.

    NOTE: Axons are described in a tuple (tick_relative, axon_addr). Axis 'tr' is used as the row   \
        coordinates while axis 'axon' is used as the column coordinates.

        | ------- AxonSeg[0] ------- | ------- AxonSeg[1] ------- | ...
    tr=0 A1[0]   A1[1]   ...  A1[99]   A2[0]   A2[1]   ... A2[199]
    tr=1 A1[100] A1[101] ... A1[199]   A2[200] A2[201] ... A2[399]

    The target axon may be Ax[100:499], where (tr=0, offset+100) is the start and (tr=2, offset+499)\
        is the end.
            offset
              | <--------- width --------> |
        | ... | ------- AxonSeg[x] ------- | ...
    tr=0  ...   Ax[0]   Ax[1]   ... Ax[199]
    tr=1  ...   Ax[200] Ax[201] ... Ax[399]
    tr=2  ...   Ax[400] Ax[401] ... Ax[599]

    When the input width is 8 bits, each A[x] occupies 8 bits. The interval of axons is 8.
    """
    addr_width = axon_seg.addr_width
    addr_offset = axon_seg.addr_offset

    # tick_relative = n_timeslot * (delay - 1) + tr_offset (start & end)
    tr_base = dest_n_timeslot * (delay - 1)
    tr_offset_start, tr_offset_stop = (
        neu_index.start // addr_width,
        neu_index.stop // addr_width,
    )
    addr_start, addr_stop = (neu_index.start % addr_width, neu_index.stop % addr_width)

    _addr_interval = 8 if is_iw8 else 1

    if tr_offset_stop == tr_offset_start:
        axon_coords = [
            AxonCoord.build(
                tr_base + tr_offset_start, (addr_offset + addr) * _addr_interval
            )
            for addr in range(addr_start, addr_stop)
        ]
    else:
        # First row: addr_start -> end
        acoords_first = [
            AxonCoord.build(
                tr_base + tr_offset_start, (addr_offset + addr) * _addr_interval
            )
            for addr in range(addr_start, addr_width)
        ]

        # Middle rows
        acoords_mid = []
        for tr in range(tr_offset_start + 1, tr_offset_stop):
            acoords_mid.extend(
                AxonCoord.build(tr_base + tr, (addr_offset + addr) * _addr_interval)
                for addr in range(addr_width)
            )

        # Last row: start -> addr_stop
        acoords_last = [
            AxonCoord.build(
                tr_base + tr_offset_stop, (addr_offset + addr) * _addr_interval
            )
            for addr in range(addr_stop)
        ]

        axon_coords = []
        axon_coords.extend(acoords_first)
        axon_coords.extend(acoords_mid)
        axon_coords.extend(acoords_last)

    return axon_coords
