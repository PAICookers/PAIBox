import warnings
from collections import defaultdict
from functools import partial
from math import ceil
from typing import Literal

from paibox.components import Neuron
from paibox.exceptions import ParamInvalidWarning, ResourceError

from .sub_utils import SubNeuron, SubSourceType
from .types import (
    AxonSegment,
    CoreAllocationOfCoreBlock,
    DendriteSegment,
    SubNeuOfCorePlm,
    SubNeuType,
)


def _place_seperately(
    seg_slices_dict: dict[Neuron, list[SubNeuType]], repl_prop: int
) -> CoreAllocationOfCoreBlock:
    neu_segs_of_cb: CoreAllocationOfCoreBlock = []

    for neu, seg_slices in seg_slices_dict.items():
        neu_segs_of_cb.extend(
            [
                [DendriteSegment(neu, seg_slice, 0, repl_prop)]
                for seg_slice in seg_slices
            ]
        )

    return neu_segs_of_cb


def _coarse_group(
    neu: SubNeuron,
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> list[SubNeuType]:
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

    sub_neus: list[SubNeuType] = []
    n_neuron = neu.num_out

    if load_type == "average":
        n_core_required = ceil(n_neuron / capacity) * neu.unrolling_factor
        dist = _average_load(n_neuron, n_core_required)
    else:
        dist = _max_capacity_load(n_neuron)

    offset = 0
    for d in dist:
        sub_neus.append(neu.index[offset : offset + d])
        offset += d

    return sub_neus


def _get_nsg_opt_core(
    seg_slices_dict: dict[Neuron, list[SubNeuType]], capacity: int, repl_prop: int
) -> CoreAllocationOfCoreBlock:
    neu_segs_of_cb: CoreAllocationOfCoreBlock = []  # The final result
    raise_warning = False

    for neu in seg_slices_dict:
        if neu.unrolling_factor > 1:
            neu.unrolling_factor = 1
            raise_warning = True

    if raise_warning:
        warnings.warn(
            "when grouping neurons with 'core' optimization, unrolling "
            "factor greater than 1 is invalid. Modified to 1.",
            ParamInvalidWarning,
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

    def backtrack(i: int, cur_addr_offset: int, taken: SubNeuOfCorePlm) -> None:
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
                DendriteSegment(
                    neu_segs_not_full[n_cur_reg].target,
                    neu_segs_not_full[n_cur_reg].index,
                    cur_addr_offset,
                    repl_prop,
                )
            )
            cur_addr_offset += neu_segs_not_full[n_cur_reg].n_neuron
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
    neu_groups: list[SubNeuron],
    capacity: int,
    load_type: Literal["average", "max_capacity"],
) -> dict[Neuron, list[SubNeuType]]:
    """Group the neuron groups by category with load balancing optimization.

    NOTE: Use load balancing optimization automatically.
    """
    seg_slices_dict: dict[Neuron, list[SubNeuType]] = defaultdict(list)

    for sub_neu in neu_groups:
        seg_slices_dict[sub_neu.target] = _coarse_group(sub_neu, capacity, load_type)

    return seg_slices_dict


_get_neu_slices_opt_core = partial(_get_neu_slices, load_type="max_capacity")
_get_neu_slices_opt_latency = partial(_get_neu_slices, load_type="average")


def _dense_reorganized(
    sub_neu_dict: dict[Neuron, list[SubNeuType]], capacity: int, repl_prop: int
) -> CoreAllocationOfCoreBlock:
    """Reorganize densely. Based on the result of 'latency' method, use greedy algorithm to \
        reorganize the incomplete neuron segments for saving cores.
    """

    def _find_neu_in_subneus_of_cplm(
        neu: Neuron, sub_neu_of_cplm: SubNeuOfCorePlm
    ) -> bool:
        return any(neu == s.target for s in sub_neu_of_cplm)

    # If there is only one type of neuron segment slices, place seperately.
    if len(sub_neu_dict) == 1:
        return _place_seperately(sub_neu_dict, repl_prop)

    cplms_of_cb: CoreAllocationOfCoreBlock = []  # The final result
    _sub_neu_sorted_list = sorted(
        sub_neu_dict.items(), key=lambda items: len(items[1]), reverse=True
    )
    # Neuron slices on index 0 requires the most cores
    _max_core_req_neu, _max_core_req_sub_neus = _sub_neu_sorted_list[0]

    _max_sub_neus_of_cplm = [
        [DendriteSegment(_max_core_req_neu, sub_neu, 0, repl_prop)]
        for sub_neu in _max_core_req_sub_neus
    ]
    cplms_of_cb.extend(_max_sub_neus_of_cplm)

    sub_neus_sorted = dict(_sub_neu_sorted_list[1:])
    for neu, sub_neus_to_allocate in sub_neus_sorted.items():
        for sub_neu_to_allocate in sub_neus_to_allocate:
            require_new_cplm = True

            for cplm in cplms_of_cb:
                cur_n_neuron = sum([sub_neu.n_neuron for sub_neu in cplm])
                # Available to place & insert for the first time

                if (
                    cur_n_neuron + len(sub_neu_to_allocate)
                ) <= capacity and not _find_neu_in_subneus_of_cplm(neu, cplm):
                    cur_addr_offset = sum([sub_neu.n_neuron for sub_neu in cplm])
                    neu_seg = DendriteSegment(
                        neu, sub_neu_to_allocate, cur_addr_offset, repl_prop
                    )
                    cplm.append(neu_seg)

                    require_new_cplm = False
                    break

            if require_new_cplm:
                neu_seg = DendriteSegment(neu, sub_neu_to_allocate, 0, repl_prop)
                cplms_of_cb.append([neu_seg])

    return cplms_of_cb


def get_dendrite_segments(
    neu_groups: list[SubNeuron],
    capacity: int,
    repl_prop: int,
    optim_target: Literal["latency", "core", "both"],
) -> CoreAllocationOfCoreBlock:
    """Get the neuron segments with a optimization strategy.

    Args:
        - neu_groups: group of neurons in the core block.
        - capacity: #N of neurons that can be accommodated in a core.
        - repl_prop: the proportion of neuron replication.
        - optim_target: optimization target. 'latency' strategy intends to optimize the latency of nodes. \
            'core' strategy intends to optimize the consumption of cores.
    """
    if optim_target == "core":
        sub_neu_dict = _get_neu_slices_opt_core(neu_groups, capacity)
        return _get_nsg_opt_core(sub_neu_dict, capacity, repl_prop)

    else:
        sub_neu_dict = _get_neu_slices_opt_latency(neu_groups, capacity)

        if optim_target == "latency":
            return _place_seperately(sub_neu_dict, repl_prop)
        else:
            return _dense_reorganized(sub_neu_dict, capacity, repl_prop)


def get_axon_segments(
    axons: list[SubSourceType], tr_max: int, fanin_base: int
) -> dict[SubSourceType, AxonSegment]:
    """Divide axons into segments by group to fit the hardware constraints.

    Args:
        - axons: the axons to be segmented.
    """
    max_n_axon = tr_max * fanin_base
    offset = 0
    axon_segments: dict[SubSourceType, AxonSegment] = dict()

    for ax in axons:
        axon_segments[ax] = AxonSegment(ax.num_out, offset, fanin_base)
        if ax.num_out + offset > max_n_axon:
            raise ResourceError(
                f"The axon segment {ax} exceeds the maximum capacity of axons in a core, "
                f"which is {max_n_axon}."
            )
        offset += ax.num_out

    return axon_segments
