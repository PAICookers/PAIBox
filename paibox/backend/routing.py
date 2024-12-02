import itertools
import math
import sys
from collections.abc import Generator, Iterator
from typing import Any, ClassVar, Union

from paicorelib import ONLINE_CORES_BASE_COORD
from paicorelib import ROUTING_DIRECTIONS_IDX as DIREC_IDX
from paicorelib import ChipCoord, Coord, HwConfig, RoutingCoord
from paicorelib import RoutingDirection as Direction
from paicorelib import RoutingLevel as Level
from paicorelib.routing_defs import MAX_ROUTING_PATH_LENGTH

from paibox.exceptions import (
    GraphBuildError,
    PAIBoxDeprecationWarning,
    ResourceError,
    RoutingError,
)

from .conf_types import CorePlmConfInChip
from .placement import CoreBlock, EmptyCorePlacement
from .types import *

if sys.version_info >= (3, 13):
    from typing import deprecated
else:
    from typing_extensions import deprecated

__all__ = ["RoutingGroup", "RoutingManager"]


def _Coord2RoutingCoord(coord: Coord) -> RoutingCoord:
    directions: list[Direction] = []
    x = coord.x
    y = coord.y

    for i in range(MAX_ROUTING_PATH_LENGTH):
        # 每个循环，提取最高位（移动了 4-i 位）到最低位，恢复 value_x 和 value_y
        shift = 4 - i
        value_x, value_y = (x >> shift) & 0b1, (y >> shift) & 0b1
        directions.append(Direction((value_x, value_y)))

    return RoutingCoord(*directions)


class RoutingGroup:
    """Each routing group should be able to route by single coord."""

    _debug_id: ClassVar[int] = 0
    """Class counter for debugging."""

    def __init__(
        self, unordered_cb: list[CoreBlock], ordered_rgrp: list["RoutingGroup"]
    ) -> None:
        self.unordered_cb: list[CoreBlock] = unordered_cb
        self.ordered_rgrp: list["RoutingGroup"] = ordered_rgrp
        self.routing_elems: list[Union[CoreBlock, "RoutingGroup"]] = (
            unordered_cb + ordered_rgrp
        )
        self.offset: list[int] = []  # TODO Change a name
        self.n_core_required: int = 0
        """The actual number of cores required by the routing group."""
        self.n_tail_waste: int = 0
        """Waste cores at the tail of the routing group."""

        axons: set[SourceNodeType] = set()
        for elem in self.routing_elems:
            axons.update(elem.axons)

        self.axons: list[SourceNodeType] = list(axons)  # unordered

        self.assigned_coords: list[Coord] = []
        """Assigned core coordinates in the routing group"""
        self.wasted_coords: list[Coord] = []
        """Wasted core coordinates in routing group"""
        self.wasted_core_plm: dict[Coord, EmptyCorePlacement] = {}
        """Wasted core placements"""

        """Status options"""
        self.is_assigned = False
        """Whether the coordinates of chip & cores are assigned."""

        # For debugging
        self._id = RoutingGroup._debug_id
        RoutingGroup._debug_id += 1

    def set_core_required(self) -> None:
        """Calculate the number of cores required for the routing group iteratively."""
        for rgrp in self.ordered_rgrp:
            rgrp.set_core_required()

        # Record the used cores of the members, but not the actual amount.
        n_core_used = 0

        # Unordered core blocks sorted in descending order, avoiding assigning waste.
        unordered_cb = sorted(
            self.unordered_cb, key=lambda x: x.n_core_required, reverse=True
        )
        for cb in unordered_cb:
            self.offset.append(self.n_core_required)
            n_core_used += cb.n_core_required

        # Ordered routing groups should be assgined first.
        ordered_rgrp = self.ordered_rgrp
        for rgrp in ordered_rgrp:
            n_core_assigned = _nearest_multiple_above(n_core_used, rgrp.n_core_required)
            self.offset.append(n_core_assigned)
            n_core_used = n_core_assigned + rgrp.n_core_required

        # Routing elements need satisfy the topological order
        self.routing_elems = unordered_cb + ordered_rgrp

        # If there are ordered routing groups, the final amount wasted is the
        # tail waste number of the LAST routing group. Otherwise, waste = 0.
        n_tail_waste = ordered_rgrp[-1].n_tail_waste if ordered_rgrp else 0
        # sub_tail_wasted = (
        #     0
        #     if isinstance(self.routing_elems[-1], CoreBlock)
        #     else self.routing_elems[-1].n_tail_waste
        # )

        # This is the amount of cores required actually.
        self.n_core_required = 1 << (n_core_used - 1).bit_length()
        self.n_tail_waste = self.n_core_required - n_core_used + n_tail_waste

    def assign_coord(
        self, chip_coord: Coord, allocated: list[Coord]
    ) -> tuple[list[Coord], list[Coord]]:
        """Assign core coordinates to the routing group."""
        cur_i = 0
        assigned_coords: list[Coord] = []
        wasted_coords: list[Coord] = []

        for elem, offset in zip(self.routing_elems, self.offset):
            if offset > cur_i:
                wasted_coords += allocated[cur_i:offset]

            cur_i = offset
            n = elem.n_core_required
            # print(
            #     f"element: {elem}, {n} cores, start at {_Coord2RoutingCoord(allocated[cur_i])}"
            # )
            assigned, wasted = elem.assign_coord(
                chip_coord, allocated[cur_i : cur_i + n]
            )
            assigned_coords += assigned
            wasted_coords += wasted
            cur_i += n

        self.assigned_coords = assigned_coords
        self.wasted_coords = wasted_coords + allocated[cur_i:]
        self.is_assigned = True

        return self.assigned_coords, self.wasted_coords

    def group_axons(self, multicast_axons: list[SourceNodeType] = []) -> None:
        """Group the axons, using list to keep the order of axons."""
        if not all(cb._lcn_locked for cb in self.core_blocks):
            raise GraphBuildError(
                "get axon segments of core block after 'lcn_ex' is locked."
            )

        private_multicast_axons = multicast_axons.copy()
        ax_shared_times: list[int] = [0] * len(self.axons)

        # Axons shared within a routing group also need to be multicast.
        for elem in self.routing_elems:
            for ax in elem.axons:
                idx = self.axons.index(ax)
                ax_shared_times[idx] += 1

        for ax, times in zip(self.axons, ax_shared_times):
            if times > 1 and ax not in private_multicast_axons:
                private_multicast_axons.append(ax)

        for elem in self.routing_elems:
            elem.group_axons(private_multicast_axons)

    @property
    def core_blocks(self) -> list[CoreBlock]:
        """Retrieve all core blocks within the routing group iteratively."""
        cbs = []

        for elem in self.routing_elems:
            if isinstance(elem, CoreBlock):
                cbs.append(elem)
            else:
                cbs += elem.core_blocks

        return cbs

    @classmethod
    def build(cls, merged_sgrp: MergedSuccGroup) -> "RoutingGroup":
        msgrp = MergedSuccGroup()
        remaining = MergedSuccGroup()

        for group in merged_sgrp.groups:
            if group.input in merged_sgrp.nodes:
                msgrp.add_group(group)
            else:
                remaining.add_group(group)

        remaining.nodes -= msgrp.nodes
        unordered_cb = CoreBlock.build_core_blocks(remaining)

        if len(msgrp.nodes) > 0:
            sub_rgrp = RoutingGroup.build(msgrp)
            ordered_rgrp = [sub_rgrp]
        else:
            ordered_rgrp = []

        return cls(unordered_cb, ordered_rgrp)

    def core_block_alloc(self) -> None:
        assert self.is_assigned, "coordinates are not assigned."

        for cb in self:
            cb.core_plm_alloc()

        # Allocate empty core placements for the wasted coordinates.
        for coord in self.wasted_coords:
            self.wasted_core_plm[coord] = EmptyCorePlacement.build(coord)

    def get_wasted_cplm_config(self) -> CorePlmConfInChip:
        return {
            coord: core_plm.export_core_plm_config()
            for coord, core_plm in self.wasted_core_plm.items()
        }

    def get_n_core_occupied(self) -> int:
        """Get the #N of cores occupied by the routing group."""
        assert self.is_assigned, "coordinates are not assigned."
        return len(self.assigned_coords) + len(self.wasted_coords)

    @property
    def chip_coord(self) -> ChipCoord:
        if not all(cb.chip_coord == self[0].chip_coord for cb in self):
            raise RoutingError(
                "chip coordinates in the routing group are not consistent."
            )

        return self[0].chip_coord

    def dump(self, i: int = 0) -> None:
        tabs = "\t" * i
        print(f"{tabs}RoutingGroup: {self} with {self.n_core_required} cores:")
        for elem in self.routing_elems:
            if isinstance(elem, RoutingGroup):
                elem.dump(i + 1)
            else:
                print(f"{tabs}\t{elem.name} with {elem.n_core_required} cores:")
                for edge in elem._parents:
                    print(
                        f"{tabs}\t\t{edge.name}: {edge.source.name} -> {edge.target.name}"
                    )

    def __contains__(self, cb: CoreBlock) -> bool:
        return cb in self.core_blocks

    def __getitem__(self, idx: int) -> CoreBlock:
        return self.core_blocks[idx]

    def __iter__(self) -> Iterator[CoreBlock]:
        return self.core_blocks.__iter__()

    def __str__(self) -> str:
        return f"RoutingGroup_{self._id}"


class RoutingManager:
    def __init__(self, chip_list: list[ChipCoord], **kwargs) -> None:
        """Initialize a routing quadtree root."""
        self.chip_list: list[ChipCoord] = chip_list
        self.used_L2_clusters: list[list[RoutingCoord]] = [
            list() for _ in range(len(chip_list))
        ]
        """Used L2 clusters in each chip. The clocks of unused L2 clusters can be turned off   \
            through the serial port to reduce power consumption.
        """
        self.n_core_total: int = 0
        self.n_core_per_chip: list[int] = [0] * len(chip_list)

    def get_insert_location(
        self, n_core_incoming: int, n_core_wasted: int
    ) -> tuple[int, int, list[Direction]]:
        """Look for the insertion location of the incoming routing group."""
        n_core_aligned = _nearest_multiple_above(self.n_core_total, n_core_incoming)

        n_core_predicted = n_core_aligned + n_core_incoming
        start_core_inchip = _num_inchip(n_core_aligned)
        end_core_inchip = _num_inchip(n_core_predicted) - n_core_wasted

        # If online cores are hit, start from the first core after the online cores
        if (
            start_core_inchip <= ONLINE_CORES_BASE_COORD
            and end_core_inchip > ONLINE_CORES_BASE_COORD
        ):
            online_end_inchip = ONLINE_CORES_BASE_COORD + HwConfig.N_CORE_ONLINE
            # The first core after the online cores
            online_end = n_core_aligned - start_core_inchip + online_end_inchip
            n_core_aligned = _nearest_multiple_above(online_end, n_core_incoming)

        core_loc = n_core_aligned

        if (chip_idx_loc := core_loc // HwConfig.N_CORE_MAX_INCHIP) >= len(
            self.chip_list
        ):
            raise ResourceError(
                f"the number of required chips exceeds the limit {len(self.chip_list)} ({chip_idx_loc+1})."
            )

        self.n_core_total = n_core_aligned + n_core_incoming
        self.n_core_per_chip[chip_idx_loc] = _num_inchip(self.n_core_total)

        routing_idx = core_loc % HwConfig.N_CORE_MAX_INCHIP
        routing_path = []

        # From L0 to L4
        for _ in range(MAX_ROUTING_PATH_LENGTH):
            routing_idx, re = divmod(routing_idx, HwConfig.N_SUB_ROUTING_NODE)
            routing_path.append(DIREC_IDX[re])

        return core_loc, chip_idx_loc, routing_path

    def place_routing_group(
        self, rgrp: RoutingGroup
    ) -> tuple[list[Coord], list[Coord]]:
        """Place a routing group in the chip list. Assign each core blocks with routing coordinates &   \
            make sure they are routable.

        Returns: a tuple of lists of assigned and wasted coordinates.
        """
        # for cb in rgrp:
        #     print(f"\t{cb.name}")
        n_core_cost = rgrp.n_core_required
        n_tail_waste = rgrp.n_tail_waste
        n_core_req = n_core_cost - n_tail_waste

        # Check whether a single routing group can be placed within a single core.
        # The largest continuous offline cores is ONLINE_CORES_BASE_COORD.
        if n_core_req > ONLINE_CORES_BASE_COORD:
            raise ResourceError(
                "the number of cores required by the routing group exceeds the hardware limit, "
                f"{n_core_req} > {ONLINE_CORES_BASE_COORD}."
            )

        core_insert_loc, chip_idx_loc, rpath_start = self.get_insert_location(
            n_core_cost, n_tail_waste
        )

        allocated_coords: list[Coord] = []
        for i, rpath in _routing_path_generator(n_core_cost, rpath_start):
            leaf_coord = RoutingCoord(*reversed(rpath))
            # Record the used L2 clusters
            if (core_insert_loc + i) % (HwConfig.N_SUB_ROUTING_NODE**Level.L2) == 0:
                L2_coord = RoutingCoord(*reversed(rpath[Level.L2 :]))
                self.used_L2_clusters[chip_idx_loc].append(L2_coord)

            allocated_coords.append(leaf_coord.to_coord())

        return rgrp.assign_coord(self.chip_list[chip_idx_loc], allocated_coords)


@deprecated(
    "'RoutingRoot' is deprecated in version 1.2.0 and will be "
    "removed in version 1.3.0. Use `RoutingManager` instead.",
    category=PAIBoxDeprecationWarning,
)
class RoutingRoot(RoutingManager):
    pass


def _nearest_multiple_above(a: int, x: int) -> int:
    """Return the nearest number greater than or equal to `a`, and is an integer multiple of `x`."""
    # (above + multiple - 1) // multiple
    return math.ceil(a / x) * x


def _num_inchip(n: int) -> int:
    return (n - 1) % HwConfig.N_CORE_MAX_INCHIP + 1


def _routing_path_generator(
    n_times: int, rpath: list[Direction]
) -> Generator[tuple[int, list[Direction]], Any, None]:
    for i in range(n_times):
        yield i, rpath

        for lx in range(len(rpath)):
            if rpath[lx] == DIREC_IDX[-1]:
                rpath[lx] = DIREC_IDX[0]
            else:
                rpath[lx] = DIREC_IDX[(DIREC_IDX.index(rpath[lx]) + 1) % len(DIREC_IDX)]
                break


# def _routing_path_generator(
#     n_times: int, rpath: RoutingPath, yield_first: bool = True
# ) -> Generator[tuple[int, RoutingPath], Any, None]:
#     _len = len(rpath)
#     lx_iter = reversed(range(_len)) if rpath.reversed else range(_len)

#     for i in range(n_times):
#         if yield_first:
#             yield i, rpath

#         for lx in lx_iter:
#             if rpath[lx] == DIREC_IDX[-1]:
#                 rpath[lx] = DIREC_IDX[0]
#             else:
#                 rpath[lx] = DIREC_IDX[(DIREC_IDX.index(rpath[lx]) + 1) % len(DIREC_IDX)]
#                 break

#         if not yield_first:
#             yield i, rpath


def _all_lx_clusters(lx: Union[Level, int]) -> list[RoutingCoord]:
    return [
        RoutingCoord(*path)
        for path in itertools.product(DIREC_IDX, repeat=MAX_ROUTING_PATH_LENGTH - lx)
    ]


def get_unused_lx(
    used_lx: list[RoutingCoord], lx: Union[Level, int] = Level.L2
) -> list[RoutingCoord]:
    all_lx = _all_lx_clusters(lx)

    for l in set(used_lx):  # make used_lx unduplicated
        all_lx.remove(l)  # keep the rest clusters in order

    return all_lx
