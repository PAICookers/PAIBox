from math import ceil
from typing import List

import pytest
from paicorelib import AxonCoord, AxonSegment

import paibox as pb
from paibox.backend.segment_utils import (
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
)
from paibox.exceptions import ResourceError

from .conftest import TestData


class TestGetNeuronSegments:
    @staticmethod
    def _coarse_group_with_load_balancing_proto(
        n_neuron: int, capacity: int, unrolling_factor: int
    ):
        def _average_load(n: int, n_part: int) -> List[int]:
            quotient = ceil(n / n_part)
            rest = n - (n_part - 1) * quotient
            return [quotient] * (n_part - 1) + [rest]

        n_core_required = ceil(n_neuron / capacity) * unrolling_factor
        dist = _average_load(n_neuron, n_core_required)

        return dist

    @pytest.mark.parametrize(
        "n_neuron, capacity, uf, expected",
        [
            (310, 100, 2, [39] * 7 + [37]),
            (1100, 500, 1, [367, 367, 366]),
            (320, 100, 3, [27] * 11 + [23]),
        ],
    )
    def test_coarse_group_proto(self, n_neuron, capacity, uf, expected):
        result = self._coarse_group_with_load_balancing_proto(n_neuron, capacity, uf)
        assert result == expected

    @staticmethod
    def _get_interval(wp, lcn_ex) -> int:
        return (1 << wp) * (1 << lcn_ex)

    @pytest.mark.parametrize(
        TestData.neu_segs_latency_test_data["args"],
        TestData.neu_segs_latency_test_data["data"],
    )
    def test_get_neu_segments_latency(self, neurons, capacity, wp, lcn_ex, expected):
        neu_segs = get_neu_segments(
            neurons, capacity, self._get_interval(wp, lcn_ex), "latency"
        )
        assert neu_segs == expected

    @pytest.mark.parametrize(
        TestData.neu_segs_core_test_data["args"],
        TestData.neu_segs_core_test_data["data"],
    )
    def test_get_neu_segments_core(self, neurons, capacity, wp, lcn_ex, expected):
        neu_segs = get_neu_segments(
            neurons, capacity, self._get_interval(wp, lcn_ex), "core"
        )
        assert neu_segs == expected

    @pytest.mark.parametrize(
        TestData.neu_segs_both_test_data["args"],
        TestData.neu_segs_both_test_data["data"],
    )
    def test_get_neu_segments_both(self, neurons, capacity, wp, lcn_ex, expected):
        neu_segs = get_neu_segments(
            neurons, capacity, self._get_interval(wp, lcn_ex), "both"
        )
        assert neu_segs == expected


@pytest.mark.parametrize(
    "axons",
    [
        [pb.LIF(600, 2), pb.LIF(800, 2), pb.LIF(256, 2)],
        [pb.LIF(384, 3), pb.LIF(383, 3), pb.LIF(385, 3)],
        [pb.LIF(1153, 2)],
        [pb.LIF(2222, 1), pb.LIF(2378, 1)],
    ],
)
def test_get_axon_segments(axons):
    from .conftest import n_axon2lcn_ex_proto

    lcn_ex = n_axon2lcn_ex_proto(sum(axon.num_out for axon in axons), 1152)

    tr_max = 1 << lcn_ex

    axon_segs = get_axon_segments(axons, tr_max, 1152)

    for axon_seg in axon_segs.values():
        assert axon_seg.addr_offset <= 1152


@pytest.mark.parametrize(
    "axons",
    [
        [pb.LIF(1151, 2), pb.LIF(1153, 2)],
        [pb.LIF(1151 * 2, 2), pb.LIF(1153 * 2, 2)],
    ],
)
def test_get_axon_segments_boundary(axons):
    """Illegal boundary cases."""
    from .conftest import n_axon2lcn_ex_proto

    lcn_ex = n_axon2lcn_ex_proto(sum(axon.num_out for axon in axons), 1152)
    tr_max = 1 << lcn_ex

    with pytest.raises(ResourceError):
        axon_segs = get_axon_segments(axons, tr_max, 1152)


@pytest.mark.parametrize(
    "neu_index, axon_seg, delay, n_timeslot, expected",
    [
        (
            slice(5, 8),
            AxonSegment(12, 3, 0),
            1,
            1 << 1,
            [
                AxonCoord(1, 2),
                AxonCoord(2, 0),
                AxonCoord(2, 1),
            ],
        ),
        (
            slice(0, 3),
            AxonSegment(12, 3, 0),
            2,
            1 << 1,
            [
                AxonCoord(2 + 0, 0),
                AxonCoord(2 + 0, 1),
                AxonCoord(2 + 0, 2),
            ],
        ),
        (
            slice(1, 5),
            AxonSegment(12, 3, 0),
            2,
            1 << 2,
            [
                AxonCoord(4 + 0, 1),
                AxonCoord(4 + 0, 2),
                AxonCoord(4 + 1, 0),
                AxonCoord(4 + 1, 1),
            ],
        ),
        (
            slice(1, 6),
            AxonSegment(12, 3, 0),
            4,
            1 << 3,
            [
                AxonCoord(24 + 0, 1),
                AxonCoord(24 + 0, 2),
                AxonCoord(24 + 1, 0),
                AxonCoord(24 + 1, 1),
                AxonCoord(24 + 1, 2),
            ],
        ),
        (
            slice(3, 10),
            AxonSegment(16, 4, 4),
            4,
            1 << 4,
            [
                AxonCoord(48 + 0, 4 + 3),
                AxonCoord(48 + 1, 4 + 0),
                AxonCoord(48 + 1, 4 + 1),
                AxonCoord(48 + 1, 4 + 2),
                AxonCoord(48 + 1, 4 + 3),
                AxonCoord(48 + 2, 4 + 0),
                AxonCoord(48 + 2, 4 + 1),
            ],
        ),
    ],
)
def test_aligned_coords(neu_index, axon_seg, delay, n_timeslot, expected):
    axon_coords = aligned_coords(neu_index, axon_seg, delay, n_timeslot)
    assert axon_coords == expected
