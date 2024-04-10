from math import ceil
from typing import List

import pytest
from paicorelib import LCN_EX, AxonCoord, AxonSegment, NeuronSegment
from paicorelib import WeightPrecision as WP

import paibox as pb
from paibox.backend.placement import NeuSeg
from paibox.backend.segment_utils import (
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
)
from paibox.exceptions import ResourceError

n1 = pb.LIF(600, 2, unrolling_factor=1)
n2 = pb.LIF(800, 2, unrolling_factor=1)
n3 = pb.LIF(320, 2, unrolling_factor=2)
n4 = pb.LIF(200, 2, unrolling_factor=3)
n5 = pb.LIF(300, 2, unrolling_factor=2)
n6 = pb.LIF(400, 2, unrolling_factor=1)
n7 = pb.LIF(500, 2, unrolling_factor=1)

neu_segs_latency_test_data = [
    # Neurons, capacity, wp, lcn_ex, expected
    # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
    (
        [n1, n2],
        512,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_1X,
        [
            [NeuSeg(n1, NeuronSegment(slice(0, 300, 1), 0))],
            [NeuSeg(n1, NeuronSegment(slice(300, 600, 1), 0))],
            [NeuSeg(n2, NeuronSegment(slice(0, 400, 1), 0))],
            [NeuSeg(n2, NeuronSegment(slice(400, 800, 1), 0))],
        ],
    ),
    (
        [n1, n2],
        256,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n1, NeuronSegment(slice(0, 200, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(200, 400, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(400, 600, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(0, 200, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(200, 400, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(400, 600, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(600, 800, 1), 0, 2))],
        ],
    ),
    (
        [n3],
        200,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n3, NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2))],
        ],
    ),
    (
        [n1, n3],
        400,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_1X,
        [
            [NeuSeg(n1, NeuronSegment(slice(0, 300, 1), 0))],
            [NeuSeg(n1, NeuronSegment(slice(300, 600, 1), 0))],
            [NeuSeg(n3, NeuronSegment(slice(160 * 0, 160 * 1, 1), 0))],
            [NeuSeg(n3, NeuronSegment(slice(160 * 1, 160 * 2, 1), 0))],
        ],
    ),
    (
        [n4, n5],
        240,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n4, NeuronSegment(slice(67 * 0, 67 * 1, 1), 0, 2))],
            [NeuSeg(n4, NeuronSegment(slice(67 * 1, 67 * 2, 1), 0, 2))],
            [NeuSeg(n4, NeuronSegment(slice(67 * 2, 200, 1), 0, 2))],
            [NeuSeg(n5, NeuronSegment(slice(75 * 0, 75 * 1, 1), 0, 2))],
            [NeuSeg(n5, NeuronSegment(slice(75 * 1, 75 * 2, 1), 0, 2))],
            [NeuSeg(n5, NeuronSegment(slice(75 * 2, 75 * 3, 1), 0, 2))],
            [NeuSeg(n5, NeuronSegment(slice(75 * 3, 75 * 4, 1), 0, 2))],
        ],
    ),
]

neu_segs_core_test_data = [
    # Neurons, capacity, wp, lcn_ex, expected
    # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
    (
        [n1, n2],
        512,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_1X,
        [
            [NeuSeg(n1, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(n2, NeuronSegment(slice(0, 512, 1), 0))],
            [
                NeuSeg(n2, NeuronSegment(slice(512, 800, 1), 0)),
                NeuSeg(n1, NeuronSegment(slice(512, 600, 1), 288)),
            ],
        ],
    ),
    (
        [n1, n2],
        256,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n1, NeuronSegment(slice(256 * 0, 256 * 1, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(256 * 1, 256 * 2, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(256 * 0, 256 * 1, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(256 * 1, 256 * 2, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(256 * 2, 256 * 3, 1), 0, 2))],
            [
                NeuSeg(n1, NeuronSegment(slice(256 * 2, 600, 1), 0, 2)),
                NeuSeg(n2, NeuronSegment(slice(256 * 3, 800, 1), 88 * 2, 2)),
            ],
        ],
    ),
    (
        [n4, n5],
        256,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            # Place the neuron segments with full capacity first
            [NeuSeg(n5, NeuronSegment(slice(0, 256, 1), 0, 2))],
            [
                NeuSeg(n4, NeuronSegment(slice(0, 200, 1), 0, 2)),
                NeuSeg(n5, NeuronSegment(slice(256, 300, 1), 200 * 2, 2)),
            ],
        ],
    ),
    (
        [n6, n7],
        512,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_1X,
        [
            [NeuSeg(n7, NeuronSegment(slice(0, 500, 1), 0, 1))],
            [NeuSeg(n6, NeuronSegment(slice(0, 400, 1), 0, 1))],
        ],
    ),
]


neu_segs_both_test_data = [
    # Neurons, capacity, wp, lcn_ex, expected
    # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
    (
        [n1, n2],
        512,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_1X,
        [
            [NeuSeg(n1, NeuronSegment(slice(0, 300, 1), 0))],
            [NeuSeg(n1, NeuronSegment(slice(300, 600, 1), 0))],
            [NeuSeg(n2, NeuronSegment(slice(0, 400, 1), 0))],
            [NeuSeg(n2, NeuronSegment(slice(400, 800, 1), 0))],
        ],
    ),
    (
        [n1, n2],
        256,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n2, NeuronSegment(slice(0, 200, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(200, 400, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(400, 600, 1), 0, 2))],
            [NeuSeg(n2, NeuronSegment(slice(600, 800, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(0, 200, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(200, 400, 1), 0, 2))],
            [NeuSeg(n1, NeuronSegment(slice(400, 600, 1), 0, 2))],
        ],
    ),
    (
        [n3],
        200,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [NeuSeg(n3, NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2))],
            [NeuSeg(n3, NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2))],
        ],
    ),
    (
        [n3, n4],
        200,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2)),
                # offset = 160
                NeuSeg(n4, NeuronSegment(slice(67 * 0, 67 * 1, 1), 160, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2)),
                # offset = 160
                NeuSeg(n4, NeuronSegment(slice(67 * 1, 67 * 2, 1), 160, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2)),
                # offset = 160
                NeuSeg(n4, NeuronSegment(slice(67 * 2, 200, 1), 160, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2)),
            ],
        ],
    ),
    (
        [n3, n4, n5],
        256,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2)),
                # offset = 160
                NeuSeg(n5, NeuronSegment(slice(75 * 0, 75 * 1, 1), 160, 2)),
                # offset = 160 + 150
                NeuSeg(n4, NeuronSegment(slice(67 * 0, 67 * 1, 1), 160 + 150, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2)),
                # offset = 160
                NeuSeg(n5, NeuronSegment(slice(75 * 1, 75 * 2, 1), 160, 2)),
                # offset = 160 + 150
                NeuSeg(n4, NeuronSegment(slice(67 * 1, 67 * 2, 1), 160 + 150, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2)),
                # offset = 160
                NeuSeg(n5, NeuronSegment(slice(75 * 2, 75 * 3, 1), 160, 2)),
                # offset = 160 + 150
                NeuSeg(n4, NeuronSegment(slice(67 * 2, 200, 1), 160 + 150, 2)),
            ],
            [
                NeuSeg(n3, NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2)),
                # offset = 160
                NeuSeg(n5, NeuronSegment(slice(75 * 3, 75 * 4, 1), 160, 2)),
            ],
        ],
    ),
    (
        [n4, n5],
        240,
        WP.WEIGHT_WIDTH_1BIT,
        LCN_EX.LCN_2X,
        [
            [
                NeuSeg(n5, NeuronSegment(slice(75 * 0, 75 * 1, 1), 0, 2)),
                NeuSeg(n4, NeuronSegment(slice(67 * 0, 67 * 1, 1), 150, 2)),
            ],
            [
                NeuSeg(n5, NeuronSegment(slice(75 * 1, 75 * 2, 1), 0, 2)),
                NeuSeg(n4, NeuronSegment(slice(67 * 1, 67 * 2, 1), 150, 2)),
            ],
            [
                NeuSeg(n5, NeuronSegment(slice(75 * 2, 75 * 3, 1), 0, 2)),
                NeuSeg(n4, NeuronSegment(slice(67 * 2, 200, 1), 150, 2)),
            ],
            [NeuSeg(n5, NeuronSegment(slice(75 * 3, 75 * 4, 1), 0, 2))],
        ],
    ),
]


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
        "neurons, capacity, wp, lcn_ex, expected", neu_segs_latency_test_data
    )
    def test_get_neu_segments_latency(self, neurons, capacity, wp, lcn_ex, expected):
        neu_segs = get_neu_segments(
            neurons, capacity, self._get_interval(wp, lcn_ex), "latency"
        )
        assert neu_segs == expected

    @pytest.mark.parametrize(
        "neurons, capacity, wp, lcn_ex, expected", neu_segs_core_test_data
    )
    def test_get_neu_segments_core(self, neurons, capacity, wp, lcn_ex, expected):
        neu_segs = get_neu_segments(
            neurons, capacity, self._get_interval(wp, lcn_ex), "core"
        )
        assert neu_segs == expected

    @pytest.mark.parametrize(
        "neurons, capacity, wp, lcn_ex, expected", neu_segs_both_test_data
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
