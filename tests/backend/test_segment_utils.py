from math import ceil

import pytest

import paibox as pb
from paibox.backend._slice import NeuronSlice, SourceSliceType
from paibox.backend.segment_utils import (
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
)
from paibox.components import Neuron
from tests.utils import make_test
from .backend_testcase import BackendTestCase as TCase


class TestGetNeuronSegments:
    @staticmethod
    def _coarse_group_with_load_balancing_proto(
        n_neuron: int, capacity: int, unrolling_factor: int
    ):
        def _average_load(n: int, n_part: int) -> list[int]:
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

    @make_test(TCase.neu_segs_latency_testcase)
    def test_get_neu_segments_latency(self, neurons, capacity, wp, lcn_ex, expected):
        neuron_slices = [NeuronSlice(neuron) for neuron in neurons]
        neu_segs = get_neu_segments(
            neuron_slices, capacity, self._get_interval(wp, lcn_ex), "latency"
        )
        assert neu_segs == expected

    @make_test(TCase.neu_segs_core_testcase)
    def test_get_neu_segments_core(self, neurons, capacity, wp, lcn_ex, expected):
        neuron_slices = [NeuronSlice(neuron) for neuron in neurons]
        neu_segs = get_neu_segments(
            neuron_slices, capacity, self._get_interval(wp, lcn_ex), "core"
        )
        assert neu_segs == expected

    @make_test(TCase.neu_segs_both_testcase)
    def test_get_neu_segments_both(self, neurons, capacity, wp, lcn_ex, expected):
        neuron_slices = [NeuronSlice(neuron) for neuron in neurons]
        neu_segs = get_neu_segments(
            neuron_slices, capacity, self._get_interval(wp, lcn_ex), "both"
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
def test_get_axon_segments(axons: list[Neuron]):
    from .conftest import n_axon2lcn_ex_proto

    lcn_ex = n_axon2lcn_ex_proto(sum(axon.num_out for axon in axons), 1152)

    tr_max = 1 << lcn_ex

    axon_slices: list[SourceSliceType] = [NeuronSlice(axon) for axon in axons]

    axon_segs = get_axon_segments(axon_slices, tr_max, 1152)

    for axon_seg in axon_segs.values():
        assert axon_seg.addr_offset <= 1152 * tr_max


@pytest.mark.parametrize(
    "axons",
    [
        [pb.LIF(1151, 2), pb.LIF(1153, 2)],
        [pb.LIF(1151 * 2, 2), pb.LIF(1153 * 2, 2)],
    ],
)
def test_get_axon_segments_boundary(axons: list[Neuron]):
    """Illegal boundary cases."""
    from .conftest import n_axon2lcn_ex_proto

    lcn_ex = n_axon2lcn_ex_proto(sum(axon.num_out for axon in axons), 1152)
    tr_max = 1 << lcn_ex

    axon_slices: list[SourceSliceType] = [NeuronSlice(axon) for axon in axons]

    axon_segs = get_axon_segments(axon_slices, tr_max, 1152)

    last_slice = axon_slices[-1]
    last_seg = axon_segs[last_slice]
    assert last_seg.addr_offset + last_seg.n_axon == (tr_max * 1152)


@make_test(TCase.aligned_coords_testcase)
def test_aligned_coords(neu_index, axon_seg, delay, n_timeslot, is_iw8, expected):
    assert aligned_coords(neu_index, axon_seg, delay, n_timeslot, is_iw8) == expected
