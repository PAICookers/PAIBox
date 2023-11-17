import numpy as np
import pytest

import paibox as pb
from paibox.backend.placement import (
    aligned_coords,
    get_axon_segments,
    get_neu_segments,
    n_axon2lcn_ex,
)
from ...paibox.exceptions import ResourceError
from paibox.libpaicore.v2 import AxonCoord, AxonSegment


@pytest.mark.parametrize(
    "input, n_col_groups, expected",
    [
        (
            np.arange(1, 17, dtype=np.int8).reshape(8, 2),
            2,
            np.array(
                [
                    [1, 13, 2, 14],
                    [3, 15, 4, 16],
                    [5, 0, 6, 0],
                    [7, 0, 8, 0],
                    [9, 0, 10, 0],
                    [11, 0, 12, 0],
                ],
                dtype=np.int8,
            ),
        ),
        (
            np.arange(1, 13, dtype=np.int8).reshape(6, 2),
            3,
            np.array([[1, 5, 9, 2, 6, 10], [3, 7, 11, 4, 8, 12]], dtype=np.int8),
        ),
        (
            np.arange(1, 25, dtype=np.int8).reshape(8, 3),
            3,
            np.array(
                [
                    [1, 10, 19, 2, 11, 20, 3, 12, 21],
                    [4, 13, 22, 5, 14, 23, 6, 15, 24],
                    [7, 16, 0, 8, 17, 0, 9, 18, 0],
                ],
                dtype=np.int8,
            ),
        ),
    ],
)
def test_get_binary_conn(input, n_col_groups, expected):
    """Convert a weight matirx into a standard binary connectivity.

    This is a test of the prototype of the original function.
    """
    cur_shape = input.shape
    expected_shape = expected.shape
    row, col = expected.shape
    o_matrix = np.zeros(expected_shape, dtype=np.int8)

    for i in range(cur_shape[1]):
        w_col = input[:, i]
        col_group = 0

        while (n_rest_axon := cur_shape[0] - row * col_group) > row:
            o_matrix[:, n_col_groups * i + col_group] = w_col[
                row * col_group : row * (col_group + 1)
            ]
            col_group += 1

            print(o_matrix)

        o_matrix[:, n_col_groups * i + col_group] = np.pad(
            w_col[row * col_group :],
            pad_width=(0, row - n_rest_axon),
            mode="constant",
            constant_values=0,
        )

        print(o_matrix)

    assert np.array_equal(o_matrix, expected)


class TestGetNeuronSegments:
    def test_get_neu_segments_catagory(
        self,
        neu_segs_test_data,
        neu_segs_expected_catagory,
    ):
        for data, expected in zip(neu_segs_test_data, neu_segs_expected_catagory):
            neu_ins, capacity, wp, lcn_ex = data
            neu_segs = get_neu_segments(
                neu_ins,
                capacity,
                weight_precision=wp,
                lcn_ex=lcn_ex,
                method="catagory",
            )

            assert neu_segs == expected
            assert neu_segs[0][0].segment.interval == (1 << wp) * (1 << lcn_ex)

    def test_get_neu_segments_dense(
        self,
        neu_segs_test_data,
        neu_segs_expected_dense,
    ):
        for data, expected in zip(neu_segs_test_data, neu_segs_expected_dense):
            neu_ins, capacity, wp, lcn_ex = data
            neu_segs = get_neu_segments(
                neu_ins,
                capacity,
                weight_precision=wp,
                lcn_ex=lcn_ex,
                method="dense",
            )

            assert neu_segs == expected
            assert neu_segs[0][0].segment.interval == (1 << wp) * (1 << lcn_ex)


@pytest.mark.parametrize(
    "axons",
    [
        [pb.neuron.LIF(600, 2), pb.neuron.LIF(800, 2), pb.neuron.LIF(256, 2)],
        [pb.neuron.LIF(384, 3), pb.neuron.LIF(383, 3), pb.neuron.LIF(385, 3)],
        [pb.neuron.LIF(1153, 2)],
    ],
)
def test_get_axon_segments(axons):
    lcn_ex = n_axon2lcn_ex(sum(axon.num_out for axon in axons), 1152)

    tr_max = 1 << lcn_ex

    axon_segs = get_axon_segments(axons, tr_max, 1152)

    for axon_seg in axon_segs.values():
        assert axon_seg.addr_offset <= 1152


@pytest.mark.parametrize(
    "axons",
    [
        [pb.neuron.LIF(1151, 2), pb.neuron.LIF(1153, 2)],
        [pb.neuron.LIF(1151 * 2, 2), pb.neuron.LIF(1153 * 2, 2)],
        [pb.neuron.LIF(2222, 2), pb.neuron.LIF(2378, 2)],
    ],
)
def test_get_axon_segments_boundary(axons):
    """Illegal boundary cases."""
    lcn_ex = n_axon2lcn_ex(sum(axon.num_out for axon in axons), 1152)
    tr_max = 1 << lcn_ex

    with pytest.raises(ResourceError):
        axon_segs = get_axon_segments(axons, tr_max, 1152)


@pytest.mark.parametrize(
    "neu_index, axon_seg, expected",
    [
        (
            slice(5, 8),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(1, 2),
                AxonCoord(2, 0),
                AxonCoord(2, 1),
            ],
        ),
        (
            slice(0, 3),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 0),
                AxonCoord(0, 1),
                AxonCoord(0, 2),
            ],
        ),
        (
            slice(1, 5),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 1),
                AxonCoord(0, 2),
                AxonCoord(1, 0),
                AxonCoord(1, 1),
            ],
        ),
        (
            slice(1, 6),
            AxonSegment(12, 3, 0),
            [
                AxonCoord(0, 1),
                AxonCoord(0, 2),
                AxonCoord(1, 0),
                AxonCoord(1, 1),
                AxonCoord(1, 2),
            ],
        ),
        (
            slice(3, 10),
            AxonSegment(16, 4, 4),
            [
                AxonCoord(0, 4 + 3),
                AxonCoord(1, 4 + 0),
                AxonCoord(1, 4 + 1),
                AxonCoord(1, 4 + 2),
                AxonCoord(1, 4 + 3),
                AxonCoord(2, 4 + 0),
                AxonCoord(2, 4 + 1),
            ],
        ),
    ],
)
def test_aligned_segments(neu_index, axon_seg, expected):
    axon_coords = aligned_coords(neu_index, axon_seg)
    assert axon_coords == expected
