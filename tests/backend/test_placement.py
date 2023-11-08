import numpy as np
import pytest

import paibox as pb


from paibox.backend.placement import (
    get_axon_segments,
    get_neu_segments,
    aligned_coords,
)
from paibox.libpaicore.v2 import AxonSegment, AxonCoord, LCN_EX


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


@pytest.mark.parametrize(
    "n_neuron_each, n, method, expected",
    [
        (
            [800, 400],
            500,
            "class",
            [
                [(0, slice(0, 500, 1))],
                [(0, slice(500, 800, 1))],
                [(1, slice(0, 400, 1))],
            ],
        ),
        (
            [400, 400],
            256,
            "class",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(256, 400, 1))],
                [(1, slice(0, 256, 1))],
                [(1, slice(256, 400, 1))],
            ],
        ),
        (
            [800, 300],
            512,
            "class",
            [
                [(0, slice(0, 512, 1))],
                [(0, slice(512, 800, 1))],
                [(1, slice(0, 300, 1))],
            ],
        ),
        (
            [300, 800],
            512,
            "class",
            [
                [(0, slice(0, 300, 1))],
                [(1, slice(0, 512, 1))],
                [(1, slice(512, 800, 1))],
            ],
        ),
        (
            [800, 300],
            512,
            "dense",
            [
                [(0, slice(0, 512, 1))],
                [(0, slice(512, 800, 1)), (1, slice(0, 224, 1))],
                [(1, slice(224, 300, 1))],
            ],
        ),
        (
            [300, 800],
            512,
            "dense",
            [
                [(0, slice(0, 300, 1)), (1, slice(0, 212, 1))],
                [(1, slice(212, 724, 1))],
                [(1, slice(724, 800, 1))],
            ],
        ),
        (
            [200, 200, 300],
            256,
            "dense",
            [
                [(0, slice(0, 200, 1)), (1, slice(0, 56, 1))],
                [(1, slice(56, 200, 1)), (2, slice(0, 112, 1))],
                [(2, slice(112, 300, 1))],
            ],
        ),
        (
            [512, 512],
            256,
            "dense",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(256, 512, 1))],
                [(1, slice(0, 256, 1))],
                [(1, slice(256, 512, 1))],
            ],
        ),
        (
            [400, 400, 400],
            256,
            "dense",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(256, 400, 1)), (1, slice(0, 112, 1))],
                [(1, slice(112, 368, 1))],
                [(1, slice(368, 400, 1)), (2, slice(0, 224, 1))],
                [(2, slice(224, 400, 1))],
            ],
        ),
        (
            [1200, 200],
            256,
            "dense",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(256, 512, 1))],
                [(0, slice(512, 768, 1))],
                [(0, slice(768, 1024, 1))],
                [(0, slice(1024, 1200, 1)), (1, slice(0, 80, 1))],
                [(1, slice(80, 200, 1))],
            ],
        ),
        (
            [1024, 200, 600, 200],
            256,
            "dense",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(256, 512, 1))],
                [(0, slice(512, 768, 1))],
                [(0, slice(768, 1024, 1))],
                [(1, slice(0, 200, 1)), (2, slice(0, 56, 1))],
                [(2, slice(56, 312, 1))],
                [(2, slice(312, 568, 1))],
                [(2, slice(568, 600, 1)), (3, slice(0, 200, 1))],
            ],
        ),
        (
            [800, 400, 400],
            1152,
            "dense",
            [
                [(0, slice(0, 800, 1)), (1, slice(0, 352, 1))],
                [(1, slice(352, 400, 1)), (2, slice(0, 400, 1))],
            ],
        ),
    ],
)
def test_group_by_combine(n_neuron_each, n, method, expected):
    box = []
    temp_box = []
    rest_of_box = 0

    for i, n_neuron in enumerate(n_neuron_each):
        if method == "dense":
            if rest_of_box == 0:
                n_left = n_neuron

            elif rest_of_box < n_neuron:
                temp_box.append((i, slice(0, rest_of_box, 1)))
                box.append(temp_box)
                n_left = n_neuron - rest_of_box
                temp_box = []

            else:
                # rest_of_box >= n_neuron
                temp_box.append((i, slice(0, n_neuron, 1)))
                rest_of_box -= n_neuron
                continue  # Go on to place the next
        else:
            n_left = n_neuron

        n_pos_start = rest_of_box
        n_box = n_left // n
        n_left_in_last_core = n_left % n

        if n_box > 0:
            for j in range(n_box):
                box.append(
                    [(i, slice(n_pos_start + j * n, n_pos_start + (j + 1) * n, 1))]
                )

        if n_left_in_last_core > 0:
            if method == "dense":
                temp_box.append((i, slice(n_pos_start + n_box * n, n_neuron, 1)))
                rest_of_box = n - n_left_in_last_core
            else:
                box.append([(i, slice(n_pos_start + n_box * n, n_neuron, 1))])
                rest_of_box = 0

    if temp_box:
        box.append(temp_box)

    assert box == expected


@pytest.mark.parametrize(
    "n_neuron_each, n, method, expected",
    [
        (
            [800, 400],
            500,
            "class",
            [
                [(0, slice(0, 500, 1))],
                [(0, slice(0, 300, 1))],
                [(1, slice(0, 400, 1))],
            ],
        ),
        (
            [400, 400],
            256,
            "class",
            [
                [(0, slice(0, 256, 1))],
                [(0, slice(0, 144, 1))],
                [(1, slice(0, 256, 1))],
                [(1, slice(0, 144, 1))],
            ],
        ),
        (
            [800, 300],
            512,
            "class",
            [
                [(0, slice(0, 512, 1))],
                [(0, slice(0, 288, 1))],
                [(1, slice(0, 300, 1))],
            ],
        ),
        # (
        #     [300, 800],
        #     512,
        #     "class",
        #     [
        #         [(0, slice(0, 300, 1))],
        #         [(1, slice(0, 512, 1))],
        #         [(1, slice(512, 800, 1))],
        #     ],
        # ),
        (
            [800, 300],
            512,
            "dense",
            [
                [(0, slice(0, 512, 1))],
                [(0, slice(0, 288, 1)), (1, slice(0, 224, 1))],
                [(1, slice(0, 76, 1))],
            ],
        ),
        # (
        #     [300, 800],
        #     512,
        #     "dense",
        #     [
        #         [(0, slice(0, 300, 1)), (1, slice(0, 212, 1))],
        #         [(1, slice(212, 724, 1))],
        #         [(1, slice(724, 800, 1))],
        #     ],
        # ),
        # (
        #     [200, 200, 300],
        #     256,
        #     "dense",
        #     [
        #         [(0, slice(0, 200, 1)), (1, slice(0, 56, 1))],
        #         [(1, slice(56, 200, 1)), (2, slice(0, 112, 1))],
        #         [(2, slice(112, 300, 1))],
        #     ],
        # ),
        # (
        #     [512, 512],
        #     256,
        #     "dense",
        #     [
        #         [(0, slice(0, 256, 1))],
        #         [(0, slice(256, 512, 1))],
        #         [(1, slice(0, 256, 1))],
        #         [(1, slice(256, 512, 1))],
        #     ],
        # ),
        # (
        #     [400, 400, 400],
        #     256,
        #     "dense",
        #     [
        #         [(0, slice(0, 256, 1))],
        #         [(0, slice(256, 400, 1)), (1, slice(0, 112, 1))],
        #         [(1, slice(112, 368, 1))],
        #         [(1, slice(368, 400, 1)), (2, slice(0, 224, 1))],
        #         [(2, slice(224, 400, 1))],
        #     ],
        # ),
        # (
        #     [1200, 200],
        #     256,
        #     "dense",
        #     [
        #         [(0, slice(0, 256, 1))],
        #         [(0, slice(256, 512, 1))],
        #         [(0, slice(512, 768, 1))],
        #         [(0, slice(768, 1024, 1))],
        #         [(0, slice(1024, 1200, 1)), (1, slice(0, 80, 1))],
        #         [(1, slice(80, 200, 1))],
        #     ],
        # ),
        # (
        #     [1024, 200, 600, 200],
        #     256,
        #     "dense",
        #     [
        #         [(0, slice(0, 256, 1))],
        #         [(0, slice(256, 512, 1))],
        #         [(0, slice(512, 768, 1))],
        #         [(0, slice(768, 1024, 1))],
        #         [(1, slice(0, 200, 1)), (2, slice(0, 56, 1))],
        #         [(2, slice(56, 312, 1))],
        #         [(2, slice(312, 568, 1))],
        #         [(2, slice(568, 600, 1)), (3, slice(0, 200, 1))],
        #     ],
        # ),
        # (
        #     [800, 400, 400],
        #     1152,
        #     "dense",
        #     [
        #         [(0, slice(0, 800, 1)), (1, slice(0, 352, 1))],
        #         [(1, slice(352, 400, 1)), (2, slice(0, 400, 1))],
        #     ],
        # ),
    ],
)
def test_group_by2(n_neuron_each, n, method, expected):
    box = []
    temp_box = []
    rest_of_box = 0

    for i, n_neuron in enumerate(n_neuron_each):
        if method == "dense":
            if rest_of_box == 0:
                n_left = n_neuron

            elif rest_of_box < n_neuron:
                temp_box.append((i, slice(0, rest_of_box, 1)))
                box.append(temp_box)
                n_left = n_neuron - rest_of_box
                temp_box = []

            else:
                # rest_of_box >= n_neuron
                temp_box.append((i, slice(0, n_neuron, 1)))
                rest_of_box -= n_neuron
                continue  # Go on to place the next
        else:
            n_left = n_neuron

        n_pos_start = rest_of_box
        n_box = n_left // n
        n_left_in_last_core = n_left % n

        if n_box > 0:
            for j in range(n_box):
                box.append(
                    [(i, slice(n_pos_start + j * n, n_pos_start + (j + 1) * n, 1))]
                )

        if n_left_in_last_core > 0:
            p = (i, slice(0, n_left_in_last_core, 1))
            if method == "dense":
                temp_box.append(p)
                rest_of_box = n - n_left_in_last_core
            else:
                box.append([p])
                rest_of_box = 0

    if temp_box:
        box.append(temp_box)

    assert box == expected


# @pytest.mark.parametrize(
#     "axons, lcn_ex, expected",
#     [
#         (
#             [pb.neuron.TonicSpiking(600, 2), pb.neuron.TonicSpiking(800, 2)],
#             LCN_EX.LCN_2X,
#             (
#                 [
#                     AxonSegment(slice(0, 300, 1), 0, 0),
#                     AxonSegment(slice(300, 600, 1), 1, 0),
#                 ],
#                 [
#                     AxonSegment(slice(0, 400, 1), 0, 300),
#                     AxonSegment(slice(400, 800, 1), 1, 300),
#                 ],
#             ),
#         ),
#         (
#             [pb.neuron.TonicSpiking(2222, 2), pb.neuron.TonicSpiking(2378, 2)],
#             LCN_EX.LCN_4X,
#             (
#                 [
#                     AxonSegment(slice(0, 556, 1), 0, 0),
#                     AxonSegment(slice(556, 556 * 2, 1), 1, 0),
#                     AxonSegment(slice(556 * 2, 556 * 3, 1), 2, 0),
#                     AxonSegment(slice(556 * 3, 2222, 1), 3, 0),
#                 ],
#                 [
#                     AxonSegment(slice(0, 595, 1), 0, 556),
#                     AxonSegment(slice(595, 595 * 2, 1), 1, 556),
#                     AxonSegment(slice(595 * 2, 595 * 3, 1), 2, 556),
#                     AxonSegment(slice(595 * 3, 2378, 1), 3, 556),
#                 ],
#             ),
#         ),
#         (
#             [pb.neuron.TonicSpiking(1151, 2), pb.neuron.TonicSpiking(1152, 2)],
#             LCN_EX.LCN_2X,
#             (
#                 [
#                     AxonSegment(slice(0, 576, 1), 0, 0),
#                     AxonSegment(slice(576, 1151, 1), 1, 0),
#                 ],
#                 [
#                     AxonSegment(slice(0, 576, 1), 0, 576),
#                     AxonSegment(slice(576, 1152, 1), 1, 576),
#                 ],
#             ),
#         ),
#     ],
# )
# def test_get_axon_segments_to_dict(axons, lcn_ex, expected):
#     segments = defaultdict(list)

#     pos = 0
#     tr_max = 1 << lcn_ex

#     for a in axons:
#         segs, pos = get_axon_segments(a, tr_max, pos)
#         segments[a] = segs

#     assert len(segments) == len(axons)

#     for i, v in enumerate(segments.values()):
#         assert v == expected[i]


def test_get_neuron_slices():
    neurons = [
        pb.neuron.TonicSpiking(300, 2),
        pb.neuron.TonicSpiking(600, 2),
        pb.neuron.TonicSpiking(800, 2),
    ]

    slices_of_neu = get_neu_segments(neurons, 512)

    print()


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
