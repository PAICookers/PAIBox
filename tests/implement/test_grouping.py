import numpy as np
import pytest

import paibox as pb


def test_weight2binary_connectivity():
    """Test for weight matrix converting to binary connectivity."""
    o = np.zeros((6, 4), np.int8)
    # o = np.array(
    #     [
    #         [1, 2, 3, 4],
    #         [5, 6, 7, 8],
    #         [9, 10, 11, 12],
    #         [13, 14, 15, 16],
    #         [17, 18, 19, 20],
    #         [21, 22, 23, 24],
    #     ],
    #     np.int8,
    # )
    # (8, 2)
    a = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], np.int8
    )

    b = np.split(a, [6], axis=0)

    assert len(b) == 2

    a_old, a_new = b[0], b[1]

    for i in range(0, 2):
        o[:, 2 * i] = a_old[:, i]
        o[:, 2 * i + 1] = np.pad(
            a_new[:, i], (0, 6 - (8 - 6)), "constant", constant_values=0
        )

    expected = np.array(
        [
            [1, 13, 2, 14],
            [3, 15, 4, 16],
            [5, 0, 6, 0],
            [7, 0, 8, 0],
            [9, 0, 10, 0],
            [11, 0, 12, 0],
        ],
        np.int8,
    )

    assert np.array_equal(o, expected)


@pytest.mark.parametrize(
    "cur_shape, expected_shape",
    [((8, 2), (6, 4)), ((120, 20), (100, 50)), ((1200, 144), (1152, 512))],
)
def test_w2bc_parameterized(cur_shape, expected_shape):
    """When LCN_EX > 1X, do w2bc. Else don't need to do so."""
    cur_total = cur_shape[0] * cur_shape[1]

    cur_matrix = np.random.randint(-128, 128, size=cur_shape, dtype=np.int8)

    o_matrix = np.zeros(expected_shape, dtype=np.int8)

    # Certainty
    assert cur_shape[0] > expected_shape[0]

    for i in range(cur_shape[1]):
        o_matrix[:, 2 * i] = cur_matrix[: expected_shape[0], i]
        o_matrix[:, 2 * i + 1] = np.pad(
            cur_matrix[expected_shape[0] :, i],
            (0, 2 * expected_shape[0] - cur_shape[0]),
            "constant",
            constant_values=0,
        )

    # o_matrix[:, :cur_shape[1]] = cur_matrix[: expected_shape[0],:]
    # o_matrix = np.insert(cur_matrix, slice(1, expected_shape[1], 1), 0, axis=1)

    for i in range(cur_shape[1]):
        assert np.array_equal(cur_matrix[: expected_shape[0], i], o_matrix[:, 2 * i])
        assert np.array_equal(
            cur_matrix[expected_shape[0] :, i],
            o_matrix[: cur_shape[0] - expected_shape[0], 2 * i + 1],
        )
