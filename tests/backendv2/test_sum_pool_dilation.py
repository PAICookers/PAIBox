import numpy as np

from paibox.backendv2.get_weight import pool1d_weight_matrix, pool2d_weight_matrix


def test_pool2d_weight_matrix_respects_dilation():
    actual = pool2d_weight_matrix(
        channels=1,
        input_shape=(1, 5, 5),
        output_shape=(1, 3, 3),
        kernel_size=(2, 2),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(2, 2),
        sign=1,
    )

    expected = np.zeros((9, 25), dtype=np.int16)
    positions = [
        [0, 2, 10, 12],
        [1, 3, 11, 13],
        [2, 4, 12, 14],
        [5, 7, 15, 17],
        [6, 8, 16, 18],
        [7, 9, 17, 19],
        [10, 12, 20, 22],
        [11, 13, 21, 23],
        [12, 14, 22, 24],
    ]
    for row, cols in enumerate(positions):
        expected[row, cols] = 1

    assert np.array_equal(actual, expected)


def test_pool1d_weight_matrix_respects_dilation():
    actual = pool1d_weight_matrix(
        channels=1,
        input_shape=(1, 7),
        output_shape=(1, 3),
        kernel_size=(3,),
        stride=(1,),
        padding=(0,),
        dilation=(2,),
        sign=1,
    )

    expected = np.zeros((3, 7), dtype=np.int16)
    expected[0, [0, 2, 4]] = 1
    expected[1, [1, 3, 5]] = 1
    expected[2, [2, 4, 6]] = 1

    assert np.array_equal(actual, expected)
