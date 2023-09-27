import numpy as np
import pytest

from paibox.synapses.connector import All2All, MatConn
from paibox.synapses.transforms import AllToAll, MaskedLinear, OneToOne


@pytest.mark.parametrize(
    "weight, expected_dtype",
    [
        (np.array([1, 2, 3], dtype=np.int8), np.int8),
        (np.array([1, 0, 1], dtype=np.bool_), np.bool_),
        (np.array([1, 0, 1], dtype=np.int8), np.bool_),
        (10, np.int8),
        (-1, np.int8),
    ],
    ids=["array_1", "array_2", "array_3", "scalar_pos", "scalar_neg"],
)
def test_OneToOne(weight, expected_dtype):
    num = 3
    f = OneToOne(num, weight)
    x = np.array([1, 0, 1], dtype=np.bool_)
    y = f(x)
    expected = x * weight

    assert f._get_dtype() == expected_dtype
    assert y.dtype <= np.int32
    assert np.array_equal(y, expected)


@pytest.mark.parametrize(
    "weight, expected_dtype",
    [(1, np.bool_), (-1, np.int8), (10, np.int8), (-100, np.int8)],
    ids=["scalar_1", "scalar_-1", "scalar_10", "scalar_-100"],
)
def test_AllToAll_weight_scalar(weight, expected_dtype):
    num_in, num_out = 10, 20
    x = np.random.randint(2, size=(10,))
    f = AllToAll(num_in, num_out, weight)
    y = f(x)
    expected = np.sum(x) * weight

    assert f._get_dtype() == expected_dtype
    assert np.array_equal(y, expected)
    assert y.dtype <= np.int32

    x = np.random.randint(2, size=(10, 1))
    with pytest.raises(ValueError):
        y = f(x)


@pytest.mark.parametrize(
    "shape, weights, expected_dtype, x",
    [
        (
            (3, 4),
            np.random.randint(2, size=(3, 4), dtype=np.bool_),
            np.bool_,
            np.random.randint(2, size=(3,), dtype=np.bool_),
        ),
        (
            (10, 20),
            np.random.randint(127, size=(10, 20), dtype=np.int8),
            np.int8,
            np.random.randint(2, size=(10,), dtype=np.bool_),
        ),
        (
            (20, 10),
            np.random.randint(2, size=(20, 10), dtype=np.int8),
            np.bool_,
            np.random.randint(2, size=(20,), dtype=np.bool_),
        ),
    ],
    ids=["weights_bool", "weights_int8_1", "weights_int8_2"],
)
def test_AllToAll_array(shape, weights, expected_dtype, x):
    # FIXME weights_bool sometimes error
    num_in, num_out = shape
    f = AllToAll(num_in, num_out, weights)
    y = f(x)
    expected = x @ weights

    assert f._get_dtype() == expected_dtype
    assert y.dtype <= np.int32
    assert np.array_equal(y, expected)


@pytest.mark.parametrize(
    "conn, weights, expected_dtype, x",
    [
        (
            MatConn(
                3,
                4,
                conn_mat=np.array(
                    [[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]], dtype=np.int8
                ),
            ),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int8),
            np.int8,
            np.array([1, 1, 1], dtype=np.bool_),
        ),
        (
            MatConn(
                10, 20, conn_mat=np.random.randint(0, 10, size=(10, 20), dtype=np.int8)
            ),
            np.random.randint(0, 10, size=(10, 20), dtype=np.int8),
            np.int8,
            np.random.randint(2, size=(10,), dtype=np.bool_),
        ),
        (
            All2All(20, 10),
            np.random.randint(2, size=(20, 10), dtype=np.int8),
            np.bool_,
            np.ones((20,), dtype=np.bool_),
        ),
    ],
    ids=["MatConn weights_int8_1", "MatConn weights_int8_2", "All2All weights_bool"],
)
def test_MaskedLinear_conn(conn, weights, expected_dtype, x):
    f = MaskedLinear(conn, weights=weights)
    y = f(x)
    expected = x @ (weights * conn.build_mat())

    assert f._get_dtype() == expected_dtype
    assert y.shape == (conn.dest_num,)
    assert y.dtype <= np.int32
    assert np.array_equal(y, expected)


def test_MaskedLinear_MatConn():
    c1 = MatConn(4, 3, conn_mat=np.ones((4, 3)))
    w1 = np.array([[4, 3, 2], [1, 2, 1], [1, 1, 3], [1, 3, 4]])
    f1 = MaskedLinear(c1, weights=w1)
    x1 = np.array([1, 0, 1, 1], np.bool_)
    y = f1(x1)
    e1 = x1 @ (w1 * np.ones((4, 3)))

    assert f1._get_dtype() == np.int8
    assert y.shape == (c1.dest_num,)
    assert y.dtype <= np.int32
    assert np.array_equal(y, e1)
