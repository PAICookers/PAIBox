import numpy as np
import pytest

from paibox.synapses.transforms import AllToAll, MaskedLinear, OneToOne


@pytest.mark.parametrize(
    "weight",
    [
        (np.array([1, 2, 3], dtype=np.int8), np.int8),
        (np.array([1, 0, 1], dtype=np.bool_), np.bool_),
        (np.array([1, 0, 1], dtype=np.int8), np.bool_),
        (10, np.int8),
        (-1, np.int8),
        (np.array([127, 0, 1], dtype=np.int8), np.int8),
        (np.array([-128, 1, 127], dtype=np.int8), np.int8)
    ],
    ids=["array_1", "array_2", "array_3", "scalar_pos", "scalar_neg", "array_127", "array_-128"],
)
def test_OneToOne_dtype(weight):
    num = 3
    f = OneToOne(num, weight)
    x = np.array([1, 0, 1], dtype=np.bool_)
    y = f(x)
    expected = x * weight

    assert y.dtype == np.int32
    assert y.shape == (num,)
    assert np.array_equal(y, expected)
    assert f.connectivity.shape == (num, num)


def test_OneToOne():
    weight = np.array([1, 2, 3, 4], dtype=np.int8)
    f = OneToOne(4, weight)
    assert f.connectivity.shape == (4, 4)

    # The last spike is an array.
    x1 = np.array([1, 2, 3, 4], dtype=np.int8)
    y = f(x1)
    assert y.shape == (4,)

    # The last spike is a scalar.
    x2 = np.array(2, dtype=np.int8)
    y = f(x2)
    assert y.shape == (4,)


@pytest.mark.parametrize(
    "weight, expected_dtype",
    [(1, np.bool_), (-1, np.int8), (10, np.int8), (-100, np.int8), (-128, np.int8), (127, np.int8),],
    ids=["scalar_1", "scalar_-1", "scalar_10", "scalar_-100", "scalar_-128", "scalar_-127"],
)
def test_AllToAll_weight_scalar(weight, expected_type):
    """Test `AllToAll` when weight is a scalar"""

    num_in, num_out = 10, 20
    x = np.random.randint(2, size=(10,))
    f = AllToAll((num_in, num_out), weight)
    y = f(x)
    expected = np.full((num_out,), np.sum(x, axis=None), dtype=np.int32) * weight

    assert f.dtype == expected_type
    assert y.dtype == np.int32
    assert y.shape == (num_out,)
    assert y.ndim == 1
    assert np.array_equal(y, expected)
    assert f.connectivity.shape == (num_in, num_out)


@pytest.mark.parametrize(
    "shape, x, weights",
    [
        (
            (3, 4),
            np.random.randint(2, size=(3,), dtype=np.bool_),
            np.random.randint(2, size=(3, 4), dtype=np.bool_),
        ),
        (
            (10, 20),
            np.random.randint(2, size=(10,), dtype=np.bool_),
            np.random.randint(127, size=(10, 20), dtype=np.int8),
        ),
        (
            (20, 10),
            np.random.randint(2, size=(20,), dtype=np.bool_),
            np.random.randint(2, size=(20, 10), dtype=np.int8),
        ),
        (
            (2, 2),
            np.array([1, 1], dtype=np.bool_),
            np.array([[1, 2], [3, 4]], dtype=np.int8),
        ),
        (
            (2, 2),
            np.array([1, 1], dtype=np.bool_),
            np.array([[127, 0], [3, -128]], dtype=np.int8),
            np.int8,
        ),
    ],
    ids=["weights_bool_1", "weights_int8_1", "weights_int8_2", "weights_int8_3", "weights_int8_4"],
)
def test_AllToAll_array(shape, x, weights):
    """Test `AllToAll` when weights is an array"""

    f = AllToAll(shape, weights)
    y = f(x)
    expected = x @ weights.astype(np.int8)

    assert f.dtype == np.int32
    assert np.array_equal(y, expected)
    assert f.connectivity.shape == shape


@pytest.mark.parametrize(
    "shape, x, weights",
    [
        (
            (3, 4),
            np.array([1, 1, 1], dtype=np.bool_),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int8),
        ),
        (
            (10, 20),
            np.random.randint(2, size=(10,), dtype=np.bool_),
            np.random.randint(0, 10, size=(10, 20), dtype=np.int8),
        ),
        (
            (20, 10),
            np.ones((20,), dtype=np.bool_),
            np.random.randint(2, size=(20, 10), dtype=np.int8),
        ),
        (
                (2, 2),
                np.array([1, 1], dtype=np.bool_),
                np.array([[127, 0], [3, -128]], dtype=np.int8),
                np.int8,
        ),
    ],
    ids=["weights_int8_1", "weights_int8_2", "weights_bool", "weights_int8_3"],
)
def test_MaskedLinear_conn(shape, x, weights):
    f = MaskedLinear(shape, weights)
    y = f(x)
    expected = x @ weights

    assert f.dtype == np.int32
    assert f.connectivity.dtype == np.int32
    assert y.shape == (shape[1],)
    assert y.dtype == np.int32
    assert np.array_equal(y, expected)
    assert f.connectivity.shape == shape
