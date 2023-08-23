import pytest
import numpy as np
from paibox.synapses import OneToOne, AllToAll
from paibox.synapses.connector import MatConn, All2All
from paibox.synapses.transforms import MaskedLinear


@pytest.mark.parametrize("weights", [np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 1])
def test_OneToOne(weights):
    num = 10
    f = OneToOne(num, weights)
    x = np.array(
        [
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        ]
    )
    y = f(x)
    expected = x * weights
    assert np.allclose(y, expected)


def test_AllToAll_weight_scalar():
    num_in, num_out, weight = 10, 20, 1
    x = np.random.randint(2, size=(10,))
    f = AllToAll(num_in, num_out, weight)
    y = f(x)
    expected = np.sum(x, keepdims=True) * weight

    assert np.allclose(y, expected)

    x = np.random.randint(2, size=(10, 1))
    with pytest.raises(ValueError):
        y = f(x)


@pytest.mark.parametrize(
    "shape, weights, x",
    [
        (
            (3, 4),
            np.random.randint(2, size=(3, 4)),
            np.random.randint(2, size=(3,)),
        ),
        (
            (10, 20),
            np.random.randint(2, size=(10, 20)),
            np.random.randint(2, size=(10,)),
        ),
        (
            (20, 10),
            np.random.randint(2, size=(20, 10)),
            np.random.randint(2, size=(20,)),
        ),
    ],
)
def test_AllToAll_array(shape, weights, x):
    num_in, num_out = shape
    f = AllToAll(num_in, num_out, weights)
    y = f(x)
    expected = np.dot(x, weights)

    assert np.allclose(y, expected)


@pytest.mark.parametrize(
    "conn, weights, x",
    [
        (
            MatConn(
                3, 4, conn_mat=np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]])
            ),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            [1, 1, 1],
        ),
        (
            MatConn(
                10, 20, conn_mat=np.random.randint(0, 10, size=(10, 20))
            ),
            np.random.randint(0, 10, size=(10, 20)),
            np.random.randint(2, size=(10,))
        ),
        (
            All2All(20, 10),
            np.random.randint(2, size=(20, 10)),
            np.ones((20,))
        )
    ],
)
def test_MaskedLinear(conn, weights, x):
    f = MaskedLinear(conn, weights=weights)
    y = f(x)
    expected = x @ (weights * conn.build_mat())

    assert y.shape == (conn.dest_num,)
    assert np.allclose(y, expected)
    
    c1 = All2All(4, 3)
    w1 = np.array([[4,3,2],[1,2,1],[1,1,3],[1,3,4]])
    f1 = MaskedLinear(c1, weights=w1)
    x1 = np.array([1,0,1,1])
    y1 = f1(x1)
    e1 = x1 @ (w1 * np.ones((4,3)))
    
    assert np.allclose(y1, e1)
    
