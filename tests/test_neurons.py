import numpy as np
import pytest

import paibox as pb
from paibox.utils import shape2num, to_shape


@pytest.mark.parametrize("shape", [1, 10, (12,), (20, 20)])
def test_neuron_instance(shape):
    n1 = pb.neuron.TonicSpikingNeuron(shape, 5)

    assert n1.shape_in == to_shape(shape)
    assert len(n1) == shape2num(shape)


@pytest.mark.parametrize(
    "shape, x, expected",
    [
        # Length of data is 16
        (
            1,
            np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.bool_),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool_),
        ),
        (
            (3,),
            np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                ],
                dtype=np.bool_,
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
                dtype=np.bool_,
            ),
        ),
    ],
)
def test_neuron_update(shape, x, expected):
    n1 = pb.neuron.TonicSpikingNeuron(shape, fire_step=3)

    # Traverse the time step
    for i in range(16):
        y = n1.update(x[i])
        assert np.allclose(y, expected[i])
