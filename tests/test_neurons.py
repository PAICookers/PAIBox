import numpy as np
import pytest

import paibox as pb
from paibox.utils import as_shape, shape2num


@pytest.mark.parametrize("shape", [5, (12,), (20, 20), (1, 2, 3)], ids=["scalar", "ndim=1", "ndim=2", "ndim=3"])
def test_neuron_instance(shape):
    # keep_size = True
    n1 = pb.neuron.TonicSpikingNeuron(shape, 5, keep_size=True)

    assert n1.shape_in == as_shape(shape)
    assert n1.shape_out == as_shape(shape)
    assert len(n1) == shape2num(shape)

    # keep_size = False
    n2 = pb.neuron.TonicSpikingNeuron(shape, 5)

    assert n2.shape_in == as_shape(shape2num(shape))
    assert n2.shape_out == as_shape(shape2num(shape))
    assert len(n2) == shape2num(shape)