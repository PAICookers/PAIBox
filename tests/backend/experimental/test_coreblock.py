import pytest

import paibox as pb
from paibox.backend.experimental.coreblock import NeuronSegment
from paibox.backend.experimental.coreblock import NeuronSegment as NS
from paibox.backend.experimental.coreblock import (
    get_neuron_segments_1,
    get_neuron_segments_2,
)

pytestmark = pytest.mark.skip(reason="Not implemented")

n1 = pb.neuron.TonicSpiking(600, 2)
n2 = pb.neuron.TonicSpiking(800, 2)
n3 = pb.neuron.TonicSpiking(200, 2)
n4 = pb.neuron.TonicSpiking(2500, 2)
n5 = pb.neuron.TonicSpiking(50, 2)
n6 = pb.neuron.TonicSpiking(400, 2)
n7 = pb.neuron.TonicSpiking(512, 2)
n8 = pb.neuron.TonicSpiking(312, 2)
n9 = pb.neuron.TonicSpiking(1024, 2)
n10 = pb.neuron.TonicSpiking(172, 2)


@pytest.mark.parametrize(
    "neurons, capacity, expected",
    [
        (
            [n1, n2],
            512,
            [
                [NS(n1, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n1, slice(512, 600, 1), slice(0, 88, 1))],
                [NS(n2, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n2, slice(512, 800, 1), slice(0, 288, 1))],
            ],
        ),
        (
            [n3, n4, n5],
            512,
            [
                [NS(n3, slice(0, 200, 1), slice(0, 200, 1))],
                [NS(n4, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n4, slice(512, 1024, 1), slice(0, 512, 1))],
                [NS(n4, slice(1024, 1536, 1), slice(0, 512, 1))],
                [NS(n4, slice(1536, 2048, 1), slice(0, 512, 1))],
                [NS(n4, slice(2048, 2500, 1), slice(0, 452, 1))],
                [NS(n5, slice(0, 50, 1), slice(0, 50, 1))],
            ],
        ),
        (
            [n3, n5],
            512,
            [
                [NS(n3, slice(0, 200, 1), slice(0, 200, 1))],
                [NS(n5, slice(0, 50, 1), slice(0, 50, 1))],
            ],
        ),
        (
            [n7, n9],
            512,
            [
                [NS(n7, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n9, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n9, slice(512, 1024, 1), slice(0, 512, 1))],
            ],
        ),
    ],
)
def test_get_neuron_segments_1(neurons, capacity, expected):
    segments_of_neurons = get_neuron_segments_1(neurons, capacity)
    assert len(segments_of_neurons) == len(expected)
    assert segments_of_neurons == expected


@pytest.mark.parametrize(
    "neurons, capacity, expected",
    [
        (
            [n3, n5],
            512,
            [
                [
                    NS(n5, slice(0, 50, 1), slice(0, 50, 1)),
                    NS(n3, slice(0, 200, 1), slice(50, 250, 1)),
                ]
            ],
        ),
        (
            [n3, n6],
            512,
            [
                [
                    NS(n3, slice(0, 200, 1), slice(0, 200, 1)),
                    NS(n6, slice(0, 312, 1), slice(200, 512, 1)),
                ],
                [NS(n6, slice(312, 400, 1), slice(0, 88, 1))],
            ],
        ),
        (
            [n1, n2],
            512,
            [
                [NS(n1, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n2, slice(0, 512, 1), slice(0, 512, 1))],
                [
                    NS(n1, slice(512, 600, 1), slice(0, 88, 1)),
                    NS(n2, slice(512, 800, 1), slice(88, 376, 1)),
                ],
            ],
        ),
        (
            [n2, n4],
            512,
            [
                [NS(n2, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n4, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n4, slice(512, 1024, 1), slice(0, 512, 1))],
                [NS(n4, slice(1024, 1536, 1), slice(0, 512, 1))],
                [NS(n4, slice(1536, 2048, 1), slice(0, 512, 1))],
                [
                    NS(n2, slice(512, 800, 1), slice(0, 288, 1)),
                    NS(n4, slice(2048, 2272, 1), slice(288, 512, 1)),
                ],
                [NS(n4, slice(2272, 2500, 1), slice(0, 228, 1))],
            ],
        ),
        (
            [n1, n6],
            512,
            [
                [NS(n1, slice(0, 512, 1), slice(0, 512, 1))],
                [
                    NS(n1, slice(512, 600, 1), slice(0, 88, 1)),
                    NS(n6, slice(0, 400, 1), slice(88, 488, 1)),
                ],
            ],
        ),
        (
            [n2, n6],
            512,
            [
                [NS(n2, slice(0, 512, 1), slice(0, 512, 1))],
                [
                    NS(n2, slice(512, 800, 1), slice(0, 288, 1)),
                    NS(n6, slice(0, 224, 1), slice(288, 512, 1)),
                ],
                [NS(n6, slice(224, 400, 1), slice(0, 176, 1))],
            ],
        ),
        (
            [n3, n4, n5],
            512,
            [
                [NS(n4, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n4, slice(512, 1024, 1), slice(0, 512, 1))],
                [NS(n4, slice(1024, 1536, 1), slice(0, 512, 1))],
                [NS(n4, slice(1536, 2048, 1), slice(0, 512, 1))],
                [
                    NS(n5, slice(0, 50, 1), slice(0, 50, 1)),
                    NS(n3, slice(0, 200, 1), slice(50, 250, 1)),
                    NS(n4, slice(2048, 2310, 1), slice(250, 512, 1)),
                ],
                [NS(n4, slice(2310, 2500, 1), slice(0, 190, 1))],
            ],
        ),
        (
            [n3, n8],
            512,
            [
                [
                    NS(n3, slice(0, 200, 1), slice(0, 200, 1)),
                    NS(n8, slice(0, 312, 1), slice(200, 512, 1)),
                ],
            ],
        ),
        (
            [n4, n6, n10],
            512,
            [
                [NS(n4, slice(0, 512, 1), slice(0, 512, 1))],
                [NS(n4, slice(512, 1024, 1), slice(0, 512, 1))],
                [NS(n4, slice(1024, 1536, 1), slice(0, 512, 1))],
                [NS(n4, slice(1536, 2048, 1), slice(0, 512, 1))],
                [
                    NS(n10, slice(0, 172, 1), slice(0, 172, 1)),
                    NS(n6, slice(0, 340, 1), slice(172, 512, 1)),
                ],
                [
                    NS(n6, slice(340, 400, 1), slice(0, 60, 1)),
                    NS(n4, slice(2048, 2500, 1), slice(60, 512, 1)),
                ],
            ],
        ),
    ],
)
def test_get_neuron_segments_2(neurons, capacity, expected):
    segments_of_neurons = get_neuron_segments_2(neurons, capacity)
    assert len(segments_of_neurons) == len(expected)
    assert segments_of_neurons == expected
