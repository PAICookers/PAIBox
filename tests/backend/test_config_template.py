import pytest

import paibox as pb
from paibox.backend.config_template import NeuronConfig
from paibox.frame.frame_params import FrameType
from paibox.libpaicore import AxonCoord, Coord


class TestNeuronConfig:
    def test_NeuronConfig_instance(self):
        n1 = pb.neuron.IF((4,), 3)

        addr_ram = slice(0, len(n1), 1)
        axon_coords = [
            AxonCoord(0, 0),
            AxonCoord(0, 1),
            AxonCoord(0, 2),
            AxonCoord(0, 3),
        ]
        dest_coords = [Coord(0, 0)]

        nc = NeuronConfig.build(n1, addr_ram, axon_coords, dest_coords)

        assert nc.frame_type is FrameType.FRAME_CONFIG
