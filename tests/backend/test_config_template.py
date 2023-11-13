import pytest

import paibox as pb
from paibox.backend.config_template import NeuronConfig
from paibox.frame.frame_params import FrameType
from paibox.libpaicore import AxonCoord, Coord, NeuronSegment
import time


class TestNeuronConfig:
    def test_NeuronConfig_instance(self):
        t1 = time.time()
        n1 = pb.neuron.IF((100,), 3)
        ns = NeuronSegment(slice(100, 200, 1), 100, 4)

        axon_coords = [AxonCoord(0, i) for i in range(0, 1000)]
        dest_coords = [Coord(0, 0)]

        nc = NeuronConfig.build(
            n1, ns.addr_ram, ns.addr_offset, axon_coords, dest_coords
        )

        assert nc.frame_type is FrameType.FRAME_CONFIG

        params_dict = nc.export_params()

        print(time.time() - t1)
        print(params_dict["addr_ram"])
