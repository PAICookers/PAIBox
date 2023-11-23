import json
from enum import Enum
from json import JSONEncoder
from typing import Any

import numpy as np
import pytest

import paibox as pb
from paibox.libpaicore import Coord


class NetForTest1(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S3 -> N3"""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.projection.InputProj(input=1, shape_out=(2000,), name="inp1")
        self.n1 = pb.neuron.TonicSpiking(2000, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(1200, 3, name="n2")
        self.n3 = pb.neuron.TonicSpiking(800, 4, name="n3")
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.synapses.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )


class NetForTest2(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2"""

    def __init__(self):
        super().__init__()
        self.inp = pb.projection.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.neuron.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(800, 3, name="n2")
        self.s1 = pb.synapses.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )


class NetForTest3(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S3 -> N3
    N1 -> S4 -> N4 -> S5 -> N2
    """

    def __init__(self):
        super().__init__()
        self.inp = pb.projection.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.neuron.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.neuron.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.neuron.TonicSpiking(300, 4, name="n4")

        self.s1 = pb.synapses.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.One2One, name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.synapses.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )
        self.s4 = pb.synapses.NoDecay(
            self.n1, self.n4, conn_type=pb.synapses.ConnType.All2All, name="s4"
        )
        self.s5 = pb.synapses.NoDecay(
            self.n4, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s5"
        )


class NetForTest4(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S4 -> N4
    N1 -> S3 -> N3
    N3 -> S5 -> N4
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.projection.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.neuron.TonicSpiking(800, 3, name="n1")
        self.n2 = pb.neuron.TonicSpiking(400, 4, name="n2")
        self.n3 = pb.neuron.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.neuron.TonicSpiking(400, 4, name="n4")
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.synapses.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )
        self.s4 = pb.synapses.NoDecay(
            self.n2, self.n4, conn_type=pb.synapses.ConnType.One2One, name="s4"
        )
        self.s5 = pb.synapses.NoDecay(
            self.n3, self.n4, conn_type=pb.synapses.ConnType.One2One, name="s5"
        )


@pytest.fixture(scope="class")
def build_example_net1():
    return NetForTest1()


@pytest.fixture(scope="class")
def build_example_net2():
    return NetForTest3()


@pytest.fixture(scope="function")
def build_example_net3():
    return NetForTest3()


@pytest.fixture(scope="class")
def build_example_net4():
    return NetForTest4()


@pytest.fixture(scope="class")
def get_mapper() -> pb.Mapper:
    return pb.Mapper()


class CustomJsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Coord):
            return o.to_tuple()
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, np.ndarray):
            return int(o)
        else:
            return super().default(o)


class TestMapperDebug:
    @pytest.fixture
    def test_simple_net(self, get_mapper, build_example_net1):
        """Go throught the backend"""
        net = build_example_net1

        mapper = get_mapper
        mapper.clear()
        mapper.build_graph(net, filter_cycle=True)
        mapper.main_phases()

    @pytest.mark.usefixtures("test_simple_net")
    def test_export_config_json(self, get_mapper, ensure_dump_dir):
        """Export all the configs into json"""
        mapper: pb.Mapper = get_mapper
        assert mapper.has_built == True

        assert len(mapper.core_blocks) == 3  # 3 layers

        _json_core_configs = dict()
        _json_core_plm_config = dict()
        _json_inp_proj_info = dict()

        for coord, core_param in mapper.core_params.items():
            _json_core_configs[coord.address] = core_param.__json__()

        for coord, cpc in mapper.core_plm_config.items():
            _json_core_plm_config[coord.address] = cpc.__json__()
        
        for inode, nd in mapper.input_cb_info.items():
            _json_inp_proj_info[inode] = nd.__json__()

        # Export parameters of cores into json
        with open(ensure_dump_dir / "core_configs.json", "w") as f:
            json.dump(
                _json_core_configs,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        # Export complete configurations of cores into json
        with open(ensure_dump_dir / "core_plm_configs.json", "w") as f:
            json.dump(
                _json_core_plm_config,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        # Export the info of input projections into json
        with open(ensure_dump_dir / "input_proj_info.json", "w") as f:
            json.dump(
                _json_inp_proj_info,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )
            
        # Export the info of output destination into json
        with open(ensure_dump_dir / "output_dest_info.json", "w") as f:
            json.dump(
                mapper.output_dest_info,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        print("OK")

    @pytest.mark.skip
    def test_CoreBlock_build(self, get_mapper, build_example_net3):
        net = build_example_net3

        mapper = get_mapper
        mapper.clear()
        mapper.build_graph(net)
        mapper.main_phases()

        print("OK")
