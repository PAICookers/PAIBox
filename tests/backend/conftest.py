import os
import random
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from paicorelib import (
    LCN_EX,
    AxonCoord,
    Coord,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronSegment,
    RoutingDirection,
    RoutingLevel,
    SNNModeEnable,
    SpikeWidthFormat,
)
from paicorelib import WeightPrecision as WP

import paibox as pb
from paibox.backend.conf_template import CoreConfig, CorePlacementConfig, NeuronConfig
from paibox.backend.routing import RoutingCluster
from paibox.generic import clear_name_cache
from paibox.node import NodeList


@pytest.fixture(scope="module")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

    yield p


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


@pytest.fixture(autouse=True)
def clean_name_dict():
    """Clean the global name dictionary after each test automatically."""
    yield
    clear_name_cache(ignore_warn=True)


@pytest.fixture
def build_example_root():
    """Example root.

    Structure:
        L3: root
        L2_1: L1_1
        L2_2: L1_2, L1_3, L1_4, L1_5
    """
    root = RoutingCluster(RoutingLevel.L3, tag="L3")

    node_l2_1 = RoutingCluster(RoutingLevel.L2, tag="L2_1")
    node_l2_2 = RoutingCluster(RoutingLevel.L2, tag="L2_2")

    node_l1_1 = RoutingCluster(RoutingLevel.L1, tag="L1_1")
    node_l1_2 = RoutingCluster(RoutingLevel.L1, tag="L1_2")
    node_l1_3 = RoutingCluster(RoutingLevel.L1, tag="L1_3")
    node_l1_4 = RoutingCluster(RoutingLevel.L1, tag="L1_4")
    node_l1_5 = RoutingCluster(RoutingLevel.L1, tag="L1_5")

    node_l2_1.add_child_to(node_l1_1, RoutingDirection.X0Y0)
    node_l2_2.add_child_to(node_l1_2, RoutingDirection.X0Y0)
    node_l2_2.add_child_to(node_l1_3, RoutingDirection.X0Y1)
    node_l2_2.add_child_to(node_l1_4, RoutingDirection.X1Y0)
    node_l2_2.add_child_to(node_l1_5, RoutingDirection.X1Y1)

    root.add_child_to(node_l2_1, RoutingDirection.X0Y0)
    root.add_child_to(node_l2_2, RoutingDirection.X0Y1)

    return root


class NetForTest1(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S3 -> N3"""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(2000,), name="inp1_1")
        self.n1 = pb.TonicSpiking(2000, 3, name="n1_1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(1200, 3, name="n2_1", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(800, 4, name="n3_1", tick_wait_start=3)
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1_1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2_1"
        )
        self.s3 = pb.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3_1"
        )


class NetForTest2(pb.Network):
    """
    INP1 -> S1 -> N1 -> S3 -> N2
    INP2 -> S2 -> N1
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1_2")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2_2")
        self.n1 = pb.TonicSpiking(30, 3, name="n1_2", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2_2", tick_wait_start=2)
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1_2"
        )
        self.s2 = pb.NoDecay(
            self.inp2, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s2_2"
        )
        self.s3 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s3_2"
        )


class NetForTest3(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2       ->       N2 -> S3 -> N3
                  N1 -> S4 -> N4 -> S5 -> N2
    """

    def __init__(self):
        super().__init__()
        self.inp = pb.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.TonicSpiking(400, 3, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(800, 3, name="n2", tick_wait_start=3)
        self.n3 = pb.TonicSpiking(400, 4, name="n3", tick_wait_start=4)
        self.n4 = pb.TonicSpiking(300, 4, name="n4", tick_wait_start=2)

        self.s1 = pb.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.One2One, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )
        self.s4 = pb.NoDecay(
            self.n1, self.n4, conn_type=pb.synapses.ConnType.All2All, name="s4"
        )
        self.s5 = pb.NoDecay(
            self.n4, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s5"
        )


class NetForTest4(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2 -> S4 -> N4
                  N1 -> S3 -> N3 -> S5 -> N4
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.TonicSpiking(800, 3, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(400, 4, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(400, 4, name="n3", tick_wait_start=2)
        self.n4 = pb.TonicSpiking(400, 4, name="n4", tick_wait_start=3)
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )
        self.s4 = pb.NoDecay(
            self.n2, self.n4, conn_type=pb.synapses.ConnType.One2One, name="s4"
        )
        self.s5 = pb.NoDecay(
            self.n3, self.n4, conn_type=pb.synapses.ConnType.One2One, name="s5"
        )


class Network_with_multi_inodes(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2
    INP2 -> S3 -> N2
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=2)

        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.inp2, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )


class Network_with_multi_onodes(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2
                  N1 -> S3 -> N3 (-> N4)
    """

    def __init__(self, connect_n4: bool = False):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(30, 4, name="n3", tick_wait_start=2)

        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )

        if connect_n4:
            self.n4 = pb.TonicSpiking(50, 4, name="n4", tick_wait_start=3)
            self.s4 = pb.NoDecay(
                self.n3, self.n4, conn_type=pb.synapses.ConnType.All2All, name="s4"
            )


class Network_with_multi_inodes_onodes(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2
    INP2 -> S3 -> N1 -> S4 -> N3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(30, 3, name="n3", tick_wait_start=2)

        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.inp2, self.n1, conn_type=pb.synapses.ConnType.All2All, name="s3"
        )
        self.s4 = pb.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s4"
        )


class Network_with_N_onodes(pb.Network):
    def __init__(self, n_onodes: int):
        super().__init__()
        self.n_onodes = n_onodes  # for check

        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.s_list = NodeList()
        self.n_list = NodeList()

        for i in range(n_onodes):
            self.n_list.append(
                pb.IF(10, threshold=10, reset_v=2, name=f"n_{i}", tick_wait_start=1)
            )

        for i in range(n_onodes):
            self.s_list.append(
                pb.NoDecay(
                    self.inp1,
                    self.n_list[i],
                    conn_type=pb.synapses.ConnType.All2All,
                    name=f"s_{i}",
                )
            )


class Network_with_Branches_4bit(pb.Network):
    """Network with branches & 4-bit weights.

    INP1 -> N1 -> N2 -> N4
               -> N3 -> N4

    Weights: 4-bit
    Strategy of grouping neurons: catagory
    """

    def __init__(self, seed: int):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.inp1 = pb.InputProj(input=1, shape_out=(10,), name="inp1")
        self.n1 = pb.TonicSpiking(10, 3, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(10, 4, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(10, 4, name="n3", tick_wait_start=2)
        self.n4 = pb.TonicSpiking(4, 4, name="n4", tick_wait_start=3)
        self.s1 = pb.NoDecay(
            self.inp1,
            self.n1,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s1",
        )
        self.s2 = pb.NoDecay(
            self.n1,
            self.n2,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s2",
        )
        self.s3 = pb.NoDecay(
            self.n1,
            self.n3,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s3",
        )
        self.s4 = pb.NoDecay(
            self.n2,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s4",
        )
        self.s5 = pb.NoDecay(
            self.n3,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s5",
        )


class Network_with_Branches_8bit(pb.Network):
    """Network with branches & 8-bit weights, using `dense`.

    Weights: 8-bit
    Strategy of grouping neurons: dense
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        rng = np.random.RandomState(seed)
        self.inp1 = pb.InputProj(input=1, shape_out=(10,), name="inp1")
        self.n1 = pb.TonicSpiking(10, 3, name="n1")
        self.n2 = pb.TonicSpiking(10, 4, name="n2")
        self.n3 = pb.TonicSpiking(10, 4, name="n3")
        self.n4 = pb.TonicSpiking(4, 4, name="n4")
        self.s1 = pb.NoDecay(
            self.inp1,
            self.n1,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s1",
        )
        self.s2 = pb.NoDecay(
            self.n1,
            self.n2,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s2",
        )
        self.s3 = pb.NoDecay(
            self.n1,
            self.n3,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s3",
        )
        self.s4 = pb.NoDecay(
            self.n2,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s4",
        )
        self.s5 = pb.NoDecay(
            self.n3,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
            conn_type=pb.synapses.ConnType.MatConn,
            name="s5",
        )


class Network_with_container(pb.DynSysGroup):
    """Network with neurons in list."""

    def __init__(self):
        super().__init__()

        self.inp = pb.InputProj(1, shape_out=(3,))

        n1 = pb.neuron.TonicSpiking((3,), 2)
        n2 = pb.neuron.TonicSpiking((3,), 3)
        n3 = pb.neuron.TonicSpiking((3,), 4)

        n_list = pb.NodeList()
        n_list.append(n1)
        n_list.append(n2)
        n_list.append(n3)
        self.n_list = n_list

        self.s1 = pb.synapses.NoDecay(
            n_list[0], n_list[1], conn_type=pb.synapses.ConnType.All2All
        )
        self.s2 = pb.synapses.NoDecay(
            n_list[1], n_list[2], conn_type=pb.synapses.ConnType.All2All
        )

        self.probe1 = pb.Probe(self.n_list[1], "output", name="n2_out")


@pytest.fixture(scope="class")
def build_example_net1():
    return NetForTest1()


@pytest.fixture(scope="class")
def build_example_net2():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net2():
    return Network_with_multi_inodes()


@pytest.fixture(scope="class")
def build_example_net3():
    return NetForTest3()


@pytest.fixture(scope="class")
def build_example_net4():
    return NetForTest4()


@pytest.fixture(scope="class")
def build_multi_onodes_net():
    return Network_with_multi_onodes()


@pytest.fixture(scope="class")
def build_multi_onodes_net2():
    return Network_with_multi_onodes(connect_n4=True)


@pytest.fixture(scope="class")
def build_multi_inodes_onodes():
    return Network_with_multi_inodes_onodes()


@pytest.fixture(scope="class", params=[30, 32, 60, 100])
def build_Network_with_N_onodes(request):
    return Network_with_N_onodes(n_onodes=request.param)


@pytest.fixture(scope="class")
def build_network_with_branches_4bit():
    return Network_with_Branches_4bit(seed=42)


@pytest.fixture(scope="class")
def build_Network_8bit_dense():
    return Network_with_Branches_8bit(seed=42)


@pytest.fixture(scope="class")
def build_Network_with_container():
    return Network_with_container()


@pytest.fixture(scope="class")
def get_mapper() -> pb.Mapper:
    return pb.Mapper()


@pytest.fixture
def MockCoreConfigDict() -> CoreConfig:
    wp = random.choice(list(WP))
    lcn_ex = random.choice(list(LCN_EX))
    iwf = random.choice(list(InputWidthFormat))
    swf = random.choice(list(SpikeWidthFormat))
    num_den = random.randint(1, 512)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, 100)
    twe = random.randint(0, 100)
    sme = random.choice(list(SNNModeEnable))
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return CoreConfig(
        "mock_core",
        wp,
        lcn_ex,
        iwf,
        swf,
        num_den,
        mpe,
        tws,
        twe,
        sme,
        target_lcn,
        test_chip_addr,
    )


@pytest.fixture
def MockNeuronConfig() -> NeuronConfig:
    n = random.randint(1, 200)
    offset = random.randint(1, 100)
    interval = random.randint(1, 2)

    neuron = pb.neuron.IF((n,), 3, reset_v=-1)
    ns = NeuronSegment(slice(0, 0 + n, 1), offset, interval)

    axon_coords = [AxonCoord(0, i) for i in range(0, n)]
    dest_coords = [Coord(0, 0), Coord(0, 1)]

    return NeuronConfig.encapsulate(
        neuron, n, ns.addr_ram, ns.addr_offset, axon_coords, dest_coords
    )


@pytest.fixture
def MockCorePlacementConfig(MockCoreConfigDict, MockNeuronConfig):
    neuron = pb.neuron.IF((100,), 3, reset_v=-1)

    cpc = CorePlacementConfig.encapsulate(
        random.randint(1, 200),
        np.random.randint(0, 100, size=(1152, 512), dtype=np.uint64),
        MockCoreConfigDict,
        {neuron: MockNeuronConfig},
    )

    return cpc


def packbits_ref(bits: np.ndarray, count: int) -> int:
    """Pack unsigned bits into a signed integer.

    This is a test of the prototype of the original function.
    """
    _bits = np.append(bits[: count - 1], bits[-1])

    result = sum(bit << i for i, bit in enumerate(_bits))
    result -= _bits[-1] << count

    return result


@pytest.fixture
def packbits8():
    return partial(packbits_ref, count=8)


@pytest.fixture
def packbits4():
    return partial(packbits_ref, count=4)


@pytest.fixture
def packbits2():
    return partial(packbits_ref, count=2)


@pytest.fixture
def packbits1():
    return partial(packbits_ref, count=1)
