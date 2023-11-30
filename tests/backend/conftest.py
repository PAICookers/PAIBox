import random
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import paibox as pb
from paibox.backend.config_template import CoreConfig, CorePlacementConfig, NeuronConfig
from paibox.backend.placement import NeuSeg
from paibox.backend.routing import RoutingNode
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    Coord,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronSegment,
    RoutingDirection,
    RoutingNodeLevel,
    SNNModeEnable,
    SpikeWidthFormat,
)
from paibox.libpaicore import WeightPrecision as WP


@pytest.fixture(scope="session")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    yield p


@pytest.fixture
def build_example_root():
    """Example root.

    Structure:
        L3: root
        L2_1: L1_1
        L2_2: L1_2, L1_3, L1_4, L1_5
    """
    root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

    node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
    node_l2_2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")

    node_l1_1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
    node_l1_2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
    node_l1_3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")
    node_l1_4 = RoutingNode(RoutingNodeLevel.L1, tag="L1_4")
    node_l1_5 = RoutingNode(RoutingNodeLevel.L1, tag="L1_5")

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
        self.n1 = pb.TonicSpiking(2000, 3, name="n1_1")
        self.n2 = pb.TonicSpiking(1200, 3, name="n2_1")
        self.n3 = pb.TonicSpiking(800, 4, name="n3_1")
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
    """INP1 -> S1 -> N1 -> S2 -> N2"""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(400,), name="inp1_2")
        self.inp2 = pb.InputProj(input=1, shape_out=(400,), name="inp2_2")
        self.n1 = pb.TonicSpiking(400, 3, name="n1_2")
        self.n2 = pb.TonicSpiking(400, 3, name="n2_2")
        self.n3 = pb.TonicSpiking(800, 3, name="n3_2")
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, conn_type=pb.synapses.ConnType.One2One, name="s1_2"
        )
        self.s2 = pb.NoDecay(
            self.inp2, self.n2, conn_type=pb.synapses.ConnType.One2One, name="s2_2"
        )
        self.s3 = pb.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s3_2"
        )
        self.s4 = pb.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.All2All, name="s4_2"
        )


class NetForTest3(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S3 -> N3
    N1 -> S4 -> N4 -> S5 -> N2
    """

    def __init__(self):
        super().__init__()
        self.inp = pb.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(300, 4, name="n4")

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
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S4 -> N4
    N1 -> S3 -> N3
    N3 -> S5 -> N4
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.TonicSpiking(800, 3, name="n1")
        self.n2 = pb.TonicSpiking(400, 4, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(400, 4, name="n4")
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


class NetForTest5(pb.Network):
    """Small 4-bits network #1.

    INP1 -> N1 -> N2 ->
               -> N3 -> N4
    """

    def __init__(self, seed: int):
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


@pytest.fixture(scope="class")
def build_example_net1():
    return NetForTest1()


@pytest.fixture(scope="class")
def build_example_net2():
    return NetForTest2()


@pytest.fixture(scope="function")
def build_example_net3():
    return NetForTest3()


@pytest.fixture(scope="class")
def build_example_net4():
    return NetForTest4()


@pytest.fixture(scope="class")
def build_small_net1():
    seed = 42

    return NetForTest5(seed)


@pytest.fixture(scope="class")
def get_mapper() -> pb.Mapper:
    return pb.Mapper()


class NeuronInstances:
    def __init__(self):
        self.n1 = pb.neuron.LIF(600, 2)
        self.n2 = pb.neuron.LIF(800, 2)
        self.n3 = pb.neuron.LIF(256, 2)
        self.n4 = pb.neuron.LIF(300, 2)


@pytest.fixture(scope="session")
def neu_ins():
    return NeuronInstances()


@pytest.fixture(scope="class")
def neu_segs_test_data(neu_ins):
    data = [
        # Neurons, capacity, wp, lcn_ex
        # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
        (
            [neu_ins.n1, neu_ins.n2],
            512,
            WP.WEIGHT_WIDTH_1BIT,
            LCN_EX.LCN_1X,
        ),
        (
            [neu_ins.n1, neu_ins.n2],
            256,
            WP.WEIGHT_WIDTH_1BIT,
            LCN_EX.LCN_2X,
        ),
        (
            [neu_ins.n3],
            64,
            WP.WEIGHT_WIDTH_8BIT,
            LCN_EX.LCN_1X,
        ),
        (
            [neu_ins.n1, neu_ins.n2, neu_ins.n3, neu_ins.n4],
            512,
            WP.WEIGHT_WIDTH_1BIT,
            LCN_EX.LCN_1X,
        ),
    ]

    return data


@pytest.fixture(scope="class")
def neu_segs_expected_catagory(neu_ins):
    expected = [
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 800, 1), 0))],
        ],
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 256, 1), 0, 2))],
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(256, 512, 1), 0, 2))],
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 256, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(256, 512, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 768, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(768, 800, 1), 0, 2))],
        ],
        [
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(0, 64, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 1, 64 * 2, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 2, 64 * 3, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 3, 64 * 4, 1), 0, 8))],
        ],
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 800, 1), 0))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(0, 256, 1), 0))],
            [NeuSeg(neu_ins.n4, NeuronSegment(slice(0, 300, 1), 0))],
        ],
    ]

    return expected


@pytest.fixture(scope="class")
def neu_segs_expected_dense(neu_ins):
    expected = [
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 512, 1), 0))],
            [
                NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 800, 1), 0)),
                NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 288)),
            ],
        ],
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 256, 1), 0, 2))],
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(256, 512, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 256, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(256, 512, 1), 0, 2))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 768, 1), 0, 2))],
            [
                NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 0, 2)),
                NeuSeg(neu_ins.n2, NeuronSegment(slice(768, 800, 1), 88, 2)),
            ],
        ],
        [
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(0, 64 * 1, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 1, 64 * 2, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 2, 64 * 3, 1), 0, 8))],
            [NeuSeg(neu_ins.n3, NeuronSegment(slice(64 * 3, 64 * 4, 1), 0, 8))],
        ],
        [
            [NeuSeg(neu_ins.n1, NeuronSegment(slice(0, 512, 1), 0))],
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(0, 512, 1), 0))],
            # 300
            [NeuSeg(neu_ins.n4, NeuronSegment(slice(0, 300, 1), 0))],
            # 288
            [NeuSeg(neu_ins.n2, NeuronSegment(slice(512, 800, 1), 0))],
            # 256 + 88
            [
                NeuSeg(neu_ins.n3, NeuronSegment(slice(0, 256, 1), 0)),
                NeuSeg(neu_ins.n1, NeuronSegment(slice(512, 600, 1), 256)),
            ],
        ],
    ]

    return expected


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
        neuron, ns.addr_ram, ns.addr_offset, axon_coords, dest_coords
    )


@pytest.fixture
def MockCorePlacementConfig(MockCoreConfigDict, MockNeuronConfig):
    neuron = pb.neuron.IF((100,), 3, reset_v=-1)

    cpc = CorePlacementConfig.encapsulate(
        np.uint64(random.randint(1, 200)),
        np.random.randint(0, 100, size=(1152, 512)),
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
