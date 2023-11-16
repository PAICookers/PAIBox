import random
from pathlib import Path

import numpy as np
import pytest

import paibox as pb
from paibox.backend.config_template import (
    CoreConfigDict,
    CorePlacementConfig,
    NeuronConfig,
)
from paibox.backend.placement import NeuSeg
from paibox.backend.routing import RoutingNode
from paibox.libpaicore import (
    LCN_EX,
    AxonCoord,
    Coord,
    InputWidthFormat,
    MaxPoolingEnable,
    NeuronSegment,
    SNNModeEnable,
    SpikeWidthFormat,
)
from paibox.libpaicore import WeightPrecision as WP
from paibox.libpaicore.v2.routing_defs import RoutingDirection, RoutingNodeLevel
from paibox.neuron.base import MetaNeuron


@pytest.fixture
def build_example_root():
    root = RoutingNode(RoutingNodeLevel.L3, tag="L3")

    node_l2_1 = RoutingNode(RoutingNodeLevel.L2, tag="L2_1")
    node_l2_2 = RoutingNode(RoutingNodeLevel.L2, tag="L2_2")
    node_l2_3 = RoutingNode(RoutingNodeLevel.L2, tag="L2_3")

    node_l1_1 = RoutingNode(RoutingNodeLevel.L1, tag="L1_1")
    node_l1_2 = RoutingNode(RoutingNodeLevel.L1, tag="L1_2")
    node_l1_3 = RoutingNode(RoutingNodeLevel.L1, tag="L1_3")

    node_l2_1.add_child_to(node_l1_1, RoutingDirection.X0Y0)
    node_l2_2.add_child_to(node_l1_2, RoutingDirection.X0Y1)
    node_l2_3.add_child_to(node_l1_3, RoutingDirection.X1Y0)

    root.add_child_to(node_l2_1, RoutingDirection.X0Y0)
    root.add_child_to(node_l2_2, RoutingDirection.X1Y1)
    root.add_child_to(node_l2_3, RoutingDirection.X1Y0)

    return root


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


@pytest.fixture(scope="session")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    yield p


@pytest.fixture
def MockCoreConfigDict() -> CoreConfigDict:
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

    return CoreConfigDict(
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
    coord = Coord(random.randint(0, 31), random.randint(0, 31))

    cpc = CorePlacementConfig.encapsulate(
        coord,
        np.uint64(random.randint(1, 200)),
        np.random.randint(0, 100, size=(1152, 512)),
        MockCoreConfigDict,
        {neuron: MockNeuronConfig},
    )

    return cpc
