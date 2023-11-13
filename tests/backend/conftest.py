import pytest

import paibox as pb
from paibox.backend.placement import NeuSeg
from paibox.backend.routing import RoutingNode
from paibox.libpaicore.v2 import LCN_EX, NeuronSegment
from paibox.libpaicore.v2.reg_types import WeightPrecisionType as WP
from paibox.libpaicore.v2.routing_defs import RoutingDirection, RoutingNodeLevel


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
