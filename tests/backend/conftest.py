import random
from functools import partial
from typing import Optional

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
from paibox.backend.conf_template import (
    CoreConfig,
    CorePlacementConfig,
    EmptyCorePlacementConfig,
    NeuronConfig,
)
from paibox.backend.placement import NeuSeg
from paibox.backend.routing import RoutingCluster
from paibox.exceptions import ResourceError
from paibox.node import NodeList
from tests.conftest import ParametrizedTestData


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
        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1_1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2_1"
        )
        self.s3 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s3_1"
        )


class NetForTest2(pb.Network):
    """Test the following situations with multiple input nodes:
        1. Two input nodes assigned within one core block.

    Structure:
        INP1 -> S1 -> N1 -> S3 -> N2
        INP2 -> S2 -> N1
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1_2")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2_2")
        self.n1 = pb.TonicSpiking(30, 3, name="n1_2", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2_2", tick_wait_start=2)
        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1_2"
        )
        self.s2 = pb.FullConn(
            self.inp2, self.n1, conn_type=pb.SynConnType.All2All, name="s2_2"
        )
        self.s3 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s3_2"
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

        self.s1 = pb.FullConn(
            self.inp, self.n1, conn_type=pb.SynConnType.One2One, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n1, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n4, self.n2, conn_type=pb.SynConnType.All2All, name="s5"
        )


class NetForTest4(pb.Network):
    """
    INP1 -> S1 -> N1 -> S2 -> N2 -> S4 -> N4
                  N1 -> S3 -> N3 -> S5 -> N4
    """

    def __init__(self, large_scale: bool = False):
        super().__init__()

        self.inp1 = pb.InputProj(input=1, shape_out=(400,), name="inp1")
        self.n1 = pb.TonicSpiking(800, 3, name="n1", tick_wait_start=1)

        if large_scale:
            self.n2 = pb.TonicSpiking(1500, 4, name="n2", tick_wait_start=2)
            self.n3 = pb.TonicSpiking(1500, 4, name="n3", tick_wait_start=2)
        else:
            self.n2 = pb.TonicSpiking(400, 4, name="n2", tick_wait_start=2)
            self.n3 = pb.TonicSpiking(400, 4, name="n3", tick_wait_start=2)

        self.n4 = pb.TonicSpiking(400, 4, name="n4", tick_wait_start=3)
        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n2, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s5"
        )


class Network_with_multi_inodes1(pb.Network):
    """Test the following situations with multiple input nodes:
        1. Two input nodes with their own core blocks.
        2. An input node assigned within one core block.
        TODO 3. The input node is input to the middle layer.

    Structure:
        INP1 -> S1 -> N1 -> S2 -> N2
             -> S3 -> N3 -> S4 -> N4 -> S5 -> N5
        INP2 -> S6 -> N6 -> S7 -> N7 -> S8 -> N5
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.inp2 = pb.InputProj(input=1, shape_out=(50,), name="inp2")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=2)
        self.n3 = pb.TonicSpiking(20, 3, name="n3", tick_wait_start=1)
        self.n4 = pb.TonicSpiking(20, 3, name="n4", tick_wait_start=2)
        self.n5 = pb.TonicSpiking(40, 3, name="n5", tick_wait_start=3)
        self.n6 = pb.TonicSpiking(40, 3, name="n6", tick_wait_start=1)
        self.n7 = pb.TonicSpiking(40, 3, name="n7", tick_wait_start=2)

        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.inp1, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n4, self.n5, conn_type=pb.SynConnType.All2All, name="s5"
        )
        self.s6 = pb.FullConn(
            self.inp2, self.n6, conn_type=pb.SynConnType.All2All, name="s6"
        )
        self.s7 = pb.FullConn(
            self.n6, self.n7, conn_type=pb.SynConnType.All2All, name="s7"
        )
        self.s8 = pb.FullConn(
            self.n7, self.n5, conn_type=pb.SynConnType.All2All, name="s8"
        )


class Network_with_multi_inodes2(pb.Network):
    """Test the following situations with multiple input nodes:
        1. One input node assigned within more than one core block.

    Structure:
        INP1 -> S1 -> N1(tws=1) -> S2 -> N2(tws=2)
             -> S3 -> N3(tws=2) -> S4 -> N4(tws=3)
             -> S5 -> N5(tws=2) -> S6 -> N6(tws=3)
                                -> S7 -> N7(tws=2/3)
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        self.n2 = pb.TonicSpiking(20, 3, name="n2", tick_wait_start=3)
        self.n3 = pb.TonicSpiking(20, 3, name="n3", tick_wait_start=2)
        self.n4 = pb.TonicSpiking(20, 3, name="n4", tick_wait_start=3)
        self.n5 = pb.TonicSpiking(20, 3, name="n5", tick_wait_start=2)
        self.n6 = pb.TonicSpiking(20, 3, name="n6", tick_wait_start=3)
        self.n7 = pb.TonicSpiking(20, 3, name="n7", tick_wait_start=2)

        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.inp1, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.inp1, self.n5, conn_type=pb.SynConnType.All2All, name="s5"
        )
        self.s6 = pb.FullConn(
            self.n5, self.n6, conn_type=pb.SynConnType.All2All, name="s6"
        )
        self.s7 = pb.FullConn(
            self.n5, self.n7, conn_type=pb.SynConnType.All2All, name="s7"
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

        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )

        if connect_n4:
            self.n4 = pb.TonicSpiking(50, 4, name="n4", tick_wait_start=3)
            self.s4 = pb.FullConn(
                self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
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

        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.inp2, self.n1, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s4"
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
                pb.FullConn(
                    self.inp1,
                    self.n_list[i],
                    conn_type=pb.SynConnType.All2All,
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
        self.s1 = pb.FullConn(
            self.inp1,
            self.n1,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s1",
        )
        self.s2 = pb.FullConn(
            self.n1,
            self.n2,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s2",
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s3",
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s4",
        )
        self.s5 = pb.FullConn(
            self.n3,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
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
        self.s1 = pb.FullConn(
            self.inp1,
            self.n1,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s1",
        )
        self.s2 = pb.FullConn(
            self.n1,
            self.n2,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s2",
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s3",
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s4",
        )
        self.s5 = pb.FullConn(
            self.n3,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
            conn_type=pb.SynConnType.MatConn,
            name="s5",
        )


class Network_with_container(pb.DynSysGroup):
    """Network with neurons in list."""

    def __init__(self):
        super().__init__()

        self.inp = pb.InputProj(1, shape_out=(3,))

        n1 = pb.TonicSpiking((3,), 2)
        n2 = pb.TonicSpiking((3,), 3)
        n3 = pb.TonicSpiking((3,), 4)

        n_list = pb.NodeList()
        n_list.append(n1)
        n_list.append(n2)
        n_list.append(n3)
        self.n_list = n_list

        self.s1 = pb.FullConn(n_list[0], n_list[1], conn_type=pb.SynConnType.All2All)
        self.s2 = pb.FullConn(n_list[1], n_list[2], conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n_list[1], "output", name="n2_out")


class ReusedStruct(pb.Network):
    """Reused structure: pre_n -> syn -> post_n, 8-bit"""

    def __init__(self, tws: int = 1, name: Optional[str] = None):
        super().__init__(name=name)

        self.pre_n = pb.LIF((10,), 10, 2, tick_wait_start=tws)
        self.post_n = pb.LIF((10,), 10, 2, tick_wait_start=tws + 1)

        w = np.random.randint(-128, 127, (10, 10), dtype=np.int8)
        self.syn = pb.FullConn(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=w
        )


class Nested_Net_level_2(pb.DynSysGroup):
    """Level 2 nested network: inp1 -> s1 -> ReusedStruct -> s2 -> ReusedStruct"""

    def __init__(self, tws: int = 1, name: Optional[str] = None):
        self.inp1 = pb.InputProj(1, shape_out=(10,))
        subnet1 = ReusedStruct(tws=tws, name="Named_Reused_0")
        subnet2 = ReusedStruct(tws=tws + 2, name="Named_Reused_1")

        self.s1 = pb.FullConn(
            self.inp1,
            subnet1.pre_n,
            conn_type=pb.SynConnType.One2One,
        )
        self.s2 = pb.FullConn(
            subnet1.post_n,
            subnet2.pre_n,
            conn_type=pb.SynConnType.One2One,
        )

        super().__init__(subnet1, subnet2, name=name)


class Nested_Net_level_3(pb.DynSysGroup):
    """Level 3 nested network: inp1 -> s1 -> Nested_Net_level_2"""

    def __init__(self):
        self.inp1 = pb.InputProj(1, shape_out=(10,))
        subnet1 = Nested_Net_level_2(name="Named_Nested_Net_level_2")

        self.s1 = pb.FullConn(
            self.inp1,
            subnet1["Named_Reused_0"].pre_n,
            conn_type=pb.SynConnType.One2One,
        )

        super().__init__(subnet1)


@pytest.fixture(scope="class")
def build_example_net1():
    return NetForTest1()


@pytest.fixture(scope="class")
def build_example_net2():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net1():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net2():
    return Network_with_multi_inodes1()


@pytest.fixture(scope="class")
def build_multi_inputproj_net3():
    return Network_with_multi_inodes2()


@pytest.fixture(scope="class")
def build_example_net3():
    return NetForTest3()


@pytest.fixture(scope="class")
def build_example_net4():
    return NetForTest4()


@pytest.fixture(scope="class")
def build_example_net4_large_scale():
    return NetForTest4(large_scale=True)


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
def build_Nested_Net_level_2():
    return Nested_Net_level_2()


@pytest.fixture(scope="class")
def build_Nested_Net_level_3():
    return Nested_Net_level_3()


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

    neuron = pb.IF((n,), 3, reset_v=-1)
    ns = NeuronSegment(slice(0, 0 + n, 1), offset, interval)

    axon_coords = [AxonCoord(0, i) for i in range(0, n)]
    dest_coords = [Coord(0, 0), Coord(0, 1)]
    pb.BACKEND_CONFIG.test_chip_addr = (10, 0)

    return NeuronConfig.encapsulate(
        neuron,
        n,
        ns.addr_ram,
        ns.addr_offset,
        axon_coords,
        dest_coords,
        pb.BACKEND_CONFIG.test_chip_addr,
    )


@pytest.fixture
def MockCorePlacementConfig(MockCoreConfigDict, MockNeuronConfig):
    neuron = pb.IF((100,), 3, reset_v=-1)

    cpc = CorePlacementConfig.encapsulate(
        random.randint(1, 200),
        np.random.randint(0, 100, size=(1152, 512), dtype=np.uint64),
        MockCoreConfigDict,
        {neuron: MockNeuronConfig},
    )

    return cpc


@pytest.fixture
def MockEmptyCorePlacementConfig(MockCoreConfigDict):
    return EmptyCorePlacementConfig.encapsulate(MockCoreConfigDict)


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


def n_axon2lcn_ex_proto(n_axon, n_fanin_max) -> LCN_EX:
    """Convert #N(of axons) to `LCN_EX` & check.

    NOTE: LCN_EX = log2[ceil(#N/fan-in per dendrite)], where `LCN_1X` = 0.
    """
    if n_axon < 1:
        raise ValueError(f"the number of axons must be positive, but got {n_axon}.")

    if (lcn := ((n_axon - 1) // n_fanin_max).bit_length()) > LCN_EX.LCN_64X:
        raise ResourceError(
            f"required LCN extension out of range {LCN_EX.LCN_64X} ({lcn}). "
        )

    return LCN_EX(lcn)


_neu_params = [
    # n1~n7
    (600, 2, 1),
    (800, 2, 1),
    (320, 2, 2),
    (200, 2, 3),
    (300, 2, 2),
    (400, 2, 1),
    (500, 2, 1),
]


def _gen_neurons_for_neu_segs():
    return [pb.LIF(p[0], p[1], unrolling_factor=p[2]) for p in _neu_params]


_nl = _gen_neurons_for_neu_segs()
_nc = _gen_neurons_for_neu_segs()
_nb = _gen_neurons_for_neu_segs()


class TestData:

    __test__ = False

    toposort_data = ParametrizedTestData(
        args="nodes",
        data=[
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n4"},
                    "n2": {"n3"},
                    "n3": {},
                    "n4": {"n2"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n5"},
                    "n2": {"n3"},
                    "n3": {"n4", "n6"},
                    "n4": {},
                    "n5": {"n3", "n6"},
                    "n6": {"n7"},
                    "n7": {"n4"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "inp2": {"n4"},
                    "n1": {"n2"},
                    "n2": {"n3"},
                    "n3": {},
                    "n4": {"n5"},
                    "n5": {"n3"},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2", "n3"},
                    "n2": {"n4"},
                    "n3": {"n4"},
                    "n4": {},
                }
            ),
            (
                {
                    "inp1": {"n1"},
                    "n1": {"n2"},
                    "n2": {"n4"},
                    "n3": {"n2"},  # Headless neuron N3
                    "n4": {},
                }
            ),
        ],
        ids=[
            "one_input_1",
            "one_input_2",
            "multi_inputs_1",
            "one_input_3",
            "headless_neuron_1",
        ],
    )

    get_longest_path_data = ParametrizedTestData(
        args="edges, expected_path, expected_distance",
        data=[
            (
                # inp1 -> n1 -> n4 -> n2 -> n3, 1+1+1+1=4
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n4": 1},
                    "n2": {"n3": 1},
                    "n3": {},
                    "n4": {"n2": 1},
                },
                ["inp1", "n1", "n4", "n2", "n3"],
                4,
            ),
            (
                # inp1 -> n1 -> n3 -> n4, 1+2+5=8
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 3, "n3": 2},
                    "n2": {"n4": 2},
                    "n3": {"n4": 5},
                    "n4": {},
                },
                ["inp1", "n1", "n3", "n4"],
                8,
            ),
            (
                # inp1 -> n1 -> n2 -> n3, 1+2+1=4
                {
                    "inp1": {"n1": 1},
                    "inp2": {"n2": 1},
                    "n1": {"n2": 2},
                    "n2": {"n3": 1},
                    "n3": {},
                },
                ["inp1", "n1", "n2", "n3"],
                4,
            ),
            (
                # inp1 -> n1 -> n3 -> n5, 1+2+1=4
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n3": 2},
                    "n2": {"n4": 1, "n5": 1},
                    "n3": {"n4": 1},
                    "n4": {},
                    "n5": {},
                },
                ["inp1", "n1", "n3", "n4"],
                4,
            ),
            (
                # inp2 -> n5 -> n4, 4+1=5
                {
                    "inp1": {"n1": 1},
                    "inp2": {"n5": 4},
                    "n1": {"n2": 1, "n3": 1},
                    "n2": {"n5": 1},
                    "n3": {"n4": 1},
                    "n4": {},
                    "n5": {"n4": 1},
                },
                ["inp2", "n5", "n4"],
                5,
            ),
            (
                {"n1": {"n2": 1}, "n2": {}},
                ["n1", "n2"],
                1,
            ),
            (
                {"n1": {}},
                ["n1"],
                0,
            ),
        ],
        ids=[
            "one_input_1",
            "one_input_2",
            "multi_inputs_1",
            "multi_outputs_1",
            "multi_inputs_outputs_1",
            "headless_neuron_1",
            "headless_neuron_2",
        ],
    )

    get_shortest_path_data = ParametrizedTestData(
        args="edges, inodes, expected_path, expected_distance",
        data=[
            (
                # inp1 -> n1 -> n2 -> n3, 1+1+1=3
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n4": 1},
                    "n2": {"n3": 1},
                    "n3": {},
                    "n4": {"n2": 1},
                },
                ["inp1"],
                ["inp1", "n1", "n2", "n3"],
                3,
            ),
            (
                # inp1 -> n1 -> n2 -> n3 -> n6 -> n7 -> n4 =
                # 1+1+3+2+2+3=12
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n5": 5},
                    "n2": {"n3": 3},
                    "n3": {"n4": 10, "n6": 2},
                    "n4": {},
                    "n5": {"n3": 5, "n6": 7},
                    "n6": {"n7": 2},
                    "n7": {"n4": 3},
                },
                ["inp1"],
                ["inp1", "n1", "n2", "n3", "n6", "n7", "n4"],
                12,
            ),
            (
                # inp2 -> n2 -> n3, 1+1=2
                {
                    "inp1": {"n1": 1},
                    "inp2": {"n2": 1},
                    "n1": {"n2": 2},
                    "n2": {"n3": 1},
                    "n3": {},
                },
                ["inp1", "inp2"],
                ["inp2", "n2", "n3"],
                2,
            ),
            (
                # inp1 -> n1 -> n2 -> n4, 1+1+1=3
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n3": 2},
                    "n2": {"n4": 1},
                    "n3": {"n4": 1},
                    "n4": {},
                },
                ["inp1"],
                ["inp1", "n1", "n2", "n4"],
                3,
            ),
            (
                # inp1 -> n1 -> n2 -> n4, 1+1+1=3
                {
                    "inp1": {"n1": 1},
                    "n1": {"n2": 1, "n3": 1},
                    "n2": {"n4": 2},
                    "n3": {"n5": 1},
                    "n4": {},
                    "n5": {},
                },
                ["inp1"],
                ["inp1", "n1", "n3", "n5"],
                3,
            ),
            (
                {"n1": {"n2": 1}, "n2": {}},
                [],
                ["n1", "n2"],
                1,
            ),
            (
                {"n1": {}},
                [],
                ["n1"],
                0,
            ),
        ],
        ids=[
            "one_input_1",
            "one_input_2",
            "multi_inputs_1",
            "multi_outputs_1",
            "multi_outputs_2",
            "headless_neuron_1",
            "headless_neuron_2",
        ],
    )

    cflags_weight_bit_opt_data = ParametrizedTestData(
        args="range, scalar, dtype, expected_wp_noopt, expected_wp_opt",
        data=[
            (
                ((0, 2), (0, 2)),
                1,
                (np.bool_, np.bool_),
                WP.WEIGHT_WIDTH_1BIT,
                WP.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -1,
                (np.bool_, np.bool_),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_2BIT,
            ),
            (
                ((0, 2), (0, 2)),
                1,
                (np.bool_, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -2,
                (np.int8, np.bool_),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_2BIT,
            ),
            (
                ((0, 2), (0, 2)),
                1,
                (np.int8, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (-2, 2)),
                -8,
                (np.bool_, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-2, 2)),
                7,
                (np.bool_, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-128, 128)),
                127,
                (np.bool_, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_8BIT,
            ),
            (
                ((-2, 2), (-8, 8)),
                7,
                (np.int8, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((-8, 8), (-8, 8)),
                -100,
                (np.int8, np.int8),
                WP.WEIGHT_WIDTH_8BIT,
                WP.WEIGHT_WIDTH_8BIT,
            ),
        ],
    )

    neu_segs_latency_test_data = ParametrizedTestData(
        args="neurons, capacity, wp, lcn_ex, expected",
        data=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nl[0], _nl[1]],
                512,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSeg(_nl[0], NeuronSegment(slice(0, 300, 1), 0))],
                    [NeuSeg(_nl[0], NeuronSegment(slice(300, 600, 1), 0))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(0, 400, 1), 0))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(400, 800, 1), 0))],
                ],
            ),
            (
                [_nl[0], _nl[1]],
                256,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nl[0], NeuronSegment(slice(0, 200, 1), 0, 2))],
                    [NeuSeg(_nl[0], NeuronSegment(slice(200, 400, 1), 0, 2))],
                    [NeuSeg(_nl[0], NeuronSegment(slice(400, 600, 1), 0, 2))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(0, 200, 1), 0, 2))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(200, 400, 1), 0, 2))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(400, 600, 1), 0, 2))],
                    [NeuSeg(_nl[1], NeuronSegment(slice(600, 800, 1), 0, 2))],
                ],
            ),
            (
                [_nl[2]],
                200,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nl[2], NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2))],
                    [NeuSeg(_nl[2], NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2))],
                    [NeuSeg(_nl[2], NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2))],
                    [NeuSeg(_nl[2], NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2))],
                ],
            ),
            (
                [_nl[0], _nl[2]],
                400,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSeg(_nl[0], NeuronSegment(slice(0, 300, 1), 0))],
                    [NeuSeg(_nl[0], NeuronSegment(slice(300, 600, 1), 0))],
                    [NeuSeg(_nl[2], NeuronSegment(slice(160 * 0, 160 * 1, 1), 0))],
                    [NeuSeg(_nl[2], NeuronSegment(slice(160 * 1, 160 * 2, 1), 0))],
                ],
            ),
            (
                [_nl[3], _nl[4]],
                240,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nl[3], NeuronSegment(slice(67 * 0, 67 * 1, 1), 0, 2))],
                    [NeuSeg(_nl[3], NeuronSegment(slice(67 * 1, 67 * 2, 1), 0, 2))],
                    [NeuSeg(_nl[3], NeuronSegment(slice(67 * 2, 200, 1), 0, 2))],
                    [NeuSeg(_nl[4], NeuronSegment(slice(75 * 0, 75 * 1, 1), 0, 2))],
                    [NeuSeg(_nl[4], NeuronSegment(slice(75 * 1, 75 * 2, 1), 0, 2))],
                    [NeuSeg(_nl[4], NeuronSegment(slice(75 * 2, 75 * 3, 1), 0, 2))],
                    [NeuSeg(_nl[4], NeuronSegment(slice(75 * 3, 75 * 4, 1), 0, 2))],
                ],
            ),
        ],
    )

    neu_segs_core_test_data = ParametrizedTestData(
        args="neurons, capacity, wp, lcn_ex, expected",
        data=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nc[0], _nc[1]],
                512,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSeg(_nc[0], NeuronSegment(slice(0, 512, 1), 0))],
                    [NeuSeg(_nc[1], NeuronSegment(slice(0, 512, 1), 0))],
                    [
                        NeuSeg(_nc[1], NeuronSegment(slice(512, 800, 1), 0)),
                        NeuSeg(_nc[0], NeuronSegment(slice(512, 600, 1), 288)),
                    ],
                ],
            ),
            (
                [_nc[0], _nc[1]],
                256,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nc[0], NeuronSegment(slice(256 * 0, 256 * 1, 1), 0, 2))],
                    [NeuSeg(_nc[0], NeuronSegment(slice(256 * 1, 256 * 2, 1), 0, 2))],
                    [NeuSeg(_nc[1], NeuronSegment(slice(256 * 0, 256 * 1, 1), 0, 2))],
                    [NeuSeg(_nc[1], NeuronSegment(slice(256 * 1, 256 * 2, 1), 0, 2))],
                    [NeuSeg(_nc[1], NeuronSegment(slice(256 * 2, 256 * 3, 1), 0, 2))],
                    [
                        NeuSeg(_nc[0], NeuronSegment(slice(256 * 2, 600, 1), 0, 2)),
                        NeuSeg(
                            _nc[1], NeuronSegment(slice(256 * 3, 800, 1), 88 * 2, 2)
                        ),
                    ],
                ],
            ),
            (
                [_nc[3], _nc[4]],
                256,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    # Place the neuron segments with full capacity first
                    [NeuSeg(_nc[4], NeuronSegment(slice(0, 256, 1), 0, 2))],
                    [
                        NeuSeg(_nc[3], NeuronSegment(slice(0, 200, 1), 0, 2)),
                        NeuSeg(_nc[4], NeuronSegment(slice(256, 300, 1), 200 * 2, 2)),
                    ],
                ],
            ),
            (
                [_nc[5], _nc[6]],
                512,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSeg(_nc[6], NeuronSegment(slice(0, 500, 1), 0, 1))],
                    [NeuSeg(_nc[5], NeuronSegment(slice(0, 400, 1), 0, 1))],
                ],
            ),
        ],
    )

    neu_segs_both_test_data = ParametrizedTestData(
        args="neurons, capacity, wp, lcn_ex, expected",
        data=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nb[0], _nb[1]],
                512,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSeg(_nb[0], NeuronSegment(slice(0, 300, 1), 0))],
                    [NeuSeg(_nb[0], NeuronSegment(slice(300, 600, 1), 0))],
                    [NeuSeg(_nb[1], NeuronSegment(slice(0, 400, 1), 0))],
                    [NeuSeg(_nb[1], NeuronSegment(slice(400, 800, 1), 0))],
                ],
            ),
            (
                [_nb[0], _nb[1]],
                256,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nb[1], NeuronSegment(slice(0, 200, 1), 0, 2))],
                    [NeuSeg(_nb[1], NeuronSegment(slice(200, 400, 1), 0, 2))],
                    [NeuSeg(_nb[1], NeuronSegment(slice(400, 600, 1), 0, 2))],
                    [NeuSeg(_nb[1], NeuronSegment(slice(600, 800, 1), 0, 2))],
                    [NeuSeg(_nb[0], NeuronSegment(slice(0, 200, 1), 0, 2))],
                    [NeuSeg(_nb[0], NeuronSegment(slice(200, 400, 1), 0, 2))],
                    [NeuSeg(_nb[0], NeuronSegment(slice(400, 600, 1), 0, 2))],
                ],
            ),
            (
                [_nb[2]],
                200,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSeg(_nb[2], NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2))],
                    [NeuSeg(_nb[2], NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2))],
                    [NeuSeg(_nb[2], NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2))],
                    [NeuSeg(_nb[2], NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2))],
                ],
            ),
            (
                [_nb[2], _nb[3]],
                200,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 0, 67 * 1, 1), 160, 2)),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 1, 67 * 2, 1), 160, 2)),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 2, 200, 1), 160, 2)),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2)),
                    ],
                ],
            ),
            (
                [_nb[2], _nb[3], _nb[4]],
                256,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 0, 80 * 1, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 0, 75 * 1, 1), 160, 2)),
                        # offset = 160 + 150
                        NeuSeg(
                            _nb[3],
                            NeuronSegment(slice(67 * 0, 67 * 1, 1), 160 + 150, 2),
                        ),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 1, 80 * 2, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 1, 75 * 2, 1), 160, 2)),
                        # offset = 160 + 150
                        NeuSeg(
                            _nb[3],
                            NeuronSegment(slice(67 * 1, 67 * 2, 1), 160 + 150, 2),
                        ),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 2, 80 * 3, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 2, 75 * 3, 1), 160, 2)),
                        # offset = 160 + 150
                        NeuSeg(
                            _nb[3], NeuronSegment(slice(67 * 2, 200, 1), 160 + 150, 2)
                        ),
                    ],
                    [
                        NeuSeg(_nb[2], NeuronSegment(slice(80 * 3, 80 * 4, 1), 0, 2)),
                        # offset = 160
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 3, 75 * 4, 1), 160, 2)),
                    ],
                ],
            ),
            (
                [_nb[3], _nb[4]],
                240,
                WP.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 0, 75 * 1, 1), 0, 2)),
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 0, 67 * 1, 1), 150, 2)),
                    ],
                    [
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 1, 75 * 2, 1), 0, 2)),
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 1, 67 * 2, 1), 150, 2)),
                    ],
                    [
                        NeuSeg(_nb[4], NeuronSegment(slice(75 * 2, 75 * 3, 1), 0, 2)),
                        NeuSeg(_nb[3], NeuronSegment(slice(67 * 2, 200, 1), 150, 2)),
                    ],
                    [NeuSeg(_nb[4], NeuronSegment(slice(75 * 3, 75 * 4, 1), 0, 2))],
                ],
            ),
        ],
    )
