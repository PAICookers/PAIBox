import random
from functools import partial
from typing import Optional

import numpy as np
import pytest
from paicorelib import (
    LCN_EX,
    Coord,
    CoordOffset,
    CoreMode,
    HwConfig,
    MaxPoolingEnable,
    RoutingCoord,
    RoutingDirection,
    RoutingLevel,
)
from paicorelib import WeightWidth as WW
from paicorelib.reg_model import TICK_WAIT_END_MAX, TICK_WAIT_START_MAX

import paibox as pb
from paibox.backend.conf_template import (
    CoreConfig,
    CorePlmConfig,
    InputNeuronDest,
    NeuronConfig,
    NeuronDest,
    NeuronDestInfo,
)
from paibox.backend.routing import RoutingCluster
from paibox.backend.types import AxonCoord, AxonSegment, NeuSegment
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
            name="s1",
        )
        self.s2 = pb.FullConn(
            self.n1,
            self.n2,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            name="s2",
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=rng.randint(-8, 8, size=(10, 10), dtype=np.int8),
            name="s3",
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
            name="s4",
        )
        self.s5 = pb.FullConn(
            self.n3,
            self.n4,
            weights=rng.randint(-8, 8, size=(10, 4), dtype=np.int8),
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
            name="s1",
        )
        self.s2 = pb.FullConn(
            self.n1,
            self.n2,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            name="s2",
        )
        self.s3 = pb.FullConn(
            self.n1,
            self.n3,
            weights=rng.randint(-128, 128, size=(10, 10), dtype=np.int8),
            name="s3",
        )
        self.s4 = pb.FullConn(
            self.n2,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
            name="s4",
        )
        self.s5 = pb.FullConn(
            self.n3,
            self.n4,
            weights=rng.randint(-128, 128, size=(10, 4), dtype=np.int8),
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


class MultichipNet1(pb.DynSysGroup):
    def __init__(self, scale: int):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1000,))

        self.n = NodeList()

        for _ in range(5):
            n = random.randint(800, 1500)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.IF((n,), thres, resetv))

        for _ in range(3):
            n = random.randint(3000, 5000)
            leakv = random.randint(-1, 1)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.LIF((n,), thres, resetv, leakv))

        for _ in range(4 * scale):
            n = random.randint(1500, 3000)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.IF((n,), thres, resetv))

        self.n_out = pb.BypassNeuron(1000)

        self.s = NodeList()

        self.s.append(
            pb.FullConn(
                self.inp1,
                self.n[0],
                np.random.randint(
                    -127, 128, size=(self.inp1.num_out, self.n[0].num_in), dtype=np.int8
                ),
            )
        )

        for i in range(7 + 4 * scale):
            self.s.append(
                pb.FullConn(
                    self.n[i],
                    self.n[i + 1],
                    np.random.randint(
                        -127,
                        128,
                        size=(self.n[i].num_out, self.n[i + 1].num_in),
                        dtype=np.int8,
                    ),
                )
            )

        self.s_out = pb.FullConn(
            self.n[-1],
            self.n_out,
            np.random.randint(
                -127, 128, size=(self.n[-1].num_out, self.n_out.num_in), dtype=np.int8
            ),
        )


class Network_branch_nodes1(pb.Network):
    """
    Before:
        INP1 -> N1 -> N2 -> N4
                         -> N5
                   -> N3 -> N5
                         -> N6
    After:
        INP1 -> N1 -> N2 -> N4
                      N2'-> N5
                   -> N3'-> N5
                   -> N3 -> N6
    """

    n_copy = 2

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(600,), name="inp1")
        self.n1 = pb.IF((600,), 10, name="n1", tick_wait_start=1)
        self.n2 = pb.IF((800,), 10, name="n2", tick_wait_start=2)
        self.n3 = pb.IF((1200,), 10, name="n3", tick_wait_start=2)
        self.n4 = pb.IF((500,), 10, name="n4", tick_wait_start=3)
        self.n5 = pb.IF((400,), 10, name="n5", tick_wait_start=3)
        self.n6 = pb.IF((200,), 10, name="n6", tick_wait_start=3)

        self.s1 = pb.FullConn(self.inp1, self.n1, name="s1")
        self.s2 = pb.FullConn(self.n1, self.n2, name="s2")
        self.s3 = pb.FullConn(self.n1, self.n3, name="s3")
        self.s4 = pb.FullConn(self.n2, self.n4, name="s4")
        self.s5 = pb.FullConn(self.n2, self.n5, name="s5")
        self.s6 = pb.FullConn(self.n3, self.n5, name="s6")
        self.s7 = pb.FullConn(self.n3, self.n6, name="s7")


class Network_branch_nodes2(pb.Network):
    """
    Before:
        INP1 -> N1 -> N2 ->
                   -------> N3 -> N4
    After:
        INP1 -> N1'-> N2 ->
             -> N1'-------> N3 -> N4
    """

    n_copy = 1

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(800,), name="inp1")
        self.n1 = pb.IF((800,), 10, name="n1", tick_wait_start=1)
        self.n2 = pb.IF((1000,), 10, name="n2", tick_wait_start=2)
        self.n3 = pb.IF((1200,), 10, name="n3", tick_wait_start=3)
        self.n4 = pb.IF((500,), 10, name="n4", tick_wait_start=4)

        self.s1 = pb.FullConn(self.inp1, self.n1, name="s1")
        self.s2 = pb.FullConn(self.n1, self.n2, name="s2")
        self.s3 = pb.FullConn(self.n1, self.n3, name="s3")
        self.s4 = pb.FullConn(self.n2, self.n3, name="s4")
        self.s5 = pb.FullConn(self.n3, self.n4, name="s5")


class Network_branch_nodes3(pb.Network):
    """
    Before:
        INP1 -> N1 -> N2 ->
                   -> N3 -> N4 -> N5
                               -> N6
                        INP2 --->
    After:
        INP1 -> N1 -> N2'-> N4 -> N5
                   -> N3'->

                      N2 ->
                      N3 -> N4'-> N6
                        INP2 --->
    """

    n_copy = 3

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(800,), name="inp1")
        self.inp2 = pb.InputProj(1, shape_out=(800,), name="inp2")

        self.n1 = pb.IF((800,), 10, name="n1", tick_wait_start=1)
        self.n2 = pb.IF((800,), 10, name="n2", tick_wait_start=2)
        self.n3 = pb.IF((800,), 10, name="n3", tick_wait_start=2)
        self.n4 = pb.IF((1000,), 10, name="n4", tick_wait_start=3)
        self.n5 = pb.IF((800,), 10, name="n5", tick_wait_start=4)
        self.n6 = pb.IF((1000,), 10, name="n6", tick_wait_start=4)

        self.s1 = pb.FullConn(self.inp1, self.n1, name="s1")
        self.s2 = pb.FullConn(self.n1, self.n2, name="s2")
        self.s3 = pb.FullConn(self.n1, self.n3, name="s3")
        self.s4 = pb.FullConn(self.n2, self.n4, name="s4")
        self.s5 = pb.FullConn(self.n3, self.n4, name="s5")
        self.s6 = pb.FullConn(self.n4, self.n5, name="s6")
        self.s7 = pb.FullConn(self.n4, self.n6, name="s7")
        self.s8 = pb.FullConn(self.inp2, self.n6, name="s8")


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
def build_MultichipNet1_s1():
    return MultichipNet1(scale=1)


@pytest.fixture(scope="class")
def build_MultichipNet1_s2():
    return MultichipNet1(scale=2)


@pytest.fixture(
    scope="function",
    params=[Network_branch_nodes1, Network_branch_nodes2, Network_branch_nodes3],
    ids=["net1", "net2", "net3"],
)
def build_Network_branch_nodes(request):
    return request.param()


@pytest.fixture(scope="class")
def get_mapper() -> pb.Mapper:
    return pb.Mapper()


@pytest.fixture
def MockCoreConfigDict() -> CoreConfig:
    wp = random.choice(list(WW))
    lcn_ex = random.choice(list(LCN_EX))

    iwf, swf, sme = random.choice(list(CoreMode)).conf

    num_den = random.randint(1, HwConfig.N_DENDRITE_MAX_SNN)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, TICK_WAIT_START_MAX)
    twe = random.randint(0, TICK_WAIT_END_MAX)
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
    n_channel = 3
    _n_per_ch = random.randint(20, 100)
    n = n_channel * _n_per_ch
    offset = random.randint(1, 100)
    interval = random.randint(1, 2)
    thres = random.randint(1, 5)
    reset_v = random.randint(-5, 5)
    leak_v = np.arange(n_channel * n).reshape((n_channel, n))
    neuron = pb.LIF((n_channel, n), thres, reset_v, bias=leak_v, keep_shape=True)
    dest_coord_start = Coord(random.randint(0, 10), random.randint(0, 10))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    _n_start = random.randint(0, 20)
    nseg = NeuSegment(
        neuron, slice(_n_start, 1 * _n_per_ch + _n_start), offset, interval
    )

    axon_coords = [AxonCoord(0, i) for i in range(nseg.n_neuron)]
    dest_coords = [dest_coord_start, dest_coord_start + CoordOffset(0, 1)]
    pb.BACKEND_CONFIG.test_chip_addr = test_chip_addr

    return NeuronConfig.encapsulate(
        nseg, axon_coords, dest_coords, pb.BACKEND_CONFIG.test_chip_addr
    )


@pytest.fixture
def MockNeuronDestInfo(MockNeuronConfig) -> NeuronDestInfo:
    return MockNeuronConfig.neuron_dest_info


@pytest.fixture
def MockNeuronDest() -> NeuronDest:
    n = random.randint(100, 1000)
    tick_relative = [0 for _ in range(n)]
    addr_axon = [i for i in range(n)]

    addr_core_x = random.randint(0, 31)
    addr_core_y = random.randint(0, 31)
    addr_core_x_ex = random.randint(0, 31)
    addr_core_y_ex = random.randint(0, 31)
    addr_chip_x = random.randint(0, 31)
    addr_chip_y = random.randint(0, 31)

    return NeuronDest(
        tick_relative,
        addr_axon,
        addr_core_x,
        addr_core_y,
        addr_core_x_ex,
        addr_core_y_ex,
        addr_chip_x,
        addr_chip_y,
    )


@pytest.fixture
def MockInputNeuronDest():
    n = random.randint(100, 1000)
    tick_relative = [0 for _ in range(n)]
    addr_axon = [i for i in range(n)]

    addr_core_x = random.randint(0, 31)
    addr_core_y = random.randint(0, 31)
    addr_core_x_ex = random.randint(0, 31)
    addr_core_y_ex = random.randint(0, 31)
    addr_chip_x = random.randint(0, 31)
    addr_chip_y = random.randint(0, 31)
    lcn = 1 << random.choice(list(LCN_EX))

    return InputNeuronDest(
        tick_relative,
        addr_axon,
        addr_core_x,
        addr_core_y,
        addr_core_x_ex,
        addr_core_y_ex,
        addr_chip_x,
        addr_chip_y,
        lcn,
    )


@pytest.fixture
def MockCorePlmConfig(MockCoreConfigDict, MockNeuronConfig):
    n = random.randint(100, 400)
    thres = random.randint(1, 5)
    reset_v = random.randint(-5, 5)
    neuron = pb.IF((n,), thres, reset_v)

    cpc = CorePlmConfig.encapsulate(
        random.randint(0, 1000),
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


def n_axon2lcn_ex_proto(n_axon, n_fanin_max) -> LCN_EX:
    if n_axon < 1:
        raise ValueError

    if (lcn := ((n_axon - 1) // n_fanin_max).bit_length()) > LCN_EX.LCN_64X:
        raise ResourceError

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


def gen_random_used_lx(n: int, lx: int) -> list[RoutingCoord]:
    used_lx = []
    d_candid = list(RoutingDirection)
    d_candid.remove(RoutingDirection.ANY)

    for _ in range(n):
        rc = random.choices(d_candid, k=5 - lx)
        used_lx.append(RoutingCoord(*rc))  # may have repeat elements

    return list(set(used_lx))


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
        args="range, scalar, dtype, expected_wp_opt",
        data=[
            (
                ((0, 2), (0, 2)),
                1,
                (np.bool_, np.bool_),
                WW.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -1,
                (np.bool_, np.bool_),
                WW.WEIGHT_WIDTH_2BIT,
            ),
            (
                ((0, 2), (0, 2)),
                1,
                (np.bool_, np.int8),
                WW.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -2,
                (np.int8, np.bool_),
                WW.WEIGHT_WIDTH_2BIT,
            ),
            (
                ((0, 2), (0, 2)),
                1,
                (np.int8, np.int8),
                WW.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (-2, 2)),
                -8,
                (np.bool_, np.int8),
                WW.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-2, 2)),
                7,
                (np.bool_, np.int8),
                WW.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-128, 128)),
                127,
                (np.bool_, np.int8),
                WW.WEIGHT_WIDTH_8BIT,
            ),
            (
                ((-2, 2), (-8, 8)),
                7,
                (np.int8, np.int8),
                WW.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((-8, 8), (-8, 8)),
                -100,
                (np.int8, np.int8),
                WW.WEIGHT_WIDTH_8BIT,
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
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSegment(_nl[0], slice(0, 300, 1), 0)],
                    [NeuSegment(_nl[0], slice(300, 600, 1), 0)],
                    [NeuSegment(_nl[1], slice(0, 400, 1), 0)],
                    [NeuSegment(_nl[1], slice(400, 800, 1), 0)],
                ],
            ),
            (
                [_nl[0], _nl[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nl[0], slice(0, 200, 1), 0, 2)],
                    [NeuSegment(_nl[0], slice(200, 400, 1), 0, 2)],
                    [NeuSegment(_nl[0], slice(400, 600, 1), 0, 2)],
                    [NeuSegment(_nl[1], slice(0, 200, 1), 0, 2)],
                    [NeuSegment(_nl[1], slice(200, 400, 1), 0, 2)],
                    [NeuSegment(_nl[1], slice(400, 600, 1), 0, 2)],
                    [NeuSegment(_nl[1], slice(600, 800, 1), 0, 2)],
                ],
            ),
            (
                [_nl[2]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nl[2], slice(80 * 0, 80 * 1, 1), 0, 2)],
                    [NeuSegment(_nl[2], slice(80 * 1, 80 * 2, 1), 0, 2)],
                    [NeuSegment(_nl[2], slice(80 * 2, 80 * 3, 1), 0, 2)],
                    [NeuSegment(_nl[2], slice(80 * 3, 80 * 4, 1), 0, 2)],
                ],
            ),
            (
                [_nl[0], _nl[2]],
                400,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSegment(_nl[0], slice(0, 300, 1), 0)],
                    [NeuSegment(_nl[0], slice(300, 600, 1), 0)],
                    [NeuSegment(_nl[2], slice(160 * 0, 160 * 1, 1), 0)],
                    [NeuSegment(_nl[2], slice(160 * 1, 160 * 2, 1), 0)],
                ],
            ),
            (
                [_nl[3], _nl[4]],
                240,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nl[3], slice(67 * 0, 67 * 1, 1), 0, 2)],
                    [NeuSegment(_nl[3], slice(67 * 1, 67 * 2, 1), 0, 2)],
                    [NeuSegment(_nl[3], slice(67 * 2, 200, 1), 0, 2)],
                    [NeuSegment(_nl[4], slice(75 * 0, 75 * 1, 1), 0, 2)],
                    [NeuSegment(_nl[4], slice(75 * 1, 75 * 2, 1), 0, 2)],
                    [NeuSegment(_nl[4], slice(75 * 2, 75 * 3, 1), 0, 2)],
                    [NeuSegment(_nl[4], slice(75 * 3, 75 * 4, 1), 0, 2)],
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
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSegment(_nc[0], slice(0, 512, 1), 0)],
                    [NeuSegment(_nc[1], slice(0, 512, 1), 0)],
                    [
                        NeuSegment(_nc[1], slice(512, 800, 1), 0),
                        NeuSegment(_nc[0], slice(512, 600, 1), 288),
                    ],
                ],
            ),
            (
                [_nc[0], _nc[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nc[0], slice(256 * 0, 256 * 1, 1), 0, 2)],
                    [NeuSegment(_nc[0], slice(256 * 1, 256 * 2, 1), 0, 2)],
                    [NeuSegment(_nc[1], slice(256 * 0, 256 * 1, 1), 0, 2)],
                    [NeuSegment(_nc[1], slice(256 * 1, 256 * 2, 1), 0, 2)],
                    [NeuSegment(_nc[1], slice(256 * 2, 256 * 3, 1), 0, 2)],
                    [
                        NeuSegment(_nc[0], slice(256 * 2, 600, 1), 0, 2),
                        NeuSegment(_nc[1], slice(256 * 3, 800, 1), 88 * 2, 2),
                    ],
                ],
            ),
            (
                [_nc[3], _nc[4]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    # Place the neuron segments with full capacity first
                    [NeuSegment(_nc[4], slice(0, 256, 1), 0, 2)],
                    [
                        NeuSegment(_nc[3], slice(0, 200, 1), 0, 2),
                        NeuSegment(_nc[4], slice(256, 300, 1), 200 * 2, 2),
                    ],
                ],
            ),
            (
                [_nc[5], _nc[6]],
                512,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSegment(_nc[6], slice(0, 500, 1), 0, 1)],
                    [NeuSegment(_nc[5], slice(0, 400, 1), 0, 1)],
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
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [NeuSegment(_nb[0], slice(0, 300, 1), 0)],
                    [NeuSegment(_nb[0], slice(300, 600, 1), 0)],
                    [NeuSegment(_nb[1], slice(0, 400, 1), 0)],
                    [NeuSegment(_nb[1], slice(400, 800, 1), 0)],
                ],
            ),
            (
                [_nb[0], _nb[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nb[1], slice(0, 200, 1), 0, 2)],
                    [NeuSegment(_nb[1], slice(200, 400, 1), 0, 2)],
                    [NeuSegment(_nb[1], slice(400, 600, 1), 0, 2)],
                    [NeuSegment(_nb[1], slice(600, 800, 1), 0, 2)],
                    [NeuSegment(_nb[0], slice(0, 200, 1), 0, 2)],
                    [NeuSegment(_nb[0], slice(200, 400, 1), 0, 2)],
                    [NeuSegment(_nb[0], slice(400, 600, 1), 0, 2)],
                ],
            ),
            (
                [_nb[2]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [NeuSegment(_nb[2], slice(80 * 0, 80 * 1, 1), 0, 2)],
                    [NeuSegment(_nb[2], slice(80 * 1, 80 * 2, 1), 0, 2)],
                    [NeuSegment(_nb[2], slice(80 * 2, 80 * 3, 1), 0, 2)],
                    [NeuSegment(_nb[2], slice(80 * 3, 80 * 4, 1), 0, 2)],
                ],
            ),
            (
                [_nb[2], _nb[3]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSegment(_nb[2], slice(80 * 0, 80 * 1, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[3], slice(67 * 0, 67 * 1, 1), 160, 2),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 1, 80 * 2, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[3], slice(67 * 1, 67 * 2, 1), 160, 2),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 2, 80 * 3, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[3], slice(67 * 2, 200, 1), 160, 2),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 3, 80 * 4, 1), 0, 2),
                    ],
                ],
            ),
            (
                [_nb[2], _nb[3], _nb[4]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSegment(_nb[2], slice(80 * 0, 80 * 1, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[4], slice(75 * 0, 75 * 1, 1), 160, 2),
                        # offset = 160 + 150
                        NeuSegment(
                            _nb[3],
                            slice(67 * 0, 67 * 1, 1),
                            160 + 150,
                            2,
                        ),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 1, 80 * 2, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[4], slice(75 * 1, 75 * 2, 1), 160, 2),
                        # offset = 160 + 150
                        NeuSegment(
                            _nb[3],
                            slice(67 * 1, 67 * 2, 1),
                            160 + 150,
                            2,
                        ),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 2, 80 * 3, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[4], slice(75 * 2, 75 * 3, 1), 160, 2),
                        # offset = 160 + 150
                        NeuSegment(_nb[3], slice(67 * 2, 200, 1), 160 + 150, 2),
                    ],
                    [
                        NeuSegment(_nb[2], slice(80 * 3, 80 * 4, 1), 0, 2),
                        # offset = 160
                        NeuSegment(_nb[4], slice(75 * 3, 75 * 4, 1), 160, 2),
                    ],
                ],
            ),
            (
                [_nb[3], _nb[4]],
                240,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        NeuSegment(_nb[4], slice(75 * 0, 75 * 1, 1), 0, 2),
                        NeuSegment(_nb[3], slice(67 * 0, 67 * 1, 1), 150, 2),
                    ],
                    [
                        NeuSegment(_nb[4], slice(75 * 1, 75 * 2, 1), 0, 2),
                        NeuSegment(_nb[3], slice(67 * 1, 67 * 2, 1), 150, 2),
                    ],
                    [
                        NeuSegment(_nb[4], slice(75 * 2, 75 * 3, 1), 0, 2),
                        NeuSegment(_nb[3], slice(67 * 2, 200, 1), 150, 2),
                    ],
                    [NeuSegment(_nb[4], slice(75 * 3, 75 * 4, 1), 0, 2)],
                ],
            ),
        ],
    )

    aligned_coords_test_data = ParametrizedTestData(
        args="neu_index, axon_seg, delay, n_timeslot, is_iw8, expected",
        data=[
            # iw1
            (
                slice(5, 8),
                AxonSegment(12, 3, 0),
                1,
                1 << 1,
                False,
                [
                    AxonCoord(1, 2),
                    AxonCoord(2, 0),
                    AxonCoord(2, 1),
                ],
            ),
            (
                slice(0, 3),
                AxonSegment(12, 3, 0),
                2,
                1 << 1,
                False,
                [AxonCoord(2 + 0, i) for i in range(3)],
            ),
            (
                slice(1, 5),
                AxonSegment(12, 3, 0),
                2,
                1 << 2,
                False,
                [
                    AxonCoord(4 + 0, 1),
                    AxonCoord(4 + 0, 2),
                    AxonCoord(4 + 1, 0),
                    AxonCoord(4 + 1, 1),
                ],
            ),
            (
                slice(1, 6),
                AxonSegment(12, 3, 0),
                4,
                1 << 3,
                False,
                [
                    AxonCoord(24 + 0, 1),
                    AxonCoord(24 + 0, 2),
                    AxonCoord(24 + 1, 0),
                    AxonCoord(24 + 1, 1),
                    AxonCoord(24 + 1, 2),
                ],
            ),
            (
                slice(3, 10),
                AxonSegment(16, 4, 4),
                4,
                1 << 4,
                False,
                [AxonCoord(48 + 0, 4 + 3)]
                + [AxonCoord(48 + 1, 4 + i) for i in range(4)]
                + [AxonCoord(48 + 2, 4 + 0), AxonCoord(48 + 2, 4 + 1)],
            ),
            # iw8
            (
                slice(5, 8),
                AxonSegment(12, 3, 0),
                1,
                1 << 1,
                True,
                [
                    AxonCoord(1, 8 * 2),
                    AxonCoord(2, 8 * 0),
                    AxonCoord(2, 8 * 1),
                ],
            ),
            (
                slice(0, 3),
                AxonSegment(12, 3, 0),
                2,
                1 << 1,
                True,
                [AxonCoord(2 + 0, 8 * i) for i in range(3)],
            ),
            (
                slice(1, 5),
                AxonSegment(12, 3, 0),
                2,
                1 << 2,
                True,
                [
                    AxonCoord(4 + 0, 8 * 1),
                    AxonCoord(4 + 0, 8 * 2),
                    AxonCoord(4 + 1, 8 * 0),
                    AxonCoord(4 + 1, 8 * 1),
                ],
            ),
            (
                slice(1, 6),
                AxonSegment(12, 3, 0),
                4,
                1 << 3,
                True,
                [
                    AxonCoord(24 + 0, 8 * 1),
                    AxonCoord(24 + 0, 8 * 2),
                    AxonCoord(24 + 1, 8 * 0),
                    AxonCoord(24 + 1, 8 * 1),
                    AxonCoord(24 + 1, 8 * 2),
                ],
            ),
            (
                slice(5, 15),
                AxonSegment(16, 8, 16),
                1,
                1 << 1,
                True,
                [AxonCoord(0, 8 * (16 + i)) for i in range(5, 8)]
                + [AxonCoord(1, 8 * (16 + i)) for i in range(7)],
            ),
            (
                slice(5, 35),
                AxonSegment(40, 10, 10),
                1,
                1 << 2,
                True,
                [AxonCoord(0, 8 * (10 + i)) for i in range(5, 10)]
                + [AxonCoord(1, 8 * (10 + i)) for i in range(10)]
                + [AxonCoord(2, 8 * (10 + i)) for i in range(10)]
                + [AxonCoord(3, 8 * (10 + i)) for i in range(5)],
            ),
        ],
    )
