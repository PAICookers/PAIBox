import random

import numpy as np
from paicorelib import LCN_EX, RoutingCoord, RoutingDirection
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox.backend.types import CustomIndex, DendriteSegment
from paibox.exceptions import ResourceError
from paibox.node import NodeList
from tests.utils import ParamTestCase, TestCase


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


class NetForTest5(pb.Network):
    def __init__(self):
        super().__init__()
        self.n1 = pb.InputProj(input=None, shape_out=(400,), name="n1")
        self.n2 = pb.TonicSpiking(400, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 3, name="n3")
        self.n4 = pb.TonicSpiking(400, 3, name="n4")
        self.n5 = pb.TonicSpiking(800, 3, name="n5")
        self.n6 = pb.TonicSpiking(400, 4, name="n6")
        self.s0 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s0"
        )
        self.s1 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n4, self.n5, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n5, self.n6, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n1, self.n6, conn_type=pb.SynConnType.All2All, name="s5"
        )
        self.s6 = pb.FullConn(
            self.n2, self.n5, conn_type=pb.SynConnType.All2All, name="s6"
        )


class NetForTest6(pb.Network):
    def __init__(self):
        super().__init__()
        self.n1 = pb.InputProj(input=None, shape_out=(400,), name="n1")
        self.n2 = pb.InputProj(input=None, shape_out=(400,), name="n2")
        self.n3 = pb.TonicSpiking(400, 3, name="n3")
        self.n4 = pb.TonicSpiking(400, 3, name="n4")
        self.s0 = pb.FullConn(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s0"
        )
        self.s1 = pb.FullConn(
            self.n1, self.n4, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n2, self.n4, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
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

    def __init__(self, connect_n4: bool = False, onode_more1152: bool = False):
        super().__init__()
        self.inp1 = pb.InputProj(input=1, shape_out=(40,), name="inp1")
        self.n1 = pb.TonicSpiking(80, 2, name="n1", tick_wait_start=1)
        if onode_more1152:
            self.n2 = pb.TonicSpiking(1800, 3, name="n2", tick_wait_start=2)
        else:
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

    def __init__(self, tws: int = 1, name: str | None = None):
        super().__init__(name=name)

        self.pre_n = pb.LIF((10,), 10, 2, tick_wait_start=tws)
        self.post_n = pb.LIF((10,), 10, 2, tick_wait_start=tws + 1)

        w = np.random.randint(-128, 127, (10, 10), dtype=np.int8)
        self.syn = pb.FullConn(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=w
        )


class Nested_Net_level_2(pb.DynSysGroup):
    """Level 2 nested network: inp1 -> s1 -> ReusedStruct -> s2 -> ReusedStruct"""

    def __init__(self, tws: int = 1, name: str | None = None):
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
        tws = 1

        for i in range(5):
            n = random.randint(800, 1500)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.IF((n,), thres, resetv, tick_wait_start=tws + i))

        tws += i + 1

        for i in range(3):
            n = random.randint(3000, 5000)
            leakv = random.randint(-1, 1)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.LIF((n,), thres, resetv, leakv, tick_wait_start=tws + i))

        tws += i + 1

        for i in range(4 * scale):
            n = random.randint(1500, 3000)
            thres = random.randint(3, 6)
            resetv = random.randint(-1, 1)

            self.n.append(pb.IF((n,), thres, resetv, tick_wait_start=tws + i))

        tws += i + 1

        self.n_out = pb.BypassNeuron(1000, tick_wait_start=tws)

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
        self.n6 = pb.IF((200,), 10, name="n6", tick_wait_start=4)

        self.s1 = pb.FullConn(self.inp1, self.n1, name="s1")
        self.s2 = pb.FullConn(self.n1, self.n2, name="s2")
        self.s3 = pb.FullConn(self.n1, self.n3, name="s3")
        self.s4 = pb.FullConn(self.n2, self.n4, name="s4")
        self.s5 = pb.FullConn(self.n3, self.n4, name="s5")
        self.s6 = pb.FullConn(self.n4, self.n5, name="s6")
        self.s7 = pb.FullConn(self.n4, self.n6, name="s7")
        self.s8 = pb.FullConn(self.inp2, self.n6, name="s8")


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


def _gen_custom_index(start: int, end: int, step: int = 1) -> list[CustomIndex]:
    return [CustomIndex(i, 0) for i in range(start, end, step)]


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


class BackendTestCase(TestCase):
    toposort_data = ParamTestCase(
        argnames="nodes",
        argvalues=[
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
    get_longest_path_data = ParamTestCase(
        argnames="edges, expected_path, expected_distance",
        argvalues=[
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
    get_shortest_path_data = ParamTestCase(
        argnames="edges, inodes, expected_path, expected_distance",
        argvalues=[
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
    cflags_weight_bit_opt_data = ParamTestCase(
        argnames="range, scalar, dtype, expected_wp_opt",
        argvalues=[
            (
                ((0, 2), (0, 2)),
                1,
                (bool, bool),
                WW.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -1,
                (bool, bool),
                WW.WEIGHT_WIDTH_2BIT,
            ),
            (
                ((0, 2), (0, 2)),
                1,
                (bool, np.int8),
                WW.WEIGHT_WIDTH_1BIT,
            ),
            (
                ((0, 2), (0, 2)),
                -2,
                (np.int8, bool),
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
                (bool, np.int8),
                WW.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-2, 2)),
                7,
                (bool, np.int8),
                WW.WEIGHT_WIDTH_4BIT,
            ),
            (
                ((0, 2), (-128, 128)),
                127,
                (bool, np.int8),
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
    neu_segs_latency_testcase = ParamTestCase(
        argnames="neurons, capacity, wp, lcn_ex, expected",
        argvalues=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nl[0], _nl[1]],
                512,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [DendriteSegment(_nl[0], _gen_custom_index(0, 300, 1), 0)],
                    [DendriteSegment(_nl[0], _gen_custom_index(300, 600, 1), 0)],
                    [DendriteSegment(_nl[1], _gen_custom_index(0, 400, 1), 0)],
                    [DendriteSegment(_nl[1], _gen_custom_index(400, 800, 1), 0)],
                ],
            ),
            (
                [_nl[0], _nl[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [DendriteSegment(_nl[0], _gen_custom_index(0, 200, 1), 0, 2)],
                    [DendriteSegment(_nl[0], _gen_custom_index(200, 400, 1), 0, 2)],
                    [DendriteSegment(_nl[0], _gen_custom_index(400, 600, 1), 0, 2)],
                    [DendriteSegment(_nl[1], _gen_custom_index(0, 200, 1), 0, 2)],
                    [DendriteSegment(_nl[1], _gen_custom_index(200, 400, 1), 0, 2)],
                    [DendriteSegment(_nl[1], _gen_custom_index(400, 600, 1), 0, 2)],
                    [DendriteSegment(_nl[1], _gen_custom_index(600, 800, 1), 0, 2)],
                ],
            ),
            (
                [_nl[2]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(80 * 0, 80 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(80 * 1, 80 * 2, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(80 * 2, 80 * 3, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(80 * 3, 80 * 4, 1), 0, 2
                        )
                    ],
                ],
            ),
            (
                [_nl[0], _nl[2]],
                400,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [DendriteSegment(_nl[0], _gen_custom_index(0, 300, 1), 0)],
                    [DendriteSegment(_nl[0], _gen_custom_index(300, 600, 1), 0)],
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(160 * 0, 160 * 1, 1), 0
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[2], _gen_custom_index(160 * 1, 160 * 2, 1), 0
                        )
                    ],
                ],
            ),
            (
                [_nl[3], _nl[4]],
                240,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        DendriteSegment(
                            _nl[3], _gen_custom_index(67 * 0, 67 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[3], _gen_custom_index(67 * 1, 67 * 2, 1), 0, 2
                        )
                    ],
                    [DendriteSegment(_nl[3], _gen_custom_index(67 * 2, 200, 1), 0, 2)],
                    [
                        DendriteSegment(
                            _nl[4], _gen_custom_index(75 * 0, 75 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[4], _gen_custom_index(75 * 1, 75 * 2, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[4], _gen_custom_index(75 * 2, 75 * 3, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nl[4], _gen_custom_index(75 * 3, 75 * 4, 1), 0, 2
                        )
                    ],
                ],
            ),
        ],
    )
    neu_segs_core_testcase = ParamTestCase(
        argnames="neurons, capacity, wp, lcn_ex, expected",
        argvalues=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nc[0], _nc[1]],
                512,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [DendriteSegment(_nc[0], _gen_custom_index(0, 512, 1), 0)],
                    [DendriteSegment(_nc[1], _gen_custom_index(0, 512, 1), 0)],
                    [
                        DendriteSegment(_nc[1], _gen_custom_index(512, 800, 1), 0),
                        DendriteSegment(_nc[0], _gen_custom_index(512, 600, 1), 288),
                    ],
                ],
            ),
            (
                [_nc[0], _nc[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        DendriteSegment(
                            _nc[0], _gen_custom_index(256 * 0, 256 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nc[0], _gen_custom_index(256 * 1, 256 * 2, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nc[1], _gen_custom_index(256 * 0, 256 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nc[1], _gen_custom_index(256 * 1, 256 * 2, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nc[1], _gen_custom_index(256 * 2, 256 * 3, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nc[0], _gen_custom_index(256 * 2, 600, 1), 0, 2
                        ),
                        DendriteSegment(
                            _nc[1], _gen_custom_index(256 * 3, 800, 1), 88, 2
                        ),
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
                    [DendriteSegment(_nc[4], _gen_custom_index(0, 256, 1), 0, 2)],
                    [
                        DendriteSegment(_nc[3], _gen_custom_index(0, 200, 1), 0, 2),
                        DendriteSegment(_nc[4], _gen_custom_index(256, 300, 1), 200, 2),
                    ],
                ],
            ),
            (
                [_nc[5], _nc[6]],
                512,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [DendriteSegment(_nc[6], _gen_custom_index(0, 500, 1), 0, 1)],
                    [DendriteSegment(_nc[5], _gen_custom_index(0, 400, 1), 0, 1)],
                ],
            ),
        ],
    )
    neu_segs_both_testcase = ParamTestCase(
        argnames="neurons, capacity, wp, lcn_ex, expected",
        argvalues=[
            # Make sure capacity * (1 << wp) * (1 << lcn_ex) <= 512
            (
                [_nb[0], _nb[1]],
                512,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_1X,
                [
                    [DendriteSegment(_nb[0], _gen_custom_index(0, 300, 1), 0)],
                    [DendriteSegment(_nb[0], _gen_custom_index(300, 600, 1), 0)],
                    [DendriteSegment(_nb[1], _gen_custom_index(0, 400, 1), 0)],
                    [DendriteSegment(_nb[1], _gen_custom_index(400, 800, 1), 0)],
                ],
            ),
            (
                [_nb[0], _nb[1]],
                256,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [DendriteSegment(_nb[1], _gen_custom_index(0, 200, 1), 0, 2)],
                    [DendriteSegment(_nb[1], _gen_custom_index(200, 400, 1), 0, 2)],
                    [DendriteSegment(_nb[1], _gen_custom_index(400, 600, 1), 0, 2)],
                    [DendriteSegment(_nb[1], _gen_custom_index(600, 800, 1), 0, 2)],
                    [DendriteSegment(_nb[0], _gen_custom_index(0, 200, 1), 0, 2)],
                    [DendriteSegment(_nb[0], _gen_custom_index(200, 400, 1), 0, 2)],
                    [DendriteSegment(_nb[0], _gen_custom_index(400, 600, 1), 0, 2)],
                ],
            ),
            (
                [_nb[2]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 0, 80 * 1, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 1, 80 * 2, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 2, 80 * 3, 1), 0, 2
                        )
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 3, 80 * 4, 1), 0, 2
                        )
                    ],
                ],
            ),
            (
                [_nb[2], _nb[3]],
                200,
                WW.WEIGHT_WIDTH_1BIT,
                LCN_EX.LCN_2X,
                [
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 0, 80 * 1, 1), 0, 2
                        ),
                        # offset = 80
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 0, 67 * 1, 1), 80, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 1, 80 * 2, 1), 0, 2
                        ),
                        # offset = 80
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 1, 67 * 2, 1), 80, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 2, 80 * 3, 1), 0, 2
                        ),
                        # offset = 80
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 2, 200, 1), 80, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 3, 80 * 4, 1), 0, 2
                        ),
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
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 0, 80 * 1, 1), 0, 2
                        ),
                        # offset = 160
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 0, 75 * 1, 1), 80, 2
                        ),
                        # offset = 160 + 150
                        DendriteSegment(
                            _nb[3],
                            _gen_custom_index(67 * 0, 67 * 1, 1),
                            80 + 75,
                            2,
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 1, 80 * 2, 1), 0, 2
                        ),
                        # offset = 160
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 1, 75 * 2, 1), 80, 2
                        ),
                        # offset = 160 + 150
                        DendriteSegment(
                            _nb[3],
                            _gen_custom_index(67 * 1, 67 * 2, 1),
                            80 + 75,
                            2,
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 2, 80 * 3, 1), 0, 2
                        ),
                        # offset = 160
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 2, 75 * 3, 1), 80, 2
                        ),
                        # offset = 160 + 150
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 2, 200, 1), 80 + 75, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[2], _gen_custom_index(80 * 3, 80 * 4, 1), 0, 2
                        ),
                        # offset = 160
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 3, 75 * 4, 1), 80, 2
                        ),
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
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 0, 75 * 1, 1), 0, 2
                        ),
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 0, 67 * 1, 1), 75, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 1, 75 * 2, 1), 0, 2
                        ),
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 1, 67 * 2, 1), 75, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 2, 75 * 3, 1), 0, 2
                        ),
                        DendriteSegment(
                            _nb[3], _gen_custom_index(67 * 2, 200, 1), 75, 2
                        ),
                    ],
                    [
                        DendriteSegment(
                            _nb[4], _gen_custom_index(75 * 3, 75 * 4, 1), 0, 2
                        )
                    ],
                ],
            ),
        ],
    )
    # aligned_coords_testcase = ParamTestCase(
    #     argnames="neu_index, axon_seg, delay, n_timeslot, is_iw8, expected",
    #     argvalues=[
    #         # iw1
    #         (
    #             slice(5, 8),
    #             AxonSegment(12, 0, 0, 1152),
    #             1,
    #             1 << 1,
    #             False,
    #             [
    #                 AxonCoord(0, 5),
    #                 AxonCoord(0, 6),
    #                 AxonCoord(0, 7),
    #             ],
    #         ),
    #         (
    #             slice(0, 3),
    #             AxonSegment(12, 0, 0, 1152),
    #             2,
    #             1 << 1,
    #             False,
    #             [AxonCoord(2 + 0, i) for i in range(3)],
    #         ),
    #         (
    #             slice(1, 5),
    #             AxonSegment(12, 0, 0, 1152),
    #             2,
    #             1 << 2,
    #             False,
    #             [
    #                 AxonCoord(4 + 0, 1),
    #                 AxonCoord(4 + 0, 2),
    #                 AxonCoord(4 + 0, 3),
    #                 AxonCoord(4 + 0, 4),
    #             ],
    #         ),
    #         (
    #             slice(1, 6),
    #             AxonSegment(12, 0, 0, 1152),
    #             4,
    #             1 << 3,
    #             False,
    #             [
    #                 AxonCoord(24 + 0, 1),
    #                 AxonCoord(24 + 0, 2),
    #                 AxonCoord(24 + 0, 3),
    #                 AxonCoord(24 + 0, 4),
    #                 AxonCoord(24 + 0, 5),
    #             ],
    #         ),
    #         (
    #             slice(3, 10),
    #             AxonSegment(16, 64, 0, 1152),
    #             4,
    #             1 << 4,
    #             False,
    #             [AxonCoord(48 + 0, 64 + i) for i in range(3, 10)],
    #         ),
    #         # iw8
    #         (
    #             slice(5, 8),
    #             AxonSegment(12, 0, 0, 144),
    #             1,
    #             1 << 1,
    #             True,
    #             [
    #                 AxonCoord(0, 8 * 5),
    #                 AxonCoord(0, 8 * 6),
    #                 AxonCoord(0, 8 * 7),
    #             ],
    #         ),
    #         (
    #             slice(0, 3),
    #             AxonSegment(12, 0, 0, 144),
    #             2,
    #             1 << 1,
    #             True,
    #             [AxonCoord(2 + 0, 8 * i) for i in range(3)],
    #         ),
    #         (
    #             slice(1, 5),
    #             AxonSegment(12, 0, 0, 144),
    #             2,
    #             1 << 2,
    #             True,
    #             [
    #                 AxonCoord(4 + 0, 8 * 1),
    #                 AxonCoord(4 + 0, 8 * 2),
    #                 AxonCoord(4 + 0, 8 * 3),
    #                 AxonCoord(4 + 0, 8 * 4),
    #             ],
    #         ),
    #         (
    #             slice(1, 6),
    #             AxonSegment(12, 0, 0, 144),
    #             4,
    #             1 << 3,
    #             True,
    #             [
    #                 AxonCoord(24 + 0, 8 * 1),
    #                 AxonCoord(24 + 0, 8 * 2),
    #                 AxonCoord(24 + 0, 8 * 3),
    #                 AxonCoord(24 + 0, 8 * 4),
    #                 AxonCoord(24 + 0, 8 * 5),
    #             ],
    #         ),
    #         (
    #             slice(5, 15),
    #             AxonSegment(16, 32, 0, 144),
    #             1,
    #             1 << 1,
    #             True,
    #             [AxonCoord(0, 8 * (32 + i)) for i in range(5, 15)],
    #         ),
    #         (
    #             slice(5, 35),
    #             AxonSegment(40, 120, 0, 144),
    #             1,
    #             1 << 2,
    #             True,
    #             [
    #                 AxonCoord((120 + i) // 144, 8 * ((120 + i) % 144))
    #                 for i in range(5, 35)
    #             ],
    #         ),
    #     ],
    # )
    prune_disconn_graph_testcase = ParamTestCase(
        argnames="graph, start_nodes, expected_graph, disconn_nodes",
        argvalues=[
            (
                {1: [2, 3], 2: [4], 3: [5], 4: [], 5: []},
                [1],
                {1: [2, 3], 2: [4], 3: [5], 4: [], 5: []},
                set(),
            ),
            (
                {
                    "A": ["B", "C"],
                    "B": ["C", "D"],
                    "C": [],
                    "D": [],
                    "E": [],
                    "F": ["G"],
                    "G": ["F", "H"],
                    "H": [],
                    "I": [],
                },
                ["A"],
                {"A": ["B", "C"], "B": ["C", "D"], "C": [], "D": []},
                {"E", "F", "G", "H", "I"},
            ),
            (
                {
                    "A": ["B", "C"],
                    "B": ["C", "D"],
                    "C": [],
                    "D": [],
                    "E": [],
                    "F": ["G"],
                    "G": ["F", "H"],
                    "H": [],
                    "I": [],
                },
                ["A", "G"],
                {
                    "A": ["B", "C"],
                    "B": ["C", "D"],
                    "C": [],
                    "D": [],
                    "F": ["G"],
                    "G": ["F", "H"],
                    "H": [],
                },
                {"E", "I"},
            ),
            (
                {
                    "A": ["B", "C"],
                    "B": ["C", "D"],
                    "C": [],
                    "D": [],
                    "E": [],
                    "F": ["G"],
                    "G": ["F", "H"],
                    "H": [],
                    "I": [],
                },
                # Even if starting from node B, A is connected to B.
                ["B"],
                {"A": ["B", "C"], "B": ["C", "D"], "C": [], "D": []},
                {"E", "F", "G", "H", "I"},
            ),
        ],
        ids=["all_connected", "1_start_node", "2_start_nodes", "start_from_middle"],
    )
