import pytest

import paibox as pb


class NetForTest1(pb.Network):
    """INP1 -> S1 -> N1 -> S2 -> N2 -> S3 -> N3"""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.projection.InputProj(input=None, shape_out=(400,), name="inp1")
        self.n1 = pb.neuron.TonicSpiking(400, 3, name="n1")
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
        self.inp = pb.projection.InputProj(input=None, shape_out=(400,), name="inp1")
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
        self.inp = pb.projection.InputProj(input=None, shape_out=(400,), name="inp1")
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
        self.inp1 = pb.projection.InputProj(input=None, shape_out=(400,), name="inp1")
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


@pytest.fixture
def build_example_net1():
    return NetForTest1()


@pytest.fixture
def build_example_net2():
    return NetForTest3()


@pytest.fixture
def build_example_net3():
    return NetForTest3()


@pytest.fixture
def build_example_net4():
    return NetForTest4()


class TestMapper:
    def test_simple_net(self, build_example_net1):
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.clear()
        mapper.build_graph(net)
        mapper.do_grouping()

        print("OK")

    def test_CoreBlock_build(self, build_example_net3):
        net = build_example_net3

        mapper = pb.Mapper()
        mapper.clear()
        mapper.build_graph(net)
        mapper.do_grouping()

        print("OK")
