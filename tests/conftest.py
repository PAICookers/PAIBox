from typing import Any, List, NotRequired, TypedDict

import pytest

import paibox as pb


class ParametrizedTestData(TypedDict):
    """Parametrized test data in dictionary format."""

    args: str
    data: List[Any]
    ids: NotRequired[List[str]]


class Input_to_N1(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2, n3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 3, tick_wait_start=2, delay=1)
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, weights=1, conn_type=pb.SynConnType.One2One
        )

        self.probe1 = pb.Probe(self.s1, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")


class NotNested_Net_Exp(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 2, tick_wait_start=2, delay=3)
        self.n2 = pb.TonicSpiking(1, 2, tick_wait_start=3)

        self.s1 = pb.NoDecay(
            self.inp1, self.n1, weights=1, conn_type=pb.SynConnType.One2One
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, weights=1, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.s2, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")
        self.probe5 = pb.Probe(self.n2, "spike", name="n2_spike")
        self.probe6 = pb.Probe(self.n2, "voltage", name="n2_v")


class Network_with_container(pb.DynSysGroup):
    """Network with neurons in list.

    n_list[0] -> s1 -> n_list[1] -> s2 -> n_list[2]
    """

    def __init__(self):
        super().__init__()

        self.inp = pb.InputProj(1, shape_out=(3,))

        n1 = pb.neuron.TonicSpiking((3,), 2)
        n2 = pb.neuron.TonicSpiking((3,), 3)
        n3 = pb.neuron.TonicSpiking((3,), 4)

        n_list: pb.NodeList[pb.neuron.Neuron] = pb.NodeList()
        n_list.append(n1)
        n_list.append(n2)
        n_list.append(n3)
        self.n_list = n_list

        self.s1 = pb.synapses.NoDecay(
            n_list[0], n_list[1], conn_type=pb.SynConnType.All2All
        )
        self.s2 = pb.synapses.NoDecay(
            n_list[1], n_list[2], conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.n_list[1], "output", name="n2_out")


class MoreInput_Net(pb.DynSysGroup):
    """Nested network, level 1.
    n1 -> s1 -> n2 -> s2 -> n4

    n3 -> s3 -> n4
    """

    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, 3)
        self.n2 = pb.neuron.TonicSpiking(2, 3)
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All
        )
        self.n3 = pb.neuron.TonicSpiking(2, 4)
        self.n4 = pb.neuron.TonicSpiking(2, 3)
        self.s2 = pb.synapses.NoDecay(
            self.n2, self.n4, conn_type=pb.SynConnType.All2All
        )
        self.s3 = pb.synapses.NoDecay(
            self.n3, self.n4, conn_type=pb.SynConnType.All2All
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
            self.inp1, self.n1, conn_type=pb.SynConnType.All2All, name="s1"
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.NoDecay(
            self.inp2, self.n1, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.NoDecay(
            self.n1, self.n3, conn_type=pb.SynConnType.All2All, name="s4"
        )


@pytest.fixture(scope="class")
def build_Input_to_N1():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net_Exp():
    return NotNested_Net_Exp()


@pytest.fixture(scope="class")
def build_Network_with_container():
    return Network_with_container()


@pytest.fixture(scope="class")
def build_multi_inodes_onodes():
    return Network_with_multi_inodes_onodes()


@pytest.fixture(scope="class")
def build_MoreInput_Net():
    return MoreInput_Net()
