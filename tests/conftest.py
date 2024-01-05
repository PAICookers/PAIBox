import numpy as np
import pytest

import paibox as pb


class Input_to_N1(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2, n3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 3, tick_wait_start=2, delay=1)
        self.s1 = pb.NoDecay(
            self.inp1, self.n1, weights=1, conn_type=pb.synapses.ConnType.One2One
        )

        self.probe1 = pb.Probe(self.s1, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")


class NotNested_Net_Exp(pb.DynSysGroup):
    """Not nested network
    inp1 -> n1 -> s1 -> n2, n3
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(1,))
        self.n1 = pb.TonicSpiking(1, 2, tick_wait_start=2, delay=3)
        self.n2 = pb.TonicSpiking(1, 2, tick_wait_start=3)

        self.s1 = pb.NoDecay(
            self.inp1, self.n1, weights=1, conn_type=pb.synapses.ConnType.One2One
        )
        self.s2 = pb.NoDecay(
            self.n1, self.n2, weights=1, conn_type=pb.synapses.ConnType.All2All
        )
        self.n3 = pb.TonicSpiking(2, 4)  # Not used

        self.probe1 = pb.Probe(self.s2, "output", name="s2_out")
        self.probe2 = pb.Probe(self.n1, "delay_registers", name="n1_reg")
        self.probe3 = pb.Probe(self.n1, "spike", name="n1_spike")
        self.probe4 = pb.Probe(self.n1, "voltage", name="n1_v")
        self.probe5 = pb.Probe(self.n2, "spike", name="n2_spike")
        self.probe6 = pb.Probe(self.n2, "voltage", name="n2_v")


class Nested_Net_Level1_1(pb.DynSysGroup):
    """Nested network, level 1.
    n1 -> s1 -> n2     n1 -> s1 -> n2
    |                  |
    subnet1 -> s2 -> subnet2
    """

    def __init__(self):
        super().__init__()

        class Subnet(pb.DynSysGroup):
            def __init__(self):
                super().__init__()
                self.n1 = pb.neuron.TonicSpiking(2, 3)
                self.n2 = pb.neuron.TonicSpiking(2, 3)
                self.s1 = pb.synapses.NoDecay(
                    self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All
                )

        self.subnet1 = Subnet()
        self.subnet2 = Subnet()
        self.s2 = pb.synapses.NoDecay(
            self.subnet1.n2, self.subnet2.n1, conn_type=pb.synapses.ConnType.All2All
        )


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
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All
        )
        self.n3 = pb.neuron.TonicSpiking(2, 4)
        self.n4 = pb.neuron.TonicSpiking(2, 3)
        self.s2 = pb.synapses.NoDecay(
            self.n2, self.n4, conn_type=pb.synapses.ConnType.All2All
        )
        self.s3 = pb.synapses.NoDecay(
            self.n3, self.n4, conn_type=pb.synapses.ConnType.All2All
        )


def output_without_shape(*args):
    return np.ones((2,), np.int8)


class _SubNet(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.inp = pb.InputProj(output_without_shape, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2, tick_wait_start=1)
        self.n2 = pb.neuron.TonicSpiking(2, fire_step=2, tick_wait_start=2)
        self.s0 = pb.synapses.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.One2One
        )
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.One2One
        )


class Nested_Net_Level1_2(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1 -> node1
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.node1 = _SubNet()
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.node1.n1, conn_type=pb.synapses.ConnType.One2One
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
def build_Nested_Net_Level1_1():
    return Nested_Net_Level1_1()


@pytest.fixture(scope="class")
def build_Nested_Net_Level1_2():
    return Nested_Net_Level1_2()


@pytest.fixture(scope="class")
def build_MoreInput_Net():
    return MoreInput_Net()
