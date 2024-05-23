import pytest

import paibox as pb


class Net_Test_Neuron_Behavior(pb.Network):
    def __init__(self, neuron):
        super().__init__()
        self.inp1 = pb.InputProj(None, shape_out=(1,))
        self.n1 = neuron
        self.s1 = pb.FullConn(self.inp1, self.n1)

        self.pb_inp_output = pb.Probe(self.inp1, "output")
        self.pb_s_output = pb.Probe(self.s1, "output")
        self.pb_n_spike = pb.Probe(self.n1, "spike")
        self.pb_n_volage = pb.Probe(self.n1, "voltage")
        self.pb_n_output = pb.Probe(self.n1, "output")


class Net2(pb.Network):
    """LIF neurons connected with more than one synapses.

    `sum_inputs()` will be called.
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(2, 2))
        self.n1 = pb.LIF((2, 2), 600, reset_v=1, leak_v=-1)
        self.s1 = pb.FullConn(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )
        self.s2 = pb.FullConn(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )
        self.s3 = pb.FullConn(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.inp1, "output")
        self.probe2 = pb.Probe(self.s1, "output")
        self.probe3 = pb.Probe(self.n1, "output")
        self.probe4 = pb.Probe(self.n1, "voltage")


class Net3(pb.Network):
    """2-layer networks, for testing start & end principle."""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(2, 2))
        self.n1 = pb.LIF((2, 2), 100, reset_v=1, leak_v=-1)
        self.n2 = pb.LIF((2, 2), 100, reset_v=1, leak_v=-1)
        self.s1 = pb.FullConn(
            self.inp1, self.n1, weights=10, conn_type=pb.SynConnType.All2All
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, weights=10, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.Probe(self.n1, "voltage", name="n1_v")
        self.probe2 = pb.Probe(self.n2, "voltage", name="n2_v")
        self.probe3 = pb.Probe(self.n1, "output", name="n1_out")
        self.probe4 = pb.Probe(self.n2, "output", name="n2_out")


@pytest.fixture(scope="class")
def build_Net2():
    return Net2()


@pytest.fixture(scope="class")
def build_Net3():
    return Net3()
