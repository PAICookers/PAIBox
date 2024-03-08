from pathlib import Path

import numpy as np
import pytest

import paibox as pb


@pytest.fixture(scope="module")
def ensure_dump_dir():
    p = Path(__file__).parent / "debug"

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink()

    yield p


def fakeout(t):
    data = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        np.bool_,
    )

    return data[t]


class Net1(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(fakeout, shape_out=(2,))
        self.n1 = pb.neuron.IF((2,), 3)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.SynConnType.One2One
        )

        self.probe1 = pb.simulator.Probe(self.inp1, "output")
        self.probe2 = pb.simulator.Probe(self.s1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "output")
        self.probe4 = pb.simulator.Probe(self.n1, "voltage")


class Net2(pb.Network):
    """LIF neurons connected with more than one synapses.

    `sum_inputs()` will be called.
    """

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(2, 2))
        self.n1 = pb.neuron.LIF((2, 2), 600, reset_v=1, leak_v=-1)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )
        self.s2 = pb.synapses.NoDecay(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )
        self.s3 = pb.synapses.NoDecay(
            self.inp1, self.n1, weights=127, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.simulator.Probe(self.inp1, "output")
        self.probe2 = pb.simulator.Probe(self.s1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "output")
        self.probe4 = pb.simulator.Probe(self.n1, "voltage")


class Net3(pb.Network):
    """2-layer networks, for testing start & end principle."""

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(1, shape_out=(2, 2))
        self.n1 = pb.neuron.LIF((2, 2), 100, reset_v=1, leak_v=-1)
        self.n2 = pb.neuron.LIF((2, 2), 100, reset_v=1, leak_v=-1)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, weights=10, conn_type=pb.SynConnType.All2All
        )
        self.s2 = pb.synapses.NoDecay(
            self.n1, self.n2, weights=10, conn_type=pb.SynConnType.All2All
        )

        self.probe1 = pb.simulator.Probe(self.n1, "voltage", name="n1_v")
        self.probe2 = pb.simulator.Probe(self.n2, "voltage", name="n2_v")
        self.probe3 = pb.simulator.Probe(self.n1, "output", name="n1_out")
        self.probe4 = pb.simulator.Probe(self.n2, "output", name="n2_out")


class TonicSpikingNet(pb.Network):
    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(fakeout, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking((2,), 3)
        self.s1 = pb.synapses.NoDecay(
            self.inp1, self.n1, conn_type=pb.SynConnType.One2One
        )

        self.probe1 = pb.simulator.Probe(self.s1, "output")
        self.probe2 = pb.simulator.Probe(self.n1, "output")
        self.probe3 = pb.simulator.Probe(self.n1, "voltage")


@pytest.fixture(scope="class")
def build_Net1():
    return Net1()


@pytest.fixture(scope="class")
def build_Net2():
    return Net2()


@pytest.fixture(scope="class")
def build_Net3():
    return Net3()


@pytest.fixture(scope="class")
def build_TonicSpikingNet():
    return TonicSpikingNet()
