import pytest

import paibox as pb


def test_GroupedSyn_build():
    """Check until the building of `GroupedSyn`."""

    class Net1(pb.Network):
        """N1 -> S1 -> N3
        N2 -> S2 -> N3
        """

        def __init__(self):
            super().__init__()
            self.n1 = pb.neuron.TonicSpiking(400, 3)
            self.n2 = pb.neuron.TonicSpiking(400, 3)
            self.n3 = pb.neuron.TonicSpiking(400, 4)

            self.s1 = pb.synapses.NoDecay(self.n1, self.n3, pb.synapses.All2All())
            self.s2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.All2All())

    class Net2(pb.Network):
        """INP1 -> S1 -> N1 -> S2 -> N2"""

        def __init__(self):
            super().__init__()
            self.inp = pb.projection.InputProj(pb.simulator.processes.Constant(1200, 1))
            self.n1 = pb.neuron.TonicSpiking(400, 3)
            self.n2 = pb.neuron.TonicSpiking(800, 3)

            self.s1 = pb.synapses.NoDecay(self.inp, self.n1, pb.synapses.All2All())
            self.s2 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.All2All())

    class Net3(pb.Network):
        """N1 -> S1 -> N3
        N2 -> S2 -> N3
        """

        def __init__(self):
            super().__init__()
            self.n1 = pb.neuron.TonicSpiking(400, 3)
            self.n2 = pb.neuron.TonicSpiking(1200, 3)
            self.n3 = pb.neuron.TonicSpiking(400, 4)

            self.s1 = pb.synapses.NoDecay(self.n1, self.n3, pb.synapses.One2One())
            self.s2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.All2All())

    class Net4(pb.Network):
        """N1 -> S1 -> N2
        N1 -> S2 -> N3
        N4 -> S3 -> N3
        N4 -> S4 -> N5
        """

        def __init__(self):
            super().__init__()
            self.inp1 = pb.projection.InputProj(pb.simulator.processes.Constant(400, 1))
            self.n2 = pb.neuron.TonicSpiking(1200, 3)
            self.n3 = pb.neuron.TonicSpiking(400, 4)
            self.n4 = pb.neuron.TonicSpiking(400, 4)
            self.n5 = pb.neuron.TonicSpiking(600, 4)

            self.s1 = pb.synapses.NoDecay(self.inp1, self.n2, pb.synapses.All2All())
            self.s2 = pb.synapses.NoDecay(self.inp1, self.n3, pb.synapses.One2One())
            self.s3 = pb.synapses.NoDecay(self.n4, self.n3, pb.synapses.All2All())
            self.s4 = pb.synapses.NoDecay(self.n4, self.n5, pb.synapses.All2All())

    # net1 = Net1()
    # net2 = Net2()
    # net3 = Net3()
    net4 = Net4()

    mapper = pb.Mapper()
    mapper.clear()
    mapper.build_graph(net4)
    mapper.do_grouping()

    print("OK")
