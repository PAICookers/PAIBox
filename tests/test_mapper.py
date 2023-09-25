import pytest

import paibox as pb


def test_grouping():
    class Net1(pb.Network):
        def __init__(self):
            super().__init__()
            self.inp = pb.projection.InputProj(pb.simulator.processes.Constant(200, 1))
            self.n1 = pb.neuron.TonicSpiking(200, 3)
            self.n2 = pb.neuron.TonicSpiking(400, 3)
            self.n3 = pb.neuron.TonicSpiking(400, 4)

            self.s1 = pb.synapses.NoDecay(self.inp, self.n1, pb.synapses.All2All())
            self.s2 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.All2All())
            self.s3 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.One2One())

    class Net2(pb.Network):
        def __init__(self):
            super().__init__()
            self.inp = pb.projection.InputProj(pb.simulator.processes.Constant(1200, 1))
            self.n1 = pb.neuron.TonicSpiking(400, 3)
            self.n2 = pb.neuron.TonicSpiking(800, 3)

            self.s1 = pb.synapses.NoDecay(self.inp, self.n1, pb.synapses.All2All())
            self.s2 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.All2All())

    # net1 = Net1()

    mapper = pb.Mapper()
    # mapper.build_graph(net1)
    # mapper.do_grouping()

    # print("OK1")

    net2 = Net2()
    mapper.clear()
    mapper.build_graph(net2)
    mapper.do_grouping()

    print("OK2")
