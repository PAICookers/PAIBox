import numpy as np
import pytest

import paibox as pb
from paibox.node import NodeDict


def test_Sequential_instance():
    n1 = pb.neuron.TonicSpikingNeuron(1, fire_step=3)
    n2 = pb.neuron.TonicSpikingNeuron(1, fire_step=5)
    sequential = pb.network.Sequential(n1, n2, name="Sequential_1_1")

    assert isinstance(sequential, pb.network.Sequential)


def test_Sequential_getitem():
    n1 = pb.neuron.TonicSpikingNeuron(10, fire_step=3, name="n1")
    n2 = pb.neuron.TonicSpikingNeuron(10, fire_step=5, name="n2")
    sequential = pb.network.Sequential(n1, n2, name="Sequential_1_2")

    assert isinstance(sequential.children, NodeDict)

    for str in ["n1", "n2"]:
        sequential[str]

    with pytest.raises(KeyError):
        sequential["n3"]

    for item in [0, 1]:
        sequential[item]

    seq = sequential[:1]

    assert seq != sequential

    seq = sequential[1:]
    seq = sequential[0:]
    seq = sequential[1:10]

    seq = sequential[["n1", "n2"]]

    sequential[1:2]  # legal


class Net1_User_Update(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.One2One())

    def update(self, x):
        y = self.n1.update(x)
        y = self.s1.update(y)
        y = self.n2.update(y)

        return y


class Net1_Default_Update(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.One2One())


class Net2_User_Update(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1
                -> n3
        n2 -> s2
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpikingNeuron(3, fire_step=2)
        self.n2 = pb.neuron.TonicSpikingNeuron(3, fire_step=2)
        self.n3 = pb.neuron.TonicSpikingNeuron(3, fire_step=2)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n3, pb.synapses.One2One())
        self.s2 = pb.synapses.NoDecay(self.n2, self.n3, pb.synapses.One2One())

    def update(self, x1, x2):
        y1 = self.n1.update(x1)
        y2 = self.n2.update(x2)
        y1_s1 = self.s1.update(y1)
        y2_s2 = self.s2.update(y2)
        y3 = self.n3.update(y1_s1 + y2_s2)

        return y3


class Net1(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.s1 = pb.synapses.NoDecay(self.n1, self.n2, pb.synapses.One2One())


class Net2(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1 -> node1
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpikingNeuron(2, fire_step=2)
        self.node1 = Net1()
        self.s1 = pb.synapses.NoDecay(self.n1, self.node1.n1, pb.synapses.One2One())


def test_DynSysGroup_nodes():
    net = Net1()

    all_nodes = net.nodes("absolute", level=1, include_self=False)
    neuron_nodes = list(all_nodes.subset(pb.neuron.Neuron).values())
    syn_nodes = list(all_nodes.subset(pb.synapses.Synapses).values())

    assert neuron_nodes == [net.n1, net.n2]
    assert syn_nodes == [net.s1]


def test_DynSysGroup_AutoUpdate_No_Nested():
    net = Net1()
    inp = pb.network.InputProj((2,), np.array([1, 1]), net.n1)

    x = np.ones((10, 2))
    expected_y_n1 = np.array(
        [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]]
    )
    expected_y_n2 = np.array(
        [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]
    )

    y = []
    for i in range(4):
        net.update()

        assert np.array_equal(net.n1.output, expected_y_n1[i])
        assert np.array_equal(net.n2.output, expected_y_n2[i])

    net.update()
    assert np.array_equal(net.n2.output, expected_y_n2[4])


@pytest.mark.parametrize("level", [1, 2], ids=["level_1", "level_2"])
def test_SynSysGroup_nodes_nested(level):
    net = Net2()
    all_nodes = net.nodes("absolute", level=level, include_self=False)

    for v in all_nodes.values():
        print(v)

    if level == 1:
        assert len(all_nodes.values()) == 3
    else:
        assert len(all_nodes.values()) == 6

    # neuron_nodes = list(all_nodes.subset(pb.neuron.Neuron).values())
    # syn_nodes = list(all_nodes.subset(pb.synapses.Synapses).values())

    # assert neuron_nodes == [net.n1, net.node1.n1, net.node1.n2]
    # assert syn_nodes == [net.s1, net.node1.s1]


def test_DynSysGroup_update():
    """
    Structure 1:
        A sequential network.
        n1 -> s1 -> n2

    Use the default `update()` function.
    
    FIXME ERROR!
    """

    def sequential_structure_user_update():
        # 10(ts) * 2(width)
        x = np.ones((10, 2))
        _y = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

        net = Net1_User_Update()

        for i in range(10):
            y = net.update(x[i])
            assert np.array_equal(y, np.ones((2,)) * _y[i])

    def sequential_structure_default_update():
        x = np.ones((10, 2))
        _y = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

        net = Net1_Default_Update()

    sequential_structure_user_update()
    sequential_structure_default_update()

    def general_structure_user_update():
        x1 = np.ones((12, 3))
        x2 = np.ones((12, 3))
        y1_s1 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        y2_s2 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        y3 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])

        net = Net2_User_Update()

        for i in range(12):
            y = net.update(x1[i], x2[i])
            assert np.array_equal(y, np.ones((3,)) * y3[i])

    general_structure_user_update()


def test_InputProj_func() -> None:

    inp = pb.network.InputProj(pb.processes.UniformGen((5,)))
    
    sim = pb.Simulator(inp)
    p1 = pb.simulator.Probe(inp, "state")
    
    sim.add_probe(p1)
    sim.run(10)

    assert sim.data[p1].shape == (10, 5)
    
    inp2 = pb.network.InputProj(pb.processes.Constant((3,), 1))
    
    sim2 = pb.Simulator(inp2)
    p2 = pb.simulator.Probe(inp2, "state")
    
    sim2.add_probe(p2)
    sim2.run(10)

def test_InputProj_user_func():  
    # 1. Define a process
    class MyProcess(pb.base.Process):
        def __init__(self, shape_out) -> None:
            super().__init__(shape_out)

        def update(self, tick, bias) -> None:
            self.output = np.ones(self.shape_out) * tick + bias
            
    # 2. Define a input projection
    my_inp = pb.network.InputProj(MyProcess((10, 10)))
    
    # 3. Simulate this input projection
    my_sim = pb.Simulator(my_inp)
    my_probe = pb.simulator.Probe(my_inp, "state")
    my_sim.add_probe(my_probe)
    my_sim.run(10, bias=1)