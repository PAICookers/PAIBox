from re import L

import numpy as np
import pytest

import paibox as pb
from paibox.node import NodeDict


def test_Collector_operations():
    s1 = pb.base.DynamicSys(name="s1")
    s2 = pb.projection.InputProj(shape_out=1, name="s2")
    s3 = pb.network.NeuDyn(name="s3")
    s4 = pb.DynSysGroup(s1, s2, name="s4")

    g1 = pb.DynSysGroup(s1, s2, s3, name="g1")
    g2 = pb.DynSysGroup(s1, s2, s4, name="g2")
    g3 = pb.DynSysGroup(s1, s2, name="g3")
    g4 = pb.DynSysGroup(s1, s4, name="g4")

    g1_nodes = g1.nodes(method="relative", level=1, include_self=False)
    g2_nodes = g2.nodes(method="relative", level=1, include_self=False)
    g3_nodes = g3.nodes(method="relative", level=1, include_self=False)
    g4_nodes = g4.nodes(method="relative", level=1, include_self=False)

    g_nodes_sum = g1_nodes + g2_nodes
    assert len(g_nodes_sum) == 4

    with pytest.raises(ValueError):
        g_nodes_sub = g1_nodes - g2_nodes

    g_nodes_sub = g1_nodes - g3_nodes

    assert len(g_nodes_sub) == 1

    assert len(g4_nodes.unique()) == 2

    assert len(g3_nodes.exclude(pb.projection.Projection)) == 1
    assert len(g1_nodes.not_subset(pb.network.NeuDyn)) == 2
    assert len(g1_nodes.include(pb.network.NeuDyn, pb.projection.Projection)) == 2


class Net(pb.DynSysGroup):
    """Not nested network
    n1->s1->n2, n3
    """

    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, 3)
        self.n2 = pb.neuron.TonicSpiking(2, 3)
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All
        )
        self.n3 = pb.neuron.TonicSpiking(2, 4)


class Nested_Net_Level_1(pb.DynSysGroup):
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


def test_DynSysGroup_flatten_nodes():
    net = Net()

    # 1. Relative + include_self == True
    nodes1 = net.nodes(method="relative", level=1, include_self=True)
    assert nodes1[""] == net
    assert len(nodes1) == 5

    # 2. Relative + include_self == False
    nodes2 = net.nodes(method="relative", level=1, include_self=False)
    assert len(nodes2) == 4

    # 3. Absolute + include_self == True
    nodes3 = net.nodes(method="absolute", level=1, include_self=True)
    assert len(nodes3) == 5

    # 4. Absolute + include_self == False
    nodes4 = net.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes4) == 4


def test_DynSysGroup_nodes_nested_level1():
    net = Nested_Net_Level_1()

    # 1. Relative + include_self == True
    nodes1 = net.nodes(method="relative", level=1, include_self=True)
    assert nodes1[""] == net
    assert len(nodes1) == 4

    # 2. Relative + include_self == False
    nodes2 = net.nodes(method="relative", level=1, include_self=False)
    assert len(nodes2) == 3

    # 3. Absolute + include_self == True
    nodes3 = net.nodes(method="absolute", level=1, include_self=True)
    assert len(nodes3) == 4

    # 4. Absolute + include_self == False
    nodes4 = net.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes4) == 3

    # 5. Find nodes from level 1 to level 2, relatively
    nodes5 = net.nodes(method="relative", level=2, include_self=False)

    # 6. Find nodes from level 1 to level 2, absolutely
    nodes6 = net.nodes(method="absolute", level=2, include_self=False)
    assert len(nodes6) == 9


def test_Sequential_build():
    n1 = pb.neuron.TonicSpiking(10, fire_step=3)
    n2 = pb.neuron.TonicSpiking(10, fire_step=5)
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.All2All)
    sequential = pb.network.Sequential(n1, s1, n2)

    assert isinstance(sequential, pb.network.Sequential)

    nodes1 = sequential.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes1) == 3

    class Seq(pb.Sequential):
        def __init__(self):
            super().__init__()
            self.n1 = pb.neuron.TonicSpiking(5, fire_step=3)
            self.n2 = pb.neuron.TonicSpiking(5, fire_step=5)
            self.s1 = pb.synapses.NoDecay(
                self.n1, self.n2, conn_type=pb.synapses.ConnType.All2All
            )

    seq = Seq()
    nodes2 = seq.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes2) == 3


def test_Sequential_getitem():
    n1 = pb.neuron.TonicSpiking(10, fire_step=3, name="n1")
    n2 = pb.neuron.TonicSpiking(10, fire_step=5, name="n2")
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.All2All)
    n3 = pb.neuron.TonicSpiking(10, fire_step=5, name="n3")
    s2 = pb.synapses.NoDecay(n2, n3, conn_type=pb.synapses.ConnType.All2All)
    sequential = pb.network.Sequential(n1, s1, n2, s2, n3, name="Sequential_2")

    assert isinstance(sequential.children, NodeDict)
    assert len(sequential) == 5

    # str
    for str in ["n1", "n2"]:
        sequential[str]

    with pytest.raises(KeyError):
        sequential["n4"]

    for item in [0, 1]:
        sequential[item]

    # Out of index
    with pytest.raises(IndexError):
        sequential[5]

    # Slice
    seq = sequential[:1]

    assert seq != sequential

    seq = sequential[1:]
    seq = sequential[0:]
    seq = sequential[1:10]
    sequential[1:2]


class Net1_User_Update(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.One2One
        )

    def update(self, x):
        y = self.n1.update(x)
        y = self.s1.update(y)
        y = self.n2.update(y)

        return y


class Net1_Default_Update(pb.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.One2One
        )


class Net2_User_Update(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1
                -> n3
        n2 -> s2
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.n3 = pb.neuron.TonicSpiking(3, fire_step=2)
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n3, conn_type=pb.synapses.ConnType.One2One
        )
        self.s2 = pb.synapses.NoDecay(
            self.n2, self.n3, conn_type=pb.synapses.ConnType.One2One
        )

    def update(self, x1, x2):
        y1 = self.n1.update(x1)
        y2 = self.n2.update(x2)
        y1_s1 = self.s1.update(y1)
        y2_s2 = self.s2.update(y2)
        y3 = self.n3.update(y1_s1 + y2_s2)

        return y3


def output_without_shape(*args):
    return np.ones((2,), np.int8)


class Net1(pb.DynSysGroup):
    def __init__(self):
        super().__init__()

        self.inp = pb.InputProj(output_without_shape, shape_out=(2,))
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.n2 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.s0 = pb.synapses.NoDecay(
            self.inp, self.n1, conn_type=pb.synapses.ConnType.One2One
        )
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.n2, conn_type=pb.synapses.ConnType.One2One
        )


class Net2(pb.DynSysGroup):
    def __init__(self):
        """
        n1 -> s1 -> node1
        """
        super().__init__()
        self.n1 = pb.neuron.TonicSpiking(2, fire_step=2)
        self.node1 = Net1()
        self.s1 = pb.synapses.NoDecay(
            self.n1, self.node1.n1, conn_type=pb.synapses.ConnType.One2One
        )


def test_DynSysGroup_AutoUpdate_No_Nested():
    net = Net1()

    expected_y_n1 = np.array(
        [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]]
    )
    expected_y_n2 = np.array(
        [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0]]
    )

    sim = pb.Simulator(net)
    p1 = pb.simulator.Probe(net.n1, "output")
    p2 = pb.simulator.Probe(net.n2, "output")

    sim.add_probe(p1)
    sim.add_probe(p2)
    sim.run(10)

    assert np.array_equal(sim.data[p1], expected_y_n1)
    assert np.array_equal(sim.data[p2], expected_y_n2)


@pytest.mark.parametrize("level", [1, 2], ids=["level_1", "level_2"])
def test_SynSysGroup_nodes_nested(level):
    net = Net2()
    all_nodes = net.nodes("absolute", level=level, include_self=False)

    for v in all_nodes.values():
        print(v)


@pytest.mark.xfail
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
