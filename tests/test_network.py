import numpy as np
import pytest

import paibox as pb
from paibox.base import DynamicSys, NeuDyn
from paibox.components import Projection


class TestNetwork_Components_Discover:
    def test_flatten_nodes(self, build_NotNested_Net):
        net = build_NotNested_Net

        # 1. Relative + include_self == True
        nodes1 = net.nodes(method="relative", level=1, include_self=True).exclude(
            pb.Probe
        )
        assert nodes1[""] == net
        assert len(nodes1) == 4

        # 2. Relative + include_self == False
        nodes2 = net.nodes(method="relative", level=1, include_self=False).exclude(
            pb.Probe
        )
        assert len(nodes2) == 3

        # 3. Absolute + include_self == True
        nodes3 = net.nodes(method="absolute", level=1, include_self=True).exclude(
            pb.Probe
        )
        assert len(nodes3) == 4

        # 4. Absolute + include_self == False
        nodes4 = net.nodes(method="absolute", level=1, include_self=False).exclude(
            pb.Probe
        )
        assert len(nodes4) == 3

    def test_nested_net_L1(self, build_Network_with_container):
        net = build_Network_with_container

        # 1. Relative + include_self == True
        nodes1 = net.nodes(method="relative", level=1, include_self=True).exclude(
            pb.Probe
        )
        assert nodes1[""] == net
        assert len(nodes1) == 7

        # 2. Relative + include_self == False
        nodes2 = net.nodes(method="relative", level=1, include_self=False).exclude(
            pb.Probe
        )
        assert len(nodes2) == 6

        # 3. Absolute + include_self == True
        nodes3 = net.nodes(method="absolute", level=1, include_self=True).exclude(
            pb.Probe
        )
        assert len(nodes3) == 7

        # 4. Absolute + include_self == False
        nodes4 = net.nodes(method="absolute", level=1, include_self=False).exclude(
            pb.Probe
        )
        assert len(nodes4) == 6

    def test_nested_net_L2(self, build_Nested_Net_L2):
        net: pb.Network = build_Nested_Net_L2

        nodes1 = net.nodes(level=1).subset(DynamicSys).unique()
        assert len(nodes1) == 5

        nodes_excluded = nodes1.not_subset(pb.DynSysGroup)
        assert len(nodes_excluded) == 5 - 2

        nodes2 = (
            net.nodes(level=2).subset(DynamicSys).unique().not_subset(pb.DynSysGroup)
        )
        assert len(nodes2) == 3 + 3 * 2

        nodes9 = (
            net.nodes(level=9).subset(DynamicSys).unique().not_subset(pb.DynSysGroup)
        )
        assert len(nodes9) == len(nodes2)

        nodes_all = net.nodes().subset(DynamicSys).unique()
        assert len(nodes_all) == 5 + 2 * 3

        assert len(net.components) == 3 + 3 * 2

        probes1 = net.nodes(level=1).subset(pb.Probe).unique()
        assert len(probes1) == 2

        probes_all = net.nodes().subset(pb.Probe).unique()
        assert len(probes_all) == 2 + 2 * 1

    def test_nested_net_L3(self, build_Nested_Net_L3):
        net: pb.Network = build_Nested_Net_L3

        nodes1 = net.nodes(level=1).subset(DynamicSys).unique()
        assert len(nodes1) == 3

        nodes2 = net.nodes(level=2).subset(DynamicSys).unique()
        assert len(nodes2) == 8

        nodes3 = (
            net.nodes(level=3).subset(DynamicSys).unique().not_subset(pb.DynSysGroup)
        )
        assert len(nodes3) == 2 + 1 * 3 + 1 * 2 * 3

        nodes_all = net.nodes().subset(DynamicSys).unique()
        assert len(nodes_all) == 3 + 1 * 5 + 1 * 2 * 3

        assert len(net.components) == 2 + 1 * 3 + 1 * 2 * 3

        probes1 = net.nodes(level=1).subset(pb.Probe).unique()
        assert len(probes1) == 3

        probes_all = net.nodes().subset(pb.Probe).unique()
        assert len(probes_all) == 3 + 1 * 2 + 1 * 2 * 1


class TestNetwork_Components_Oprations:
    def test_Collector_operations(self):
        s1 = DynamicSys(name="s1")
        s2 = pb.InputProj(1, shape_out=1, name="s2")
        s3 = NeuDyn(name="s3")
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

        assert len(g3_nodes.exclude(Projection)) == 1
        assert len(g1_nodes.not_subset(NeuDyn)) == 2
        assert len(g1_nodes.include(NeuDyn, Projection)) == 2

    def test_add_components(self, build_NotNested_Net_Exp):
        net: pb.Network = build_NotNested_Net_Exp
        n3 = pb.LIF((3,), 10)
        s1 = pb.FullConn(net.n1, n3, name="s1_new")
        s2 = pb.FullConn(net.n2, n3, name="s2_new")

        # Add extra components into the network after initialization
        setattr(net, n3.name, n3)  # key is 'LIF_0'
        # Or added by user
        # net.n3 = n3 # key is 'n3'
        net._add_components(s_insert=s1)
        nodes = net.nodes(level=1, include_self=False).subset(DynamicSys).unique()
        assert n3.name in nodes
        assert s1.name in nodes
        assert getattr(net, "s_insert", False)

        net._add_components(s2)
        assert getattr(net, s2.name, False)


@pytest.mark.skip(reason="'Sequential' is not used")
def test_Sequential_build():
    n1 = pb.TonicSpiking(10, fire_step=3)
    n2 = pb.TonicSpiking(10, fire_step=5)
    s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.All2All)
    sequential = pb.network.Sequential(n1, s1, n2)

    assert isinstance(sequential, pb.network.Sequential)

    nodes1 = sequential.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes1) == 3

    class Seq(pb.network.Sequential):
        def __init__(self):
            super().__init__()
            self.n1 = pb.TonicSpiking(5, fire_step=3)
            self.n2 = pb.TonicSpiking(5, fire_step=5)
            self.s1 = pb.FullConn(self.n1, self.n2, conn_type=pb.SynConnType.All2All)

    seq = Seq()
    nodes2 = seq.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes2) == 3


@pytest.mark.skip(reason="'Sequential' is not used")
def test_Sequential_getitem():
    n1 = pb.TonicSpiking(10, fire_step=3, name="n1")
    n2 = pb.TonicSpiking(10, fire_step=5, name="n2")
    s1 = pb.FullConn(n1, n2, conn_type=pb.SynConnType.All2All)
    n3 = pb.TonicSpiking(10, fire_step=5, name="n3")
    s2 = pb.FullConn(n2, n3, conn_type=pb.SynConnType.All2All)
    sequential = pb.network.Sequential(n1, s1, n2, s2, n3, name="Sequential_2")

    assert isinstance(sequential.children, pb.NodeDict)
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


class TestNetwork_Principles:
    def test_control_group(self, monkeypatch, build_Input_to_N1):
        net = build_Input_to_N1
        sim = pb.Simulator(net, start_time_zero=True)

        # _delay = 1, default
        # Spike at T - tws + 1 = pos * N
        sim.run(10)
        assert sim.data[net.probe3][7] == True
        sim.reset()

        monkeypatch.setattr(net.n1, "_tws", 3)
        sim.run(10)
        assert sim.data[net.probe3][8] == True
        sim.reset()

    def test_exp_group(self, build_NotNested_Net_Exp):
        net = build_NotNested_Net_Exp
        sim = pb.Simulator(net, start_time_zero=False)

        # N1 will spike at T = 3,5,7,9 with delay 3
        sim.run(10)

        # T = 10, (10, 1)
        assert np.array_equal(
            sim.data[net.probe2][-1][:10],
            np.array(
                [[0], [0], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.bool_
            ),
        )

        sim.reset()
