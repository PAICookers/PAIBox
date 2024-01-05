import pytest
import numpy as np

import paibox as pb
from paibox.node import NodeDict


class TestNetworkNodes:
    def test_Collector_operations(self):
        s1 = pb.base.DynamicSys(name="s1")
        s2 = pb.InputProj(1, shape_out=1, name="s2")
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

    def test_flatten_hzynet(self, build_MoreInput_Net):
        net = build_MoreInput_Net

        nodes1 = net.nodes(method="relative", level=1, include_self=True)
        assert nodes1[""] == net
        assert len(nodes1) == 8

        # 2. Relative + include_self == False
        nodes2 = net.nodes(method="relative", level=1, include_self=False)
        assert len(nodes2) == 7

        # 3. Absolute + include_self == True
        nodes3 = net.nodes(method="absolute", level=1, include_self=True)
        assert len(nodes3) == 8

        # 4. Absolute + include_self == False
        nodes4 = net.nodes(method="absolute", level=1, include_self=False)
        assert len(nodes4) == 7

    def test_Network_flatten_nodes(self, build_NotNested_Net):
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

    def test_Network_nested_level_1(self, build_Nested_Net_Level1_1):
        net = build_Nested_Net_Level1_1

        # 1. Relative + include_self == True
        nodes1 = net.nodes(method="relative", level=1, include_self=True)
        assert nodes1[""] == net
        # for k,v in nodes1.items():
        #     print(k,v)
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
        assert len(nodes5) == 11
        # 6. Find nodes from level 1 to level 2, absolutely
        nodes6 = net.nodes(method="absolute", level=2, include_self=False)
        assert len(nodes6) == 9


@pytest.mark.skip(reason="'Sequential is not used'")
def test_Sequential_build():
    n1 = pb.neuron.TonicSpiking(10, fire_step=3)
    n2 = pb.neuron.TonicSpiking(10, fire_step=5)
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.synapses.ConnType.All2All)
    sequential = pb.network.Sequential(n1, s1, n2)

    assert isinstance(sequential, pb.network.Sequential)

    nodes1 = sequential.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes1) == 3

    class Seq(pb.network.Sequential):
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


@pytest.mark.skip(reason="'Sequential is not used'")
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


@pytest.mark.parametrize("level", [1, 2], ids=["level_1", "level_2"])
def test_SynSysGroup_nodes_nested(level, build_Nested_Net_Level1_2):
    net = build_Nested_Net_Level1_2
    all_nodes = net.nodes("absolute", level=level, include_self=False)

    for v in all_nodes.values():
        print(v)


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
