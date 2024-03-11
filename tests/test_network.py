from typing import Optional

import numpy as np
import pytest

import paibox as pb
from paibox.base import DynamicSys, NeuDyn
from paibox.exceptions import PAIBoxWarning
from paibox.node import NodeDict


class Nested_Net_level_1(pb.DynSysGroup):
    """Level 1 nested network: pre_n -> syn -> post_n"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

        self.pre_n = pb.LIF((10,), 10)
        self.post_n = pb.LIF((10,), 10)

        w = np.random.randint(-128, 127, (10, 10), dtype=np.int8)
        self.syn = pb.NoDecay(
            self.pre_n, self.post_n, conn_type=pb.SynConnType.All2All, weights=w
        )


class TestNetwork_Components_Discover:
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

    def test_nested_net_level_1(self, build_Network_with_container):
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

        # Simulation
        sim = pb.Simulator(net)
        sim.run(10)
        sim.reset()

    def test_nested_net_level_2(self):
        class Nested_Net_level_2(pb.DynSysGroup):
            """Level 2 nested network: inp1 -> s1 -> Nested_Net_level_1 -> s2 -> Nested_Net_level_1"""

            def __init__(self):
                self.inp1 = pb.InputProj(1, shape_out=(10,))
                subnet1 = Nested_Net_level_1()
                subnet2 = Nested_Net_level_1(name="Named_SubNet")
                self.s1 = pb.NoDecay(
                    self.inp1,
                    subnet1.pre_n,
                    conn_type=pb.SynConnType.One2One,
                )
                self.s2 = pb.NoDecay(
                    subnet1.post_n,
                    subnet2.pre_n,
                    conn_type=pb.SynConnType.One2One,
                )

                super().__init__(subnet1, subnet2)

        net = Nested_Net_level_2()
        nodes = net.nodes(level=1, include_self=False).subset(DynamicSys).unique()
        nodes_excluded = (
            net.nodes(level=1, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )
        nodes2 = (
            net.nodes(level=2, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )
        nodes9 = (
            net.nodes(level=9, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )

        assert isinstance(net[f"{Nested_Net_level_1.__name__}_0"], pb.Network)
        assert isinstance(net["Named_SubNet"], pb.Network)

        assert len(nodes) == 5
        assert len(nodes_excluded) == 3
        assert len(nodes2) == 3 + 3 * 2
        assert len(nodes9) == len(nodes2)

    def test_nested_net_level_3(self):
        class Nested_Net_level_2(pb.DynSysGroup):
            """Level 2 nested network: -> s1 -> Nested_Net_level_1"""

            def __init__(self, n: pb.neuron.Neuron):
                subnet = Nested_Net_level_1()
                self.s1 = pb.NoDecay(
                    n,
                    subnet.pre_n,
                    conn_type=pb.SynConnType.One2One,
                )

                super().__init__(subnet)

        class Nested_Net_level_3(pb.DynSysGroup):
            """Level 3 nested network: inp1 -> s1 -> n1 -> Nested_Net_level_2 -> s1 -> Nested_Net_level_1"""

            def __init__(self):
                self.inp1 = pb.InputProj(1, shape_out=(10,))
                self.n1 = pb.LIF((10,), 10)

                net_level2 = Nested_Net_level_2(self.n1)
                self.s1 = pb.NoDecay(
                    self.inp1,
                    self.n1,
                    conn_type=pb.SynConnType.One2One,
                )

                super().__init__(net_level2)

        net = Nested_Net_level_3()
        nodes_excluded = (
            net.nodes(level=1, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )
        nodes2 = (
            net.nodes(level=2, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )
        nodes3 = (
            net.nodes(level=3, include_self=False)
            .subset(DynamicSys)
            .unique()
            .not_subset(pb.DynSysGroup)
        )

        assert len(nodes_excluded) == 3
        assert len(nodes2) == 3 + 1
        assert len(nodes3) == 3 + 1 + 3


class TestNetwork_Components_Oprations:
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

    def test_add_components(self, build_NotNested_Net_Exp):
        net: pb.Network = build_NotNested_Net_Exp
        n3 = pb.LIF((3,), 10)
        s1 = pb.synapses.NoDecay(net.n1, n3, conn_type=pb.SynConnType.All2All)
        s2 = pb.synapses.NoDecay(net.n2, n3, conn_type=pb.SynConnType.All2All)

        with pytest.raises(ValueError):
            net.diconnect_neudyn_succ(n3)

        # Add extra components into the network after initialization
        setattr(net, n3.name, n3)  # key is 'LIF_0'
        # Or added by user
        # net.n3 = n3 # key is 'n3'
        net.add_components(s_insert=s1)
        nodes = net.nodes(level=1, include_self=False).subset(DynamicSys).unique()
        assert n3.name in nodes
        assert s1.name in nodes
        assert getattr(net, "s_insert", False)

        net.add_components(s2)
        assert getattr(net, s2.name, False)

    def test_disconnect_neudyn_from(self, build_Network_with_container):
        net: pb.Network = build_Network_with_container

        # Disconnet the n_list[0] -> s1 -> n_list[1]
        # Nothing to disconnect so a warning is raised
        with pytest.warns(PAIBoxWarning):
            removed = net.disconnect_neudyn_from(
                net.n_list[0], net.n_list[2], remove=False
            )
            assert removed == []

        nodes = net.nodes(level=1, include_self=False).subset(DynamicSys).unique()
        assert net.n_list[0].name in nodes

        # Remove the target synapse
        removed = net.disconnect_neudyn_from(net.n_list[0], net.n_list[1], remove=True)
        assert len(removed) == 1
        assert not getattr(net, "s1", False)

    def test_disconnect_neudyn_succ(self, build_multi_inodes_onodes):
        net: pb.Network = build_multi_inodes_onodes

        removed = net.diconnect_neudyn_succ(net.n1, remove=True)

        assert len(removed) == 2
        assert not getattr(net, "s2", False)
        assert not getattr(net, "s4", False)
        assert getattr(net, "s1", False)
        assert getattr(net, "s3", False)

    def test_disconnect_neudyn_pred(self, build_multi_inodes_onodes):
        net: pb.Network = build_multi_inodes_onodes

        removed = net.diconnect_neudyn_pred(net.n1, remove=True)

        assert len(removed) == 2
        assert not getattr(net, "s1", False)
        assert not getattr(net, "s3", False)
        assert getattr(net, "s2", False)
        assert getattr(net, "s4", False)

    def test_insert_neudyn(self, build_Network_with_container):
        net: pb.Network = build_Network_with_container

        # Insert n3 between n_list[0] & n_list[1]
        n_insert = pb.LIF((3,), 10)
        s_insert1 = pb.synapses.NoDecay(
            net.n_list[0], n_insert, conn_type=pb.SynConnType.All2All
        )
        s_insert2 = pb.synapses.NoDecay(
            n_insert, net.n_list[1], conn_type=pb.SynConnType.All2All
        )

        # Replace s1 with s_insert1->n_insert->s_insert2
        net.insert_neudyn(
            net.n_list[0],
            net.n_list[1],
            (n_insert, s_insert1, s_insert2),
            replace=True,
            remove=False,
        )

        nodes = net.nodes(level=1, include_self=False).subset(DynamicSys).unique()
        assert n_insert.name in nodes
        assert s_insert1.name in nodes
        assert s_insert2.name in nodes
        assert getattr(net, f"{s_insert1.name}", False)

        assert getattr(net, "s1", False)  # s1 is still in the network
        assert net.s1.name in nodes

        assert getattr(net, f"{s_insert2.name}", False) in list(
            net.n_list[1].master_nodes.values()
        )

    @pytest.mark.skip(reason="Not implemented")
    def test_Subnets(self, build_Network_with_subnet):
        net = build_Network_with_subnet

        # 1. Relative + include_self == True, level 1
        nodes1 = (
            net.nodes(method="absolute", level=1, include_self=False)
            .subset(DynamicSys)
            .unique()
        )
        nodes1_sub = nodes1.subset(NeuDyn)

        # 2. Relative + include_self == True, level 2
        nodes2 = (
            net.nodes(method="absolute", level=2, include_self=False)
            .subset(DynamicSys)
            .unique()
        )

        nodes3 = (
            net.nodes(method="absolute", level=7, include_self=False)
            .subset(DynamicSys)
            .unique()
        )

        print()


@pytest.mark.skip(reason="'Sequential is not used'")
def test_Sequential_build():
    n1 = pb.neuron.TonicSpiking(10, fire_step=3)
    n2 = pb.neuron.TonicSpiking(10, fire_step=5)
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.SynConnType.All2All)
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
                self.n1, self.n2, conn_type=pb.SynConnType.All2All
            )

    seq = Seq()
    nodes2 = seq.nodes(method="absolute", level=1, include_self=False)
    assert len(nodes2) == 3


@pytest.mark.skip(reason="'Sequential is not used'")
def test_Sequential_getitem():
    n1 = pb.neuron.TonicSpiking(10, fire_step=3, name="n1")
    n2 = pb.neuron.TonicSpiking(10, fire_step=5, name="n2")
    s1 = pb.synapses.NoDecay(n1, n2, conn_type=pb.SynConnType.All2All)
    n3 = pb.neuron.TonicSpiking(10, fire_step=5, name="n3")
    s2 = pb.synapses.NoDecay(n2, n3, conn_type=pb.SynConnType.All2All)
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
