from math import ceil

import numpy as np
import pytest
from paicorelib import ONLINE_CORES_BASE_COORD, Coord, HwConfig
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox.backend._slice import node_sl_lst_overlap
from paibox.backend.conf_exporting import *
from paibox.backend.mapper import merge_cycles
from paibox.exceptions import ResourceError

from .conftest import TestData


class TestGraphInfo:
    def test_multi_inputproj1(self, ensure_dump_dir, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

        assert len(mapper.graph_info["input"]) == 2

    def test_multi_inputproj2(self, ensure_dump_dir, build_multi_inputproj_net2):
        net = build_multi_inputproj_net2
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

        assert len(mapper.graph_info["input"]) == 2

    def test_multi_inputproj3(
        self, monkeypatch, ensure_dump_dir, build_multi_inputproj_net3
    ):
        net = build_multi_inputproj_net3
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

        assert len(mapper.graph_info["input"]) == 1
        assert len(mapper.core_blocks) == 6

        monkeypatch.setattr(net.n7, "_tws", 3)  # n7.tws: 2 -> 3
        mapper.clear()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.core_blocks) == 5  # n6 & n7 grouped in one core block.

    def test_multi_onodes(self, ensure_dump_dir, build_multi_onodes_net):
        net = build_multi_onodes_net
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"]) == 2

        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

    def test_multi_onodes2(self, ensure_dump_dir, build_multi_onodes_net2):
        net = build_multi_onodes_net2
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"]) == 2

        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

    def test_multi_onodes_more1152(
        self, ensure_dump_dir, build_multi_onodes_net_more1152
    ):
        net = build_multi_onodes_net_more1152

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"]) == 2

        mapper.export(fp=ensure_dump_dir, split_by_chip=True)

    def test_multi_inodes_onodes(self, build_multi_inodes_onodes):
        net = build_multi_inodes_onodes
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["input"]) == 2
        assert len(mapper.graph_info["output"]) == 2

    def test_nested_net_L2_compile(self, build_Nested_Net_level_2):
        net = build_Nested_Net_level_2
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph.nodes.keys()) == 5
        assert len(mapper.graph_info["input"]) == 1
        assert len(mapper.graph_info["output"]) == 1

    def test_nested_net_L3_compile(self, build_Nested_Net_level_3):
        net = build_Nested_Net_level_3
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph.edges.keys()) == 5
        assert len(mapper.graph_info["input"]) == 2
        assert len(mapper.graph_info["output"]) == 1

    def test_ANN_network_compile(self, build_ANN_Network_1, ensure_dump_dir):
        net = build_ANN_Network_1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

        assert 1


class TestMapperDeployment:
    def test_build_graph(self, build_example_net1, build_example_net2):
        """Build more than one networks."""
        net1 = build_example_net1
        net2 = build_example_net2

        mapper = pb.Mapper()
        mapper.build(net1, net2)
        mapper.compile()

        assert len(mapper.graph.nodes.keys()) == 8
        assert len(mapper.graph_info["input"]) == 3
        assert len(mapper.graph_info["output"]) == 2

    @pytest.fixture
    def compile_simple_net(self, build_example_net1):
        """Reused fixture."""
        net = build_example_net1

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        return mapper

    def test_export_config_json(self, ensure_dump_dir, compile_simple_net):
        """Export all the configs into json"""
        mapper = compile_simple_net
        assert mapper.graph.has_built == True

        assert len(mapper.core_blocks) == 3  # 3 layers
        assert mapper.graph_info["inherent_timestep"] == 3

        mapper.export(fp=ensure_dump_dir, split_by_chip=False)

    def test_find_neuron(self, compile_simple_net):
        mapper: pb.Mapper = compile_simple_net
        assert mapper.graph.has_built == True

        mapper.find_neuron(mapper.graph._raw_networks[0].n3)

        assert 1

    def test_find_axon(self, compile_simple_net):
        mapper = compile_simple_net
        assert mapper.graph.has_built == True

        mapper.find_axon(mapper.graph._raw_networks[0].n2)

        assert 1

    def test_network_with_container(self, build_Network_with_container):
        net: pb.Network = build_Network_with_container
        mapper = pb.Mapper()
        mapper.build(net)
        assert len(mapper.graph.nodes) == 4

        mapper.compile()

        # The input node is isolated
        assert len(mapper.graph_info["input"]) == 0
        assert len(mapper.graph_info["output"]) == 1

    def test_network_axons_out_of_range(self):
        class Net(pb.Network):
            def __init__(self):
                super().__init__()
                self.inp = pb.InputProj(1, shape_out=(300, 300))
                self.n1 = pb.IF((300, 300), 1, name="n1")
                self.n2 = pb.IF((300,), 1, name="n2")

                self.s1 = pb.FullConn(
                    self.inp, self.n1, conn_type=pb.SynConnType.All2All
                )
                self.s2 = pb.FullConn(
                    self.n1, self.n2, conn_type=pb.SynConnType.All2All
                )

        net = Net()
        mapper = pb.Mapper()
        mapper.build(net)

        with pytest.raises(ResourceError):
            mapper.compile()  # 300*300 > 1152*64

    @pytest.mark.parametrize(
        "n_networks", [16, 50, 400, 512, 513, 600, 900, 1008, 1009, 1023]
    )
    def test_multi_networks(self, n_networks, monkeypatch, ensure_dump_dir):
        class Net(pb.Network):
            # This network will be placed in 1 core only.
            def __init__(self):
                super().__init__()
                self.inp = pb.InputProj(1, shape_out=(3,))
                self.n1 = pb.IF((3,), 10)
                self.s1 = pb.FullConn(self.inp, self.n1)

        nets = [Net() for _ in range(n_networks)]

        if n_networks > 1008:
            clist = [Coord(0, 0), Coord(0, 1)]
            monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)

        mapper = pb.Mapper()
        mapper.build(*nets)
        graph_info = mapper.compile(use_exp_features=True)

        assert graph_info["n_core_occupied"] == n_networks

        rtotal = sum(mapper.routing_mgr.n_core_per_chip)
        r1 = mapper.routing_mgr.n_core_per_chip[0]

        if n_networks > 1008:
            r2 = mapper.routing_mgr.n_core_per_chip[1]
            assert rtotal == r1 + r2
            assert r1 == 1024
            assert r2 == n_networks - 1008
        elif n_networks > ONLINE_CORES_BASE_COORD:
            assert rtotal == r1 == n_networks + 16
        else:
            assert rtotal == r1 == n_networks

        mapper.export(fp=ensure_dump_dir)

    @pytest.mark.parametrize("layer", [63, 64])
    def test_bypass_linear(self, layer, ensure_dump_dir, monkeypatch):
        class Net(pb.Network):
            def __init__(self, layer: int):
                super().__init__()
                weight1 = np.eye(1000, dtype=np.int8) * 127
                self.i1 = pb.InputProj(input=1, shape_out=(weight1.shape[0],))
                self.n_list = pb.NodeList()
                self.s_list = pb.NodeList()
                self.p_list = pb.NodeList()

                for i in range(layer):
                    self.n_list.append(
                        pb.IF(
                            weight1.shape[1],
                            threshold=127,
                            tick_wait_start=i + 1,
                        )
                    )
                    if i == 0:
                        self.s_list.append(
                            pb.FullConn(self.i1, self.n_list[i], weights=weight1)
                        )
                    else:
                        self.s_list.append(
                            pb.FullConn(
                                self.n_list[i - 1], self.n_list[i], weights=weight1
                            )
                        )

        net = Net(layer)

        # Output to (2,0)
        monkeypatch.setattr(pb.BACKEND_CONFIG, "output_chip_addr", (2, 0))

        if layer > 64 - 1:
            clist = [Coord(1, 0), Coord(0, 0)]
        else:
            clist = [Coord(1, 0)]

        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)

        mapper = pb.Mapper()
        mapper.build(net)
        graph_info = mapper.compile()
        mapper.export(fp=ensure_dump_dir, use_hw_sim=False)

        assert 1


class TestMapper_Export:
    def test_export_multi_nodes_more_than_32(
        self, build_Network_with_N_onodes, ensure_dump_dir
    ):
        net = build_Network_with_N_onodes
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

        assert len(mapper.graph_info["output"]) == net.n_onodes

    def test_export_empty_cplm(self, build_example_net4_large_scale, ensure_dump_dir):
        net = build_example_net4_large_scale
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        mapper.export(fp=ensure_dump_dir)

        assert len(mapper.routing_manager.ordered_rgrps[1].wasted_coords) == 2


class TestMapper_Compile:
    @pytest.mark.xfail(reason="change the hardware limit may cause unexpected errors.")
    def test_grouping_optim_latency(
        self, monkeypatch, build_Network_8bit_dense, ensure_dump_dir
    ):
        monkeypatch.setattr(HwConfig, "N_NEURON_MAX_SNN", 8 * 8)
        monkeypatch.setattr(HwConfig, "N_FANIN_PER_DENDRITE_SNN", 6)

        net = build_Network_8bit_dense

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(grouping_optim_target="latency")

        # Export complete configurations of cores into json
        export_core_plm_conf_json(
            mapper.core_plm_config, ensure_dump_dir, "core_plm_configs_dense.json"
        )

    def test_grouping_optim_core(self, monkeypatch, build_example_net4):
        net = build_example_net4

        monkeypatch.setattr(net.n1, "unrolling_factor", 2)
        monkeypatch.setattr(net.n2, "unrolling_factor", 3)
        monkeypatch.setattr(net.n4, "unrolling_factor", 4)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(grouping_optim_target="core")

        for cb in mapper.core_blocks:
            if node_sl_lst_overlap(net.n1, cb.dest):
                assert cb.n_core_required == ceil(
                    net.n1.num_out / HwConfig.N_DENDRITE_MAX_SNN
                )
            elif node_sl_lst_overlap(net.n2, cb.dest):
                assert cb.n_core_required == 1 + 1

            elif node_sl_lst_overlap(net.n4, cb.dest):
                assert cb.n_core_required == ceil(
                    net.n4.num_out / HwConfig.N_DENDRITE_MAX_SNN
                )

    def test_grouping_optim_both(self, monkeypatch, build_example_net4):
        net = build_example_net4

        monkeypatch.setattr(net.n1, "unrolling_factor", 2)
        monkeypatch.setattr(net.n2, "unrolling_factor", 3)
        monkeypatch.setattr(net.n4, "unrolling_factor", 4)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(grouping_optim_target="both")

        assert (
            mapper.core_blocks[0].n_core_required
            == ceil(net.n1.num_out / HwConfig.N_DENDRITE_MAX_SNN) * 2
        )

        assert mapper.core_blocks[1].n_core_required == ceil(
            net.n2.num_out / HwConfig.N_DENDRITE_MAX_SNN
        ) * 3 + ceil(net.n3.num_out / HwConfig.N_DENDRITE_MAX_SNN)

        assert (
            mapper.core_blocks[2].n_core_required
            == ceil(net.n4.num_out / HwConfig.N_DENDRITE_MAX_SNN) * 4
        )

    def test_ordered_axons(self, build_example_net5):
        net = build_example_net5
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        for cb in mapper.core_blocks:
            if node_sl_lst_overlap(net.n3, cb.dest):
                assert len(cb.ordered_axons) > len(cb.source)
            elif node_sl_lst_overlap(net.n4, cb.dest):
                assert len(cb.ordered_axons) > len(cb.source)
            elif node_sl_lst_overlap(net.n5, cb.dest):
                assert len(cb.ordered_axons) > len(cb.source)
            else:
                assert len(cb.ordered_axons) == len(cb.source)

    def test_merge_cycles(self, build_example_net5):
        net = build_example_net5
        mapper = pb.Mapper()
        mapper.build(net)
        merged_sgrps_with_cycle = mapper.graph.graph_partition()
        """merged_sgrps_with_cycle:

        MergedSuccGroup:
              Nodes: n2, n6
              Group of n1:
                      Edge s0: n1 -> n2
                      Edge s5: n1 -> n6
              Group of n5:
                      Edge s4: n5 -> n6

        MergedSuccGroup:
              Nodes: n5, n3
              Group of n2:
                      Edge s1: n2 -> n3
                      Edge s6: n2 -> n5
              Group of n4:
                      Edge s3: n4 -> n5

        MergedSuccGroup:
              Nodes: n4
              Group of n3:
                      Edge s2: n3 -> n4
        """
        assert len(merged_sgrps_with_cycle) == 3

        """
        MergedSuccGroup:
              Nodes: n2, n6, n3, n4, n5
              Group of n1:
                      Edge s0: n1 -> n2
                      Edge s5: n1 -> n6
              Group of n5:
                      Edge s4: n5 -> n6
              Group of n2:
                      Edge s1: n2 -> n3
                      Edge s6: n2 -> n5
              Group of n4:
                      Edge s3: n4 -> n5
              Group of n3:
                      Edge s2: n3 -> n4
        """

        merged_sgrps = merge_cycles(merged_sgrps_with_cycle)

        assert len(merged_sgrps) == 1

    def test_partition(self, build_example_net6):
        net = build_example_net6
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()
        for cb in mapper.core_blocks:
            if node_sl_lst_overlap(net.n3, cb.dest):
                assert len(cb.ordered_axons) == 2
            if node_sl_lst_overlap(net.n4, cb.dest):
                assert len(cb.ordered_axons) == 3

    def test_set_target_chip(self, build_example_net1, monkeypatch):
        monkeypatch.setattr(
            pb.BACKEND_CONFIG, "target_chip_addr", [Coord(1, 0), (0, 0), (1, 1)]
        )
        net = build_example_net1
        monkeypatch.setattr(net.n2, "target_chip_idx", 1)
        monkeypatch.setattr(net.n3, "target_chip_idx", 2)

        mapper = pb.Mapper()
        mapper.build(net)
        info = mapper.compile()
        assert len(info["members"].keys()) == 3
        mapper.clear()

        monkeypatch.setattr(
            net.n3, "target_chip_idx", len(pb.BACKEND_CONFIG.target_chip_addr)
        )
        mapper.build(net)

        with pytest.raises(ResourceError):
            info = mapper.compile()


class TestMapper_cflags:
    @pytest.mark.parametrize(
        TestData.cflags_weight_bit_opt_data["args"],
        TestData.cflags_weight_bit_opt_data["data"],
    )
    def test_cflags_weight_bit_opt(self, range, scalar, dtype, expected_wp_opt):
        # s1, s2, s3 will be grouped in one core block.
        class Net(pb.Network):
            def __init__(self):
                super().__init__()
                self.n1 = pb.TonicSpiking(10, 3, name="n1", tick_wait_start=1)
                self.n2 = pb.TonicSpiking(10, 4, name="n2", tick_wait_start=2)
                self.n3 = pb.TonicSpiking(10, 4, name="n3", tick_wait_start=2)
                self.n4 = pb.TonicSpiking(10, 4, name="n4", tick_wait_start=2)
                self.s1 = pb.FullConn(
                    self.n1,
                    self.n2,
                    weights=np.random.randint(*range[0], size=(10,), dtype=dtype[0]),
                    conn_type=pb.SynConnType.One2One,
                    name="s1",
                )
                self.s2 = pb.FullConn(
                    self.n1,
                    self.n3,
                    weights=np.random.randint(*range[1], size=(10, 10), dtype=dtype[1]),
                    name="s2",
                )
                self.s3 = pb.FullConn(
                    self.n1,
                    self.n4,
                    weights=scalar,
                    conn_type=pb.SynConnType.All2All,
                    name="s3",
                )

        net = Net()
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(weight_bit_optimization=False)
        assert mapper.core_blocks[0].weight_width == WW.WEIGHT_WIDTH_8BIT

        mapper.clear()
        mapper.build(net)
        mapper.compile(weight_bit_optimization=True)
        assert mapper.core_blocks[0].weight_width == max(
            s.weight_width for s in (net.s1, net.s2, net.s3)
        )
        assert mapper.core_blocks[0].weight_width == expected_wp_opt

    def test_core_estimate_only(self, build_example_net4):
        net = build_example_net4

        mapper = pb.Mapper()
        mapper.build(net)
        graph_info = mapper.compile(core_estimate_only=True)

        assert graph_info["n_core_required"] > 0
        assert graph_info["members"] == {}


from tests.utils import measure_time


class TestMapper_Multichip:
    @pytest.mark.xfail(reason="Network may too large.", raises=ResourceError)
    def test_multichip_1(self, ensure_dump_dir, monkeypatch, build_MultichipNet1_s1):
        """Multichip network of scale 1"""
        clist = [Coord(0, 0), Coord(0, 1), Coord(1, 1)]
        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)
        assert pb.BACKEND_CONFIG.n_target_chips == len(clist)

        net = build_MultichipNet1_s1
        mapper = pb.Mapper()
        mapper.build(net)

        with measure_time("test_multichip_1"):
            mapper.compile(weight_bit_optimization=False)

        mapper.export(fp=ensure_dump_dir, split_by_chip=False, read_voltage=net.n_out)

        print("Total cores occupied:", mapper.n_core_occupied)

        assert 1

    @pytest.mark.skip(reason="Too slow, ~15 mins.")
    @pytest.mark.xfail(reason="Network may too large.", raises=ResourceError)
    def test_multichip_2(self, ensure_dump_dir, monkeypatch, build_MultichipNet1_s2):
        """Multichip network of scale 2"""
        clist = [Coord(0, 0), Coord(0, 1), Coord(1, 0)]
        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)
        assert pb.BACKEND_CONFIG.n_target_chips == len(clist)

        net = build_MultichipNet1_s2
        mapper = pb.Mapper()
        mapper.build(net)

        with measure_time("test_multichip_2"):
            mapper.compile(weight_bit_optimization=False)

        mapper.export(fp=ensure_dump_dir, split_by_chip=False)

        print("Total cores occupied:", mapper.n_core_occupied)

        assert 1
