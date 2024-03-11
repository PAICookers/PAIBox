import json
from enum import Enum
from json import JSONEncoder
from math import ceil
from typing import Any

import numpy as np
import pytest
from paicorelib import Coord, HwConfig, WeightPrecision

import paibox as pb
from paibox.backend.conf_template import CoreConfig, NeuronDest, NeuronDestInfo
from paibox.synapses import SynSys


class CustomJsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Coord):
            return o.address
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, CoreConfig):
            return o.__json__()
        elif isinstance(o, NeuronDest):
            return o.__json__()
        elif isinstance(o, NeuronDestInfo):
            return o.model_dump(by_alias=True)
        else:
            return super().default(o)


class TestGraphInfo:
    def test_multi_inputproj(
        self, get_mapper, ensure_dump_dir, build_multi_inputproj_net
    ):
        net = build_multi_inputproj_net
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()
        mapper.export(
            fp=ensure_dump_dir,
            format="txt",
            split_by_coordinate=True,
            export_core_params=True,
        )

        assert len(mapper.graph_info["input"]) == 2

    def test_multi_inputproj2(
        self, get_mapper, ensure_dump_dir, build_multi_inputproj_net2
    ):
        net = build_multi_inputproj_net2
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()
        mapper.export(
            fp=ensure_dump_dir,
            format="txt",
            split_by_coordinate=True,
            export_core_params=True,
        )

        assert len(mapper.graph_info["input"]) == 2

    def test_multi_output_nodes(
        self, get_mapper, ensure_dump_dir, build_multi_onodes_net
    ):
        net = build_multi_onodes_net
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"]) == 2

        mapper.export(
            fp=ensure_dump_dir,
            format="txt",
            split_by_coordinate=True,
            export_core_params=True,
        )

    def test_multi_output_nodes2(
        self, get_mapper, ensure_dump_dir, build_multi_onodes_net2
    ):
        net = build_multi_onodes_net2
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"]) == 2

        mapper.export(
            fp=ensure_dump_dir,
            format="txt",
            split_by_coordinate=True,
            export_core_params=True,
        )

    def test_multi_inodes_onodes(
        self, get_mapper, ensure_dump_dir, build_multi_inodes_onodes
    ):
        net = build_multi_inodes_onodes
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["input"]) == 2
        assert len(mapper.graph_info["output"]) == 2

    def test_nested_net_L2_compile(self, get_mapper, build_Nested_Net_level_2):
        net = build_Nested_Net_level_2
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph.nodes.keys()) == 5
        assert len(mapper.graph_info["input"]) == 1
        assert len(mapper.graph_info["output"]) == 1

    def test_nested_net_L3_compile(self, get_mapper, build_Nested_Net_level_3):
        net2 = build_Nested_Net_level_3
        mapper: pb.Mapper = get_mapper
        mapper.build(net2)
        mapper.compile()

        assert len(mapper.graph.edges.keys()) == 5
        assert len(mapper.graph_info["input"]) == 2
        assert len(mapper.graph_info["output"]) == 1


class TestMapperDebug:
    def test_build_graph(self, get_mapper, build_example_net1, build_example_net2):
        """Build more than one networks."""
        net1 = build_example_net1
        net2 = build_example_net2

        mapper: pb.Mapper = get_mapper
        mapper.build(net1, net2)
        mapper.compile()

        assert len(mapper.graph.nodes.keys()) == 8
        assert len(mapper.graph_info["input"]) == 3
        assert len(mapper.graph_info["output"]) == 2

    @pytest.fixture
    def compile_simple_net(self, get_mapper, build_example_net1):
        """Reused fixture."""
        net = build_example_net1

        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

    @pytest.mark.usefixtures("compile_simple_net")
    def test_export_config_json(self, ensure_dump_dir, get_mapper):
        """Export all the configs into json"""
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        assert len(mapper.core_blocks) == 3  # 3 layers
        assert mapper.graph_info["inherent_timestep"] == 3

        mapper.export(
            fp=ensure_dump_dir, export_core_params=True, split_by_coordinate=False
        )
        print()

    @pytest.mark.usefixtures("compile_simple_net")
    def test_find_neuron(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        mapper.find_neuron(net.n3)

        print()

    @pytest.mark.usefixtures("compile_simple_net")
    def test_find_axon(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        mapper.find_axon(net.n2)

        print()

    def test_network_with_container(self, get_mapper, build_Network_with_container):
        net: pb.Network = build_Network_with_container
        mapper: pb.Mapper = get_mapper
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph.nodes.keys()) == 4
        # Input projectioon is discnnected!
        assert len(mapper.graph_info["input"]) == 0
        assert len(mapper.graph_info["output"]) == 1


class TestMapper_Export:
    def test_export_multi_nodes_more_than_32(self, build_Network_with_N_onodes):
        net = build_Network_with_N_onodes
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert len(mapper.graph_info["output"].keys()) == net.n_onodes


class TestMapper_Weight4:
    @pytest.mark.skipif(
        hasattr(SynSys, "CFLAG_ENABLE_WP_OPTIMIZATION"), reason="Breaking change"
    )
    def test_mapper_weight4(
        self, monkeypatch, ensure_dump_dir, build_network_with_branches_4bit, packbits8
    ):
        # Use monkey patch to change the settings of `HwConfig` when running the test.
        monkeypatch.setattr(HwConfig, "N_DENDRITE_MAX_SNN", 8 * 8)
        monkeypatch.setattr(HwConfig, "N_FANIN_PER_DENDRITE_SNN", 6)

        net = build_network_with_branches_4bit

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        configs = mapper.export(write_to_file=False, fp=ensure_dump_dir, format="npy")

        assert mapper.n_core_required == 11

        from paibox.backend.checker import ConfigChecker

        cplm00 = mapper.core_blocks[0].core_placements[Coord(0, 0)]
        cplm01 = mapper.core_blocks[0].core_placements[Coord(0, 1)]
        cplm10 = mapper.core_blocks[0].core_placements[Coord(1, 0)]

        n_config_core00 = ConfigChecker.n_config_estimate(
            cplm00.n_neuron, cplm00.weight_precision, cplm00.lcn_ex
        )
        n_config_core01 = ConfigChecker.n_config_estimate(
            cplm01.n_neuron, cplm01.weight_precision, cplm01.lcn_ex
        )
        n_config_core10 = ConfigChecker.n_config_estimate(
            cplm10.n_neuron, cplm10.weight_precision, cplm10.lcn_ex
        )

        assert n_config_core00 == configs[Coord(0, 0)].size
        assert n_config_core01 == configs[Coord(0, 1)].size
        assert n_config_core10 == configs[Coord(1, 0)].size

        # The #N of config frames of each core.

        original_w1 = net.s1.connectivity
        original_w2 = net.s2.connectivity
        original_w3 = net.s3.connectivity
        original_w4 = net.s4.connectivity
        original_w5 = net.s5.connectivity

        # Folded weight of s1
        w11_folded = mapper.core_blocks[0].core_placements[Coord(0, 0)]._weights_folded
        w12_folded = mapper.core_blocks[0].core_placements[Coord(0, 1)]._weights_folded
        w13_folded = mapper.core_blocks[0].core_placements[Coord(1, 0)]._weights_folded

        # Splited & folded weight of s2 & s3
        w21_folded = mapper.core_blocks[1].core_placements[Coord(2, 0)]._weights_folded
        w22_folded = mapper.core_blocks[1].core_placements[Coord(2, 1)]._weights_folded
        w23_folded = mapper.core_blocks[1].core_placements[Coord(3, 0)]._weights_folded
        w24_folded = mapper.core_blocks[1].core_placements[Coord(3, 1)]._weights_folded
        w25_folded = mapper.core_blocks[1].core_placements[Coord(2, 2)]._weights_folded
        w26_folded = mapper.core_blocks[1].core_placements[Coord(2, 3)]._weights_folded

        # Splited & folded weight of s4 & 5
        w31_folded = mapper.core_blocks[2].core_placements[Coord(0, 2)]._weights_folded
        w32_folded = mapper.core_blocks[2].core_placements[Coord(0, 3)]._weights_folded

        # Unpacked weight of s1
        w11_unpacked = mapper.core_blocks[0].core_placements[Coord(0, 0)].weight_ram
        w12_unpacked = mapper.core_blocks[0].core_placements[Coord(0, 1)].weight_ram
        w13_unpacked = mapper.core_blocks[0].core_placements[Coord(1, 0)].weight_ram

        for i in range(10):
            for j in range(4):
                n_in_col = w11_folded.shape[0]
                now_i = i % n_in_col

                offset_j = i // n_in_col
                now_j = offset_j + j * 2

                expected = original_w1[i, j]
                wij = w11_folded[now_i, now_j]

                assert expected == wij

                # wij = w11_folded[now_i, now_j * 8 : (now_j + 1) * 8]
                # packed = packbits8(wij)
                # assert expected == packed

        print("OK")


class TestMapper_Grouping_Optim:
    def test_grouping_optim_latency(
        self, monkeypatch, build_Network_8bit_dense, ensure_dump_dir
    ):
        monkeypatch.setattr(HwConfig, "N_DENDRITE_MAX_SNN", 8 * 8)
        monkeypatch.setattr(HwConfig, "N_FANIN_PER_DENDRITE_SNN", 6)

        net = build_Network_8bit_dense

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(grouping_optim_target="latency")

        _json_core_plm_config = dict()

        for coord, cpc in mapper.core_plm_config.items():
            _json_core_plm_config[coord.address] = cpc.__json__()

        # Export complete configurations of cores into json
        with open(ensure_dump_dir / "core_plm_configs_dense.json", "w") as f:
            json.dump(
                _json_core_plm_config,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

    def test_grouping_optim_core(self, monkeypatch, build_example_net4):
        net = build_example_net4

        monkeypatch.setattr(net.n1, "unrolling_factor", 2)
        monkeypatch.setattr(net.n2, "unrolling_factor", 3)
        monkeypatch.setattr(net.n4, "unrolling_factor", 4)

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(grouping_optim_target="core")

        assert mapper.core_blocks[0].n_core_required == ceil(
            net.n1.num_out / HwConfig.N_DENDRITE_MAX_SNN
        )

        assert mapper.core_blocks[1].n_core_required == 1 + 1

        assert mapper.core_blocks[2].n_core_required == ceil(
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


class TestMapper_cflags:
    def test_cflags_weight_bit_optimization(self, build_network_with_branches_4bit):
        net = build_network_with_branches_4bit
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(weight_bit_optimization=True)
        assert (
            mapper.core_blocks[0].weight_precision == WeightPrecision.WEIGHT_WIDTH_4BIT
        )

        mapper.clear()
        mapper.build(net)
        mapper.compile(weight_bit_optimization=False)
        assert (
            mapper.core_blocks[0].weight_precision == WeightPrecision.WEIGHT_WIDTH_8BIT
        )
