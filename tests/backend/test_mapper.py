import json
from enum import Enum
from json import JSONEncoder
from typing import Any

import numpy as np
import pytest
from paicorelib import Coord, HwConfig

import paibox as pb
from paibox.backend.conf_template import CoreConfig, NeuronDest, NeuronDestInfo


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
    def test_multi_inputproj(self, get_mapper, build_example_net2):
        net = build_example_net2
        mapper: pb.Mapper = get_mapper

        mapper.build(net)
        mapper.compile()

        assert mapper.graph_info.get("input") is not None

        assert len(mapper.graph_info["input"]) == 2  # type: ignore


class TestMapperDebug:
    def test_build_graph(self, get_mapper, build_example_net1, build_example_net2):
        """Build more than one networks."""
        net1 = build_example_net1
        net2 = build_example_net2

        mapper: pb.Mapper = get_mapper
        mapper.clear()
        mapper.build(net1, net2)

        assert mapper.graph.has_built == True

    @pytest.fixture
    def test_simple_net(self, get_mapper, build_example_net1):
        """Go throught the backend"""
        net = build_example_net1

        mapper: pb.Mapper = get_mapper
        mapper.clear()
        mapper.build(net)
        mapper.compile()

    @pytest.mark.usefixtures("test_simple_net")
    def test_export_config_json(self, get_mapper, ensure_dump_dir):
        """Export all the configs into json"""
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        assert len(mapper.core_blocks) == 3  # 3 layers
        assert mapper.get_inherent_timestep() == 3

        mapper.export(fp=ensure_dump_dir, export_core_params=True)
        print()

    @pytest.mark.usefixtures("test_simple_net")
    def test_find_neuron(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        mapper.find_neuron(net.n3)

        print()

    @pytest.mark.usefixtures("test_simple_net")
    def test_find_axon(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.graph.has_built == True

        mapper.find_axon(net.n2)

        print()


class TestMapper_Weight4:
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


class TestMapper_NeuronSeg_Dense:
    def test_neuron_seg_dense(
        self, monkeypatch, build_Network_8bit_dense, ensure_dump_dir
    ):
        monkeypatch.setattr(HwConfig, "N_DENDRITE_MAX_SNN", 8 * 8)
        monkeypatch.setattr(HwConfig, "N_FANIN_PER_DENDRITE_SNN", 6)

        net = build_Network_8bit_dense

        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(method="dense")

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
