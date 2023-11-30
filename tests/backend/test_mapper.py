import json
from enum import Enum
from json import JSONEncoder
from typing import Any

import numpy as np
import pytest

import paibox as pb
from paibox.libpaicore import Coord


class CustomJsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Coord):
            return o.to_tuple()
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, np.ndarray):
            return int(o)
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

        assert mapper.has_built == True

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
        assert mapper.has_built == True

        assert len(mapper.core_blocks) == 3  # 3 layers
        assert mapper.get_inherent_timestep() == 4

        _json_core_configs = dict()
        _json_core_plm_config = dict()
        _json_inp_proj_info = dict()
        _json_out_proj_info = dict()

        for coord, core_param in mapper.core_params.items():
            _json_core_configs[coord.address] = core_param.__json__()

        for coord, cpc in mapper.core_plm_config.items():
            _json_core_plm_config[coord.address] = cpc.__json__()

        if input_info := mapper.graph_info.get("input"):
            for inode, nd in input_info.items():
                _json_inp_proj_info[inode] = nd.__json__()

        if output_info := mapper.graph_info.get("output"):
            for onode, nd_with_coord in output_info.items():
                _json_out_proj_info[onode] = dict()
                for coord, nd in nd_with_coord.items():
                    _json_out_proj_info[onode][coord] = nd.__json__()

        # Export parameters of cores into json
        with open(ensure_dump_dir / "core_configs.json", "w") as f:
            json.dump(
                _json_core_configs,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        # Export complete configurations of cores into json
        with open(ensure_dump_dir / "core_plm_configs.json", "w") as f:
            json.dump(
                _json_core_plm_config,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        # Export the info of input projections into json
        with open(ensure_dump_dir / "input_proj_info.json", "w") as f:
            json.dump(
                _json_inp_proj_info,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        # Export the info of output destination into json
        with open(ensure_dump_dir / "output_dest_info.json", "w") as f:
            json.dump(
                _json_out_proj_info,
                f,
                ensure_ascii=True,
                indent=4,
                cls=CustomJsonEncoder,
            )

        print("OK")

    @pytest.mark.usefixtures("test_simple_net")
    def test_find_neuron(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.has_built == True

        mapper.find_neuron(net.n3)

        print()

    @pytest.mark.usefixtures("test_simple_net")
    def test_find_axon(self, get_mapper, build_example_net1):
        net: pb.Network = build_example_net1
        mapper: pb.Mapper = get_mapper
        assert mapper.has_built == True

        mapper.find_axon(net.n2)

        print()


class TestMapper_Weight8:
    def test_mapper_weight8(self, monkeypatch, build_small_net1, packbits8):
        # Use monkey patch to change the settings of `HwConfig` when running the test.
        monkeypatch.setattr(pb.HwConfig, "N_DENDRITE_MAX_SNN", 8 * 8)
        monkeypatch.setattr(pb.HwConfig, "N_FANIN_PER_DENDRITE_SNN", 6)

        net = build_small_net1

        # Core required
        # inp1 -> n1: 10*10, LCN_2X, 3
        # n1 -> n2 & n1 -> n3: 10*20, LCN_2X, 6
        # n2 -> n4 & n3 -> n4: 20*4, LCN_4X, 2

        mapper = pb.Mapper()
        mapper.clear()
        mapper.build(net)
        mapper.compile()

        assert mapper.n_core_required == 11

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
