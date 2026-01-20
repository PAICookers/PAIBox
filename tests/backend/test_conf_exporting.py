import random

import numpy as np
import pytest
from paicorelib import (
    LCN_EX,
    ChipCoord,
    Coord,
    CoordOffset,
    CoreMode,
    HwConfig,
    MaxPoolingEnable,
    NeuDestInfo,
    OffCoreCfg,
    OfflineCoreRegLim,
    get_replication_id,
)
from paicorelib import WeightWidth as WW

import paibox as pb
from paibox.backend.conf_exporting import (
    export_aux_gh_info,
    export_core_params_json,
    export_core_plm_conf_json,
    export_input_conf_json,
    export_neuconf_json,
    export_output_conf_json,
    export_used_L2_clusters,
    get_clk_en_L2_dict,
)
from paibox.backend.conf_types import (
    CoreConfig,
    CorePlmConfig,
    GraphInfo,
    InputNeuronDest,
    OfflineCoreConfig,
    OfflineCorePlmConfig,
    OfflineNeuConfig,
    OfflineNeuDestInfo,
)
from paibox.backend.types import AxonCoord, DendriteSegment
from paibox.base import DataFlowFormat
from tests.utils import file_not_exist_fail

from .backend_testcase import _gen_custom_index
from .conftest import gen_random_used_lx

try:
    import orjson as json
except ModuleNotFoundError:
    import json

TICK_WAIT_END_MAX = OfflineCoreRegLim.TICK_WAIT_END_MAX
TICK_WAIT_START_MAX = OfflineCoreRegLim.TICK_WAIT_START_MAX


def _gen_random_core_config() -> OfflineCoreConfig:
    wp = random.choice(list(WW))
    lcn_ex = random.choice(list(LCN_EX)[:-1])

    iwf, swf, sme = random.choice(list(CoreMode)).conf

    num_den = random.randint(1, OffCoreCfg.N_DENDRITE_MAX_SNN)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, TICK_WAIT_START_MAX)
    twe = random.randint(0, TICK_WAIT_END_MAX)
    target_lcn = random.choice(list(LCN_EX)[:-1])
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return OfflineCoreConfig(
        "mock_core",
        wp,
        lcn_ex,
        iwf,
        swf,
        num_den,
        mpe,
        tws,
        twe,
        sme,
        target_lcn,
        test_chip_addr,
    )


def _gen_random_neuron_config(
    n_per_channel: int, n_channel: int = 3
) -> OfflineNeuConfig:
    n = n_channel * n_per_channel
    offset = random.randint(1, 20)
    interval = random.randint(1, 2)
    thres = random.randint(1, 5)
    reset_v = random.randint(-5, 5)
    leak_v = np.arange(n_channel * n).reshape((n_channel, n))
    neuron = pb.LIF((n_channel, n), thres, reset_v, bias=leak_v, keep_shape=True)
    dest_coord_start = Coord(random.randint(0, 10), random.randint(0, 10))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    _n_start = random.randint(0, 10)
    nseg = DendriteSegment(
        neuron, _gen_custom_index(_n_start, 1 * n_per_channel), offset, interval
    )

    axon_coords = [AxonCoord(0, i) for i in range(nseg.n_neuron)]
    dest_coords = [dest_coord_start, dest_coord_start + CoordOffset(0, 1)]
    pb.BACKEND_CONFIG.test_chip_addr = test_chip_addr
    base_coord, rid = get_replication_id(dest_coords)

    return OfflineNeuConfig(
        nseg, axon_coords, base_coord, rid, pb.BACKEND_CONFIG.test_chip_addr
    )


def _gen_random_neuron_dest_info(n: int) -> OfflineNeuDestInfo:
    tick_relative = [0 for _ in range(n)]
    addr_axon = [i for i in range(n)]

    addr_core_x = random.randint(0, 31)
    addr_core_y = random.randint(0, 31)
    addr_core_x_ex = random.randint(0, 31)
    addr_core_y_ex = random.randint(0, 31)
    addr_chip_x = random.randint(0, 31)
    addr_chip_y = random.randint(0, 31)

    dest_info = {
        "tick_relative": tick_relative,
        "addr_axon": addr_axon,
        "addr_core_x": addr_core_x,
        "addr_core_y": addr_core_y,
        "addr_core_x_ex": addr_core_x_ex,
        "addr_core_y_ex": addr_core_y_ex,
        "addr_chip_x": addr_chip_x,
        "addr_chip_y": addr_chip_y,
    }

    return OfflineNeuDestInfo.model_validate(dest_info, strict=True)


def _gen_input_neuron_dest(n: int) -> InputNeuronDest:
    tick_relative = [0 for _ in range(n)]
    addr_axon = [i for i in range(n)]

    addr_core_x = random.randint(0, 31)
    addr_core_y = random.randint(0, 31)
    addr_core_x_ex = random.randint(0, 31)
    addr_core_y_ex = random.randint(0, 31)
    addr_chip_x = random.randint(0, 31)
    addr_chip_y = random.randint(0, 31)
    lcn = 1 << random.choice(list(LCN_EX))

    return InputNeuronDest(
        tick_relative,
        addr_axon,
        addr_core_x,
        addr_core_y,
        addr_core_x_ex,
        addr_core_y_ex,
        addr_chip_x,
        addr_chip_y,
        lcn,
    )


def _gen_random_core_plm_config(n_neuron: int) -> CorePlmConfig:
    thres = random.randint(1, 5)
    reset_v = random.randint(-5, 5)
    neuron = pb.IF((n_neuron,), thres, reset_v)

    cpc = OfflineCorePlmConfig.encapsulate(
        random.randint(0, 1000),
        np.random.randint(
            np.iinfo(np.uint64).min,
            np.iinfo(np.uint64).max,
            size=(512, 18),
            dtype=np.uint64,
        ),
        _gen_random_core_config(),
        {neuron: _gen_random_neuron_config(n_neuron, 1)},
    )

    return cpc


@pytest.fixture
def setup_clist_for_used_L2(monkeypatch):
    clist = [Coord(0, 0), Coord(0, 1), Coord(2, 2)]
    monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)


class TestConfExporting:
    def test_export_core_params_json(self, ensure_dump_dir):
        core_params: dict[ChipCoord, dict[Coord, CoreConfig]] = {
            ChipCoord(1, 1): {
                Coord(0, 0): _gen_random_core_config(),
                Coord(0, 1): _gen_random_core_config(),
            },
            ChipCoord(2, 2): {Coord(0, 0): _gen_random_core_config()},
        }

        export_core_params_json(core_params, ensure_dump_dir)

    @pytest.mark.parametrize("n_per_channel, n_channel", [(100, 3), (200, 2), (240, 1)])
    def test_OfflineNeuConfig_conf_json(
        self, ensure_dump_dir, n_per_channel, n_channel
    ):
        nconf = _gen_random_neuron_config(n_per_channel, n_channel)
        mock_n = pb.IF(1, 1)
        export_neuconf_json({mock_n: nconf}, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_input_conf_json(self, ensure_dump_dir, n_neuron):
        iconf = {"n1": _gen_input_neuron_dest(n_neuron)}
        export_input_conf_json(iconf, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_output_conf_json(self, ensure_dump_dir, n_neuron):
        oconf: dict[str, dict[Coord, NeuDestInfo]] = {
            "n1": {Coord(3, 2): _gen_random_neuron_dest_info(n_neuron)}
        }
        export_output_conf_json(oconf, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_core_plm_conf_json(self, ensure_dump_dir, n_neuron):
        chip_coord = Coord(1, 1)
        core_coord = Coord(10, 10)

        core_plm_conf = {
            chip_coord: {core_coord: _gen_random_core_plm_config(n_neuron)}
        }
        export_core_plm_conf_json(core_plm_conf, ensure_dump_dir)

        fp = ensure_dump_dir / "core_plm.json"
        file_not_exist_fail(fp)

        with open(fp, "rb") as f:
            core_plm_conf_json = json.loads(f.read())
            assert list(core_plm_conf_json.keys())[0] == str(chip_coord)

    def test_export_used_L2_clusters(self, ensure_dump_dir, setup_clist_for_used_L2):
        n_lx_max = HwConfig.N_SUB_ROUTING_NODE ** (5 - 2)
        n = random.randint(1, n_lx_max)
        used_L2 = []

        for _ in range(len(pb.BACKEND_CONFIG.target_chip_addr)):
            used_L2.append(gen_random_used_lx(n, 2))

        clk_en_L2_dict = get_clk_en_L2_dict(pb.BACKEND_CONFIG.target_chip_addr, used_L2)

        export_used_L2_clusters(clk_en_L2_dict, ensure_dump_dir)

    def test_export_aux_gh_info(self, ensure_dump_dir, setup_clist_for_used_L2):
        n_lx_max = HwConfig.N_SUB_ROUTING_NODE ** (5 - 2)
        n = random.randint(1, n_lx_max)
        used_L2 = []

        for _ in range(len(pb.BACKEND_CONFIG.target_chip_addr)):
            used_L2.append(gen_random_used_lx(n, 2))

        unused_gh_info = {"input": {}, "output": {}, "members": {}}
        aux_gh_info = GraphInfo(
            **unused_gh_info,  # type: ignore
            **{
                "name": "test_export_aux_gh_info",
                "n_core_occupied": 100,
                "n_core_required": 120,
                "inherent_timestep": 10,
                "output_flow_format": {
                    "output_1": DataFlowFormat(10, 1, 1, False),
                    "output_2": DataFlowFormat(8, 2, 10, False),
                },
                "misc": {
                    "clk_en_L2": get_clk_en_L2_dict(
                        pb.BACKEND_CONFIG.target_chip_addr, used_L2
                    ),
                    "target_chip_list": pb.BACKEND_CONFIG.target_chip_addr,
                },
            },
        )

        export_aux_gh_info(aux_gh_info, ensure_dump_dir, export_clk_en_L2=True)

    @pytest.mark.parametrize("read_type", ["object", "name"])
    def test_export_neuron_phy_loc(
        self, ensure_dump_dir, build_multi_onodes_net_more1152, read_type
    ):
        net = build_multi_onodes_net_more1152
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        neu_to_read = []

        neuron_in_net_maybe = ["n1", "n2", "n3", "n4"]
        for n in neuron_in_net_maybe:
            if hasattr(net, n):  # prevent the network from being changed unnoticed
                if read_type == "object":  # Passing in the neuron objects
                    neu_to_read.append(getattr(net, n))
                else:  # Passing in the neuron names
                    neu_to_read.append(n)

        if len(neu_to_read) == 0:
            pytest.skip("Nothing to read. Skip.")

        mapper.export(fp=ensure_dump_dir, read_voltage=neu_to_read)


@pytest.mark.parametrize(
    "index, offset, expected",
    [
        (_gen_custom_index(0, 200), 100, (_gen_custom_index(0, 200), None)),
        (_gen_custom_index(200, 400), 512, (None, _gen_custom_index(200, 400))),
        (
            _gen_custom_index(0, 600),
            100,
            (_gen_custom_index(0, 412), _gen_custom_index(412, 600)),
        ),
        (
            _gen_custom_index(100, 400),
            300,
            (_gen_custom_index(100, 312), _gen_custom_index(312, 400)),
        ),
    ],
)
def test_OfflineNeuConfig_mapped_on_ram(index, offset, expected):
    n = len(index)
    neuron = pb.ANNNeuron((n,), bias=9, keep_shape=True)
    dest_coord_start = Coord(random.randint(0, 10), random.randint(0, 10))

    nseg = DendriteSegment(neuron, index, offset)
    axon_coords = [AxonCoord(0, i) for i in range(n)]
    dest_coords = [dest_coord_start, dest_coord_start + CoordOffset(0, 1)]
    base_coord, rid = get_replication_id(dest_coords)

    neu_config1 = OfflineNeuConfig(
        nseg, axon_coords, base_coord, rid, pb.BACKEND_CONFIG.test_chip_addr
    )

    if (
        neu_config1.neu_seg.offset + neu_config1.neu_seg.n_neuron
        <= OffCoreCfg.ADDR_RAM_MAX + 1
    ):
        result1 = neu_config1
        result2 = None

        assert result1.neu_seg.index == expected[0]
        assert result2 == expected[1]

    elif (n_on_nram := OffCoreCfg.ADDR_RAM_MAX + 1 - neu_config1.neu_seg.offset) > 0:
        s1 = slice(None, n_on_nram)
        s2 = slice(n_on_nram, None)
        result1 = neu_config1[s1]
        result2 = neu_config1[s2]

        assert result1.neu_seg.index == expected[0]
        assert result2.neu_seg.index == expected[1]
    else:
        result1 = None
        result2 = neu_config1

        assert result1 == expected[0]
        assert result2.neu_seg.index == expected[1]
