import random
import numpy as np
import pytest
import paibox as pb

from paicorelib import Coord, CoordOffset, CoreMode, HwConfig, LCN_EX, MaxPoolingEnable
from paicorelib import WeightWidth as WW
from paibox.backend.conf_types import (
    CoreConfig,
    CorePlmConfig,
    InputNeuronDest,
    NeuronConfig,
    NeuronDestInfo,
)
from paibox.backend.conf_exporting import *
from paibox.backend.types import AxonCoord, NeuSegment
from paicorelib.reg_model import TICK_WAIT_END_MAX, TICK_WAIT_START_MAX

from .conftest import gen_random_used_lx

try:
    import orjson as json
except ModuleNotFoundError:
    import json


def _gen_random_core_config() -> CoreConfig:
    wp = random.choice(list(WW))
    lcn_ex = random.choice(list(LCN_EX))

    iwf, swf, sme = random.choice(list(CoreMode)).conf

    num_den = random.randint(1, HwConfig.N_DENDRITE_MAX_SNN)
    mpe = random.choice(list(MaxPoolingEnable))
    tws = random.randint(0, TICK_WAIT_START_MAX)
    twe = random.randint(0, TICK_WAIT_END_MAX)
    target_lcn = random.choice(list(LCN_EX))
    test_chip_addr = Coord(random.randint(0, 31), random.randint(0, 31))

    return CoreConfig(
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


def _gen_random_neuron_config(n_per_channel: int, n_channel: int = 3) -> NeuronConfig:
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
    nseg = NeuSegment(
        neuron, slice(_n_start, 1 * n_per_channel + _n_start), offset, interval
    )

    axon_coords = [AxonCoord(0, i) for i in range(nseg.n_neuron)]
    dest_coords = [dest_coord_start, dest_coord_start + CoordOffset(0, 1)]
    pb.BACKEND_CONFIG.test_chip_addr = test_chip_addr

    return NeuronConfig(
        nseg, axon_coords, dest_coords, pb.BACKEND_CONFIG.test_chip_addr
    )


def _gen_random_neuron_dest_info(n: int) -> NeuronDestInfo:
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

    return NeuronDestInfo.model_validate(dest_info, strict=True)


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

    cpc = CorePlmConfig.encapsulate(
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


class TestConfExporting:
    def test_export_core_params_json(self, ensure_dump_dir):
        core_params = {
            Coord(1, 1): {
                Coord(0, 0): _gen_random_core_config(),
                Coord(0, 1): _gen_random_core_config(),
            },
            Coord(2, 2): {Coord(0, 0): _gen_random_core_config()},
        }

        export_core_params_json(core_params, ensure_dump_dir)

    @pytest.mark.parametrize("n_per_channel, n_channel", [(100, 3), (200, 2), (240, 1)])
    def test_NeuronConfig_conf_json(self, ensure_dump_dir, n_per_channel, n_channel):
        nconf = _gen_random_neuron_config(n_per_channel, n_channel)
        mock_n = pb.IF(1, 1)
        export_neuconf_json({mock_n: nconf}, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_input_conf_json(self, ensure_dump_dir, n_neuron):
        iconf = {"n1": _gen_input_neuron_dest(n_neuron)}
        export_input_conf_json(iconf, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_output_conf_json(self, ensure_dump_dir, n_neuron):
        oconf = {"n1": {0: _gen_random_neuron_dest_info(n_neuron)}}
        export_output_conf_json(oconf, ensure_dump_dir)

    @pytest.mark.parametrize("n_neuron", [100, 200, 300])
    def test_export_core_plm_conf_json(self, ensure_dump_dir, n_neuron):
        chip_coord = Coord(1, 1)
        core_coord = Coord(10, 10)

        core_plm_conf = {
            chip_coord: {core_coord: _gen_random_core_plm_config(n_neuron)}
        }
        export_core_plm_conf_json(core_plm_conf, ensure_dump_dir)

        with open(ensure_dump_dir / "core_plm.json", "rb") as f:
            core_plm_conf_json = json.loads(f.read())
            assert list(core_plm_conf_json.keys())[0] == str(chip_coord)

    def test_export_used_L2_clusters(self, ensure_dump_dir, monkeypatch):
        clist = [Coord(0, 0), Coord(0, 1), Coord(2, 2)]
        monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)

        n_lx_max = HwConfig.N_SUB_ROUTING_NODE ** (5 - 2)
        n = random.randint(1, n_lx_max)
        used_L2 = []

        for _ in range(len(clist)):
            used_L2.append(gen_random_used_lx(n, 2))

        clk_en_L2_dict = get_clk_en_L2_dict(pb.BACKEND_CONFIG.target_chip_addr, used_L2)

        export_used_L2_clusters(clk_en_L2_dict, ensure_dump_dir)


@pytest.mark.parametrize(
    "index, offset, expected",
    [
        (slice(0, 200), 100, (slice(0, 200), None)),
        (slice(200, 400), 512, (None, slice(200, 400))),
        (slice(0, 600), 100, (slice(0, 412), slice(412, 600))),
        (slice(100, 400), 300, (slice(100, 312), slice(312, 400))),
    ],
)
def test_NeuronConfig_mapped_on_ram(index, offset, expected):
    n = index.stop - index.start
    neuron = pb.ANNNeuron((n,), bias=9, keep_shape=True)
    dest_coord_start = Coord(random.randint(0, 10), random.randint(0, 10))

    nseg = NeuSegment(neuron, index, offset)
    axon_coords = [AxonCoord(0, i) for i in range(n)]
    dest_coords = [dest_coord_start, dest_coord_start + CoordOffset(0, 1)]

    neu_config1 = NeuronConfig(
        nseg, axon_coords, dest_coords, pb.BACKEND_CONFIG.test_chip_addr
    )

    if (
        neu_config1.neu_seg.offset + neu_config1.neu_seg.n_neuron
        <= HwConfig.ADDR_RAM_MAX + 1
    ):
        result1 = neu_config1
        result2 = None

        assert result1.neu_seg.index == expected[0]
        assert result2 == expected[1]

    elif (n_on_nram := HwConfig.ADDR_RAM_MAX + 1 - neu_config1.neu_seg.offset) > 0:
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
