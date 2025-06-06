import json
import timeit
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from paicorelib import Coord
from paicorelib import ReplicationId as RId
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineWorkFrame1Format as Off_WF1F
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import OfflineTestOutFrame3
from paicorelib.framelib.utils import print_frame

from paibox.runtime import PAIBoxRuntime
from paibox.runtime.runtime import LENGTH_EX_MULTIPLE_KEY, get_length_ex_onode
from tests.utils import file_not_exist_fail

TEST_CONF_DIR = Path(__file__).parent / "test_data"


def test_get_length_ex_onode():
    output_dest_info = {
        "n2_1": {
            "4": {
                "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                "tick_relative": [0] * 8,
                "addr_core_x": 0,
                "addr_core_y": 0,
                "addr_core_x_ex": 0,
                "addr_core_y_ex": 0,
                "addr_chip_x": 1,
                "addr_chip_y": 0,
            }
        },
        "n3_1": {
            "5": {
                "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                "tick_relative": [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4,
                "addr_core_x": 0,
                "addr_core_y": 1,  # y increases
                "addr_core_x_ex": 0,
                "addr_core_y_ex": 0,
                "addr_chip_x": 1,
                "addr_chip_y": 0,
            }
        },
    }
    assert get_length_ex_onode(output_dest_info["n2_1"]) == 1
    assert get_length_ex_onode(output_dest_info["n3_1"]) == 4


class TestRuntime:
    def test_gen_input_frames_info_by_dict(self):
        fp = TEST_CONF_DIR / "input_proj_info1.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            input_proj_info = json.load(f)

        n_input_node = len(input_proj_info.keys())
        assert n_input_node == 2

        n_ts = 8
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, input_proj_info=input_proj_info
        )
        for i in common_part:
            print_frame(i)

        assert len(common_part) == 2
        assert common_part[0].size == 64 * n_ts
        assert common_part[1].size == 32 * n_ts

    def test_gen_input_frames_info_by_kwds(self):
        n_ts = 16
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, (0, 0), 33, RId(0, 0), [0] * 8 + [1] * 8, list(range(16))  # type: ignore
        )
        print_frame(common_part)

        assert len(common_part) == 16 * n_ts

    def test_encode(self):
        data = list(range(8))
        n_ts = 16
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, Coord(0, 0), Coord(1, 0), RId(0, 0), [0] * 4 + [1] * 4, list(range(8))
        )

        input_spike = PAIBoxRuntime.encode(data, common_part, n_ts)

        data_in_spike = (input_spike >> Off_WF1F.DATA_OFFSET) & Off_WF1F.DATA_MASK
        # Encode data with none-zero values.
        assert 0 not in data_in_spike

        axons_in_spike = (input_spike >> Off_WF1F.AXON_OFFSET) & Off_WF1F.AXON_MASK
        # Except axon with data=0
        assert np.array_equal(axons_in_spike, [1, 2, 3, 4, 5, 6, 7] * n_ts)

    def test_decode_by_dict(self):
        # oframe_info `list[FrameArrayType]`, return `list[NDArray[np.uint8]]`
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
            "n3_1": {
                "5": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 1,  # y increases
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
            # Not occured
            "n4_1": {
                "6": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 2,  # y increases
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
        }
        n_ts = 2
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                # o1(x=0,y=0), o2(0,1)
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000001_00000000_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000100_00000000_00001001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000111_00000000_00001010,
            ],
            dtype=np.uint64,
        )
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_info, flatten=False)

        expected = np.zeros((len(oframe_info), n_ts, 8), dtype=np.uint8)
        expected[0][0][3] = 7
        expected[0][0][5] = 8
        expected[0][1][0] = 1
        expected[1][0][1] = 2
        expected[1][0][4] = 9
        expected[1][0][7] = 10

        assert np.array_equal(data, expected)

    def test_decode_by_dict2(self):
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            }
        }

        n_ts = 2
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
            ],
            dtype=np.uint64,
        )
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_info, flatten=False)

        expected = np.zeros((len(oframe_info), n_ts, 8), dtype=np.uint8)
        expected[0][0][3] = 7
        expected[0][0][5] = 8
        expected[0][1][0] = 1

        assert np.array_equal(data, expected)

    def test_decode_by_kwds(self):
        # oframe_info is `FrameArrayType`, return `NDArray[np.uint8]`
        n_axon_max = 100
        n_ts_max = 64

        for n_axon in range(1, n_axon_max):
            for n_ts in range(1, n_ts_max):
                oframe_info = PAIBoxRuntime.gen_output_frames_info(
                    n_ts, Coord(1, 0), Coord(0, 0), RId(0, 0), list(range(n_axon))
                )

                n_chosen = np.random.randint(1, n_axon * n_ts + 1)

                choice_idx = np.random.choice(
                    range(n_axon * n_ts), n_chosen, replace=False
                )

                # choice_idx = [1, 0, 2]
                random = np.random.randint(0, 256, (n_axon * n_ts,), dtype=np.uint8)

                output_frames = oframe_info + random
                shuffle_frame = output_frames[choice_idx]

                expected = np.zeros((n_axon * n_ts,), dtype=np.uint8)
                expected[choice_idx] = random[choice_idx]

                data = PAIBoxRuntime.decode(
                    n_ts, shuffle_frame, oframe_info, flatten=True
                )

                assert np.array_equal(data, expected)

    @pytest.mark.parametrize(
        "n_axons, n_ts", [(1000, 1), (1000, 16), (1000, 32), (1000, 64)]
    )
    def test_decode_perf(self, n_axons, n_ts):
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": list(range(n_axons)),
                    "tick_relative": [0] * n_axons,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
        }
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )
        test_frames = np.zeros((n_axons,), dtype=np.uint64)

        for i in range(n_axons):
            _data = np.random.randint(0, 256, dtype=np.uint8)
            test_frames[i] = (
                (FH.WORK_TYPE1 << Off_WF1F.GENERAL_HEADER_OFFSET)
                | (Coord(1, 0).address << Off_WF1F.GENERAL_CHIP_ADDR_OFFSET)
                | (i << Off_WF1F.AXON_OFFSET)
                | np.uint64(_data)
            )

        t = timeit.timeit(
            lambda: PAIBoxRuntime.decode(n_ts, test_frames, oframe_info),
            number=100,
        )
        print(f"n_axons: {n_axons}, n_ts: {n_ts}, time: {t/100:.5f}s")

    def test_gen_output_frames_info_by_dict1(self):
        fp = TEST_CONF_DIR / "output_dest_info1.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 1

        common_part = PAIBoxRuntime.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 800

    def test_gen_output_frames_info_by_dict2(self):
        fp = TEST_CONF_DIR / "output_dest_info2.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 2

        common_part = PAIBoxRuntime.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 104

    def test_gen_output_frames_info_by_kwds(self):
        n_ts = 16
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, (1, 0), (0, 0), (0, 0), [0, 1, 2, 3, 4, 5, 6, 7]
        )

        assert oframe_info.size == 8 * n_ts

    def test_get_length_ex_onode(self):
        onode_attrs = {
            "(0, 1)": {LENGTH_EX_MULTIPLE_KEY: [0, 0, 0, 0, 0]},
            "(0, 2)": {LENGTH_EX_MULTIPLE_KEY: [0, 0, 0, 0, 0]},
            "(1, 2)": {LENGTH_EX_MULTIPLE_KEY: [0, 1, 1, 1, 1]},
            "(1, 3)": {LENGTH_EX_MULTIPLE_KEY: [1, 1, 1, 2, 2]},
        }
        n_ex_onode = get_length_ex_onode(onode_attrs)

        assert n_ex_onode == 3

    def test_gen_output_frames_info(self):
        fp = TEST_CONF_DIR / "output_dest_info.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000000_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000000_00001000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000001_00010111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000001001_00000001_00011000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000010_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000010_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000010_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000011_00000001,
            ],
            dtype=np.uint64,
        )
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)

        expected = np.zeros((len(oframe_infos), n_ts, 10), dtype=np.uint8)
        expected[0][0][0] = 1
        expected[0][1][0] = 7
        expected[0][0][2] = 8
        expected[0][0][3] = 3
        expected[0][1][3] = 23
        expected[0][1][9] = 24
        expected[0][2][0] = 1
        expected[0][2][2] = 2
        expected[0][2][3] = 3
        expected[0][3][0] = 1

        assert np.array_equal(data, expected)

    def test_gen_output_frames_info_more1152(self):
        fp = TEST_CONF_DIR / "output_dest_info_more1152.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_dest_info = json.load(f)

        n_ts = 2
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000000_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000000_00001000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000001_00010111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000000_00011000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000010_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000010_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000010_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000011_00000001,
            ],
            dtype=np.uint64,
        )
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)
        expected = np.zeros((len(oframe_infos), n_ts, 1300), dtype=np.uint8)
        expected[0][0][0] = 1
        expected[0][0][1152] = 7
        expected[0][0][2] = 8
        expected[0][0][3] = 3
        expected[0][0][1155] = 23
        expected[0][0][1] = 24
        expected[0][1][1] = 1
        expected[0][1][2] = 2
        expected[0][1][3] = 3
        expected[0][1][1152] = 1

        assert np.array_equal(data, expected)

    def test_gen_output_frames_info_more1152_multi_onodes(self):
        fp = TEST_CONF_DIR / "output_dest_info_more1152_multi.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                # o1(x=0,y=0), o2(0,1)
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000000_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000111,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000001000_00000001_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000000_00001000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000011,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000100_00000000_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000001_00010111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000000_00011000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000010_00000001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000100000_00000011_00001001,
                0b1000_00001_00000_00000_00001_00000_00000_000_01000000000_00000000_10000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000010_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000010_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000011_00000001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000010011_00000010_00000110,
            ],
            dtype=np.uint64,
        )
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)
        expected_o1 = np.zeros((n_ts, 1300), dtype=np.uint8)
        expected_o1[0][0] = 1
        expected_o1[0][1152] = 7
        expected_o1[0][2] = 8
        expected_o1[0][3] = 3
        expected_o1[0][1155] = 23
        expected_o1[0][1] = 24
        expected_o1[1][1] = 1
        expected_o1[1][2] = 2
        expected_o1[1][3] = 3
        expected_o1[1][1152] = 1

        expected_o2 = np.zeros((n_ts, 1200), dtype=np.uint8)
        expected_o2[0][4] = 1
        expected_o2[1][8] = 3
        expected_o2[2][19] = 6
        expected_o2[3][32] = 9
        expected_o2[0][1 << 9] = 1 << 7

        assert np.array_equal(data[0], expected_o1)
        assert np.array_equal(data[1], expected_o2)

    def test_decode_zero_oframes(self):
        # Even if zero output frames are given, it should be decoded correctly.
        fp = TEST_CONF_DIR / "output_dest_info_more1152.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        zero_oframes = np.array([], dtype=np.uint64)
        data = PAIBoxRuntime.decode(n_ts, zero_oframes, oframe_infos, flatten=False)

        assert all(d.all() == 0 for d in data)


REQUIRED_PLIB_VERSION = "1.4.1"  # Required version for neuron voltage decoding
from paicorelib import __version__ as plib_version


def get_neu_phy_files() -> list[Path]:
    return list(TEST_CONF_DIR.glob("neuron_phy_loc[0-9]*.json"))


def _shuffle_otframe3(otframe3: list[OfflineTestOutFrame3]):
    rng = np.random.default_rng()
    otframe3_np = np.asarray(otframe3)
    rng.shuffle(otframe3_np)
    return otframe3_np


def get_n_neuron_from_phy_loc(neu_phy_loc: dict[str, dict[str, Any]]) -> int:
    n_neuron = 0
    for chip_loc in neu_phy_loc.values():
        n_neuron += sum(core_loc["n_neuron"] for core_loc in chip_loc.values())

    return n_neuron


def get_contiguous_reading_models_dir() -> list[Path]:
    return list((TEST_CONF_DIR / "contiguous_reading_models").glob("model[0-9]*"))


"""If necessary, enable the following variable to recompile the actual networks for contiguous voltage  \
    decoding tests.
"""
COMPILE_CONTIGUOUS_DECOING_MODEL = 0
import paibox as pb
from paibox.components.synapses.transforms import ConnType


# Run 1 timestep to check the voltage decoding
class Net1_one2one_8b(pb.Network):
    def __init__(self, n: int):
        super().__init__()
        self.inp1 = pb.InputProj(None, (n,))
        self.n1 = pb.IF((n,), 1000, tick_wait_start=1)
        w = np.arange(1, n + 1, dtype=np.int8)
        self.s1 = pb.FullConn(self.inp1, self.n1, w, conn_type=ConnType.One2One)


class Net2_triu_1b(pb.Network):
    def __init__(self, n: int):
        super().__init__()
        self.inp1 = pb.InputProj(None, (n,))
        self.n1 = pb.IF((n,), 10000)
        w = np.triu(np.ones((n, n), dtype=np.bool_), k=0)  # w1
        self.s1 = pb.FullConn(self.inp1, self.n1, w)


class TestReadNeuronVoltage:
    dest_info = dict(
        addr_chip_x=1,
        addr_chip_y=1,
        addr_core_x=2,
        addr_core_y=2,
        addr_core_x_ex=0,
        addr_core_y_ex=0,
        tick_relative=[0],
        addr_axon=[1],
    )  # ramdon, read only

    neu_attrs = dict(
        reset_mode=1,
        reset_v=0,
        leak_post=0,
        threshold_mask_ctrl=0,
        threshold_neg_mode=0,
        threshold_neg=100,
        threshold_pos=100,
        leak_reversal_flag=0,
        leak_det_stoch=0,
        leak_v=3,
        weight_det_stoch=0,
        bit_truncate=8,
        voltage=0,  # voltage will be set
    )

    @pytest.mark.parametrize("fp", get_neu_phy_files())
    def test_gen_read_attr_frames(self, fp):
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            neu_phy_locs = json.load(f)

        for neu_phy_loc in neu_phy_locs.values():
            # read mode: one-by-one
            tframe3 = PAIBoxRuntime.gen_read_neuron_attrs_frames(
                neu_phy_loc, reading_mode="onebyone"
            )

            n_neuron = get_n_neuron_from_phy_loc(neu_phy_loc)
            assert len(tframe3) == n_neuron

            # read mode: contiguous
            tframe3_2 = PAIBoxRuntime.gen_read_neuron_attrs_frames(
                neu_phy_loc, reading_mode="contiguous"
            )
            n_itf = 0
            for chip_loc in neu_phy_loc.values():
                for core_loc in chip_loc.values():
                    if core_loc["interval"] > 1 or core_loc["n_neuron"] == 1:
                        n_itf += 1
                    else:
                        n_itf += 2

            assert len(tframe3_2) == n_itf

    @pytest.mark.skipif(
        plib_version < f"{REQUIRED_PLIB_VERSION}",
        reason=f"requires paicorelib >= {REQUIRED_PLIB_VERSION}",
    )
    def test_decode_voltage_onebyone1(self, monkeypatch):
        fp = TEST_CONF_DIR / "neuron_phy_loc1.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            neu_phy_locs = json.load(f)
            assert len(neu_phy_locs) == 1

        n_neuron = 100
        interval = 8
        core_coords = [Coord(0, 0), Coord(0, 1)]
        expected_v = np.random.randint(-500, 500, size=(n_neuron,), dtype=np.int32)
        supposed_addr = [
            interval * i for i in range(n_neuron // len(core_coords))
        ] * len(core_coords)

        otframe3: list[OfflineTestOutFrame3] = []
        for i, (v, addr) in enumerate(zip(expected_v, supposed_addr)):
            core_coord = core_coords[i // 50]
            monkeypatch.setitem(self.neu_attrs, "voltage", v)

            otframe3.append(
                OfflineFrameGen.gen_testout_frame3(
                    Coord(1, 1),
                    core_coord,
                    RId(0, 0),
                    addr,
                    1,
                    attrs=self.neu_attrs,
                    dest_info=self.dest_info,
                    repeat=1,
                )
            )

        # Shuffle the order of the test out frames
        shuffled = _shuffle_otframe3(otframe3)

        for neu_phy_loc in neu_phy_locs.values():
            decoded_v = PAIBoxRuntime.decode_voltage(
                neu_phy_loc, *[f.value for f in shuffled], reading_mode="onebyone"
            )

            assert np.array_equal(decoded_v, expected_v)

    @pytest.mark.skipif(
        plib_version < f"{REQUIRED_PLIB_VERSION}",
        reason=f"requires paicorelib >= {REQUIRED_PLIB_VERSION}",
    )
    def test_decode_voltage_onebyone2(self, monkeypatch):
        fp = TEST_CONF_DIR / "neuron_phy_loc2.json"
        file_not_exist_fail(fp)

        with open(fp, "r") as f:
            neu_phy_locs = json.load(f)
            assert len(neu_phy_locs) == 1

        n_neuron = 100
        interval = 16
        core_coords = [Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1)]
        expected_v = np.random.randint(-500, 500, size=(n_neuron,), dtype=np.int32)
        supposed_addr = [
            interval * i for i in range(n_neuron // len(core_coords))
        ] * len(core_coords)

        otframe3: list[OfflineTestOutFrame3] = []
        for i, (v, addr) in enumerate(zip(expected_v, supposed_addr)):
            core_coord = core_coords[i // 25]
            monkeypatch.setitem(self.neu_attrs, "voltage", v)

            otframe3.append(
                OfflineFrameGen.gen_testout_frame3(
                    Coord(1, 1),
                    core_coord,
                    RId(0, 0),
                    addr,
                    1,
                    attrs=self.neu_attrs,
                    dest_info=self.dest_info,
                    repeat=1,
                )
            )

        # Shuffle the order of the test out frames
        shuffled = _shuffle_otframe3(otframe3)

        for neu_phy_loc in neu_phy_locs.values():
            decoded_v = PAIBoxRuntime.decode_voltage(
                neu_phy_loc, *[f.value for f in shuffled], reading_mode="onebyone"
            )

            assert np.array_equal(decoded_v, expected_v)

    """Use real network to decode voltages contiguously from the actual output test frames made on the chip.
        The directory is at `runtime/test_data/real_models/model{x}`.
        The real output test frames are obtained from the chip & saved in `otf3.npz`([arr1, arr2, ...]).
    """

    @pytest.mark.skipif(
        plib_version < f"{REQUIRED_PLIB_VERSION}",
        reason=f"requires paicorelib >= {REQUIRED_PLIB_VERSION}",
    )
    @pytest.mark.parametrize("test_model_dir", get_contiguous_reading_models_dir())
    def test_decode_voltage_contiguous(self, test_model_dir: Path):
        fp = test_model_dir / "neuron_phy_loc.json"
        file_not_exist_fail(fp)

        otf3_fp = test_model_dir / "otf3.npz"
        file_not_exist_fail(otf3_fp)

        with open(fp, "r") as f:
            neu_phy_locs = json.load(f)

        weight_all1 = ["model3", "model4"]

        for neu_phy_loc in neu_phy_locs.values():
            n_neuron = get_n_neuron_from_phy_loc(neu_phy_loc)

            _loaded = np.load(otf3_fp)
            otframes = list(_loaded.values())
            decoded_v = PAIBoxRuntime.decode_voltage(
                neu_phy_loc, *otframes, reading_mode="contiguous"
            )

            if test_model_dir.name in weight_all1:
                expected_v = np.arange(1, n_neuron + 1, dtype=np.int32)
            else:
                expected_v = np.arange(1, n_neuron + 1, dtype=np.int8).astype(np.int32)

            assert np.array_equal(decoded_v, expected_v)

    @pytest.mark.skipif(COMPILE_CONTIGUOUS_DECOING_MODEL == 0, reason="skip by default")
    @pytest.mark.parametrize(
        "idx, model, n_neuron, wbit_opt",
        [
            (1, Net1_one2one_8b, 64, False),
            (2, Net1_one2one_8b, 180, False),
            (3, Net2_triu_1b, 100, True),
            (4, Net2_triu_1b, 2000, True),  # lcn2
        ],
    )
    def test_compile_decode_voltage_contiguous_models(
        self, idx, model, n_neuron, wbit_opt
    ):
        pb.BACKEND_CONFIG.target_chip_addr = (0, 0)
        pb.BACKEND_CONFIG.output_chip_addr = (2, 0)

        net = model(n_neuron)
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile(weight_bit_optimization=wbit_opt)
        mapper.export(
            fp=TEST_CONF_DIR / "real_models" / f"model{idx}", read_voltage=net.n1
        )
