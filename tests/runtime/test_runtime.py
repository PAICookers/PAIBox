import itertools
import json
import timeit
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    ChipCoord,
    Coord,
    OffCoreCfg,
    OnCoreCfg,
)
from paicorelib import ReplicationId as RId
from paicorelib import __version__ as plib_ver
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineWorkFrame1Format as Off_WF1F
from paicorelib.framelib.frame_defs import OnlineWorkFrame1Format_1 as On_WF1_1F
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import OfflineTestOutFrame3
from paicorelib.framelib.types import FRAME_DTYPE, PAYLOAD_DATA_DTYPE
from paicorelib.framelib.utils import print_frame

from tests.utils import (
    file_not_exist_fail,
    skip_if_in_ci_env,
    skip_if_version_less_than,
)

try:
    from paibox.runtime import PAIBoxRuntime
except ImportError:
    pytestmark = pytest.mark.skip(reason="Skip if runtime module import failed")

from paibox.runtime.runtime import (
    LENGTH_EX_MULTIPLE_KEY,
    VOLTAGE_DTYPE,
    get_length_ex_onode,
)

TEST_DATA_CFG_DIR = Path(__file__).parent / "test_data"


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
        fp = TEST_DATA_CFG_DIR / "input_proj_info1.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            input_proj_info = json.load(f)

        n_input_node = len(input_proj_info.keys())
        assert n_input_node == 2

        n_ts = 8
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, input_proj_info=input_proj_info
        )

        assert len(common_part) == 2
        assert common_part[0].size == 64 * n_ts
        assert common_part[1].size == 32 * n_ts

        # Check mismatch length
        with pytest.raises(ValueError, match="length"):
            n_ts = 127
            common_part = PAIBoxRuntime.gen_input_frames_info(
                n_ts, input_proj_info=input_proj_info
            )

        with pytest.raises(ValueError, match="tick_relative"):
            n_ts = 128
            common_part = PAIBoxRuntime.gen_input_frames_info(
                n_ts, input_proj_info=input_proj_info
            )

    def test_gen_input_frames_info_by_kwds(self):
        n_ts = 16
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, (0, 0), 33, RId(0, 0), [0] * 8 + [1] * 8, list(range(16))
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
        # 0 is not encoded in frames
        assert np.array_equal(axons_in_spike, [1, 2, 3, 4, 5, 6, 7] * n_ts)

    def test_decode_by_dict(self, fixed_rng):
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                }
            },
            "n3_1": {
                "5": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 1,  # y increases
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                }
            },
            "n4_1": {
                "6": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 2,  # y increases
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                }
            },
        }
        n_ts = 2
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                # Output to core (0,0) & (0,1) on chip (1,0)
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000001_00000000_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000100_00000000_00001001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000111_00000000_00001010,
                # Output to core (0,2) on chip (1,0) decode with `strict=False`
                0b1000_00001_00000_00000_00010_00000_00000_000_00000000101_00000000_00000011,
                0b1000_00001_00000_00000_00010_00000_00000_000_00000000100_00000001_00000001,
                0b1000_00001_00000_00000_00010_00000_00000_000_00000000101_00000001_00001011,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_info, flatten=False)

        expected = np.zeros((len(oframe_info), n_ts, 8), dtype=PAYLOAD_DATA_DTYPE)
        # [onode][ts][axon]
        has_data_pos = {
            (0, 0, 3): 7,
            (0, 0, 5): 8,
            (0, 1, 0): 1,
            (1, 0, 1): 2,
            (1, 0, 4): 9,
            (1, 0, 7): 10,
            (2, 0, 5): 3,
            (2, 1, 4): 1,
            (2, 1, 5): 11,
        }
        for pos, val in has_data_pos.items():
            expected[pos] = val

        assert np.array_equal(data, expected)

    def test_decode_by_dict2(self, fixed_rng):
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
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_info, flatten=False)

        expected = np.zeros((len(oframe_info), n_ts, 8), dtype=PAYLOAD_DATA_DTYPE)
        expected[0][0][3] = 7
        expected[0][0][5] = 8
        expected[0][1][0] = 1

        assert np.array_equal(data, expected)

    def test_decode_by_kwds(self, fixed_rng):
        n_axon_max = 100
        n_ts_max = 32
        assert n_ts_max <= OffCoreCfg.N_TIMESLOT_MAX

        for n_axon, n_ts in itertools.product(range(1, n_axon_max), range(1, n_ts_max)):
            oframe_info = PAIBoxRuntime.gen_output_frames_info(
                n_ts, ChipCoord(1, 0), Coord(0, 0), RId(0, 0), list(range(n_axon))
            )

            n_chosen = 1 + fixed_rng.integers(n_axon * n_ts)
            choice_idx = fixed_rng.choice(range(n_axon * n_ts), n_chosen, replace=False)

            # choice_idx = [1, 0, 2]
            random = fixed_rng.integers(
                np.iinfo(PAYLOAD_DATA_DTYPE).max,
                size=(n_axon * n_ts,),
                dtype=PAYLOAD_DATA_DTYPE,
            )
            output_frames = oframe_info + random
            shuffle_frame = output_frames[choice_idx]

            expected = np.zeros((n_axon * n_ts,), dtype=PAYLOAD_DATA_DTYPE)
            expected[choice_idx] = random[choice_idx]

            data = PAIBoxRuntime.decode(n_ts, shuffle_frame, oframe_info, flatten=True)

            assert np.array_equal(data, expected)

    @skip_if_in_ci_env()
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
        test_frames = np.zeros((n_axons,), dtype=FRAME_DTYPE)

        for i in range(n_axons):
            _data = np.random.randint(0, 256, dtype=np.uint8)
            test_frames[i] = (
                (FH.WORK_TYPE1 << Off_WF1F.GENERAL_HEADER_OFFSET)
                | (Coord(1, 0).address << Off_WF1F.GENERAL_CHIP_ADDR_OFFSET)
                | (i << Off_WF1F.AXON_OFFSET)
                | FRAME_DTYPE(_data)
            )

        t = timeit.timeit(
            lambda: PAIBoxRuntime.decode(n_ts, test_frames, oframe_info),
            number=100,
        )
        print(f"n_axons: {n_axons}, n_ts: {n_ts}, time: {t/100:.5f}s")

    def test_gen_output_frames_info_by_dict1(self):
        fp = TEST_DATA_CFG_DIR / "output_dest_info1.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 1

        n_ts = 4
        common_part = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 800 * n_ts

        with pytest.raises(ValueError):
            common_part = PAIBoxRuntime.gen_output_frames_info(
                OffCoreCfg.N_TIMESLOT_MAX, output_dest_info=output_proj_info
            )

    def test_gen_output_frames_info_by_dict2(self):
        fp = TEST_DATA_CFG_DIR / "output_dest_info2.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
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

    def test_gen_output_frames_info(self, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "output_dest_info.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
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
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)

        expected = np.zeros((len(oframe_infos), n_ts, 10), dtype=PAYLOAD_DATA_DTYPE)
        # [onode][ts][axon]
        has_data_pos = {
            (0, 0, 0): 1,
            (0, 1, 0): 7,
            (0, 0, 2): 8,
            (0, 0, 3): 3,
            (0, 1, 3): 23,
            (0, 1, 9): 24,
            (0, 2, 0): 1,
            (0, 2, 2): 2,
            (0, 2, 3): 3,
            (0, 3, 0): 1,
        }
        for pos, val in has_data_pos.items():
            expected[pos] = val

        assert np.array_equal(data, expected)

    def test_gen_output_frames_info_more1152(self, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "output_dest_info_more1152.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_dest_info = json.load(f)

        n_ts = 2
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                # Output to core (0,0) on chip (1,0)
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000000_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000000_00001000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000001_00010111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000000_00011000,
                # At ts=2
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000010_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000010_00000010_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000010_00000011,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000011_00000001,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)

        expected = np.zeros((len(oframe_infos), n_ts, 1300), dtype=PAYLOAD_DATA_DTYPE)
        # [onode][ts][axon]
        has_data_pos = {
            (0, 0, 0): 1,
            (0, 0, 1): 24,
            (0, 0, 2): 8,
            (0, 0, 3): 3,
            (0, 0, 1152): 7,
            (0, 0, 1155): 23,
            (0, 1, 1): 1,
            (0, 1, 2): 2,
            (0, 1, 3): 3,
            (0, 1, 1152): 1,
        }
        for pos, val in has_data_pos.items():
            expected[pos] = val

        assert np.array_equal(data, expected)

    def test_gen_output_frames_info_more1152_multi_onodes(self, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "output_dest_info_more1152_multi.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                # Output to core (0,0)
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
                # Output to core (0,1)
                0b1000_00001_00000_00000_00001_00000_00000_000_00000001000_00000001_00000011,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000100_00000000_00000001,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000100000_00000011_00001001,
                0b1000_00001_00000_00000_00001_00000_00000_000_01000000000_00000000_10000000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000010011_00000010_00000110,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(n_ts, output_frames, oframe_infos, flatten=False)

        expected_o1 = np.zeros((n_ts, 1300), dtype=PAYLOAD_DATA_DTYPE)
        expected_o2 = np.zeros((n_ts, 1200), dtype=PAYLOAD_DATA_DTYPE)

        has_data_pos_o1 = {
            (0, 0): 1,
            (0, 1): 24,
            (0, 2): 8,
            (0, 3): 3,
            (0, 1152): 7,
            (0, 1155): 23,
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 3,
            (1, 1152): 1,
        }
        has_data_pos_o2 = {
            (0, 1 << 9): 1 << 7,
            (0, 4): 1,
            (1, 8): 3,
            (2, 19): 6,
            (3, 32): 9,
        }

        for pos, val in has_data_pos_o1.items():
            expected_o1[pos] = val

        for pos, val in has_data_pos_o2.items():
            expected_o2[pos] = val

        assert np.array_equal(data[0], expected_o1)
        assert np.array_equal(data[1], expected_o2)

    def test_decode_zero_oframes(self):
        # Even if zero output frames are given, it should be decoded correctly.
        fp = TEST_DATA_CFG_DIR / "output_dest_info_more1152.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        zero_oframes = np.array([], dtype=FRAME_DTYPE)
        data = PAIBoxRuntime.decode(n_ts, zero_oframes, oframe_infos, flatten=False)

        assert all(d.all() == 0 for d in data)


REQUIRED_PLIB_VER_ONLINE_SUPPORT = "1.5.0b1"  # Online cores support


@skip_if_version_less_than("paicorelib", REQUIRED_PLIB_VER_ONLINE_SUPPORT)
class TestRuntimeOnline:
    def test_gen_input_frames_info_by_dict(self):
        fp = TEST_DATA_CFG_DIR / "input_proj_info_online.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            input_proj_info = json.load(f)

        n_input_node = len(input_proj_info.keys())
        assert n_input_node == 1

        n_ts = 2
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts, input_proj_info=input_proj_info, is_online=True
        )

        assert len(common_part) == 1
        assert common_part[0].size == 1200 * n_ts

        # Check mismatch length
        with pytest.raises(ValueError, match="length"):
            n_ts = 3
            common_part = PAIBoxRuntime.gen_input_frames_info(
                n_ts, input_proj_info=input_proj_info, is_online=True
            )

        # Check lcn * ts
        with pytest.raises(ValueError, match="timeslot"):
            n_ts = 4
            common_part = PAIBoxRuntime.gen_input_frames_info(
                n_ts, input_proj_info=input_proj_info, is_online=True
            )

    def test_gen_input_frames_info_by_kwds(self):
        n_ts = 3
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts,
            (0, 0),
            33,
            RId(0, 0),
            [0] * 8 + [1] * 8,  # mas ts = 2
            list(range(16)),
            is_online=True,
        )
        print_frame(common_part)

        assert len(common_part) == 16 * n_ts

    def test_encode(self):
        data = list(range(8))
        n_ts = 3
        common_part = PAIBoxRuntime.gen_input_frames_info(
            n_ts,
            Coord(0, 0),
            Coord(1, 0),
            RId(0, 0),
            [0] * 4 + [1] * 4,  # max ts = 2
            list(range(8)),
            is_online=True,
        )

        input_spike = PAIBoxRuntime.encode(data, common_part, n_ts, is_online=True)

        axons_in_spike = (input_spike >> On_WF1_1F.AXON_OFFSET) & On_WF1_1F.AXON_MASK
        # 0 is not encoded in frames
        assert np.array_equal(axons_in_spike, [1, 2, 3, 4, 5, 6, 7] * n_ts)

    def test_decode_by_dict(self, fixed_rng):
        # oframe_info `list[FrameArrayType]`, return `list[NDArray[np.uint8]]`
        output_dest_info = {
            "STDPLIF_1": {
                "(30,28)": {
                    "addr_axon": list(range(8)),
                    "tick_relative": [0] * 8,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                },
                "(30,29)": {
                    "addr_axon": list(range(8, 16)),
                    "tick_relative": [0] * 8,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                },
            },
            "STDPLIF_2": {
                "6": {
                    "addr_axon": list(range(16)),
                    "tick_relative": [0] * 16,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                    "addr_core_x": 0,
                    "addr_core_y": 1,  # y increase
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                }
            },
        }
        n_ts = 2
        oframe_info = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info, is_online=True
        )

        output_frames = np.array(
            [
                # Output to core (0,0) on chip (1,0)
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000001_00000_001_00000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000100_00000_000_00000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000111_00000_001_00000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000_001_00000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000110_00000_000_00000000,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000001000_00000_001_00000000,
                # Output to core (0,1) on chip (1,0)
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000101_00000_001_00000000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000100_00000_000_00000000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000000111_00000_001_00000000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000001011_00000_001_00000000,
                0b1000_00001_00000_00000_00001_00000_00000_000_00000001111_00000_000_00000000,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(
            n_ts, output_frames, oframe_info, is_online=True, flatten=False
        )

        expected = np.zeros((len(oframe_info), n_ts, 16), dtype=PAYLOAD_DATA_DTYPE)
        # [onode][ts][axon]
        spike_pos = [
            (0, 0, 4),
            (0, 0, 6),
            (0, 1, 1),
            (0, 1, 5),
            (0, 1, 7),
            (0, 1, 8),
            (1, 0, 4),
            (1, 0, 15),
            (1, 1, 5),
            (1, 1, 7),
            (1, 1, 11),
        ]
        for pos in spike_pos:
            expected[pos] = 1

        assert np.array_equal(data, expected)

    def test_decode_by_kwds(self, fixed_rng):
        n_axon_max = 200
        n_ts_max = 8
        assert n_ts_max <= OnCoreCfg.N_TIMESLOT_MAX

        for n_axon, n_ts in itertools.product(range(1, n_axon_max), range(1, n_ts_max)):
            oframe_info = PAIBoxRuntime.gen_output_frames_info(
                n_ts,
                ChipCoord(1, 0),
                Coord(28, 28),
                RId(0, 0),
                list(range(n_axon)),
                is_online=True,
            )

            n_chosen = 1 + fixed_rng.integers(n_axon * n_ts)
            choice_idx = fixed_rng.choice(range(n_axon * n_ts), n_chosen, replace=False)
            output_frames = oframe_info + 1
            shuffle_frame = output_frames[choice_idx]

            expected = np.zeros((n_axon * n_ts,), dtype=PAYLOAD_DATA_DTYPE)
            expected[choice_idx] = 1

            data = PAIBoxRuntime.decode(
                n_ts,
                shuffle_frame,
                oframe_info,
                is_online=True,
                strict=False,
                flatten=True,
            )

            assert np.array_equal(data, expected)

    def test_gen_output_frames_info_by_dict(self):
        fp = TEST_DATA_CFG_DIR / "output_dest_info_online.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 1

        n_ts = 7
        common_part = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_proj_info, is_online=True
        )
        assert sum(part.size for part in common_part) == 200 * n_ts

        with pytest.raises(ValueError):
            common_part = PAIBoxRuntime.gen_output_frames_info(
                OnCoreCfg.N_TIMESLOT_MAX,
                output_dest_info=output_proj_info,
                is_online=True,
            )

    def test_gen_output_frames_info(self, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "output_dest_info_online.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_dest_info = json.load(f)

        n_ts = 4
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info, is_online=True
        )

        output_frames = np.array(
            [
                # Output to core (0,0) on chip (9,9)
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000000_00000_000_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000000_00000_001_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000010_00000_000_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000011_00000_000_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000011_00000_001_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000011_00000_010_00000000,
                0b1000_01001_01001_00000_00000_00000_00000_000_00000000000_00000_011_00000000,
                # Output to core (0,1) on chip (9,9), decode with `strict=False`
                0b1000_01001_01001_00000_00001_00000_00000_000_00000011111_00000_001_00000000,
                0b1000_01001_01001_00000_00001_00000_00000_000_00000001111_00000_011_00000000,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(
            n_ts, output_frames, oframe_infos, is_online=True, flatten=False
        )

        expected = np.zeros((len(oframe_infos), n_ts, 200), dtype=PAYLOAD_DATA_DTYPE)
        has_spike_pos = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 3),
            (0, 1, 0),
            (0, 1, 3),
            (0, 2, 3),
            (0, 3, 0),
        ]
        for pos in has_spike_pos:
            expected[pos] = 1

        assert np.array_equal(data, expected)

    def test_gen_output_frames_info_more1152(self, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "output_dest_info_online_more1152.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
            output_dest_info = json.load(f)

        n_ts = 2
        oframe_infos = PAIBoxRuntime.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info, is_online=True
        )

        output_frames = np.array(
            [
                # Output to core (0,0) on chip (2,0)
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000000_00000_000_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000000_00000_001_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000010_00000_000_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000011_00000_000_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000011_00000_001_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000011111_00000_001_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000001111_00000_001_00000000,
                # At ts=2
                0b1000_00010_00000_00000_00000_00000_00000_000_00000001111_00000_010_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000000001_00000_011_00000000,
                0b1000_00010_00000_00000_00000_00000_00000_000_00000010000_00000_011_00000000,
            ],
            dtype=FRAME_DTYPE,
        )
        fixed_rng.shuffle(output_frames)
        data = PAIBoxRuntime.decode(
            n_ts, output_frames, oframe_infos, is_online=True, flatten=False
        )

        expected = np.zeros((len(oframe_infos), n_ts, 1200), dtype=PAYLOAD_DATA_DTYPE)
        # [onode][ts][axon]
        has_spike_pos = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 3),
            (0, 0, 1152),
            (0, 0, 1152 + 3),
            (0, 0, 1152 + 15),
            (0, 0, 1152 + 31),
            (0, 1, 15),
            (0, 1, 1152 + 1),
            (0, 1, 1152 + 16),
        ]
        for pos in has_spike_pos:
            expected[pos] = 1

        assert np.array_equal(data, expected)


REQUIRED_PLIB_VER_V_DECODING = "1.4.1"  # Required version for neuron voltage decoding


def get_neu_phy_files() -> list[Path]:
    return list(TEST_DATA_CFG_DIR.glob("neuron_phy_loc[0-9]*.json"))


def shuffle_otf3(otf3: list[OfflineTestOutFrame3], rng: np.random.Generator):
    otf3_np = np.asarray(otf3)
    rng.shuffle(otf3_np)
    return otf3_np


def get_n_neuron_from_phy_loc(neu_phy_loc: dict[str, dict[str, Any]]) -> int:
    n_neuron = 0
    for chip_loc in neu_phy_loc.values():
        n_neuron += sum(core_loc["n_neuron"] for core_loc in chip_loc.values())

    return n_neuron


def get_contiguous_reading_models_dir() -> list[Path]:
    return list((TEST_DATA_CFG_DIR / "contiguous_reading_models").glob("model[0-9]*"))


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
        w = np.triu(np.ones((n, n), dtype=np.bool), k=0)  # w1
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
        reset_mode=RM.MODE_NORMAL,
        reset_v=0,
        leak_comparison=LCM.LEAK_BEFORE_COMP,
        thres_mask_bits=0,
        neg_thres_mode=NTM.MODE_RESET,
        neg_threshold=100,
        pos_threshold=100,
        leak_direction=LDM.MODE_FORWARD,
        leak_integration_mode=LIM.MODE_DETERMINISTIC,
        leak_v=3,
        syn_integration_mode=SIM.MODE_DETERMINISTIC,
        bit_trunc=8,
        voltage=0,  # voltage will be set
    )

    @pytest.mark.parametrize("fp", get_neu_phy_files())
    def test_gen_read_attr_frames(self, fp):
        file_not_exist_fail(fp)

        with fp.open("r") as f:
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

    @skip_if_version_less_than("paicorelib", REQUIRED_PLIB_VER_V_DECODING)
    def test_decode_voltage_onebyone1(self, monkeypatch, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "neuron_phy_loc1.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
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
        shuffled = shuffle_otf3(otframe3, fixed_rng)

        for neu_phy_loc in neu_phy_locs.values():
            decoded_v = PAIBoxRuntime.decode_voltage(
                neu_phy_loc, *[f.value for f in shuffled], reading_mode="onebyone"
            )

            assert np.array_equal(decoded_v, expected_v)

    @skip_if_version_less_than("paicorelib", REQUIRED_PLIB_VER_V_DECODING)
    def test_decode_voltage_onebyone2(self, monkeypatch, fixed_rng):
        fp = TEST_DATA_CFG_DIR / "neuron_phy_loc2.json"
        file_not_exist_fail(fp)

        with fp.open("r") as f:
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
        shuffled = shuffle_otf3(otframe3, fixed_rng)

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
        plib_ver < f"{REQUIRED_PLIB_VER_V_DECODING}",
        reason=f"requires paicorelib >= {REQUIRED_PLIB_VER_V_DECODING}",
    )
    @pytest.mark.parametrize("test_model_dir", get_contiguous_reading_models_dir())
    def test_decode_voltage_contiguous(self, test_model_dir: Path):
        fp = test_model_dir / "neuron_phy_loc.json"
        file_not_exist_fail(fp)

        otf3_fp = test_model_dir / "otf3.npz"
        file_not_exist_fail(otf3_fp)

        with fp.open("r") as f:
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
                expected_v = np.arange(1, n_neuron + 1, dtype=VOLTAGE_DTYPE)
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
            fp=TEST_DATA_CFG_DIR / "real_models" / f"model{idx}", read_voltage=net.n1
        )
