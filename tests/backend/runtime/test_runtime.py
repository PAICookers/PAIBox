import json
import time
from pathlib import Path

import numpy as np
import pytest

from paibox.backend import RuntimeDecoder, RuntimeEncoder
from paibox.backend.runtime.libframe.utils import print_frame
from paibox.libpaicore import Coord
from paibox.libpaicore import FrameHeader as FH
from paibox.libpaicore import ReplicationId as RId
from paibox.libpaicore import SpikeFrameFormat as SFF


class TestRuntimeEncoder:
    def test_gen_input_frames_info_by_dict(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "input_proj_info1.json", "r") as f:
            input_proj_info = json.load(f)

        n_input_node = len(input_proj_info.keys())
        assert n_input_node == 2

        n_ts = 8
        common_part = RuntimeEncoder.gen_input_frames_info(
            n_ts, input_proj_info=input_proj_info
        )
        assert len(common_part) == 2

        assert common_part[0].size == 64 * n_ts
        assert common_part[1].size == 32 * n_ts

    def test_gen_input_frames_info_by_kwds(self):
        n_ts = 16
        common_part = RuntimeEncoder.gen_input_frames_info(
            n_ts, (0, 0), 33, RId(0, 0), [0] * 8 + [1] * 8, list(range(16))  # type: ignore
        )

        assert len(common_part) == 16 * n_ts

    def test_encode(self):
        data = list(range(8))
        n_ts = 16
        common_part = RuntimeEncoder.gen_input_frames_info(
            n_ts, Coord(0, 0), Coord(1, 0), RId(0, 0), [0] * 4 + [1] * 4, list(range(8))
        )

        input_spike = RuntimeEncoder.encode(data, n_ts, common_part)

        data_in_spike = (input_spike >> SFF.DATA_OFFSET) & SFF.DATA_MASK
        # Encode data with none-zero values.
        assert 0 not in data_in_spike

        axons_in_spike = (input_spike >> SFF.AXON_OFFSET) & SFF.AXON_MASK
        # Except axon with data=0
        assert np.array_equal(axons_in_spike, [1, 2, 3, 4, 5, 6, 7] * n_ts)


class TestRuntimeDecoder:
    def test_decode_spike_by_dict(self):
        # oframe_info is `List[FrameArrayType]`, return `List[NDArray[np.uint8]]`
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
                    "addr_core_x": 1,
                    "addr_core_y": 0,
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
                    "addr_core_x": 2,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
        }
        n_ts = 2
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000001_00000000_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000100_00000000_00001001,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000111_00000000_00001010,
            ],
            dtype=np.uint64,
        )
        data = RuntimeDecoder.decode_spike_less1152(
            n_ts, output_frames, oframe_info, flatten=False
        )

        expected = [
            np.array(
                [[0, 0, 0, 7, 0, 8, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
            ),
            np.array(
                [[0, 2, 0, 0, 9, 0, 0, 10], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
            ),
            np.zeros((2, 8), dtype=np.uint8),
        ]
        assert np.array_equal(data, expected)

    def test_decode_spike_by_dict2(self):
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
        oframe_info = RuntimeDecoder.gen_output_frames_info(
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
        data = RuntimeDecoder.decode_spike_less1152(
            n_ts, output_frames, oframe_info, flatten=False
        )

        expected = [
            np.array(
                [[0, 0, 0, 7, 0, 8, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
            )
        ]

        assert np.array_equal(data, expected)

    def test_decode_spike_by_kwds(self):
        # oframe_info is `FrameArrayType`, return `NDArray[np.uint8]`
        n_axon_max = 100
        n_ts_max = 64

        for n_axon in range(1, n_axon_max):
            for n_ts in range(1, n_ts_max):
                oframe_info = RuntimeDecoder.gen_output_frames_info(
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

                expected = expected.reshape(-1, n_ts).T.flatten()

                data = RuntimeDecoder.decode_spike_less1152(
                    n_ts, shuffle_frame, oframe_info, flatten=True
                )

                assert np.array_equal(data, expected)

    @pytest.mark.skip(reason="Is it useful?")
    def test_decode_spike_perf(self):
        n_axons = 1152
        n_ts = 1

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
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )
        test_frames = np.zeros((n_axons,), dtype=np.uint64)

        for i in range(n_axons):
            _data = np.random.randint(0, 256, dtype=np.uint8)
            test_frames[i] = (
                (FH.WORK_TYPE1 << SFF.GENERAL_HEADER_OFFSET)
                | (Coord(1, 0).address << SFF.GENERAL_CHIP_ADDR_OFFSET)
                | (i << SFF.AXON_OFFSET)
                | _data
            )

        t1 = time.perf_counter()
        data = RuntimeDecoder.decode_spike_less1152(n_ts, test_frames, oframe_info)
        t2 = time.perf_counter()

        print(t2 - t1)

    def test_gen_output_frames_info_by_dict1(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "output_dest_info1.json", "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 1

        common_part = RuntimeDecoder.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 800

    def test_gen_output_frames_info_by_dict2(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "output_dest_info2.json", "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 2

        common_part = RuntimeDecoder.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 104

    def test_gen_output_frames_info_by_kwds(self):
        n_ts = 16
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, (1, 0), (0, 0), (0, 0), [0, 1, 2, 3, 4, 5, 6, 7]
        )

        assert oframe_info.size == 8 * n_ts
