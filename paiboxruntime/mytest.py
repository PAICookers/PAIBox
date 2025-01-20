
import json
import time
from pathlib import Path

import numpy as np
import pytest

from runtime import PAIBoxRuntime
from paicorelib import Coord, to_coordoffset
from paicorelib import ReplicationId as RId
from paicorelib.framelib.frame_defs import FrameHeader as FH, SpikeFrameFormat as SFF
from paicorelib.framelib.utils import print_frame

class TestRuntimeDecoder:

    def test_output_more1152_gen_frame(self):
        with open("config/output_dest_more1152_info.json", "r", encoding="utf-8") as file:
            output_dest_info = json.load(file)

        n_ts = 2
        lcn = 2 #从output_shape计算
        oframe_infos = PAIBoxRuntime.gen_output_frames_more1152_info(
            lcn, n_ts, output_dest_info=output_dest_info
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
        data = PAIBoxRuntime.decode_spike(
            n_ts, output_frames, oframe_infos, flatten=False
        )
        expected = [
            np.zeros((2, 1300), dtype=np.uint8),
        ]
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
        # print(data)

    def test_output_gen_frame(self):
        with open("config/output_dest_info.json", "r", encoding="utf-8") as file:
            output_dest_info = json.load(file)

        n_ts = 4
        lcn = 1#从output_shape计算
        oframe_infos = PAIBoxRuntime.gen_output_frames_more1152_info(
            lcn, n_ts, output_dest_info=output_dest_info
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
        data = PAIBoxRuntime.decode_spike(
            n_ts, output_frames, oframe_infos, flatten=False
        )
        print(data)
        
        expected = [
            np.zeros((n_ts, 10), dtype=np.uint8),
        ]
        
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
        print(expected[0])
        assert np.array_equal(data, expected)
TestRuntimeDecoder().test_output_more1152_gen_frame()
TestRuntimeDecoder().test_output_gen_frame()

