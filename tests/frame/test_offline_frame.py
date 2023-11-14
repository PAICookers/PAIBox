from paibox.frame.base_frame import *
from paibox.frame.offline_frame import OfflineConfigFrame1, OfflineConfigFrame2, OfflineConfigFrame3
from paibox.frame.params import *
from paibox.frame.util import print_frame
from paibox.libpaicore.v2 import Coord, ReplicationId

import numpy as np
import pytest
@pytest.mark.parametrize(
    "chip_coord,core_coord,core_e_coord,random_seed",
    [
        (
            Coord(1, 2),
            Coord(3,4),
            ReplicationId(5, 5),
            0b1111_1001_1010_1011_1100_1101_1110_011111_0000_0001_0010_0011_0100_0101_101011
        )
    ]
)
def test_OfflineConfigFrame1(chip_coord,core_coord,core_e_coord,random_seed):
    frames = OfflineConfigFrame1(chip_coord,core_coord,core_e_coord,random_seed)
    print(frames)
    print_frame(frames.value)

@pytest.mark.parametrize(
    "chip_coord,core_coord,core_e_coord,parameter_reg",
    [
        (
            Coord(1, 2),
            Coord(3, 4),
            ReplicationId(5, 5),
            {
                "weight_width":       2,
                "LCN":                4,
                "input_width":        1,
                "spike_width":        1,
                "neuron_num":         13,
                "pool_max":           1,
                "tick_wait_start":    15,
                "tick_wait_end":      15,
                "snn_en":             1,
                "targetLCN":          4,
                "test_chip_addr":     10
            }
        )
    ]
)
def test_OfflineConfigFrame2(chip_coord,core_coord,core_e_coord,parameter_reg):
    frames = OfflineConfigFrame2(chip_coord,core_coord,core_e_coord,parameter_reg)
    print(frames)
    print_frame(frames.value)
    
    
@pytest.mark.parametrize(
    "chip_coord,core_coord,core_e_coord,sram_start_addr,neuron_ram,neuron_num",
    [
        (
            Coord(1, 2),
            Coord(3, 4),
            ReplicationId(5, 5),
            np.uint64(4),
            {
                "tick_relative": [1,2],
                "addr_axon": [2],
                "addr_core_x": [3],
                "addr_core_y": [4],
                "addr_core_x_ex": [5],
                "addr_core_y_ex": [6],
                "addr_chip_x": [7],
                "addr_chip_y": [8],
                "reset_mode": [2],
                "reset_v": [30],
                "leak_post": [1],
                "threshold_mask_ctrl": [5,1],
                "threshold_neg_mode": [1],
                "threshold_neg": [29],
                "threshold_pos": [30],
                "leak_reversal_flag": [1],
                "leak_det_stoch": [1],
                "leak_v": [30],
                "weight_det_stoch": [1],
                "bit_truncate": [5],
                "vjt_pre": [40],
            },
            2
        )
    ],
)
def test_OfflineConfigFrame3(
    chip_coord, core_coord, core_e_coord, sram_start_addr, neuron_ram, neuron_num
):
    frame = OfflineConfigFrame3(
        chip_coord,
        core_coord,
        core_e_coord,
        sram_start_addr,
        neuron_ram,
        neuron_num
    )
    print(frame)
