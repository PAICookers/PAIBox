import numpy as np
import pytest

from paibox.backend.runtime.libframe.base import *
from paibox.backend.runtime.libframe.frames import (
    OfflineConfigFrame1,
    OfflineConfigFrame2,
    OfflineConfigFrame3,
    OfflineTestOutFrame1,
    OfflineTestOutFrame2,
    OfflineTestOutFrame3,
)

from paibox.backend.runtime.libframe.utils import print_frame
from paibox.libpaicore import Coord, ReplicationId

pexpect = pytest.importorskip("pexpect")

@pytest.mark.parametrize(
    "chip_coord,core_coord,core_e_coord,random_seed",
    [
        (
            Coord(0, 0),
            Coord(0, 0),
            ReplicationId(0, 0),
            0b11000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000_0001,
        )
    ],
)
def test_OfflineConfigFrame1(chip_coord, core_coord, core_e_coord, random_seed):
    frames = OfflineConfigFrame1(chip_coord, core_coord, core_e_coord, random_seed)
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
                "weight_width": 2,
                "LCN": 4,
                "input_width": 1,
                "spike_width": 1,
                "neuron_num": 13,
                "pool_max": 1,
                "tick_wait_start": 15,
                "tick_wait_end": 15,
                "snn_en": 1,
                "targetLCN": 4,
                "test_chip_addr": 10,
            },
        )
    ],
)
def test_OfflineConfigFrame2(chip_coord, core_coord, core_e_coord, parameter_reg):
    frames = OfflineConfigFrame2(chip_coord, core_coord, core_e_coord, parameter_reg)
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
                "attrs": {
                    "reset_mode": 2,
                    "reset_v": 30,
                    "leak_post": 1,
                    "threshold_mask_ctrl": 5,
                    "threshold_neg_mode": 1,
                    "threshold_neg": 29,
                    "threshold_pos": 30,
                    "leak_reversal_flag": 1,
                    "leak_det_stoch": 1,
                    "leak_v": 30,
                    "weight_det_stoch": 1,
                    "bit_truncate": 5,
                    "vjt_pre": 40,
                },
                "dest_info": {
                    "tick_relative": [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    "addr_axon": [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                        32,
                        33,
                        34,
                        35,
                        36,
                        37,
                        38,
                        39,
                        40,
                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                        51,
                        52,
                        53,
                        54,
                        55,
                        56,
                        57,
                        58,
                        59,
                        60,
                        61,
                        62,
                        63,
                        64,
                        65,
                        66,
                        67,
                        68,
                        69,
                        70,
                        71,
                        72,
                        73,
                        74,
                        75,
                        76,
                        77,
                        78,
                        79,
                        80,
                        81,
                        82,
                        83,
                        84,
                        85,
                        86,
                        87,
                        88,
                        89,
                        90,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96,
                        97,
                        98,
                        99,
                        100,
                        101,
                        102,
                        103,
                        104,
                        105,
                        106,
                        107,
                        108,
                        109,
                        110,
                        111,
                        112,
                        113,
                        114,
                        115,
                        116,
                        117,
                        118,
                        119,
                        120,
                        121,
                        122,
                        123,
                        124,
                        125,
                        126,
                        127,
                        128,
                        129,
                        130,
                        131,
                        132,
                        133,
                        134,
                        135,
                        136,
                        137,
                        138,
                        139,
                        140,
                        141,
                        142,
                        143,
                        144,
                        145,
                        146,
                        147,
                        148,
                        149,
                        150,
                        151,
                        152,
                        153,
                        154,
                        155,
                    ],
                    "addr_core_x": 3,
                    "addr_core_y": 4,
                    "addr_core_x_ex": 5,
                    "addr_core_y_ex": 6,
                    "addr_chip_x": 7,
                    "addr_chip_y": 8,
                },
            },
            2,
        )
    ],
)
def test_OfflineConfigFrame3(
    chip_coord, core_coord, core_e_coord, sram_start_addr, neuron_ram, neuron_num
):
    frame = OfflineConfigFrame3(
        chip_coord, core_coord, core_e_coord, sram_start_addr, neuron_ram, neuron_num
    )
    print(frame)
    print_frame(frame.value)


@pytest.mark.parametrize(
    "chip_coord,core_coord,core_neuron_ram",
    [
        (
            Coord(1, 2),
            Coord(3, 4),
            {
                "IF_1": {
                    "attrs": {
                        "reset_mode": 0,
                        "reset_v": -1,
                        "leak_post": 1,
                        "threshold_mask_ctrl": 0,
                        "threshold_neg_mode": 1,
                        "threshold_neg": 0,
                        "threshold_pos": 3,
                        "leak_reversal_flag": 0,
                        "leak_det_stoch": 0,
                        "leak_v": 0,
                        "weight_det_stoch": 0,
                        "bit_truncate": 0,
                        "vjt_pre": 0,
                    },
                    "dest_info": {
                        "tick_relative": [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        "addr_axon": [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                            28,
                            29,
                            30,
                            31,
                            32,
                            33,
                            34,
                            35,
                            36,
                            37,
                            38,
                            39,
                            40,
                            41,
                            42,
                            43,
                            44,
                            45,
                            46,
                            47,
                            48,
                            49,
                            50,
                            51,
                            52,
                            53,
                            54,
                            55,
                            56,
                            57,
                            58,
                            59,
                            60,
                            61,
                            62,
                            63,
                            64,
                            65,
                            66,
                            67,
                            68,
                            69,
                            70,
                            71,
                            72,
                            73,
                            74,
                            75,
                            76,
                            77,
                            78,
                            79,
                            80,
                            81,
                            82,
                            83,
                            84,
                            85,
                            86,
                            87,
                            88,
                            89,
                            90,
                            91,
                            92,
                            93,
                            94,
                            95,
                            96,
                            97,
                            98,
                            99,
                            100,
                            101,
                            102,
                            103,
                            104,
                            105,
                            106,
                            107,
                            108,
                            109,
                            110,
                            111,
                            112,
                            113,
                            114,
                            115,
                            116,
                            117,
                            118,
                            119,
                            120,
                            121,
                            122,
                            123,
                            124,
                            125,
                            126,
                            127,
                            128,
                            129,
                            130,
                            131,
                            132,
                            133,
                            134,
                            135,
                            136,
                            137,
                            138,
                            139,
                            140,
                            141,
                            142,
                            143,
                            144,
                            145,
                            146,
                            147,
                            148,
                            149,
                            150,
                            151,
                            152,
                            153,
                            154,
                            155,
                        ],
                        "addr_core_x": 0,
                        "addr_core_y": 0,
                        "addr_core_x_ex": 0,
                        "addr_core_y_ex": 1,
                        "addr_chip_x": 1,
                        "addr_chip_y": 0,
                    },
                    "addr_offset": 38,
                    "neuron_num": 277,
                },
                "IF_2": {
                    "attrs": {
                        "reset_mode": 0,
                        "reset_v": -1,
                        "leak_post": 1,
                        "threshold_mask_ctrl": 0,
                        "threshold_neg_mode": 1,
                        "threshold_neg": 0,
                        "threshold_pos": 3,
                        "leak_reversal_flag": 0,
                        "leak_det_stoch": 0,
                        "leak_v": 0,
                        "weight_det_stoch": 0,
                        "bit_truncate": 0,
                        "vjt_pre": 0,
                    },
                    "dest_info": {
                        "tick_relative": [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        "addr_axon": [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                            28,
                            29,
                            30,
                            31,
                            32,
                            33,
                            34,
                            35,
                            36,
                            37,
                            38,
                            39,
                            40,
                            41,
                            42,
                            43,
                            44,
                            45,
                            46,
                            47,
                            48,
                            49,
                            50,
                            51,
                            52,
                            53,
                            54,
                            55,
                            56,
                            57,
                            58,
                            59,
                            60,
                            61,
                            62,
                            63,
                            64,
                            65,
                            66,
                            67,
                            68,
                            69,
                            70,
                            71,
                            72,
                            73,
                            74,
                            75,
                            76,
                            77,
                            78,
                            79,
                            80,
                            81,
                            82,
                            83,
                            84,
                            85,
                            86,
                            87,
                            88,
                            89,
                            90,
                            91,
                            92,
                            93,
                            94,
                            95,
                            96,
                            97,
                            98,
                            99,
                            100,
                            101,
                            102,
                            103,
                            104,
                            105,
                            106,
                            107,
                            108,
                            109,
                            110,
                            111,
                            112,
                            113,
                            114,
                            115,
                            116,
                            117,
                            118,
                            119,
                            120,
                            121,
                            122,
                            123,
                            124,
                            125,
                            126,
                            127,
                            128,
                            129,
                            130,
                            131,
                            132,
                            133,
                            134,
                            135,
                            136,
                            137,
                            138,
                            139,
                            140,
                            141,
                            142,
                            143,
                            144,
                            145,
                            146,
                            147,
                            148,
                            149,
                            150,
                            151,
                            152,
                            153,
                            154,
                            155,
                        ],
                        "addr_core_x": 0,
                        "addr_core_y": 0,
                        "addr_core_x_ex": 0,
                        "addr_core_y_ex": 1,
                        "addr_chip_x": 1,
                        "addr_chip_y": 0,
                    },
                    "addr_offset": 38,
                    "neuron_num": 277,
                },
            },
        )
    ],
)
def test_OfflineConfigFrame3Group(chip_coord, core_coord, core_neuron_ram):
    frame = OfflineConfigFrame3Group(
        chip_coord=chip_coord, core_coord=core_coord, core_neuron_ram=core_neuron_ram
    )
    print(frame)
    print_frame(frame.value)


def test_OfflineTestOutFrame1():
    frame = OfflineTestOutFrame1(
        np.array(
            [
                0b0100_00000_00000_00000_00000_00000_00000_0000000000_0000000000_0000000000,
                0b0100_00000_00000_00000_00000_00000_00000_0000000000_0000000000_0000000000,
                0b0100_00000_00000_00000_00000_00000_00000_0111000000_0000000000_0000000000,
            ]
        )
    )
    print(bin(frame.random_seed))
    print(len(bin(frame.random_seed)) - 2)


def test_OfflineTestOutFrame2():
    frame = np.array(
        [
            0b0101_00000_00000_00000_00000_00000_00000_0000000000_0000000000_0000000000,
            0b0101_00000_00000_00000_00000_00000_00000_0000000000_0000000000_0000000000,
            0b0101_00000_00000_00000_00000_00000_00000_0111000000_0000000000_0000000000,
        ]
    )
    frame = OfflineTestOutFrame2(frame)


def test_OfflineTestOutFrame3():
    frame = np.array(
        [
            0b0110000010001000011001000010100101000000010000000000000000001000,
            0b0000000000000000000000011110100101000000000000000000000000101000,
            0b1100000000000000000000000011101000000000000000000000000111101100,
            0b1100100001010011000111010001000000000000000000000000001111010010,
            0b0000000000000000000000000000000000000000000000000100000000010000,
            0b0000000000000000000000011110100101000000000000000000000000101000,
            0b1100000000000000000000000011101000000000000000000000000111101100,
            0b1100100001010011000111010001000000000000000000000000001111010000,
            0b0000000000000000000000000000000000000000000000001000000000010000,
        ]
    ).astype(np.uint64)
    frame = OfflineTestOutFrame3(frame)
    print(frame.neuron_ram["vjt_pre"])
    print(type(frame.neuron_ram["vjt_pre"][0]))