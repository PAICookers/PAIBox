import numpy as np
import pytest

from paibox.libpaicore.v2.frame.base import (
    Frame,
    FrameFactory,
    FramePackage,
)
from paibox.libpaicore.v2.frame.params import FrameFormat as FF, FrameHeader as FH
from paibox.libpaicore.v2.frame.utils import print_frame
from paibox.libpaicore import Coord, ReplicationId as RId


class TestFrameBasicObj:
    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload",
        [
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(0, 0),
                np.asarray(list(range(5)), dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                np.asarray([3], dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                np.random.randint(0, 128, size=(8,), dtype=np.uint64),
            ),
            (
                FH.CONFIG_TYPE1,
                Coord(3, 4),
                Coord(1, 2),
                RId(4, 4),
                np.asarray([12, 12], dtype=np.uint64),
            ),
        ],
    )
    def test_Frame_instance(self, header, chip_coord, core_coord, rid, payload):
        frame = Frame(header, chip_coord, core_coord, rid, payload)
        print(frame)
        print_frame(frame.value)

        assert len(frame) == payload.size

    @pytest.mark.parametrize(
        "header, chip_coord, core_coord, rid, payload, packages",
        [
            (
                FH.CONFIG_TYPE1,
                Coord(1, 2),
                Coord(3, 4),
                RId(5, 5),
                4,
                np.array([1, 2, 3], dtype=np.uint64),
            )
        ],
    )
    def test_FramePackage_instance(
        self, header, chip_coord, core_coord, rid, payload, packages
    ):
        framepackage = FramePackage(
            header, chip_coord, core_coord, rid, payload, packages
        )
        print(framepackage)
        print_frame(framepackage.value)

        assert len(framepackage) == packages.size + 1


class TestFrameFactory:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (
                np.array(
                    [
                        0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001,
                        0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000010,
                    ],
                    dtype=np.uint64,
                ),
                False,
            ),
            (
                np.array(
                    [
                        0b0100_00001_00010_00001_00100_00011_00011_0000000000_0100000000_0000000010,
                        0b0100_00001_00010_00001_00100_00011_00011_0000000000_0100000000_1111111111,
                    ]
                ),
                True,
            ),
        ],
    )
    def test_is_package_format(self, value, expected):
        header0 = FH(
            (int(value[0]) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK
        )
        bit19 = (
            int(value[0]) >> FF.DATA_PACKAGE_TYPE_OFFSET
        ) & FF.DATA_PACKAGE_TYPE_MASK

        assert FrameFactory._is_package_format(header0, bit19) == False

    @pytest.mark.parametrize(
        "value",
        [
            np.array(
                [
                    0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001,
                    0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000010,
                ],
                dtype=np.uint64,
            ),
            [
                0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001,
                0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000010,
            ],
        ],
    )
    def test_Frame_decode(self, value):
        result = FrameFactory.decode(value)

        assert isinstance(result, Frame)

    def test_Frame_decode_config1(self, mock_gen_random_config1):
        test_frame = mock_gen_random_config1

        result = FrameFactory.decode(test_frame.value)

        assert isinstance(result, Frame)
        assert np.array_equal(result.value, test_frame.value)

    def test_FramePackage_decode(self, mock_gen_random_config1):
        test_frame = mock_gen_random_config1
