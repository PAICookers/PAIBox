import numpy as np
import pytest

from paibox.frame.base_frame import *
from paibox.frame.params import *
from paibox.frame.util import print_frame
from paibox.libpaicore.v2 import Coord, ReplicationId


@pytest.mark.parametrize(
    "header,chip_coord,core_coord,core_e_coord,payload",
    [
        (
            FrameHead.CONFIG_TYPE1,
            Coord(1, 2),
            Coord(3, 4),
            ReplicationId(5, 5),
            [1, 2, 3, 4],
        ),
        (FrameHead.CONFIG_TYPE1, Coord(1, 2), Coord(3, 4), ReplicationId(5, 5), [3]),
        (
            FrameHead.CONFIG_TYPE1,
            [Coord(1, 2)],
            [Coord(3, 4)],
            [ReplicationId(5, 5)],
            np.array([32]),
        ),
        (
            FrameHead.CONFIG_TYPE1,
            [Coord(1, 2), Coord(3, 4)],
            [Coord(3, 4), Coord(1, 2)],
            [ReplicationId(5, 5), ReplicationId(4, 4)],
            [12, 12],
        ),
    ],
    ids=["more_load_1", "one_load_1", "one_load_2", "more_load_2"],
)
def test_Frame(header, chip_coord, core_coord, core_e_coord, payload):
    frame = Frame(
        header=header,
        chip_coord=chip_coord,
        core_coord=core_coord,
        core_ex_coord=core_e_coord,
        payload=payload,
    )
    print(frame)
    print_frame(frame.value)
    # print(frame.value)

@pytest.mark.parametrize(
    "value",
    [
        (
            np.array([
                0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001,
                0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000010,
            ]).astype(np.uint64)
        ),
        
    ],
    ids=["one_load_1"],
)
def test_Frame_by_value(value):
    frame = Frame(
        value = value
    )
    print(frame)
    print_frame(frame.value)
# @pytest.mark.parametrize(
#     "value",
#     [
#         (
#             np.array([0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001,
#                       0b0100_00001_00010_00001_00100_00011_00011_0000000000_0000000000_0000000001])
#         ),
        
#     ],
#     ids=["one_load_1"],
# )
# def test_Frame2(value):
#     frame = Frame.get_frame_by_value(value)
#     print(frame)
#     for f,v in zip(frame.value,value):
#         assert f == v
#     # print(frame.value)
    
    


@pytest.mark.parametrize(
    "header,chip_coord,core_coord,core_ex_coord,payload,data",
    [
        (
            FrameHead.CONFIG_TYPE1,
            Coord(1, 2),
            Coord(3, 4),
            ReplicationId(5, 5),
            4,
            [1, 2, 3],
        )
    ],
)
def test_FramePackage(header, chip_coord, core_coord, core_ex_coord, payload, data):
    framepackage = FramePackage(
        header=header, chip_coord=chip_coord, core_coord=core_coord,core_ex_coord=core_ex_coord, payload=payload ,data_package=data
    )
    print(framepackage)
    print_frame(framepackage.value)


@pytest.mark.parametrize(
    "value",
    [
        (
            np.array([38390725162827780,1,2,3]).astype(np.uint64)
        )
    ]
    
)
def test_FramePackage_by_value(value):
    framepackage = FramePackage(value = value)
    print(framepackage)
    print_frame(framepackage.value)