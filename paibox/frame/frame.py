from frame_params import FrameMask
from typing import Optional, Tuple, Union, List
from frame_params import *
from coord import Coord


def Coord2Addr(coord: Coord) -> int:
    return (coord.x << 5) | coord.y


def Addr2Coord(addr: int) -> Coord:
    return Coord(addr >> 5, addr & ((1 << 5) - 1))


# 用于配置帧2型配置各个寄存器参数，tick_wait_start、test_chip_addr进行了拆分
def config_frame2_split(x: int, high: int, low: int) -> Tuple[int, int]:
    highbit = x >> (low)
    lowbit_mask = (1 << (low)) - 1
    lowbit = x & lowbit_mask
    return highbit, lowbit


# 配置帧2型 est_chip_addr拆分
def test_chip_coord_split(coord: Coord) -> Tuple[int, int]:
    addr = Coord2Addr(coord)
    high3 = (
        addr >> ConfigFrame2Mask.TEST_CHIP_ADDR_COMBINATION_OFFSET
    ) & ConfigFrame2Mask.TEST_CHIP_ADDR_HIGH3_MASK
    low7 = addr & ConfigFrame2Mask.TEST_CHIP_ADDR_LOW7_MASK

    return high3, low7


# 配置帧2型 est_chip_addr合并
def test_chip_addr_combine(high3: int, low7: int) -> Coord:
    _high3 = high3 & ConfigFrame2Mask.TEST_CHIP_ADDR_HIGH3_MASK
    _low7 = low7 & ConfigFrame2Mask.TEST_CHIP_ADDR_LOW7_MASK

    addr = (_high3 << ConfigFrame2Mask.TEST_CHIP_ADDR_COMBINATION_OFFSET) | _low7

    return Addr2Coord(addr)


class FrameGen:
    # 通用Frame生成函数
    @staticmethod
    def _GenFrame(
        header: int, chip_addr: int, core_addr: int, core_ex_addr: int, payload: int
    ) -> int:
        header = header & FrameMask.GENERAL_HEADER_MASK
        chip_addr = chip_addr & FrameMask.GENERAL_CHIP_ADDR_MASK
        core_addr = core_addr & FrameMask.GENERAL_CORE_ADDR_MASK
        core_ex_addr = core_ex_addr & FrameMask.GENERAL_CORE_EX_ADDR_MASK
        payload = payload & FrameMask.GENERAL_PAYLOAD_MASK

        res = (
            (header << FrameMask.GENERAL_HEADER_OFFSET)
            | (chip_addr << FrameMask.GENERAL_CHIP_ADDR_OFFSET)
            | (core_addr << FrameMask.GENERAL_CORE_ADDR_OFFSET)
            | (core_ex_addr << FrameMask.GENERAL_CORE_EX_ADDR_OFFSET)
            | (payload << FrameMask.GENERAL_PAYLOAD_OFFSET)
        )

        return res

    # 配置帧
    @staticmethod
    def GenConfigFrame(
        header: FrameHead,
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: Coord,
        payload: int = None,
        parameter_reg: dict = None,
    ) -> int:
        chip_addr = Coord2Addr(chip_coord)
        core_addr = Coord2Addr(core_coord)
        core_ex_addr = Coord2Addr(core_ex_coord)

        # 配置帧1型
        if header == FrameHead.CONFIG_TYPE1.value:
            if payload is None:
                raise ValueError("payload is None")
            if parameter_reg is not None:
                raise ValueError("parameter_reg is not need")

            return FrameGen._GenFrame(
                header, chip_addr, core_addr, core_ex_addr, payload
            )

        # 配置帧2型
        elif header == FrameHead.CONFIG_TYPE2.value:
            if payload is not None:
                raise ValueError("payload is not need")
            if parameter_reg is None:
                raise ValueError("parameter_reg is None")

        # 配置帧3型
        elif header == FrameHead.CONFIG_TYPE3.value:
            pass
        # 配置帧4型
        elif header == FrameHead.CONFIG_TYPE4.value:
            pass

    # 配置帧2型
    def GenConfigFrame2(
        header: FrameHead,
        chip_coord: Coord,
        core_coord: Coord,
        core_star_coord: Coord,
        test_chip_coord: Coord,
        weight_width,
        lcn,
        input_width,
        spike_width,
        neuron_num,
        pool_max,
        tick_wait_start,
        tick_wait_end,
        snn_en,
        target_lcn,
        test_chip_addr,
    ) -> Tuple[int, ...]:
        ConfigFrameGroup: List[int] = []
        high3, low7 = test_chip_coord_split(test_chip_coord)


if __name__ == "__main__":
    # head = FrameHead.CONFIG_TYPE1.value
    # chip_coord = Coord(1,2)
    # core_coord = Coord(3,4)
    # core_ex_coord = Coord(5,6)
    # payload = 1
    # x = FrameGen.GenConfigFrame(head, chip_coord, core_coord, core_ex_coord, payload)
    x = config_frame2_split(0b0111, 2, 2)
    print(x)
