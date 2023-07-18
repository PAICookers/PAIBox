import numpy as np
from typing import Optional, Tuple, Union, List, Type, Union
# from .frame_params import *
from core.coord import Coord
from frame.frame_params import *

def Coord2Addr(coord: Coord) -> int:
    return (coord.x << 5) | coord.y


def Addr2Coord(addr: int) -> Coord:
    return Coord(addr >> 5, addr & ((1 << 5) - 1))


def _bin_split(x: int, high:int ,low: int) -> Tuple[int, int]:
    """用于配置帧2\3型配置各个参数，对需要拆分的配置进行拆分

    Args:
        x (int): 输入待拆分参数
        low (int): 低位长度

    Returns:
        Tuple[int, int]: 返回拆分后的高位和低位
    """
    high_mask = (1 << high) -1
    highbit = x >> (low) & high_mask
    lowbit_mask = (1 << low) - 1
    lowbit = x & lowbit_mask
    return highbit, lowbit


# 配置帧2型 test_chip_addr拆分


def test_chip_coord_split(coord: Coord) -> Tuple[int, int]:
    addr = Coord2Addr(coord)
    high3 = (addr >> ParameterRegMask.TEST_CHIP_ADDR_COMBINATION_OFFSET
                & ParameterRegMask.TEST_CHIP_ADDR_HIGH3_MASK)
    low7 = addr & ParameterRegMask.TEST_CHIP_ADDR_LOW7_MASK

    return high3, low7


# 配置帧2型 est_chip_addr合并
def test_chip_addr_combine(high3: int, low7: int) -> Coord:
    _high3 = high3 & ParameterRegMask.TEST_CHIP_ADDR_HIGH3_MASK
    _low7 = low7 & ParameterRegMask.TEST_CHIP_ADDR_LOW7_MASK

    addr = (_high3 << ParameterRegMask.TEST_CHIP_ADDR_COMBINATION_OFFSET) | _low7

    return Addr2Coord(addr)


# 帧生成
class FrameGen:

    @staticmethod
    def _GenFrame(header: int, chip_addr: int, core_addr: int,
                    core_ex_addr: int, payload: int) -> int:
        # 通用Frame生成函数
        if len(bin(payload)[2:]) > 30:
            raise ValueError("len of payload:{} ,payload need to be less than 30 bits".format(len(bin(payload)[2:])))
        
        header = header & FrameMask.GENERAL_HEADER_MASK
        chip_addr = chip_addr & FrameMask.GENERAL_CHIP_ADDR_MASK
        core_addr = core_addr & FrameMask.GENERAL_CORE_ADDR_MASK
        core_ex_addr = core_ex_addr & FrameMask.GENERAL_CORE_EX_ADDR_MASK
        payload = payload & FrameMask.GENERAL_PAYLOAD_MASK

        res = ((header << FrameMask.GENERAL_HEADER_OFFSET)
                | (chip_addr << FrameMask.GENERAL_CHIP_ADDR_OFFSET)
                | (core_addr << FrameMask.GENERAL_CORE_ADDR_OFFSET)
                | (core_ex_addr << FrameMask.GENERAL_CORE_EX_ADDR_OFFSET)
                | (payload << FrameMask.GENERAL_PAYLOAD_OFFSET))

        return res

    @staticmethod
    def GenConfigFrame(
            header: FrameHead,
            chip_coord: Coord,
            core_coord: Coord,
            core_ex_coord: Coord,
            # 配置帧1型
            payload: Optional[int] = None,
            # 配置帧2型
            parameter_reg: Optional[dict] = None,
            # 配置帧3\4型
            sram_start_addr: Optional[int] = None,
            data_package_num: Optional[int] = None,
            # 配置帧3型
            neuron_ram: Optional[dict] = None,
            # 配置帧4型
            weight_ram: Optional[np.ndarray] = None
    ) -> Union[None, np.ndarray]:
        """生成配置帧

        Args:
            header (FrameHead): 帧头
            chip_coord (Coord): chip地址
            core_coord (Coord): core地址
            core_ex_coord (Coord): core*地址
            payload (int, optional): Load参数（配置帧1型）. Defaults to None.
            parameter_reg (dict, optional): parameter_reg（配置帧2型）. Defaults to None.
            sram_start_addr (int, optional): 配置帧3型参数SRAM起始地址. Defaults to None.
            data_package_num (int, optional): 配置帧3型参数数据包数目. Defaults to None.
            neuron_ram (dict, optional): 配置帧3型参数. Defaults to None.
            weight_ram (np.ndarray, optional): 配置帧4型参数. Defaults to None.

        Returns:
            Type[np.array]：配置帧
        """

        chip_addr = Coord2Addr(chip_coord)
        core_addr = Coord2Addr(core_coord)
        core_ex_addr = Coord2Addr(core_ex_coord)

        # 配置帧1型
        if header == FrameHead.CONFIG_TYPE1:
            if payload is None:
                raise ValueError("payload is None")
            
            
            if parameter_reg is not None:
                raise ValueError("parameter_reg is not need")

            if sram_start_addr is not None:
                raise ValueError("sram_start_addr is not need")
            if data_package_num is not None:
                raise ValueError("data_package_num is not need")
            if neuron_ram is not None:
                raise ValueError("neuron_ram is not need")
            if weight_ram is not None:
                raise ValueError("weight_ram is not need")

            return np.array([FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)])

        # 配置帧2型
        elif header == FrameHead.CONFIG_TYPE2:

            ConfigFrameGroup = np.array([], dtype=np.uint64)

            if parameter_reg is None:
                raise ValueError("parameter_reg is None")

            if payload is not None:
                raise ValueError("payload is not need")

            if sram_start_addr is not None:
                raise ValueError("sram_start_addr is not need")
            if data_package_num is not None:
                raise ValueError("data_package_num is not need")
            if neuron_ram is not None:
                raise ValueError("neuron_ram is not need")
            if weight_ram is not None:
                raise ValueError("weight_ram is not need")

            tick_wait_start_high8, tick_wait_start_low7 = _bin_split(parameter_reg["tick_wait_start"], 8, 7)
            test_chip_addr_high3, test_chip_addr_low7 = _bin_split(parameter_reg["test_chip_addr"], 3, 7)

            # frame 1
            reg_frame1 = (
                    ((parameter_reg["weight_width"] & ParameterRegMask.WEIGHT_WIDTH_MASK) << ParameterRegMask.WEIGHT_WIDTH_OFFSET) 
                |   ((parameter_reg["LCN"] & ParameterRegMask.LCN_MASK) << ParameterRegMask.LCN_OFFSET)
                |   ((parameter_reg["input_width"] & ParameterRegMask.INPUT_WIDTH_MASK) << ParameterRegMask.INPUT_WIDTH_OFFSET)
                |   ((parameter_reg["spike_width"] & ParameterRegMask.SPIKE_WIDTH_MASK) << ParameterRegMask.SPIKE_WIDTH_OFFSET)
                |   ((parameter_reg["neuron_num"] & ParameterRegMask.NEURON_NUM_MASK) << ParameterRegMask.NEURON_NUM_OFFSET)
                |   ((parameter_reg["pool_max"] & ParameterRegMask.POOL_MAX_MASK) << ParameterRegMask.POOL_MAX_OFFSET)
                |   ((tick_wait_start_high8 & ParameterRegMask.TICK_WAIT_START_HIGH8_MASK) << ParameterRegMask.TICK_WAIT_START_HIGH8_OFFSET)
            )


            ConfigFrameGroup = np.append(ConfigFrameGroup,FrameGen._GenFrame(header.value, chip_addr, core_addr,core_ex_addr, reg_frame1))
            # frame 2
            reg_frame2 = (
                    ((tick_wait_start_low7 & ParameterRegMask.TICK_WAIT_START_LOW7_MASK) << ParameterRegMask.TICK_WAIT_START_LOW7_OFFSET)
                |   ((parameter_reg["tick_wait_end"] & ParameterRegMask.TICK_WAIT_END_MASK) << ParameterRegMask.TICK_WAIT_END_OFFSET)
                |   ((parameter_reg["snn_en"] & ParameterRegMask.SNN_EN_MASK) << ParameterRegMask.SNN_EN_OFFSET)
                |   ((parameter_reg["targetLCN"] & ParameterRegMask.TARGET_LCN_MASK) << ParameterRegMask.TARGET_LCN_OFFSET)
                |   ((test_chip_addr_high3 & ParameterRegMask.TEST_CHIP_ADDR_HIGH3_MASK) << ParameterRegMask.TEST_CHIP_ADDR_HIGH3_OFFSET)
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup,FrameGen._GenFrame(header.value, chip_addr, core_addr,core_ex_addr, reg_frame2))
            #ConfigFrameGroup.append(FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, reg_frame2))
            # frame 3
            reg_frame3 = (
                    ((test_chip_addr_low7 & ParameterRegMask.TEST_CHIP_ADDR_LOW7_MASK) << ParameterRegMask.TEST_CHIP_ADDR_LOW7_OFFSET)
            )
            
            ConfigFrameGroup = np.append(ConfigFrameGroup,FrameGen._GenFrame(header.value, chip_addr, core_addr,core_ex_addr, reg_frame3))
            #ConfigFrameGroup.append(FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, reg_frame3))

            return ConfigFrameGroup

        # 配置帧3型
        elif header == FrameHead.CONFIG_TYPE3:
            if sram_start_addr is None:
                raise ValueError("sram_start_addr is None")
            if data_package_num is None:
                raise ValueError("data_package_num is None")
            if neuron_ram is None:
                raise ValueError("neuron_ram is None")

            if payload is not None:
                raise ValueError("payload is not need")
            if parameter_reg is not None:
                raise ValueError("parameter_reg is not need")
            if weight_ram is not None:
                raise ValueError("weight_ram is not need")

            # 数据包起始帧
            ConfigFrameGroup = np.array([], dtype=np.uint64)
            neuron_ram_load = (
                    ((sram_start_addr & FrameMask.DATA_PACKAGE_SRAM_NEURON_ADDR_MASK) << FrameMask.DATA_PACKAGE_SRAM_NEURON_ADDR_OFFSET)
                |   ((0b0 & FrameMask.DATA_PACKAGE_TYPE_MASK) << FrameMask.DATA_PACKAGE_TYPE_OFFSET)
                |   ((data_package_num & FrameMask.DATA_PACKAGE_NUM_MASK) << FrameMask.DATA_PACKAGE_NUM_OFFSET)
            )
            
            start_frame = FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, neuron_ram_load)
            
            ConfigFrameGroup = np.append(ConfigFrameGroup , start_frame)
            #ConfigFrameGroup.append(start_frame)

            leak_v_high2, leak_v_low28 = _bin_split(neuron_ram["leak_v"], 2, 28)
            threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = _bin_split(neuron_ram["threshold_mask_ctrl"], 4, 1)
            addr_core_x_high3, addr_core_x_low2 = _bin_split(neuron_ram["addr_core_x"],3 , 2)
            
            # 1
            ram_frame1 = int(
                    ((neuron_ram["vjt_pre"] & ParameterRAMMask.VJT_PRE_MASK) << ParameterRAMMask.VJT_PRE_OFFSET)
                |   ((neuron_ram["bit_truncate"] & ParameterRAMMask.BIT_TRUNCATE_MASK) << ParameterRAMMask.BIT_TRUNCATE_OFFSET)
                |   ((neuron_ram["weight_det_stoch"] & ParameterRAMMask.WEIGHT_DET_STOCH_MASK) << ParameterRAMMask.WEIGHT_DET_STOCH_OFFSET)
                |   ((leak_v_low28 & ParameterRAMMask.LEAK_V_LOW28_MASK) << ParameterRAMMask.LEAK_V_LOW28_OFFSET)
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, ram_frame1) 
            #ConfigFrameGroup.append(ram_frame1)
            # 2
            ram_frame2 = int(
                    ((leak_v_high2 & ParameterRAMMask.LEAK_V_HIGH2_MASK) << ParameterRAMMask.LEAK_V_HIGH2_OFFSET)
                |   ((neuron_ram["leak_det_stoch"] & ParameterRAMMask.LEAK_DET_STOCH_MASK) << ParameterRAMMask.LEAK_DET_STOCH_OFFSET)
                |   ((neuron_ram["leak_reversal_flag"] & ParameterRAMMask.LEAK_REVERSAL_FLAG_MASK) << ParameterRAMMask.LEAK_REVERSAL_FLAG_OFFSET)
                |   ((neuron_ram["threshold_pos"] & ParameterRAMMask.THRESHOLD_POS_MASK) << ParameterRAMMask.THRESHOLD_POS_OFFSET)
                |   ((neuron_ram["threshold_neg"] & ParameterRAMMask.THRESHOLD_NEG_MASK) << ParameterRAMMask.THRESHOLD_NEG_OFFSET)
                |   ((neuron_ram["threshold_neg_mode"] & ParameterRAMMask.THRESHOLD_NEG_MODE_MASK) << ParameterRAMMask.THRESHOLD_NEG_MODE_OFFSET)
                |   ((threshold_mask_ctrl_low1 & ParameterRAMMask.THRESHOLD_MASK_CTRL_LOW1_MASK) << ParameterRAMMask.THRESHOLD_MASK_CTRL_LOW1_OFFSET)
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, ram_frame2)
            #ConfigFrameGroup.append(ram_frame2)
            # 3
            ram_frame3 = int(
                    ((threshold_mask_ctrl_high4 & ParameterRAMMask.THRESHOLD_MASK_CTRL_HIGH4_MASK) << ParameterRAMMask.THRESHOLD_MASK_CTRL_HIGH4_OFFSET)
                |   ((neuron_ram["leak_post"] & ParameterRAMMask.LEAK_POST_MASK) << ParameterRAMMask.LEAK_POST_OFFSET)
                |   ((neuron_ram["reset_v"] & ParameterRAMMask.RESET_V_MASK) << ParameterRAMMask.RESET_V_OFFSET)
                |   ((neuron_ram["reset_mode"] & ParameterRAMMask.RESET_MODE_MASK) << ParameterRAMMask.RESET_MODE_OFFSET)
                |   ((neuron_ram["addr_chip_y"] & ParameterRAMMask.ADDR_CHIP_Y_MASK) << ParameterRAMMask.ADDR_CHIP_Y_OFFSET)
                |   ((neuron_ram["addr_chip_x"] & ParameterRAMMask.ADDR_CHIP_X_MASK) << ParameterRAMMask.ADDR_CHIP_X_OFFSET)
                |   ((neuron_ram["addr_core_y_ex"] & ParameterRAMMask.ADDR_CORE_Y_EX_MASK) << ParameterRAMMask.ADDR_CORE_Y_EX_OFFSET)
                |   ((neuron_ram["addr_core_x_ex"] & ParameterRAMMask.ADDR_CORE_X_EX_MASK) << ParameterRAMMask.ADDR_CORE_X_EX_OFFSET)
                |   ((neuron_ram["addr_core_y"] & ParameterRAMMask.ADDR_CORE_Y_MASK) << ParameterRAMMask.ADDR_CORE_Y_OFFSET)
                |   ((addr_core_x_low2 & ParameterRAMMask.ADDR_CORE_X_LOW2_MASK) << ParameterRAMMask.ADDR_CORE_X_LOW2_OFFSET)
            )
            
            ConfigFrameGroup = np.append(ConfigFrameGroup, ram_frame3)
            # ConfigFrameGroup.append(ram_frame3)
            # 4
            ram_frame4 = int(
                    ((addr_core_x_high3 & ParameterRAMMask.ADDR_CORE_X_HIGH3_MASK) << ParameterRAMMask.ADDR_CORE_X_HIGH3_OFFSET)
                |   ((neuron_ram["addr_axon"] & ParameterRAMMask.ADDR_AXON_MASK) << ParameterRAMMask.ADDR_AXON_OFFSET)
                |   ((neuron_ram["tick_relative"] & ParameterRAMMask.TICK_RELATIVE_MASK) << ParameterRAMMask.TICK_RELATIVE_OFFSET)
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, ram_frame4)
            #ConfigFrameGroup.append(ram_frame4)
            
            return ConfigFrameGroup

        # 配置帧4型
        elif header == FrameHead.CONFIG_TYPE4:
            if sram_start_addr is None:
                raise ValueError("sram_start_addr is None")
            if data_package_num is None:
                raise ValueError("data_package_num is None")
            if weight_ram is None:
                raise ValueError("weight_ram is None")

            if payload is not None:
                raise ValueError("payload is not need")
            if parameter_reg is not None:
                raise ValueError("parameter_reg is not need")
            if neuron_ram is not None:
                raise ValueError("neuron_ram is not need")

            ConfigFrameGroup = np.array([], dtype=np.int64)

            weight_ram_load = (
                    ((sram_start_addr & FrameMask.DATA_PACKAGE_SRAM_NEURON_ADDR_MASK) << FrameMask.DATA_PACKAGE_SRAM_NEURON_ADDR_OFFSET)
                |   ((0b0 & FrameMask.DATA_PACKAGE_TYPE_MASK) << FrameMask.DATA_PACKAGE_TYPE_OFFSET)
                |   ((data_package_num & FrameMask.DATA_PACKAGE_NUM_MASK) << FrameMask.DATA_PACKAGE_NUM_OFFSET)
            )
            
            start_frame = FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, weight_ram_load)
            ConfigFrameGroup = np.concatenate((ConfigFrameGroup,start_frame,weight_ram))

            # ConfigFrameGroup.append(start_frame)

            return ConfigFrameGroup

if __name__ == "__main__":
    x = _bin_split(0b1011,2,2)
    print(x)
