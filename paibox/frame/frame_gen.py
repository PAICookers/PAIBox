from typing import Optional, Tuple, Union

import numpy as np

from ..core.coord import Coord
from .frame_params import ParameterRAMMask as RAMMask
from .frame_params import ParameterRegMask as RegMask
from .frame_params import *


def Coord2Addr(coord: Coord) -> int:
    return (coord.x << 5) | coord.y


def Addr2Coord(addr: int) -> Coord:
    return Coord(addr >> 5, addr & ((1 << 5) - 1))


def _bin_split(x: int, high: int, low: int) -> Tuple[int, int]:
    """用于配置帧2/3型配置各个参数，对需要拆分的配置进行拆分

    Args:
        x (int): 输入待拆分参数
        low (int): 低位长度

    Returns:
        Tuple[int, int]: 返回拆分后的高位和低位
    """
    high_mask = (1 << high) - 1
    highbit = x >> (low) & high_mask

    lowbit_mask = (1 << low) - 1
    lowbit = x & lowbit_mask

    return highbit, lowbit


class FrameGen:
    """Frame Generator"""

    @staticmethod
    def _GenFrame(
        header: int, chip_addr: int, core_addr: int, core_ex_addr: int, payload: int
    ) -> np.ndarray:
        # 通用Frame生成函数
        if len(bin(payload)[2:]) > 30:
            raise ValueError(
                "len of payload:{} ,payload need to be less than 30 bits".format(
                    len(bin(payload)[2:])
                )
            )

        header = header & FrameMask.GENERAL_HEADER_MASK
        chip_addr = chip_addr & FrameMask.GENERAL_CHIP_ADDR_MASK
        core_addr = core_addr & FrameMask.GENERAL_CORE_ADDR_MASK
        core_ex_addr = core_ex_addr & FrameMask.GENERAL_CORE_EX_ADDR_MASK
        payload = payload & FrameMask.GENERAL_PAYLOAD_MASK

        res = np.uint64(
            (header << FrameMask.GENERAL_HEADER_OFFSET)
            | (chip_addr << FrameMask.GENERAL_CHIP_ADDR_OFFSET)
            | (core_addr << FrameMask.GENERAL_CORE_ADDR_OFFSET)
            | (core_ex_addr << FrameMask.GENERAL_CORE_EX_ADDR_OFFSET)
            | (payload << FrameMask.GENERAL_PAYLOAD_OFFSET)
        )

        return np.array([res])

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
        weight_ram: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        
        """生成配置帧

        Args:
            header (FrameHead): 帧头
            chip_coord (Coord): chip地址
            core_coord (Coord): core地址
            core_ex_coord (Coord): core*地址
            payload (Optional[int], optional): Load参数（配置帧1型）. Defaults to None.
            parameter_reg (Optional[dict], optional): parameter_reg（配置帧2型）. Defaults to None.
            sram_start_addr (Optional[int], optional): 配置帧3型参数SRAM起始地址. Defaults to None.
            data_package_num (Optional[int], optional): 配置帧3型参数数据包数目. Defaults to None.
            neuron_ram (Optional[dict], optional): 配置帧3型参数. Defaults to None.
            weight_ram (np.ndarray, optional): 配置帧4型参数. Defaults to None.

        Returns:
            np.ndarray：配置帧
        """
        chip_addr = Coord2Addr(chip_coord)
        core_addr = Coord2Addr(core_coord)
        core_ex_addr = Coord2Addr(core_ex_coord)

        # *配置帧1型
        if header is FrameHead.CONFIG_TYPE1:
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

            return FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)


        # *配置帧2型
        elif header is FrameHead.CONFIG_TYPE2:
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

            tick_wait_start_high8, tick_wait_start_low7 = _bin_split(
                parameter_reg["tick_wait_start"], 8, 7
            )
            test_chip_addr_high3, test_chip_addr_low7 = _bin_split(
                parameter_reg["test_chip_addr"], 3, 7
            )

            # frame 1
            reg_frame1 = int(
                (
                    (parameter_reg["weight_width"] & RegMask.WEIGHT_WIDTH_MASK)
                    << RegMask.WEIGHT_WIDTH_OFFSET
                )
                | ((parameter_reg["LCN"] & RegMask.LCN_MASK) << RegMask.LCN_OFFSET)
                | (
                    (parameter_reg["input_width"] & RegMask.INPUT_WIDTH_MASK)
                    << RegMask.INPUT_WIDTH_OFFSET
                )
                | (
                    (parameter_reg["spike_width"] & RegMask.SPIKE_WIDTH_MASK)
                    << RegMask.SPIKE_WIDTH_OFFSET
                )
                | (
                    (parameter_reg["neuron_num"] & RegMask.NEURON_NUM_MASK)
                    << RegMask.NEURON_NUM_OFFSET
                )
                | (
                    (parameter_reg["pool_max"] & RegMask.POOL_MAX_MASK)
                    << RegMask.POOL_MAX_OFFSET
                )
                | (
                    (tick_wait_start_high8 & RegMask.TICK_WAIT_START_HIGH8_MASK)
                    << RegMask.TICK_WAIT_START_HIGH8_OFFSET
                )
            )
            
            ConfigFrameGroup = np.append(
                ConfigFrameGroup,
                FrameGen._GenFrame(
                    header.value, chip_addr, core_addr, core_ex_addr, reg_frame1
                )
            )
            
            # frame 2
            reg_frame2 = int(
                (
                    (tick_wait_start_low7 & RegMask.TICK_WAIT_START_LOW7_MASK)
                    << RegMask.TICK_WAIT_START_LOW7_OFFSET
                )
                | (
                    (parameter_reg["tick_wait_end"] & RegMask.TICK_WAIT_END_MASK)
                    << RegMask.TICK_WAIT_END_OFFSET
                )
                | (
                    (parameter_reg["snn_en"] & RegMask.SNN_EN_MASK)
                    << RegMask.SNN_EN_OFFSET
                )
                | (
                    (parameter_reg["targetLCN"] & RegMask.TARGET_LCN_MASK)
                    << RegMask.TARGET_LCN_OFFSET
                )
                | (
                    (test_chip_addr_high3 & RegMask.TEST_CHIP_ADDR_HIGH3_MASK)
                    << RegMask.TEST_CHIP_ADDR_HIGH3_OFFSET
                )
            )

            ConfigFrameGroup = np.append(
                ConfigFrameGroup,
                FrameGen._GenFrame(
                    header.value, chip_addr, core_addr, core_ex_addr, reg_frame2
                )
            )
            # frame 3
            reg_frame3 = int((
                test_chip_addr_low7 & RegMask.TEST_CHIP_ADDR_LOW7_MASK
            ) << RegMask.TEST_CHIP_ADDR_LOW7_OFFSET)

            ConfigFrameGroup = np.append(
                ConfigFrameGroup,
                FrameGen._GenFrame(
                    header.value, chip_addr, core_addr, core_ex_addr, reg_frame3
                )
            )
            
            return ConfigFrameGroup

        # *配置帧3型
        elif header is FrameHead.CONFIG_TYPE3:
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
                (
                    (sram_start_addr & ConfigFrame3Mask.DATA_PACKAGE_SRAM_NEURON_ADDR_MASK)
                    << ConfigFrame3Mask.DATA_PACKAGE_SRAM_NEURON_ADDR_OFFSET
                )
                | (
                    (0b0 & ConfigFrame3Mask.DATA_PACKAGE_TYPE_MASK)
                    << ConfigFrame3Mask.DATA_PACKAGE_TYPE_OFFSET
                )
                | (
                    (data_package_num & ConfigFrame3Mask.DATA_PACKAGE_NUM_MASK)
                    << ConfigFrame3Mask.DATA_PACKAGE_NUM_OFFSET
                )
            )

            start_frame = FrameGen._GenFrame(
                header.value, chip_addr, core_addr, core_ex_addr, neuron_ram_load
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, start_frame)
            # ConfigFrameGroup.append(start_frame)

            leak_v_high2, leak_v_low28 = _bin_split(neuron_ram["leak_v"], 2, 28)
            threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = _bin_split(
                neuron_ram["threshold_mask_ctrl"], 4, 1
            )
            addr_core_x_high3, addr_core_x_low2 = _bin_split(
                neuron_ram["addr_core_x"], 3, 2
            )

            # 1
            ram_frame1 = int(
                (
                    (neuron_ram["vjt_pre"] & RAMMask.VJT_PRE_MASK)
                    << RAMMask.VJT_PRE_OFFSET
                )
                | (
                    (neuron_ram["bit_truncate"] & RAMMask.BIT_TRUNCATE_MASK)
                    << RAMMask.BIT_TRUNCATE_OFFSET
                )
                | (
                    (neuron_ram["weight_det_stoch"] & RAMMask.WEIGHT_DET_STOCH_MASK)
                    << RAMMask.WEIGHT_DET_STOCH_OFFSET
                )
                | (
                    (leak_v_low28 & RAMMask.LEAK_V_LOW28_MASK)
                    << RAMMask.LEAK_V_LOW28_OFFSET
                )
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, np.uint64(ram_frame1))
            
            # 2
            ram_frame2 = int(
                (
                    (leak_v_high2 & RAMMask.LEAK_V_HIGH2_MASK)
                    << RAMMask.LEAK_V_HIGH2_OFFSET
                )
                | (
                    (neuron_ram["leak_det_stoch"] & RAMMask.LEAK_DET_STOCH_MASK)
                    << RAMMask.LEAK_DET_STOCH_OFFSET
                )
                | (
                    (neuron_ram["leak_reversal_flag"] & RAMMask.LEAK_REVERSAL_FLAG_MASK)
                    << RAMMask.LEAK_REVERSAL_FLAG_OFFSET
                )
                | (
                    (neuron_ram["threshold_pos"] & RAMMask.THRESHOLD_POS_MASK)
                    << RAMMask.THRESHOLD_POS_OFFSET
                )
                | (
                    (neuron_ram["threshold_neg"] & RAMMask.THRESHOLD_NEG_MASK)
                    << RAMMask.THRESHOLD_NEG_OFFSET
                )
                | (
                    (neuron_ram["threshold_neg_mode"] & RAMMask.THRESHOLD_NEG_MODE_MASK)
                    << RAMMask.THRESHOLD_NEG_MODE_OFFSET
                )
                | (
                    (threshold_mask_ctrl_low1 & RAMMask.THRESHOLD_MASK_CTRL_LOW1_MASK)
                    << RAMMask.THRESHOLD_MASK_CTRL_LOW1_OFFSET
                )
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, np.uint64(ram_frame2))

            # 3
            ram_frame3 = int(
                (
                    (threshold_mask_ctrl_high4 & RAMMask.THRESHOLD_MASK_CTRL_HIGH4_MASK)
                    << RAMMask.THRESHOLD_MASK_CTRL_HIGH4_OFFSET
                )
                | (
                    (neuron_ram["leak_post"] & RAMMask.LEAK_POST_MASK)
                    << RAMMask.LEAK_POST_OFFSET
                )
                | (
                    (neuron_ram["reset_v"] & RAMMask.RESET_V_MASK)
                    << RAMMask.RESET_V_OFFSET
                )
                | (
                    (neuron_ram["reset_mode"] & RAMMask.RESET_MODE_MASK)
                    << RAMMask.RESET_MODE_OFFSET
                )
                | (
                    (neuron_ram["addr_chip_y"] & RAMMask.ADDR_CHIP_Y_MASK)
                    << RAMMask.ADDR_CHIP_Y_OFFSET
                )
                | (
                    (neuron_ram["addr_chip_x"] & RAMMask.ADDR_CHIP_X_MASK)
                    << RAMMask.ADDR_CHIP_X_OFFSET
                )
                | (
                    (neuron_ram["addr_core_y_ex"] & RAMMask.ADDR_CORE_Y_EX_MASK)
                    << RAMMask.ADDR_CORE_Y_EX_OFFSET
                )
                | (
                    (neuron_ram["addr_core_x_ex"] & RAMMask.ADDR_CORE_X_EX_MASK)
                    << RAMMask.ADDR_CORE_X_EX_OFFSET
                )
                | (
                    (neuron_ram["addr_core_y"] & RAMMask.ADDR_CORE_Y_MASK)
                    << RAMMask.ADDR_CORE_Y_OFFSET
                )
                | (
                    (addr_core_x_low2 & RAMMask.ADDR_CORE_X_LOW2_MASK)
                    << RAMMask.ADDR_CORE_X_LOW2_OFFSET
                )
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, np.uint64(ram_frame3))

            # 4
            ram_frame4 = int(
                (
                    (addr_core_x_high3 & RAMMask.ADDR_CORE_X_HIGH3_MASK)
                    << RAMMask.ADDR_CORE_X_HIGH3_OFFSET
                )
                | (
                    (neuron_ram["addr_axon"] & RAMMask.ADDR_AXON_MASK)
                    << RAMMask.ADDR_AXON_OFFSET
                )
                | (
                    (neuron_ram["tick_relative"] & RAMMask.TICK_RELATIVE_MASK)
                    << RAMMask.TICK_RELATIVE_OFFSET
                )
            )

            ConfigFrameGroup = np.append(ConfigFrameGroup, np.uint64(ram_frame4))

            return ConfigFrameGroup

        # *配置帧4型
        elif header is FrameHead.CONFIG_TYPE4:
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

            ConfigFrameGroup = np.array([], dtype=np.uint64)

            weight_ram_load = (
                (
                    (sram_start_addr & ConfigFrame4Mask.DATA_PACKAGE_SRAM_NEURON_ADDR_MASK)
                    << ConfigFrame4Mask.DATA_PACKAGE_SRAM_NEURON_ADDR_OFFSET
                )
                | (
                    (0b0 & ConfigFrame4Mask.DATA_PACKAGE_TYPE_MASK)
                    << ConfigFrame4Mask.DATA_PACKAGE_TYPE_OFFSET
                )
                | (
                    (data_package_num & ConfigFrame4Mask.DATA_PACKAGE_NUM_MASK)
                    << ConfigFrame4Mask.DATA_PACKAGE_NUM_OFFSET
                )
            )

            start_frame = FrameGen._GenFrame(
                header.value, chip_addr, core_addr, core_ex_addr, weight_ram_load
            )
            ConfigFrameGroup = np.concatenate(
                (ConfigFrameGroup, start_frame, weight_ram)
            )
            return ConfigFrameGroup
    
        else:
            raise ValueError(f"header is not defined: {header}")

    @staticmethod
    def GenTestFrame():
        pass
    
    @staticmethod
    def GenWorkFrame(
        header: FrameHead,
        chip_coord: Coord,
        core_coord: Optional[Coord] = None,
        core_ex_coord: Optional[Coord] = None,
        # 工作帧1型
        axon: Optional[int] = None,
        time_slot: Optional[int] = None,
        data:Optional[int] = None,
        # 工作帧2型
        time: Optional[int] = None
    ) -> np.ndarray:
        """工作帧

        Args:
            header (FrameHead): 帧头
            chip_coord (Coord): chip地址
            core_coord (Optional[Coord], optional): core地址. Defaults to None.
            core_ex_coord (Optional[Coord], optional): core*地址. Defaults to None.
            axon (Optional[int], optional): 工作帧1型参数. Defaults to None.
            time_slot (Optional[int], optional): 工作帧1型参数. Defaults to None.
            data (Optional[int], optional): 工作帧1型参数. Defaults to None.
            time (Optional[int], optional): 工作帧2型参数. Defaults to None.
        """
        chip_addr = Coord2Addr(chip_coord)
        
        if header is FrameHead.WORK_TYPE1:
            
            if core_coord is None:
                raise ValueError("core_coord is None")
            if core_ex_coord is None:
                raise ValueError("core_ex_coord is None")
            
            if axon is None:
                raise ValueError("axon is None")
            if time_slot is None:
                raise ValueError("time_slot is None")
            if data is None:
                raise ValueError("data is None")
            
            if time is not None:
                raise ValueError("time is not need")
            
            core_addr = Coord2Addr(core_coord)
            core_ex_addr = Coord2Addr(core_ex_coord)
            
            payload =  (
                ((0x00 & WorkFrame1Mask.RESERVED_MASK) << WorkFrame1Mask.RESERVED_OFFSET)
            |   ((axon & WorkFrame1Mask.AXON_MASK) << WorkFrame1Mask.AXON_OFFSET)
            |   ((time_slot & WorkFrame1Mask.TIME_SLOT_MASK) << WorkFrame1Mask.TIME_SLOT_OFFSET)
            |   ((data & WorkFrame1Mask.DATA_MASK) << WorkFrame1Mask.DATA_OFFSET)
            )
            
            return FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)
            
        elif header is FrameHead.WORK_TYPE2:
            
            if core_coord is not None:
                raise ValueError("core_coord is not need")
            if core_ex_coord is not None:
                raise ValueError("core_ex_coord is not need")
            
            if time is None:
                raise ValueError("time is None")
            
            if axon is not None:
                raise ValueError("axon is not need")
            if time_slot is not None:
                raise ValueError("time_slot is not need")
            if data is not None:
                raise ValueError("data is not need")
            
            payload = ((time & WorkFrame2Mask.TIME_MASK) << WorkFrame2Mask.TIME_OFFSET)
            core_addr = ((0x000 & WorkFrame2Mask.GENERAL_CORE_ADDR_MASK) << WorkFrame2Mask.GENERAL_CORE_ADDR_OFFSET)
            core_ex_addr = ((0x000 & WorkFrame2Mask.GENERAL_CORE_EX_ADDR_MASK) << WorkFrame2Mask.GENERAL_CORE_EX_ADDR_OFFSET)
            
            return FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)
            
        elif header is FrameHead.WORK_TYPE3:
            
            if core_coord is not None:
                raise ValueError("core_coord is not need")
            if core_ex_coord is not None:
                raise ValueError("core_ex_coord is not need")
            if time is not None:
                raise ValueError("time is not need")
            if axon is not None:
                raise ValueError("axon is not need")
            if time_slot is not None:
                raise ValueError("time_slot is not need")
            if data is not None:
                raise ValueError("data is not need")
            
            core_addr = ((0x000 & WorkFrame3Mask.GENERAL_CORE_ADDR_MASK) << WorkFrame3Mask.GENERAL_CORE_ADDR_OFFSET)
            core_ex_addr = ((0x000 & WorkFrame3Mask.GENERAL_CORE_EX_ADDR_MASK) << WorkFrame3Mask.GENERAL_CORE_EX_ADDR_OFFSET)
            payload = (0x00000000 & WorkFrame3Mask.GENERAL_PAYLOAD_MASK) << WorkFrame3Mask.GENERAL_PAYLOAD_OFFSET
            
            return FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)
        
        elif header is FrameHead.WORK_TYPE4:
            
            if core_coord is not None:
                raise ValueError("core_coord is not need")
            if core_ex_coord is not None:
                raise ValueError("core_ex_coord is not need")
            if time is not None:
                raise ValueError("time is not need")
            if axon is not None:
                raise ValueError("axon is not need")
            if time_slot is not None:
                raise ValueError("time_slot is not need")
            if data is not None:
                raise ValueError("data is not need")
            
            core_addr = ((0x000 & WorkFrame4Mask.GENERAL_CORE_ADDR_MASK) << WorkFrame4Mask.GENERAL_CORE_ADDR_OFFSET)
            core_ex_addr = ((0x000 & WorkFrame4Mask.GENERAL_CORE_EX_ADDR_MASK) << WorkFrame4Mask.GENERAL_CORE_EX_ADDR_OFFSET)
            payload = (0x00000000 & WorkFrame4Mask.GENERAL_PAYLOAD_MASK) << WorkFrame4Mask.GENERAL_PAYLOAD_OFFSET
            
            return FrameGen._GenFrame(header.value, chip_addr, core_addr, core_ex_addr, payload)
        
        else:
            raise ValueError(f"header is not defined: {header}")
