from .base_frame import *
from .params import *
from .params import ParameterRAMFormat as RAMMask
from .params import ParameterRegFormat as RegMask
from .util import *
from paibox.libpaicore.v2 import *
from typing import Union, List

"""Offline Config Frame"""
class OfflineConfigFrame1(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_e_coord: Union[List[ReplicationId], ReplicationId],
        random_seed: Union[List[int], int, np.ndarray],
    ):
        header = [FrameHead.CONFIG_TYPE1]
        super().__init__(header, chip_coord, core_coord, core_e_coord, random_seed)


class OfflineConfigFrame2(FrameGroup):
    def __init__(self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, parameter_reg: dict) -> None:
        header = FrameHead.CONFIG_TYPE2

        tick_wait_start_high8, tick_wait_start_low7 = bin_split(parameter_reg["tick_wait_start"], 8, 7)
        test_chip_addr_high3, test_chip_addr_low7 = bin_split(parameter_reg["test_chip_addr"], 3, 7)
        reg_frame1 = int(
            ((parameter_reg["weight_width"] & RegMask.WEIGHT_WIDTH_MASK) << RegMask.WEIGHT_WIDTH_OFFSET)
            | ((parameter_reg["LCN"] & RegMask.LCN_MASK) << RegMask.LCN_OFFSET)
            | ((parameter_reg["input_width"] & RegMask.INPUT_WIDTH_MASK) << RegMask.INPUT_WIDTH_OFFSET)
            | ((parameter_reg["spike_width"] & RegMask.SPIKE_WIDTH_MASK) << RegMask.SPIKE_WIDTH_OFFSET)
            | ((parameter_reg["neuron_num"] & RegMask.NEURON_NUM_MASK) << RegMask.NEURON_NUM_OFFSET)
            | ((parameter_reg["pool_max"] & RegMask.POOL_MAX_MASK) << RegMask.POOL_MAX_OFFSET)
            | ((tick_wait_start_high8 & RegMask.TICK_WAIT_START_HIGH8_MASK) << RegMask.TICK_WAIT_START_HIGH8_OFFSET)
        )

        reg_frame2 = int(
            ((tick_wait_start_low7 & RegMask.TICK_WAIT_START_LOW7_MASK) << RegMask.TICK_WAIT_START_LOW7_OFFSET)
            | ((parameter_reg["tick_wait_end"] & RegMask.TICK_WAIT_END_MASK) << RegMask.TICK_WAIT_END_OFFSET)
            | ((parameter_reg["snn_en"] & RegMask.SNN_EN_MASK) << RegMask.SNN_EN_OFFSET)
            | ((parameter_reg["targetLCN"] & RegMask.TARGET_LCN_MASK) << RegMask.TARGET_LCN_OFFSET)
            | ((test_chip_addr_high3 & RegMask.TEST_CHIP_ADDR_HIGH3_MASK) << RegMask.TEST_CHIP_ADDR_HIGH3_OFFSET)
        )

        reg_frame3 = int((test_chip_addr_low7 & RegMask.TEST_CHIP_ADDR_LOW7_MASK) << RegMask.TEST_CHIP_ADDR_LOW7_OFFSET)

        self.reg_list = [reg_frame1, reg_frame2, reg_frame3]

        super().__init__(header, chip_coord, core_coord, core_e_coord, self.reg_list)

    @property
    def parameter_reg(self):
        return self.reg_list


class OfflineConfigFrame3(FramePackage):
    def __init__(
        self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, neuron_ram: dict
    ):
        header = FrameHead.CONFIG_TYPE3
        neuron_ram_load = (
            ((sram_start_addr & ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_MASK) << ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_OFFSET)
            | ((0b0 & ConfigFrame3Format.DATA_PACKAGE_TYPE_MASK) << ConfigFrame3Format.DATA_PACKAGE_TYPE_OFFSET)
            | ((data_package_num & ConfigFrame3Format.DATA_PACKAGE_NUM_MASK) << ConfigFrame3Format.DATA_PACKAGE_NUM_OFFSET)
        )

        leak_v_high2, leak_v_low28 = bin_split(neuron_ram["leak_v"], 2, 28)
        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_split(neuron_ram["threshold_mask_ctrl"], 4, 1)
        addr_core_x_high3, addr_core_x_low2 = bin_split(neuron_ram["addr_core_x"], 3, 2)
        # 1
        ram_frame1 = int(
            ((neuron_ram["vjt_pre"] & RAMMask.VJT_PRE_MASK) << RAMMask.VJT_PRE_OFFSET)
            | ((neuron_ram["bit_truncate"] & RAMMask.BIT_TRUNCATE_MASK) << RAMMask.BIT_TRUNCATE_OFFSET)
            | ((neuron_ram["weight_det_stoch"] & RAMMask.WEIGHT_DET_STOCH_MASK) << RAMMask.WEIGHT_DET_STOCH_OFFSET)
            | ((leak_v_low28 & RAMMask.LEAK_V_LOW28_MASK) << RAMMask.LEAK_V_LOW28_OFFSET)
        )

        # 2
        ram_frame2 = int(
            ((leak_v_high2 & RAMMask.LEAK_V_HIGH2_MASK) << RAMMask.LEAK_V_HIGH2_OFFSET)
            | ((neuron_ram["leak_det_stoch"] & RAMMask.LEAK_DET_STOCH_MASK) << RAMMask.LEAK_DET_STOCH_OFFSET)
            | ((neuron_ram["leak_reversal_flag"] & RAMMask.LEAK_REVERSAL_FLAG_MASK) << RAMMask.LEAK_REVERSAL_FLAG_OFFSET)
            | ((neuron_ram["threshold_pos"] & RAMMask.THRESHOLD_POS_MASK) << RAMMask.THRESHOLD_POS_OFFSET)
            | ((neuron_ram["threshold_neg"] & RAMMask.THRESHOLD_NEG_MASK) << RAMMask.THRESHOLD_NEG_OFFSET)
            | ((neuron_ram["threshold_neg_mode"] & RAMMask.THRESHOLD_NEG_MODE_MASK) << RAMMask.THRESHOLD_NEG_MODE_OFFSET)
            | ((threshold_mask_ctrl_low1 & RAMMask.THRESHOLD_MASK_CTRL_LOW1_MASK) << RAMMask.THRESHOLD_MASK_CTRL_LOW1_OFFSET)
        )

        # 3
        ram_frame3 = int(
            ((threshold_mask_ctrl_high4 & RAMMask.THRESHOLD_MASK_CTRL_HIGH4_MASK) << RAMMask.THRESHOLD_MASK_CTRL_HIGH4_OFFSET)
            | ((neuron_ram["leak_post"] & RAMMask.LEAK_POST_MASK) << RAMMask.LEAK_POST_OFFSET)
            | ((neuron_ram["reset_v"] & RAMMask.RESET_V_MASK) << RAMMask.RESET_V_OFFSET)
            | ((neuron_ram["reset_mode"] & RAMMask.RESET_MODE_MASK) << RAMMask.RESET_MODE_OFFSET)
            | ((neuron_ram["addr_chip_y"] & RAMMask.ADDR_CHIP_Y_MASK) << RAMMask.ADDR_CHIP_Y_OFFSET)
            | ((neuron_ram["addr_chip_x"] & RAMMask.ADDR_CHIP_X_MASK) << RAMMask.ADDR_CHIP_X_OFFSET)
            | ((neuron_ram["addr_core_y_ex"] & RAMMask.ADDR_CORE_Y_EX_MASK) << RAMMask.ADDR_CORE_Y_EX_OFFSET)
            | ((neuron_ram["addr_core_x_ex"] & RAMMask.ADDR_CORE_X_EX_MASK) << RAMMask.ADDR_CORE_X_EX_OFFSET)
            | ((neuron_ram["addr_core_y"] & RAMMask.ADDR_CORE_Y_MASK) << RAMMask.ADDR_CORE_Y_OFFSET)
            | ((addr_core_x_low2 & RAMMask.ADDR_CORE_X_LOW2_MASK) << RAMMask.ADDR_CORE_X_LOW2_OFFSET)
        )

        # 4
        ram_frame4 = int(
            ((addr_core_x_high3 & RAMMask.ADDR_CORE_X_HIGH3_MASK) << RAMMask.ADDR_CORE_X_HIGH3_OFFSET)
            | ((neuron_ram["addr_axon"] & RAMMask.ADDR_AXON_MASK) << RAMMask.ADDR_AXON_OFFSET)
            | ((neuron_ram["tick_relative"] & RAMMask.TICK_RELATIVE_MASK) << RAMMask.TICK_RELATIVE_OFFSET)
        )

        super().__init__(header, chip_coord, core_coord, core_e_coord, neuron_ram_load, [ram_frame1, ram_frame2, ram_frame3, ram_frame4])


class OfflineConfigFrame4(FramePackage):
    def __init__(
        self, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, weight_ram: List
    ):
        header = FrameHead.CONFIG_TYPE4
        weight_ram_load = (
            ((sram_start_addr & ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_MASK) << ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_OFFSET)
            | ((0b0 & ConfigFrame4Format.DATA_PACKAGE_TYPE_MASK) << ConfigFrame4Format.DATA_PACKAGE_TYPE_OFFSET)
            | ((data_package_num & ConfigFrame4Format.DATA_PACKAGE_NUM_MASK) << ConfigFrame4Format.DATA_PACKAGE_NUM_OFFSET)
        )

        super().__init__(header, chip_coord, core_coord, core_e_coord, weight_ram_load, weight_ram)



# TODO:test frame
class OfflineTestInFrame1(Frame):
    pass

class OfflineTestOutFrame1(Frame):
    pass

class OfflineTestInFrame2(Frame):
    pass

class OfflineTestOutFrame2(Frame):
    pass

class OfflineTestInFrame3(Frame):
    pass

class OfflineTestOutFrame3(Frame):
    pass

class OfflineTestInFrame4(Frame):
    pass

class OfflineTestOutFrame4(Frame):
    pass

"""Offline Work Frame"""
class OfflineWorkFrame1(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_e_coord: Union[List[ReplicationId], ReplicationId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        data: np.ndarray,
    ):
        header = [FrameHead.WORK_TYPE1]
        self.data = data.reshape(-1).astype(np.uint64)

        self.axon = np.array([axon], dtype=np.uint64) if isinstance(axon, int) else np.array(axon, dtype=np.uint64)
        self.time_slot = np.array([time_slot], dtype=np.uint64) if isinstance(time_slot, int) else np.array(time_slot, dtype=np.uint64)

        payload = (
            (np.uint64((0x00 & WorkFrame1Format.RESERVED_MASK) << WorkFrame1Format.RESERVED_OFFSET))
            | (self.axon & np.uint64(WorkFrame1Format.AXON_MASK) << np.uint64(WorkFrame1Format.AXON_OFFSET))
            | (self.time_slot & np.uint64(WorkFrame1Format.TIME_SLOT_MASK) << np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET))
            | (self.data & np.uint64(WorkFrame1Format.DATA_MASK) << np.uint64(WorkFrame1Format.DATA_OFFSET))
        )
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)  # type: ignore

    @property
    def value(self) -> np.ndarray:
        return self.frame

    def __repr__(self) -> str:
        info = (
            "Header:    FrameHead.WORK_TYPE1\n"
            + "Chip address".ljust(16)
            + "Core address".ljust(16)
            + "Core_E address".ljust(16)
            + "axon".ljust(16)
            + "time_slot".ljust(16)
            + "data".ljust(16)
            + "\n"
        )
        content = [
            f"{chip_coord}".ljust(16)
            + f"{core_coord}".ljust(16)
            + f"{core_e_coord}".ljust(16)
            + f"{axon}".ljust(16)
            + f"{time_slot}".ljust(16)
            + f"{data}".ljust(16)
            + "\n"
            for chip_coord, core_coord, core_e_coord, axon, time_slot, data in zip(
                self.chip_coord, self.core_coord, self.core_e_coord, self.axon, self.time_slot, self.data
            )
        ]

        return info + "".join(content)

    @staticmethod
    def gen_frame_fast(frameinfo: np.ndarray, data: np.ndarray) -> np.ndarray:
        indexes = np.nonzero(data)
        spike_frame_info = frameinfo[indexes]
        data = data[indexes]
        frame = spike_frame_info | data

        return frame


# class OfflineWorkFrame1(Frame):
#     def __init__(
#         self,
#         data: np.ndarray,
#         chip_coord: Optional[Union[List[Coord], Coord]] = None,
#         core_coord: Optional[Union[List[Coord], Coord]] = None,
#         core_e_coord: Optional[Union[List[ReplicationId], ReplicationId]] = None,
#         axon: Optional[Union[List[int], int]] = None,
#         time_slot: Optional[Union[List[int], int]] = None,
#         frameinfo: Optional[np.ndarray] = None,
#     ):
#         header = [FrameHead.WORK_TYPE1]
#         self.data = np.array([data], dtype=np.uint64) if isinstance(data, int) else np.array(data, dtype=np.uint64)
#         if frameinfo is not None:
#             if any([chip_coord, core_coord, core_e_coord, axon, time_slot]):
#                 raise ValueError("frameinfo和chip_coord、core_coord、core_e_coord、axon、time_slot不能同时输入")

#             self.frameinfo = frameinfo
#             data = data.reshape(-1)
#             indexes = np.nonzero(data)
#             spike_frame_info = self.frameinfo[indexes]
#             data = data[indexes]
#             self.frame = spike_frame_info | data

#         else:
#             if not all([chip_coord, core_coord, core_e_coord, axon, time_slot]):
#                 raise ValueError("chip_coord、core_coord、core_e_coord、axon、time_slot必须同时输入")

#             self.axon = np.array([axon], dtype=np.uint64) if isinstance(axon, int) else np.array(axon, dtype=np.uint64)
#             self.time_slot = np.array([time_slot], dtype=np.uint64) if isinstance(time_slot, int) else np.array(time_slot, dtype=np.uint64)

#             payload = (
#                 (np.uint64((0x00 & WorkFrame1Format.RESERVED_MASK) << WorkFrame1Format.RESERVED_OFFSET))
#                 | (self.axon & np.uint64(WorkFrame1Format.AXON_MASK) << np.uint64(WorkFrame1Format.AXON_OFFSET))
#                 | (self.time_slot & np.uint64(WorkFrame1Format.TIME_SLOT_MASK) << np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET))
#                 | (self.data & np.uint64(WorkFrame1Format.DATA_MASK) << np.uint64(WorkFrame1Format.DATA_OFFSET))
#             )
#             super().__init__(header, chip_coord, core_coord, core_e_coord, payload)  # type: ignore

#     @property
#     def value(self) -> np.ndarray:
#         return self.frame

#     def __repr__(self) -> str:
#         if self.frameinfo is not None:
#             self._frameinfo_decode()

#         return (
#             f"Frame info:\n"
#             f"Head:             {self.header}\n"
#             f"Chip address:     {self.chip_coord}\n"
#             f"Core address:     {self.core_coord}\n"
#             f"Core_E address:   {self.core_e_coord}\n"
#             f"axon:             {self.axon}\n"
#             f"time_slot:        {self.time_slot}\n"
#             f"data:             {self.data}\n")


#     def _frameinfo_decode(self):
#         self.header = [FrameHead.WORK_TYPE1]
#         self.chip_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CHIP_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CHIP_ADDR_OFFSET)
#         self.chip_coord = [Coord.from_addr(add) for add in self.chip_address]
#         self.core_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_OFFSET)
#         self.core_coord = [Coord.from_addr(add) for add in self.core_address]
#         self.core_e_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CORE_E_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CORE_E_ADDR_OFFSET)
#         self.core_e_coord = [Coord.from_addr(add) for add in self.core_e_address]
#         self.axon = self.frameinfo & np.uint64(WorkFrame1Format.AXON_MASK) >> np.uint64(WorkFrame1Format.AXON_OFFSET)
#         self.time_slot = self.frameinfo & np.uint64(WorkFrame1Format.TIME_SLOT_MASK) >> np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET)
#     # return self.header,self.chip_address,self.core_address,self.core_e_address,self.axon,self.time_slot


class OfflineWorkFrame2(Frame):
    def __init__(self, chip_coord: Coord, time: Union[List[int], int, np.ndarray]):
        header = [FrameHead.WORK_TYPE2]
        core_coord = [Coord(0, 0)]
        core_e_coord = [ReplicationId(0, 0)]

        super().__init__(header, chip_coord, core_coord, core_e_coord, time)


class OfflineWorkFrame3(Frame):
    def __init__(self, chip_coord: Union[List[Coord], Coord]):
        header = [FrameHead.WORK_TYPE3]
        core_coord = [Coord(0, 0)]
        core_e_coord = [ReplicationId(0, 0)]
        payload = [0]
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)


class OfflineWorkFrame4(Frame):
    def __init__(self, chip_coord: Union[List[Coord], Coord]):
        header = [FrameHead.WORK_TYPE4]
        core_coord = [Coord(0, 0)]
        core_e_coord = [ReplicationId(0, 0)]
        payload = [0]
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)
