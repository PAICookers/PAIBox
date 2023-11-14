import warnings
from typing import Dict, List, Union

import numpy as np

from paibox.libpaicore.v2 import Coord, ReplicationId

from .base_frame import Frame, FramePackage
from .params import ConfigFrame3Format, ConfigFrame4Format, FrameFormat, FrameHead
from .params import ParameterRAMFormat as RAMF
from .params import ParameterRegFormat as RegF
from .params import WorkFrame1Format
from .util import bin_array_split, bin_split

"""Offline Config Frame"""


class OfflineConfigFrame1(Frame):
    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        random_seed: Union[int, np.uint64],
    ):
        header = [FrameHead.CONFIG_TYPE1]
        if random_seed > FrameFormat.GENERAL_MASK:
            warnings.warn(
                f"seed {random_seed} is too large, truncated into 64 bits!", UserWarning
            )
        self.random_seed = np.uint64(random_seed) & FrameFormat.GENERAL_MASK
        payload = np.array(
            [
                (self.random_seed >> np.uint64(34)) & FrameFormat.GENERAL_PAYLOAD_MASK,
                (self.random_seed >> np.uint64(4)) & FrameFormat.GENERAL_PAYLOAD_MASK,
                self.random_seed & np.uint64((1 << 4) - 1),
            ]
        )

        super().__init__(header, chip_coord, core_coord, core_ex_coord, payload)

    def __repr__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:             {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_EX address:  {self.core_ex_coord}\n"
            f"random_seed:      {self.random_seed}\n"
        )


class OfflineConfigFrame2(Frame):
    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        parameter_reg: dict,
    ) -> None:
        """_summary_

        Args:
            chip_coord (Coord): _description_
            core_coord (Coord): _description_
            core_ex_coord (ReplicationId): _description_
            parameter_reg (dict(np.ndarray,list)):
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
        """
        header = FrameHead.CONFIG_TYPE2
        for key, value in parameter_reg.items():
            parameter_reg[key] = np.uint64(value)
        tick_wait_start_high8, tick_wait_start_low7 = bin_split(
            parameter_reg["tick_wait_start"], 8, 7
        )
        test_chip_addr_high3, test_chip_addr_low7 = bin_split(
            parameter_reg["test_chip_addr"], 3, 7
        )
        reg_frame1 = (
            (parameter_reg["weight_width"] & RegF.WEIGHT_WIDTH_MASK)
            << RegF.WEIGHT_WIDTH_OFFSET
            | ((parameter_reg["LCN"] & RegF.LCN_MASK) << RegF.LCN_OFFSET)
            | (
                (parameter_reg["input_width"] & RegF.INPUT_WIDTH_MASK)
                << RegF.INPUT_WIDTH_OFFSET
            )
            | (
                (parameter_reg["spike_width"] & RegF.SPIKE_WIDTH_MASK)
                << RegF.SPIKE_WIDTH_OFFSET
            )
            | (
                (parameter_reg["neuron_num"] & RegF.NEURON_NUM_MASK)
                << RegF.NEURON_NUM_OFFSET
            )
            | ((parameter_reg["pool_max"] & RegF.POOL_MAX_MASK) << RegF.POOL_MAX_OFFSET)
            | (
                (tick_wait_start_high8 & RegF.TICK_WAIT_START_HIGH8_MASK)
                << RegF.TICK_WAIT_START_HIGH8_OFFSET
            )
        )

        reg_frame2 = (
            (
                (tick_wait_start_low7 & RegF.TICK_WAIT_START_LOW7_MASK)
                << RegF.TICK_WAIT_START_LOW7_OFFSET
            )
            | (
                (parameter_reg["tick_wait_end"] & RegF.TICK_WAIT_END_MASK)
                << RegF.TICK_WAIT_END_OFFSET
            )
            | ((parameter_reg["snn_en"] & RegF.SNN_EN_MASK) << RegF.SNN_EN_OFFSET)
            | (
                (parameter_reg["targetLCN"] & RegF.TARGET_LCN_MASK)
                << RegF.TARGET_LCN_OFFSET
            )
            | (
                (test_chip_addr_high3 & RegF.TEST_CHIP_ADDR_HIGH3_MASK)
                << RegF.TEST_CHIP_ADDR_HIGH3_OFFSET
            )
        )

        reg_frame3 = (
            test_chip_addr_low7 & RegF.TEST_CHIP_ADDR_LOW7_MASK
        ) << RegF.TEST_CHIP_ADDR_LOW7_OFFSET

        self.reg_list = [reg_frame1, reg_frame2, reg_frame3]

        super().__init__(header, chip_coord, core_coord, core_ex_coord, self.reg_list)

    @property
    def parameter_reg(self):
        return self.reg_list


class OfflineConfigFrame3(FramePackage):
    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: np.uint64,
        neuron_ram: dict,
        neuron_num: int,
    ):
        """_summary_

        Args:
            chip_coord (Coord): _description_
            core_coord (Coord): _description_
            core_ex_coord (ReplicationId): _description_
            sram_start_addr (int): _description_
            data_package_num (int): _description_
            neuron_ram (dict):
                "tick_relative":            8,
                "addr_axon":                11,
                "addr_core_x":              5,
                "addr_core_y":              5,
                "addr_core_x_ex":           5,
                "addr_core_y_ex":           5,
                "addr_chip_x":              5,
                "addr_chip_y":              5,
                "reset_mode":               2,
                "reset_v":                  30,
                "leak_post":                1,
                "threshold_mask_ctrl":      5,
                "threshold_neg_mode":       1,
                "threshold_neg":            29,
                "threshold_pos":            29,
                "leak_reversal_flag":       1,
                "leak_det_stoch":           1,
                "leak_v":                   30,
                "weight_det_stoch":         1,
                "bit_truncate":             5,
                "vjt_pre":                  30
        """

        header = FrameHead.CONFIG_TYPE3
        self.data_package_num = neuron_num * 4
        for key, value in neuron_ram.items():
            neuron_ram[key] = np.array(value).astype(np.uint64)
        neuron_ram_load = (
            (
                (sram_start_addr & ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_MASK)
                << ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | (
                (np.uint64(0b0) & ConfigFrame3Format.DATA_PACKAGE_TYPE_MASK)
                << ConfigFrame3Format.DATA_PACKAGE_TYPE_OFFSET
            )
            | (
                (
                    np.uint64(self.data_package_num)
                    & ConfigFrame3Format.DATA_PACKAGE_NUM_MASK
                )
                << ConfigFrame3Format.DATA_PACKAGE_NUM_OFFSET
            )
        )

        leak_v_high2, leak_v_low28 = bin_array_split(neuron_ram["leak_v"], 2, 28)
        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_array_split(
            neuron_ram["threshold_mask_ctrl"], 4, 1
        )
        addr_core_x_high3, addr_core_x_low2 = bin_array_split(
            neuron_ram["addr_core_x"], 3, 2
        )
        # 1
        ram_frame1 = (
            ((neuron_ram["vjt_pre"] & RAMF.VJT_PRE_MASK) << RAMF.VJT_PRE_OFFSET)
            | (
                (neuron_ram["bit_truncate"] & RAMF.BIT_TRUNCATE_MASK)
                << RAMF.BIT_TRUNCATE_OFFSET
            )
            | (
                (neuron_ram["weight_det_stoch"] & RAMF.WEIGHT_DET_STOCH_MASK)
                << RAMF.WEIGHT_DET_STOCH_OFFSET
            )
            | ((leak_v_low28 & RAMF.LEAK_V_LOW28_MASK) << RAMF.LEAK_V_LOW28_OFFSET)
        )

        # 2
        ram_frame2 = (
            ((leak_v_high2 & RAMF.LEAK_V_HIGH2_MASK) << RAMF.LEAK_V_HIGH2_OFFSET)
            | (
                (neuron_ram["leak_det_stoch"] & RAMF.LEAK_DET_STOCH_MASK)
                << RAMF.LEAK_DET_STOCH_OFFSET
            )
            | (
                (neuron_ram["leak_reversal_flag"] & RAMF.LEAK_REVERSAL_FLAG_MASK)
                << RAMF.LEAK_REVERSAL_FLAG_OFFSET
            )
            | (
                (neuron_ram["threshold_pos"] & RAMF.THRESHOLD_POS_MASK)
                << RAMF.THRESHOLD_POS_OFFSET
            )
            | (
                (neuron_ram["threshold_neg"] & RAMF.THRESHOLD_NEG_MASK)
                << RAMF.THRESHOLD_NEG_OFFSET
            )
            | (
                (neuron_ram["threshold_neg_mode"] & RAMF.THRESHOLD_NEG_MODE_MASK)
                << RAMF.THRESHOLD_NEG_MODE_OFFSET
            )
            | (
                (threshold_mask_ctrl_low1 & RAMF.THRESHOLD_MASK_CTRL_LOW1_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_LOW1_OFFSET
            )
        )

        # 3
        ram_frame3 = (
            (
                (threshold_mask_ctrl_high4 & RAMF.THRESHOLD_MASK_CTRL_HIGH4_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_HIGH4_OFFSET
            )
            | ((neuron_ram["leak_post"] & RAMF.LEAK_POST_MASK) << RAMF.LEAK_POST_OFFSET)
            | ((neuron_ram["reset_v"] & RAMF.RESET_V_MASK) << RAMF.RESET_V_OFFSET)
            | (
                (neuron_ram["reset_mode"] & RAMF.RESET_MODE_MASK)
                << RAMF.RESET_MODE_OFFSET
            )
            | (
                (neuron_ram["addr_chip_y"] & RAMF.ADDR_CHIP_Y_MASK)
                << RAMF.ADDR_CHIP_Y_OFFSET
            )
            | (
                (neuron_ram["addr_chip_x"] & RAMF.ADDR_CHIP_X_MASK)
                << RAMF.ADDR_CHIP_X_OFFSET
            )
            | (
                (neuron_ram["addr_core_y_ex"] & RAMF.ADDR_CORE_Y_EX_MASK)
                << RAMF.ADDR_CORE_Y_EX_OFFSET
            )
            | (
                (neuron_ram["addr_core_x_ex"] & RAMF.ADDR_CORE_X_EX_MASK)
                << RAMF.ADDR_CORE_X_EX_OFFSET
            )
            | (
                (neuron_ram["addr_core_y"] & RAMF.ADDR_CORE_Y_MASK)
                << RAMF.ADDR_CORE_Y_OFFSET
            )
            | (
                (addr_core_x_low2 & RAMF.ADDR_CORE_X_LOW2_MASK)
                << RAMF.ADDR_CORE_X_LOW2_OFFSET
            )
        )

        # 4
        ram_frame4 = (
            (
                (addr_core_x_high3 & RAMF.ADDR_CORE_X_HIGH3_MASK)
                << RAMF.ADDR_CORE_X_HIGH3_OFFSET
            )
            | ((neuron_ram["addr_axon"] & RAMF.ADDR_AXON_MASK) << RAMF.ADDR_AXON_OFFSET)
            | (
                (neuron_ram["tick_relative"] & RAMF.TICK_RELATIVE_MASK)
                << RAMF.TICK_RELATIVE_OFFSET
            )
        )

        ram = [ram_frame1, ram_frame2, ram_frame3, ram_frame4]
        l = [len(i) for i in ram]
        ram = np.array([np.pad(i, (0, max(l) - len(i)), "wrap") for i in ram]).reshape(
            -1, order="F"
        )

        super().__init__(
            header,
            chip_coord,
            core_coord,
            core_ex_coord,
            neuron_ram_load,
            ram,
        )


class OfflineConfigFrame4(FramePackage):
    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: np.uint64,
        data_package_num: np.uint64,
        weight_ram: List,
    ):
        header = FrameHead.CONFIG_TYPE4
        weight_ram_load = (
            (
                (sram_start_addr & ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_MASK)
                << ConfigFrame4Format.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | (
                (np.uint64(0b0) & ConfigFrame4Format.DATA_PACKAGE_TYPE_MASK)
                << ConfigFrame4Format.DATA_PACKAGE_TYPE_OFFSET
            )
            | (
                (data_package_num & ConfigFrame4Format.DATA_PACKAGE_NUM_MASK)
                << ConfigFrame4Format.DATA_PACKAGE_NUM_OFFSET
            )
        )

        super().__init__(
            header, chip_coord, core_coord, core_ex_coord, weight_ram_load, weight_ram
        )


# TODO:test frame
class OfflineTestInFrame1(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
    ):
        header = FrameHead.TEST_TYPE1
        super().__init__(header, chip_coord, core_coord, core_ex_coord, np.array([0]))


class OfflineTestOutFrame1(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
        data: np.ndarray,
    ):
        header = FrameHead.TEST_TYPE1
        super().__init__(header, chip_coord, core_coord, core_ex_coord, data)


class OfflineTestInFrame2(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
    ):
        header = FrameHead.TEST_TYPE2

        super().__init__(header, chip_coord, core_coord, core_ex_coord, np.array([0]))


class OfflineTestOutFrame2(Frame):
    pass


class OfflineTestInFrame3(Frame):
    def __init__(
        self,
        chip_coord,
        core_coord,
        core_ex_coord,
        sram_start_addr: np.uint64,
        data_package_num: np.uint64,
    ):
        neuron_ram_load = np.array(
            [
                (
                    (sram_start_addr & ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_MASK)
                    << ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_OFFSET
                )
                | (
                    (np.uint64(0b0) & ConfigFrame3Format.DATA_PACKAGE_TYPE_MASK)
                    << ConfigFrame3Format.DATA_PACKAGE_TYPE_OFFSET
                )
                | (
                    (data_package_num & ConfigFrame3Format.DATA_PACKAGE_NUM_MASK)
                    << ConfigFrame3Format.DATA_PACKAGE_NUM_OFFSET
                )
            ]
        )

        header = FrameHead.TEST_TYPE3
        super().__init__(header, chip_coord, core_coord, core_ex_coord, neuron_ram_load)


class OfflineTestOutFrame3(Frame):
    pass


class OfflineTestInFrame4(Frame):
    def __init__(
        self,
        chip_coord,
        core_coord,
        core_ex_coord,
        sram_start_addr: int,
        data_package_num: int,
    ):
        neuron_ram_load = np.array(
            [
                (
                    (
                        (
                            sram_start_addr
                            & ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_MASK
                        )
                        << ConfigFrame3Format.DATA_PACKAGE_SRAM_NEURON_OFFSET
                    )
                    | (
                        (0b1 & ConfigFrame3Format.DATA_PACKAGE_TYPE_MASK)
                        << ConfigFrame3Format.DATA_PACKAGE_TYPE_OFFSET
                    )
                    | (
                        (data_package_num & ConfigFrame3Format.DATA_PACKAGE_NUM_MASK)
                        << ConfigFrame3Format.DATA_PACKAGE_NUM_OFFSET
                    )
                )
            ]
        )
        header = FrameHead.TEST_TYPE4
        super().__init__(header, chip_coord, core_coord, core_ex_coord, neuron_ram_load)


class OfflineTestOutFrame4(Frame):
    pass


"""Offline Work Frame"""


class OfflineWorkFrame1(Frame):
    def __init__(
        self,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        data: np.ndarray,
    ):
        header = [FrameHead.WORK_TYPE1]
        self.data = data.reshape(-1).astype(np.uint64)

        self.axon = (
            np.array([axon], dtype=np.uint64)
            if isinstance(axon, int)
            else np.array(axon, dtype=np.uint64)
        )
        self.time_slot = (
            np.array([time_slot], dtype=np.uint64)
            if isinstance(time_slot, int)
            else np.array(time_slot, dtype=np.uint64)
        )

        payload = (
            (
                np.uint64(
                    (0x00 & WorkFrame1Format.RESERVED_MASK)
                    << WorkFrame1Format.RESERVED_OFFSET
                )
            )
            | (self.axon & WorkFrame1Format.AXON_MASK << WorkFrame1Format.AXON_OFFSET)
            | (
                self.time_slot
                & WorkFrame1Format.TIME_SLOT_MASK << WorkFrame1Format.TIME_SLOT_OFFSET
            )
            | (self.data & WorkFrame1Format.DATA_MASK << WorkFrame1Format.DATA_OFFSET)
        )
        super().__init__(header, chip_coord, core_coord, core_ex_coord, payload)  # type: ignore

    @property
    def value(self) -> np.ndarray:
        return self.frame

    def __repr__(self) -> str:
        info = (
            "Header:    FrameHead.WORK_TYPE1\n"
            + "Chip address".ljust(16)
            + "Core address".ljust(16)
            + "Core_EX address".ljust(16)
            + "axon".ljust(16)
            + "time_slot".ljust(16)
            + "data".ljust(16)
            + "\n"
        )
        content = [
            f"{chip_coord}".ljust(16)
            + f"{core_coord}".ljust(16)
            + f"{core_ex_coord}".ljust(16)
            + f"{axon}".ljust(16)
            + f"{time_slot}".ljust(16)
            + f"{data}".ljust(16)
            + "\n"
            for chip_coord, core_coord, core_ex_coord, axon, time_slot, data in zip(
                self.chip_coord,
                self.core_coord,
                self.core_ex_coord,
                self.axon,
                self.time_slot,
                self.data,
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
#         core_ex_coord: Optional[Union[List[ReplicationId], ReplicationId]] = None,
#         axon: Optional[Union[List[int], int]] = None,
#         time_slot: Optional[Union[List[int], int]] = None,
#         frameinfo: Optional[np.ndarray] = None,
#     ):
#         header = [FrameHead.WORK_TYPE1]
#         self.data = np.array([data], dtype=np.uint64) if isinstance(data, int) else np.array(data, dtype=np.uint64)
#         if frameinfo is not None:
#             if any([chip_coord, core_coord, core_ex_coord, axon, time_slot]):
#                 raise ValueError("frameinfo和chip_coord、core_coord、core_ex_coord、axon、time_slot不能同时输入")

#             self.frameinfo = frameinfo
#             data = data.reshape(-1)
#             indexes = np.nonzero(data)
#             spike_frame_info = self.frameinfo[indexes]
#             data = data[indexes]
#             self.frame = spike_frame_info | data

#         else:
#             if not all([chip_coord, core_coord, core_ex_coord, axon, time_slot]):
#                 raise ValueError("chip_coord、core_coord、core_ex_coord、axon、time_slot必须同时输入")

#             self.axon = np.array([axon], dtype=np.uint64) if isinstance(axon, int) else np.array(axon, dtype=np.uint64)
#             self.time_slot = np.array([time_slot], dtype=np.uint64) if isinstance(time_slot, int) else np.array(time_slot, dtype=np.uint64)

#             payload = (
#                 (np.uint64((0x00 & WorkFrame1Format.RESERVED_MASK) << WorkFrame1Format.RESERVED_OFFSET))
#                 | (self.axon & np.uint64(WorkFrame1Format.AXON_MASK) << np.uint64(WorkFrame1Format.AXON_OFFSET))
#                 | (self.time_slot & np.uint64(WorkFrame1Format.TIME_SLOT_MASK) << np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET))
#                 | (self.data & np.uint64(WorkFrame1Format.DATA_MASK) << np.uint64(WorkFrame1Format.DATA_OFFSET))
#             )
#             super().__init__(header, chip_coord, core_coord, core_ex_coord, payload)  # type: ignore

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
#             f"Core_EX address:   {self.core_ex_coord}\n"
#             f"axon:             {self.axon}\n"
#             f"time_slot:        {self.time_slot}\n"
#             f"data:             {self.data}\n")


#     def _frameinfo_decode(self):
#         self.header = [FrameHead.WORK_TYPE1]
#         self.chip_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CHIP_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CHIP_ADDR_OFFSET)
#         self.chip_coord = [Coord.from_addr(add) for add in self.chip_address]
#         self.core_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_OFFSET)
#         self.core_coord = [Coord.from_addr(add) for add in self.core_address]
#         self.core_ex_address = self.frameinfo & np.uint64(WorkFrame1Format.GENERAL_CORE_EX_ADDR_MASK) >> np.uint64(WorkFrame1Format.GENERAL_CORE_EX_ADDR_OFFSET)
#         self.core_ex_coord = [Coord.from_addr(add) for add in self.core_ex_address]
#         self.axon = self.frameinfo & np.uint64(WorkFrame1Format.AXON_MASK) >> np.uint64(WorkFrame1Format.AXON_OFFSET)
#         self.time_slot = self.frameinfo & np.uint64(WorkFrame1Format.TIME_SLOT_MASK) >> np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET)
#     # return self.header,self.chip_address,self.core_address,self.core_ex_address,self.axon,self.time_slot


class OfflineWorkFrame2(Frame):
    def __init__(self, chip_coord: Coord, time: Union[List[int], int, np.ndarray]):
        header = [FrameHead.WORK_TYPE2]
        core_coord = [Coord(0, 0)]
        core_ex_coord = [ReplicationId(0, 0)]

        super().__init__(header, chip_coord, core_coord, core_ex_coord, time)


class OfflineWorkFrame3(Frame):
    def __init__(self, chip_coord: Union[List[Coord], Coord]):
        header = [FrameHead.WORK_TYPE3]
        core_coord = [Coord(0, 0)]
        core_ex_coord = [ReplicationId(0, 0)]
        payload = [0]
        super().__init__(header, chip_coord, core_coord, core_ex_coord, payload)


class OfflineWorkFrame4(Frame):
    def __init__(self, chip_coord: Union[List[Coord], Coord]):
        header = [FrameHead.WORK_TYPE4]
        core_coord = [Coord(0, 0)]
        core_ex_coord = [ReplicationId(0, 0)]
        payload = [0]
        super().__init__(header, chip_coord, core_coord, core_ex_coord, payload)
