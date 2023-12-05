import warnings
from functools import reduce
from typing import Any, ClassVar, Dict, List, Union

import numpy as np

from ._types import FRAME_DTYPE, FrameArrayType
from .base import Frame, FramePackage
from .utils import params_check, params_check2

from paibox.libpaicore import (
    Coord,
    ConfigFrame3Format as CF3F,
    ConfigFrame4Format as CF4F,
    FrameFormat as FF,
    FrameHeader as FH,
    ParameterRAMFormat as RAMF,
    ParameterRegFormat as RegF,
    ReplicationId as RId,
    WorkFrame1Format,
)
from paibox.libpaicore.v2.reg_model import ParamsRegDictChecker
from paibox.libpaicore.v2.ram_model import (
    NeuronAttrsDictChecker,
    NeuronDestInfoDictChecker,
)
from paibox.utils import bin_split


__all__ = [
    "OfflineConfigFrame1",
    "OfflineConfigFrame2",
    "OfflineConfigFrame3",
    "OfflineConfigFrame4",
]


class OfflineConfigFrame1(Frame):
    """Offline config frame type I"""

    header: ClassVar[FH] = FH.CONFIG_TYPE1

    def __init__(
        self, chip_coord: Coord, core_coord: Coord, rid: RId, random_seed: int
    ) -> None:
        if random_seed > FF.GENERAL_MASK:
            warnings.warn(
                f"Random seed {random_seed} is too large, truncated into 64 bits!",
                UserWarning,
            )

        self._random_seed = random_seed & FF.GENERAL_MASK

        payload = self._random_seed_split()

        super().__init__(self.header, chip_coord, core_coord, rid, payload)

    def _random_seed_split(self) -> FrameArrayType:
        return np.asarray(
            [
                (self._random_seed >> 34) & FF.GENERAL_PAYLOAD_MASK,
                (self._random_seed >> 4) & FF.GENERAL_PAYLOAD_MASK,
                (self._random_seed & ((1 << 4) - 1)) << 26,
            ],
            dtype=FRAME_DTYPE,
        )

    @property
    def random_seed(self) -> FRAME_DTYPE:
        return FRAME_DTYPE(self._random_seed)


class OfflineConfigFrame2(Frame):
    header: ClassVar[FH] = FH.CONFIG_TYPE2

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        params_reg_dict: Dict[str, Any],
    ) -> None:
        payload = self._payload_reorganized(params_reg_dict)

        super().__init__(self.header, chip_coord, core_coord, rid, payload)

    @staticmethod
    @params_check(ParamsRegDictChecker)
    def _payload_reorganized(reg_dict: Dict[str, Any]) -> FrameArrayType:
        # High 8 bits & low 7 bits of tick_wait_start
        tws_high8, tws_low7 = bin_split(reg_dict["tick_wait_start"], 7, 8)
        # High 3 bits & low 7 bits of test_chip_addrs
        tca_high3, tca_low7 = bin_split(reg_dict["test_chip_addr"], 7, 3)

        reg_frame1 = (
            (reg_dict["weight_width"] & RegF.WEIGHT_WIDTH_MASK)
            << RegF.WEIGHT_WIDTH_OFFSET
            | ((reg_dict["LCN"] & RegF.LCN_MASK) << RegF.LCN_OFFSET)
            | (
                (reg_dict["input_width"] & RegF.INPUT_WIDTH_MASK)
                << RegF.INPUT_WIDTH_OFFSET
            )
            | (
                (reg_dict["spike_width"] & RegF.SPIKE_WIDTH_MASK)
                << RegF.SPIKE_WIDTH_OFFSET
            )
            | (
                (reg_dict["neuron_num"] & RegF.NEURON_NUM_MASK)
                << RegF.NEURON_NUM_OFFSET
            )
            | ((reg_dict["pool_max"] & RegF.POOL_MAX_MASK) << RegF.POOL_MAX_OFFSET)
            | (
                (tws_high8 & RegF.TICK_WAIT_START_HIGH8_MASK)
                << RegF.TICK_WAIT_START_HIGH8_OFFSET
            )
        )

        reg_frame2 = (
            (
                (tws_low7 & RegF.TICK_WAIT_START_LOW7_MASK)
                << RegF.TICK_WAIT_START_LOW7_OFFSET
            )
            | (
                (reg_dict["tick_wait_end"] & RegF.TICK_WAIT_END_MASK)
                << RegF.TICK_WAIT_END_OFFSET
            )
            | ((reg_dict["snn_en"] & RegF.SNN_EN_MASK) << RegF.SNN_EN_OFFSET)
            | (
                (reg_dict["target_LCN"] & RegF.TARGET_LCN_MASK)
                << RegF.TARGET_LCN_OFFSET
            )
            | (
                (tca_high3 & RegF.TEST_CHIP_ADDR_HIGH3_MASK)
                << RegF.TEST_CHIP_ADDR_HIGH3_OFFSET
            )
        )

        reg_frame3 = (
            tca_low7 & RegF.TEST_CHIP_ADDR_LOW7_MASK
        ) << RegF.TEST_CHIP_ADDR_LOW7_OFFSET

        return np.asarray([reg_frame1, reg_frame2, reg_frame3], dtype=FRAME_DTYPE)

    @property
    def params_reg(self) -> FrameArrayType:
        return self.payload


class OfflineConfigFrame3(FramePackage):
    header: ClassVar[FH] = FH.CONFIG_TYPE3

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: int,
        neuron_num: int,
        neuron_attrs: Dict[str, Any],
        neuron_dest_info: Dict[str, Any],
        tick_relative: List[int],
        addr_axon: List[int],
    ) -> None:
        payload = np.uint64(
            (
                (sram_start_addr & CF3F.DATA_PACKAGE_SRAM_NEURON_MASK)
                << CF3F.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((0 & CF3F.DATA_PACKAGE_TYPE_MASK) << CF3F.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (neuron_num * 4 & CF3F.DATA_PACKAGE_NUM_MASK)
                << CF3F.DATA_PACKAGE_NUM_OFFSET
            )
        )
        packages = self._packages_reorganized(
            neuron_attrs, neuron_dest_info, neuron_num, tick_relative, addr_axon
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload, packages)

    @staticmethod
    @params_check2(NeuronAttrsDictChecker, NeuronDestInfoDictChecker)
    def _packages_reorganized(
        attrs: Dict[str, Any],
        dest_info: Dict[str, Any],
        neuron_num: int,
        tick_relative: List[int],
        addr_axon: List[int],
    ) -> FrameArrayType:
        assert len(tick_relative) == len(addr_axon)

        _packages = np.zeros((neuron_num * 4,), dtype=FRAME_DTYPE)

        leak_v_high2, leak_v_low28 = bin_split(attrs["leak_post"], 28, 2)
        threshold_mask_ctrl_high4, threshold_mask_ctrl_low1 = bin_split(
            attrs["threshold_mask_ctrl"], 1, 4
        )
        addr_core_x_high3, addr_core_x_low2 = bin_split(dest_info["addr_core_x"], 2, 3)

        # Packages #1
        ram_frame1 = (
            ((attrs["vjt_pre"] & RAMF.VJT_PRE_MASK) << RAMF.VJT_PRE_OFFSET)
            | (
                (attrs["bit_truncate"] & RAMF.BIT_TRUNCATE_MASK)
                << RAMF.BIT_TRUNCATE_OFFSET
            )
            | (
                (attrs["weight_det_stoch"] & RAMF.WEIGHT_DET_STOCH_MASK)
                << RAMF.WEIGHT_DET_STOCH_OFFSET
            )
            | ((leak_v_low28 & RAMF.LEAK_V_LOW28_MASK) << RAMF.LEAK_V_LOW28_OFFSET)
        )

        # Packages #2
        ram_frame2 = (
            ((leak_v_high2 & RAMF.LEAK_V_HIGH2_MASK) << RAMF.LEAK_V_HIGH2_OFFSET)
            | (
                (attrs["leak_det_stoch"] & RAMF.LEAK_DET_STOCH_MASK)
                << RAMF.LEAK_DET_STOCH_OFFSET
            )
            | (
                (attrs["leak_reversal_flag"] & RAMF.LEAK_REVERSAL_FLAG_MASK)
                << RAMF.LEAK_REVERSAL_FLAG_OFFSET
            )
            | (
                (attrs["threshold_pos"] & RAMF.THRESHOLD_POS_MASK)
                << RAMF.THRESHOLD_POS_OFFSET
            )
            | (
                (attrs["threshold_neg"] & RAMF.THRESHOLD_NEG_MASK)
                << RAMF.THRESHOLD_NEG_OFFSET
            )
            | (
                (attrs["threshold_neg_mode"] & RAMF.THRESHOLD_NEG_MODE_MASK)
                << RAMF.THRESHOLD_NEG_MODE_OFFSET
            )
            | (
                (threshold_mask_ctrl_low1 & RAMF.THRESHOLD_MASK_CTRL_LOW1_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_LOW1_OFFSET
            )
        )

        # Packages #3
        ram_frame3 = (
            (
                (threshold_mask_ctrl_high4 & RAMF.THRESHOLD_MASK_CTRL_HIGH4_MASK)
                << RAMF.THRESHOLD_MASK_CTRL_HIGH4_OFFSET
            )
            | ((attrs["leak_post"] & RAMF.LEAK_POST_MASK) << RAMF.LEAK_POST_OFFSET)
            | ((attrs["reset_v"] & RAMF.RESET_V_MASK) << RAMF.RESET_V_OFFSET)
            | ((attrs["reset_mode"] & RAMF.RESET_MODE_MASK) << RAMF.RESET_MODE_OFFSET)
            | (
                (dest_info["addr_chip_y"] & RAMF.ADDR_CHIP_Y_MASK)
                << RAMF.ADDR_CHIP_Y_OFFSET
            )
            | (
                (dest_info["addr_chip_x"] & RAMF.ADDR_CHIP_X_MASK)
                << RAMF.ADDR_CHIP_X_OFFSET
            )
            | (
                (dest_info["addr_core_y_ex"] & RAMF.ADDR_CORE_Y_EX_MASK)
                << RAMF.ADDR_CORE_Y_EX_OFFSET
            )
            | (
                (dest_info["addr_core_x_ex"] & RAMF.ADDR_CORE_X_EX_MASK)
                << RAMF.ADDR_CORE_X_EX_OFFSET
            )
            | (
                (dest_info["addr_core_y"] & RAMF.ADDR_CORE_Y_MASK)
                << RAMF.ADDR_CORE_Y_OFFSET
            )
            | (
                (addr_core_x_low2 & RAMF.ADDR_CORE_X_LOW2_MASK)
                << RAMF.ADDR_CORE_X_LOW2_OFFSET
            )
        )
        _package_common = np.array(
            [ram_frame1, ram_frame2, ram_frame3], dtype=FRAME_DTYPE
        )

        # Iterate destination infomation of every neuron
        for i in range(neuron_num):
            # Packages #4
            ram_frame4 = (
                (
                    (addr_core_x_high3 & RAMF.ADDR_CORE_X_HIGH3_MASK)
                    << RAMF.ADDR_CORE_X_HIGH3_OFFSET
                )
                | ((tick_relative[i] & RAMF.ADDR_AXON_MASK) << RAMF.ADDR_AXON_OFFSET)
                | (
                    (addr_axon[i] & RAMF.TICK_RELATIVE_MASK)
                    << RAMF.TICK_RELATIVE_OFFSET
                )
            )
            _packages[4 * i : 4 * i + 3] = _package_common
            _packages[4 * (i + 1) - 1] = ram_frame4

        return _packages


class OfflineConfigFrame4(FramePackage):
    header: ClassVar[FH] = FH.CONFIG_TYPE4

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: int,
        data_package_num: int,
        weight_ram: FrameArrayType,
    ) -> None:
        payload = np.uint64(
            (
                (sram_start_addr & CF4F.DATA_PACKAGE_SRAM_NEURON_MASK)
                << CF4F.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((0 & CF4F.DATA_PACKAGE_TYPE_MASK) << CF4F.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (data_package_num & CF4F.DATA_PACKAGE_NUM_MASK)
                << CF4F.DATA_PACKAGE_NUM_OFFSET
            )
        )

        # [N * 18] -> [(N*18), ]
        _weight_ram = weight_ram.flatten()

        super().__init__(self.header, chip_coord, core_coord, rid, payload, _weight_ram)


class OfflineTestInFrame1(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE1

    def __init__(self, chip_coord: Coord, core_coord: Coord, rid: RId) -> None:
        super().__init__(
            self.header, chip_coord, core_coord, rid, np.asarray(0, dtype=FRAME_DTYPE)
        )


# class OfflineTestOutFrame1(Frame):
#     def __init__(self, value: np.ndarray):
#         super().__init__(value)
#         if self.header != FH.TEST_TYPE1:
#             raise ValueError("The header of the frame is not TEST_TYPE1")

#         if (self.payload[-1] >> FRAME_DTYPE(26)) > FRAME_DTYPE(0b1111):
#             warnings.warn(
#                 "The payload of the frame 3 is too large, the length of random seed is shorter than 64"
#             )

#         self.random_seed = reduce(
#             lambda x, y: (x << FRAME_DTYPE(30)) + y, self.payload[:-1]
#         )
#         self.random_seed = (self.random_seed << FRAME_DTYPE(4)) + (
#             self.payload[-1] >> FRAME_DTYPE(26)
#         )
#         self.random_seed = self.random_seed & FF.GENERAL_MASK


class OfflineTestInFrame2(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE2

    def __init__(self, chip_coord: Coord, core_coord: Coord, rid: RId) -> None:
        super().__init__(
            self.header, chip_coord, core_coord, rid, np.asarray(0, dtype=FRAME_DTYPE)
        )


# class OfflineTestOutFrame2(Frame):
#     def __init__(self, value: np.ndarray):
#         super().__init__(value=value)
#         if self.header != FH.TEST_TYPE2:
#             raise ValueError("The header of the frame is not TEST_TYPE2")

#         self.neuron_ram = {}


class OfflineTestInFrame3(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE3

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: int,
        data_package_num: int,
    ) -> None:
        payload = np.asarray(
            (
                (sram_start_addr & CF3F.DATA_PACKAGE_SRAM_NEURON_MASK)
                << CF3F.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((0 & CF3F.DATA_PACKAGE_TYPE_MASK) << CF3F.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (data_package_num & CF3F.DATA_PACKAGE_NUM_MASK)
                << CF3F.DATA_PACKAGE_NUM_OFFSET
            ),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)


# class OfflineTestOutFrame3(FramePackage):
#     def __init__(self, value: np.ndarray):
#         super().__init__(value=value)
#         if self.header != FH.TEST_TYPE3:
#             raise ValueError("The header of the frame is not TEST_TYPE3")

#         self.data_package_num = int(
#             (self.value[0] >> CF3F.DATA_PACKAGE_NUM_OFFSET) & CF3F.DATA_PACKAGE_NUM_MASK
#         )

#         self.neuron_ram = {}
#         for i in range(1, len(self.value), 4):
#             self.neuron_ram["vjt_pre"] = np.append(
#                 self.neuron_ram.get("vjt_pre", []),
#                 ((self.value[i] >> RAMF.VJT_PRE_OFFSET) & RAMF.VJT_PRE_MASK),
#             )

#         for key, value in self.neuron_ram.items():
#             self.neuron_ram[key] = self.neuron_ram[key].astype(np.int32)


class OfflineTestInFrame4(Frame):
    header: ClassVar[FH] = FH.TEST_TYPE4

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        sram_start_addr: int,
        data_package_num: int,
    ):
        payload = np.asarray(
            (
                (sram_start_addr & CF3F.DATA_PACKAGE_SRAM_NEURON_MASK)
                << CF3F.DATA_PACKAGE_SRAM_NEURON_OFFSET
            )
            | ((1 & CF3F.DATA_PACKAGE_TYPE_MASK) << CF3F.DATA_PACKAGE_TYPE_OFFSET)
            | (
                (data_package_num & CF3F.DATA_PACKAGE_NUM_MASK)
                << CF3F.DATA_PACKAGE_NUM_OFFSET
            ),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)


# class OfflineTestOutFrame4(Frame):
#     pass


class OfflineWorkFrame1(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE1

    def __init__(
        self,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        data: np.ndarray,
    ):
        self.data = data.reshape(-1).astype(FRAME_DTYPE)
        self.axon = (
            np.array([axon], dtype=FRAME_DTYPE)
            if isinstance(axon, int)
            else np.array(axon, dtype=FRAME_DTYPE)
        )
        self.time_slot = (
            np.array([time_slot], dtype=FRAME_DTYPE)
            if isinstance(time_slot, int)
            else np.array(time_slot, dtype=FRAME_DTYPE)
        )

        payload = np.asarray(
            ((0 & WorkFrame1Format.RESERVED_MASK) << WorkFrame1Format.RESERVED_OFFSET)
            | ((self.axon & WorkFrame1Format.AXON_MASK) << WorkFrame1Format.AXON_OFFSET)
            | (
                (self.time_slot & WorkFrame1Format.TIME_SLOT_MASK)
                << WorkFrame1Format.TIME_SLOT_OFFSET
            )
            | (
                (self.data & WorkFrame1Format.DATA_MASK) << WorkFrame1Format.DATA_OFFSET
            ),
            dtype=FRAME_DTYPE,
        )

        super().__init__(self.header, chip_coord, core_coord, rid, payload)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @staticmethod
    def gen_frame_fast(frameinfo: np.ndarray, data: np.ndarray) -> np.ndarray:
        indexes = np.nonzero(data)
        spike_frame_info = frameinfo[indexes]
        data = data[indexes]
        frame = spike_frame_info | data

        return frame


class OfflineWorkFrame2(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE2

    def __init__(self, chip_coord: Coord, n_sync: int) -> None:
        super().__init__(
            self.header,
            chip_coord,
            Coord(0, 0),
            RId(0, 0),
            np.asarray(n_sync, dtype=FRAME_DTYPE),
        )


class OfflineWorkFrame3(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE3

    def __init__(self, chip_coord: Coord) -> None:
        super().__init__(self.header, chip_coord, Coord(0, 0), RId(0, 0), 0)


class OfflineWorkFrame4(Frame):
    header: ClassVar[FH] = FH.WORK_TYPE4

    def __init__(self, chip_coord: Coord) -> None:
        super().__init__(self.header, chip_coord, Coord(0, 0), RId(0, 0), 0)


WorkFrame = OfflineWorkFrame1
SyncFrame = OfflineWorkFrame2
ClearFrame = OfflineWorkFrame3
InitialFrame = OfflineWorkFrame4
