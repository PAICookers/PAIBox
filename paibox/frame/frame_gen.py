from typing import Optional

import numpy as np

from paibox.libpaicore.v2 import Coord

from .base_frame import *
from .offline_frame import *
from .params import *
from .util import *


class FrameGenOffline:
    """Offline Frame Generator"""

    @staticmethod
    def gen_config_frame1(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        random_seed: Union[int, np.uint64],
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord, core_coord, core_ex_coord, random_seed)

    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        parameter_reg: dict,
    ) -> OfflineConfigFrame2:
        return OfflineConfigFrame2(chip_coord, core_coord, core_ex_coord, parameter_reg)

    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: np.uint64,
        neuron_ram: dict,
        neuron_num: int,
    ) -> OfflineConfigFrame3:
        return OfflineConfigFrame3(
            chip_coord,
            core_coord,
            core_ex_coord,
            sram_start_addr,
            neuron_ram,
            neuron_num,
        )

    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: np.uint64,
        data_package_num: np.uint64,
        weight_ram: List,
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord,
            core_coord,
            core_ex_coord,
            sram_start_addr,
            data_package_num,
            weight_ram,
        )

    @staticmethod
    def gen_testin_frame1(
        chip_coord: Coord, core_coord: Coord, core_ex_coord: ReplicationId
    ) -> OfflineTestInFrame1:
        return OfflineTestInFrame1(chip_coord, core_coord, core_ex_coord)

    @staticmethod
    def gen_testout_frame1():
        pass

    @staticmethod
    def gen_testin_frame2(
        chip_coord: Coord, core_coord: Coord, core_ex_coord: ReplicationId
    ) -> OfflineTestInFrame2:
        return OfflineTestInFrame2(chip_coord, core_coord, core_ex_coord)

    @staticmethod
    def gen_testout_frame2():
        pass

    @staticmethod
    def gen_testin_frame3(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: np.uint64,
        data_package_num: np.uint64,
    ) -> OfflineTestInFrame3:
        return OfflineTestInFrame3(
            chip_coord, core_coord, core_ex_coord, sram_start_addr, data_package_num
        )

    @staticmethod
    def gen_testout_frame3():
        pass

    @staticmethod
    def gen_testin_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        core_ex_coord: ReplicationId,
        sram_start_addr: int,
        data_package_num: int,
    ):
        return OfflineTestInFrame4(
            chip_coord, core_coord, core_ex_coord, sram_start_addr, data_package_num
        )

    @staticmethod
    def gen_testout_frame4():
        pass

    @staticmethod
    def gen_work_frame1(
        data: np.ndarray,
        chip_coord: Optional[Union[List[Coord], Coord]] = None,
        core_coord: Optional[Union[List[Coord], Coord]] = None,
        core_ex_coord: Optional[Union[List[ReplicationId], ReplicationId]] = None,
        axon: Optional[Union[List[int], int]] = None,
        time_slot: Optional[Union[List[int], int]] = None,
        frameinfo: Optional[np.ndarray] = None,
    ) -> Union[OfflineWorkFrame1, np.ndarray]:
        if frameinfo is not None:
            if any([chip_coord, core_coord, core_ex_coord, axon, time_slot]):
                raise ValueError(
                    "frameinfo和chip_coord、core_coord、core_ex_coord、axon、time_slot不能同时输入"
                )
            return OfflineWorkFrame1.gen_frame_fast(frameinfo, data)
        else:
            if not all([chip_coord, core_coord, core_ex_coord, axon, time_slot]):
                raise ValueError(
                    "chip_coord、core_coord、core_ex_coord、axon、time_slot必须同时输入"
                )
            return OfflineWorkFrame1(chip_coord, core_coord, core_ex_coord, axon, time_slot, data)  # type: ignore

    @staticmethod
    def gen_work_frame2(
        chip_coord: Coord, time: Union[List[int], int, np.ndarray]
    ) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, time)

    @staticmethod
    def gen_work_frame3(
        chip_coord: Union[List[Coord], Coord],
    ) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: Union[List[Coord], Coord]) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)
