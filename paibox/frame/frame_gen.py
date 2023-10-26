from typing import Optional, Tuple
from .base_frame import *
import numpy as np

from paibox.libpaicore.v2 import Coord
from .util import *
from .params import ParameterRAMFormat as RAMMask
from .params import ParameterRegFormat as RegMask
from .params import *
from .offline_frame import *


class FrameGenOffline:
    """Offline Frame Generator"""

    @staticmethod
    def gen_config_frame1(
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_e_coord: Union[List[ReplicationId], ReplicationId],
        random_seed: Union[List[int], int, np.ndarray],
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord, core_coord, core_e_coord, random_seed)

    @staticmethod
    def gen_config_frame2(chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, parameter_reg: dict) -> FrameGroup:
        return OfflineConfigFrame2(chip_coord, core_coord, core_e_coord, parameter_reg)

    @staticmethod
    def gen_config_frame3(
        chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, neuron_ram: dict
    ) -> OfflineConfigFrame3:
        return OfflineConfigFrame3(chip_coord, core_coord, core_e_coord, sram_start_addr, data_package_num, neuron_ram)

    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, sram_start_addr: int, data_package_num: int, weight_ram: List
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(chip_coord, core_coord, core_e_coord, sram_start_addr, data_package_num, weight_ram)

    @staticmethod
    def gen_work_frame1(
        data: np.ndarray,
        chip_coord: Optional[Union[List[Coord], Coord]] = None,
        core_coord: Optional[Union[List[Coord], Coord]] = None,
        core_e_coord: Optional[Union[List[ReplicationId], ReplicationId]] = None,
        axon: Optional[Union[List[int], int]] = None,
        time_slot: Optional[Union[List[int], int]] = None,
        frameinfo: Optional[np.ndarray] = None,
    ) -> Union[OfflineWorkFrame1,np.ndarray]:
        if frameinfo is not None:
            if any([chip_coord, core_coord, core_e_coord, axon, time_slot]):
                raise ValueError("frameinfo和chip_coord、core_coord、core_e_coord、axon、time_slot不能同时输入")
            return OfflineWorkFrame1.gen_frame_fast(frameinfo, data)
        else:
            if not all([chip_coord, core_coord, core_e_coord, axon, time_slot]):
                raise ValueError("chip_coord、core_coord、core_e_coord、axon、time_slot必须同时输入")
            return OfflineWorkFrame1(chip_coord,core_coord,core_e_coord,axon,time_slot,data) # type: ignore

    @staticmethod
    def gen_work_frame2(chip_coord: Coord, time:Union[List[int], int, np.ndarray]) -> OfflineWorkFrame2:
        return OfflineWorkFrame2(chip_coord, time)

    @staticmethod
    def gen_work_frame3(chip_coord: Union[List[Coord], Coord],) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)

    @staticmethod
    def gen_work_frame4(chip_coord: Union[List[Coord], Coord]) -> OfflineWorkFrame4:
        return OfflineWorkFrame4(chip_coord)
