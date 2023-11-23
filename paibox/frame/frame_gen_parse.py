from typing import Optional

import numpy as np

from paibox.libpaicore.v2 import Coord

from .base_frame import *
from .offline_frame import *
from .params import *
from .util import *


class OfflineFrameGen:
    """Offline Frame Generator"""

    @staticmethod
    def gen_config_frame1(
        core_plm_config: dict,
    ) -> np.ndarray:
        core_ex_coord = ReplicationId(0, 0)
        all_frames_value = np.array([]).astype(np.uint64)
        for key, value in core_plm_config.items():
            chip_coord = Coord.from_addr(value["chip_coord"])
            core_coord = Coord.from_addr(value["coord"])
            random_seed = value["random_seed"]
            frame = OfflineConfigFrame1(
                chip_coord=chip_coord,
                core_coord=core_coord,
                core_ex_coord=core_ex_coord,
                random_seed=random_seed,
            )
            all_frames_value = np.append(all_frames_value, frame.value)
        return all_frames_value

    @staticmethod
    def gen_config_frame2(
        core_plm_config: dict,
    ) -> np.ndarray:
        core_ex_coord = ReplicationId(0, 0)
        all_frames_value = np.array([]).astype(np.uint64)
        for key, value in core_plm_config.items():
            chip_coord = Coord.from_addr(value["chip_coord"])
            core_coord = Coord.from_addr(value["coord"])
            parameter_reg = value
            frame = OfflineConfigFrame2(
                chip_coord=chip_coord,
                core_coord=core_coord,
                core_ex_coord=core_ex_coord,
                parameter_reg=parameter_reg,
            )
            all_frames_value = np.append(all_frames_value, frame.value)
        return all_frames_value

    @staticmethod
    def gen_config_frame3(
        core_plm_config: dict,
    ) -> np.ndarray:
        core_ex_coord = ReplicationId(0, 0)
        all_frames_value = np.array([]).astype(np.uint64)
        for key,value in core_plm_config.items():
            chip_coord = Coord.from_addr(value["chip_coord"])
            core_coord = Coord.from_addr(value["coord"])
            # neuron_ram = value["neuron_ram"]
            frame=OfflineConfigFrame3Group(
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            core_neuron_ram=value["neuron_ram"],
        )
            all_frames_value = np.append(all_frames_value, frame.value)
        return all_frames_value

    @staticmethod
    def gen_config_frame4(
        core_coord: Coord,
        sram_start_addr: np.uint64,
        data_package_num: np.uint64,
        weight_ram: List,
        chip_coord: Coord = Coord(0, 0),
        core_ex_coord: ReplicationId = ReplicationId(0, 0),
    ) -> OfflineConfigFrame4:
        return OfflineConfigFrame4(
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            sram_start_addr=sram_start_addr,
            data_package_num=data_package_num,
            weight_ram=weight_ram,
        )

    @staticmethod
    def gen_testin_frame1(
        chip_coord: Coord, core_coord: Coord, core_ex_coord: ReplicationId
    ) -> OfflineTestInFrame1:
        return OfflineTestInFrame1(chip_coord, core_coord, core_ex_coord)

    @staticmethod
    def gen_testin_frame2(
        chip_coord: Coord, core_coord: Coord, core_ex_coord: ReplicationId
    ) -> OfflineTestInFrame2:
        return OfflineTestInFrame2(chip_coord, core_coord, core_ex_coord)

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


class OfflineFrameParser:
    @staticmethod
    def parse(value):
        header = OfflineFrameParser.get_header(value)

        if header == FrameHead.TEST_TYPE1:
            return OfflineTestOutFrame1(value=value)
        elif header == FrameHead.TEST_TYPE2:
            return OfflineTestOutFrame2(value=value)
        elif header == FrameHead.TEST_TYPE3:
            return OfflineTestOutFrame3(value=value)
        elif header == FrameHead.TEST_TYPE4:
            pass

        else:
            raise ValueError("The header of the frame is not supported.")

    @staticmethod
    def get_header(value):
        return FrameHead(
            (value[0] >> FrameFormat.GENERAL_HEADER_OFFSET)
            & FrameFormat.GENERAL_HEADER_MASK
        )
