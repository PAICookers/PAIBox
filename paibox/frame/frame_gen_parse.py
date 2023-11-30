import re
from typing import Optional

import numpy as np
from matplotlib.font_manager import weight_dict

from paibox import frame
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
            if "chip_coord" in value:
                chip_coord = Coord.from_addr(value["chip_coord"])
            else:
                chip_coord = Coord(0, 0)
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
            if "chip_coord" in value:
                chip_coord = Coord.from_addr(value["chip_coord"])
            else:
                chip_coord = Coord(0, 0)
            core_coord = Coord.from_addr(value["coord"])
            parameter_reg = value.copy()
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
        all_frames_value = np.array([]).astype(np.uint64)
        core_ex_coord = ReplicationId(0, 0)
        for key, value in core_plm_config.items():
            if "chip_coord" in value:
                chip_coord = Coord.from_addr(value["chip_coord"])
            else:
                chip_coord = Coord(0, 0)
            core_coord = Coord.from_addr(value["coord"])
            # neuron_ram = value["neuron_ram"]
            frame = OfflineConfigFrame3Group(
                chip_coord=chip_coord,
                core_coord=core_coord,
                core_ex_coord=core_ex_coord,
                core_neuron_ram=value["neuron_ram"],
            )
            all_frames_value = np.append(all_frames_value, frame.value)
        return all_frames_value

    @staticmethod
    def gen_config_frame4(
        core_plm_config: dict,
    ) -> np.ndarray:
        all_frames_value = np.array([]).astype(np.uint64)
        core_ex_coord = ReplicationId(0, 0)
        sram_start_addr = np.uint64(0)

        for key, value in core_plm_config.items():
            if "chip_coord" in value:
                chip_coord = Coord.from_addr(value["chip_coord"])
            else:
                chip_coord = Coord(0, 0)
            core_coord = Coord.from_addr(value["coord"])
            data_package_num = np.uint64(18 * 512)
            weight_ram = value["weight_ram"]

            frame = OfflineConfigFrame4(
                chip_coord=chip_coord,
                core_coord=core_coord,
                core_ex_coord=core_ex_coord,
                sram_start_addr=sram_start_addr,
                data_package_num=data_package_num,
                weight_ram=weight_ram,
            )
            all_frames_value = np.append(all_frames_value, frame.value)
        return all_frames_value

    @staticmethod
    def gen_config_frame(core_plm_config: dict) -> np.ndarray:
        frame1 = OfflineFrameGen.gen_config_frame1(core_plm_config)
        frame2 = OfflineFrameGen.gen_config_frame2(core_plm_config)
        frame3 = OfflineFrameGen.gen_config_frame3(core_plm_config)
        frame4 = OfflineFrameGen.gen_config_frame4(core_plm_config)
        all_frame = np.concatenate((frame1, frame2, frame3, frame4), axis=0)

        return all_frame

    @staticmethod
    def gen_reset_frame(chip_coord, core_coord, core_ex_coord=ReplicationId(0, 0)):
        """每次推理或配置前先发送复位帧，再进行配置"""
        frame_array = np.array([]).astype(np.uint64)
        frame1 = Frame(
            header=FrameHead.CONFIG_TYPE1,
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            payload=0,
        )
        frame2 = Frame(
            header=FrameHead.CONFIG_TYPE1,
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            payload=0,
        )
        frame3 = OfflineFrameGen.gen_work_frame4(chip_coord)
        frame4 = Frame(
            header=FrameHead.CONFIG_TYPE1,
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            payload=0,
        )
        frame5 = OfflineWorkFrame1(
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            axon=0,
            time_slot=0,
            data=np.array([0]),
        )
        for frame in [frame1, frame2, frame3, frame4, frame5]:
            frame_array = np.append(frame_array, frame.value)
        return frame_array

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
        input_proj_info: dict, axon, time_slot, data
    ) -> OfflineWorkFrame1:
        chip_coord = Coord(
            input_proj_info["addr_chip_x"], input_proj_info["addr_chip_y"]
        )
        core_coord = Coord(
            input_proj_info["addr_core_x"], input_proj_info["addr_core_y"]
        )
        core_ex_coord = ReplicationId(
            input_proj_info["addr_core_x_ex"], input_proj_info["addr_core_y_ex"]
        )

        return OfflineWorkFrame1(
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            axon=axon,
            time_slot=time_slot,
            data=data,
        )

    @staticmethod
    def gen_work_frame1_fast(
        frameinfo: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        return OfflineWorkFrame1.gen_frame_fast(frameinfo=frameinfo, data=data)

    @staticmethod
    def gen_frameinfo(
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        header = [FrameHead.WORK_TYPE1]
        if not isinstance(chip_coord, list):
            chip_coord = [chip_coord]
        if not isinstance(core_coord, list):
            core_coord = [core_coord]
        if not isinstance(core_ex_coord, list):
            core_ex_coord = [core_ex_coord]
        if not isinstance(axon, list):
            axon = [axon]
        if not isinstance(time_slot, list):
            time_slot = [time_slot]

        header_value = np.array([head.value for head in header]).astype(np.uint64)
        chip_address = np.array([coord.address for coord in chip_coord]).astype(
            np.uint64
        )
        core_address = np.array([coord.address for coord in core_coord]).astype(
            np.uint64
        )
        core_ex_address = np.array([coord.address for coord in core_ex_coord]).astype(
            np.uint64
        )
        axon_array = np.array(axon, dtype=np.uint64)
        time_slot_array = np.array(time_slot, dtype=np.uint64)

        temp_header = header_value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_address = chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_address = core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_ex_address = core_ex_address & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
        temp_reserve = 0x00 & WorkFrame1Format.RESERVED_MASK
        temp_axon_array = axon_array & WorkFrame1Format.AXON_MASK
        temp_time_slot_array = time_slot_array & WorkFrame1Format.TIME_SLOT_MASK

        frameinfo = (
            (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
            | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
            | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
            | (temp_core_ex_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
            | (temp_reserve << WorkFrame1Format.RESERVED_OFFSET)
            | (temp_axon_array << WorkFrame1Format.AXON_OFFSET)
            | (temp_time_slot_array << WorkFrame1Format.TIME_SLOT_OFFSET)
        )

        if save_path is not None:
            np.save(save_path, frameinfo)

        return frameinfo

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

        if header == FrameHead.WORK_TYPE1:
            pass
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
