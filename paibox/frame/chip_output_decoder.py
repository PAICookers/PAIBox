"""
    将输出的帧解码为输出数据
"""
from typing import List, Optional, Union
import numpy as np

from paibox.frame.offline_frame import OfflineWorkFrame1
from paibox.libpaicore.v2.coordinate import Coord, ReplicationId
from .params import FrameFormat, FrameHead, WorkFrame1Format


class ChipOutputDecoder:
    def decode(
        self,
        frames: np.ndarray,
    ) -> dict:
        """将输出数据帧解码，以字典形式返回，按照time_slot排序

        Args:
            frames (Union[np.ndarray,OfflineWorkFrame1]): _description_

        Returns:
            dict: {
                core_addr: {
                    axonid: {
                        time_slot: np.ndarray,
                        data: np.ndarray
                    }
                }
            }

            eg: {
                1:{
                    2: {
                        time_slot : [1,3,9],
                        data : [4,5,6]
                    }
                }
            }
        """
        header_list = []
        for frame in frames:
            header_list.append(
                FrameHead(
                    (frame >> FrameFormat.GENERAL_HEADER_OFFSET)
                    & FrameFormat.GENERAL_HEADER_MASK
                )
            )

        if all(element == FrameHead.WORK_TYPE1 for element in header_list):
            return self.decode_spike(frames)
        else:
            return {}

    def decode_spike(
        self,
        frames: Union[np.ndarray, OfflineWorkFrame1],
    ) -> dict:
        if isinstance(frames, OfflineWorkFrame1):
            frames = frames.value

        frames = frames.astype(np.uint64)
        for frame in frames:
            header = FrameHead(
                (frame >> FrameFormat.GENERAL_HEADER_OFFSET)
                & FrameFormat.GENERAL_HEADER_MASK
            )

            if header != FrameHead.WORK_TYPE1:
                raise ValueError(
                    "The header of the frame is not WORK_TYPE1, please check the input frames."
                )

        axons = (frames >> WorkFrame1Format.AXON_OFFSET) & WorkFrame1Format.AXON_MASK
        time_slots = (
            frames >> WorkFrame1Format.TIME_SLOT_OFFSET
        ) & WorkFrame1Format.TIME_SLOT_MASK
        data = (frames >> WorkFrame1Format.DATA_OFFSET) & WorkFrame1Format.DATA_MASK
        core_addr = (
            frames >> WorkFrame1Format.GENERAL_CORE_ADDR_OFFSET
        ) & WorkFrame1Format.GENERAL_CORE_ADDR_MASK

        res = {}

        unique_axon = np.unique(axons)
        unique_core_addr = np.unique(core_addr)

        for core_value in unique_core_addr:
            axon_positions = {}  # 存储所有的axon在frames中的位置
            res[core_value] = {}
            core_addr_positions = np.where(core_addr == core_value)[
                0
            ]  # 获取value在原来的core_addr中的位置
            core_axons = axons[core_addr_positions]  # 将当前core的frames信息筛选出来
            core_time_slots = time_slots[core_addr_positions]
            core_data = data[core_addr_positions]

            for axon_value in unique_axon:
                # print(np.where(axons == value)[0])
                positions = np.where(core_axons == axon_value)[
                    0
                ]  # 获取当前core中的当前axon在筛选后的frames信息（core_axons）中的位置
                if len(positions) > 0:
                    axon_positions[axon_value] = positions

            for axon_value, positions in axon_positions.items():
                res[core_value][axon_value] = {}
                res[core_value][axon_value]["time_slot"] = core_time_slots[positions]
                res[core_value][axon_value]["data"] = core_data[positions]

                sorted_indices = np.argsort(res[core_value][axon_value]["time_slot"])

                res[core_value][axon_value]["time_slot"] = res[core_value][axon_value][
                    "time_slot"
                ][sorted_indices]
                res[core_value][axon_value]["data"] = res[core_value][axon_value][
                    "data"
                ][sorted_indices]

        return res

    @staticmethod
    def decode_spike_fast(out_frame, frame_info, axon_num, time_step):
        frame_info = np.sort(frame_info)
        out_frame = np.sort(out_frame)
        out_frame_info = out_frame & ((1 << 64) - 1 - WorkFrame1Format.DATA_MASK)

        same_frame_info = np.in1d(frame_info, out_frame_info)
        idx = np.where(same_frame_info == True)
        out_data = np.zeros((time_step * axon_num), dtype=np.uint64)
        out_data[idx] = out_frame & WorkFrame1Format.DATA_MASK
        out_data = out_data
        return out_data.reshape(time_step,axon_num)

    @staticmethod
    def gen_frameinfo(
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        chip_coord: Union[List[Coord], Coord] = Coord(0, 0),
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
        core_e_address = np.array([coord.address for coord in core_ex_coord]).astype(
            np.uint64
        )
        axon_array = np.array(axon, dtype=np.uint64)
        time_slot_array = np.array(time_slot, dtype=np.uint64)

        pack_list = zip(
            header_value,
            chip_address,
            core_address,
            core_e_address,
            axon_array,
            time_slot_array,
        )
        pack_list = sorted(pack_list, key=lambda x: (x[5], x[4]))

        (
            header_value,
            chip_address,
            core_address,
            core_e_address,
            axon_array,
            time_slot_array,
        ) = map(np.array, zip(*pack_list))

        temp_header = header_value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_address = chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_address = core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_e_address = core_e_address & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
        temp_reserve = 0x00 & WorkFrame1Format.RESERVED_MASK
        temp_axon_array = axon_array & WorkFrame1Format.AXON_MASK
        temp_time_slot_array = time_slot_array & WorkFrame1Format.TIME_SLOT_MASK

        frameinfo = (
            (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
            | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
            | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
            | (temp_core_e_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
            | (temp_reserve << WorkFrame1Format.RESERVED_OFFSET)
            | (temp_axon_array << WorkFrame1Format.AXON_OFFSET)
            | (temp_time_slot_array << WorkFrame1Format.TIME_SLOT_OFFSET)
        )

        if save_path is not None:
            np.save(save_path, frameinfo)

        return frameinfo
