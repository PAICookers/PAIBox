from typing import List, Optional, Union

import numpy as np

from .frame_gen import OfflineFrameGen
from .libframe.frames import OfflineWorkFrame1
from paibox.libpaicore import (
    Coord,
    FrameFormat,
    FrameHeader,
    SpikeFrameFormat as SFF,
    ReplicationId as RId,
)


__all__ = ["RuntimeEncoder", "RuntimeDecoder"]


class RuntimeEncoder:
    def __init__(
        self,
        chip_coord: Optional[Coord] = None,
        time_step: Optional[int] = None,
        frameinfo: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
    ):
        if chip_coord is not None:
            self.chip_coord = chip_coord
        if time_step is not None:
            self.time_step = time_step
        if frameinfo is not None:
            self.frameinfo = frameinfo.astype(np.uint64)
        if data is not None:
            self.data = data.astype(np.uint64)

    @classmethod
    def get_encoder(
        cls,
        chip_coord: Coord,
        time_step: int,
        frameinfo: np.ndarray,
        data: np.ndarray,
    ):
        return cls(chip_coord, time_step, frameinfo, data)

    def encode(self):
        work1_frames = OfflineFrameGen.gen_work_frame1_fast(self.frameinfo, self.data)
        work2 = OfflineFrameGen.gen_work_frame2(self.chip_coord, self.time_step)

        return np.concatenate((work1_frames, work2.value), axis=0)  # type: ignore

    def __call__(
        self,
        chip_coord: Coord,
        time_step: int,
        frameinfo: np.ndarray,
        data: np.ndarray,
    ):
        if frameinfo.shape[0] != data.shape[0]:
            raise ValueError("frameinfo and data must have the same length")

        return self.get_encoder(chip_coord, time_step, frameinfo, data).encode()

    @staticmethod
    def gen_frameinfo(
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_e_coord: Union[List[RId], RId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        header = [FrameHeader.WORK_TYPE1]
        if not isinstance(chip_coord, list):
            chip_coord = [chip_coord]
        if not isinstance(core_coord, list):
            core_coord = [core_coord]
        if not isinstance(core_e_coord, list):
            core_e_coord = [core_e_coord]
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
        core_e_address = np.array([coord.address for coord in core_e_coord]).astype(
            np.uint64
        )
        axon_array = np.array(axon, dtype=np.uint64)
        time_slot_array = np.array(time_slot, dtype=np.uint64)

        temp_header = header_value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_address = chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_address = core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_e_address = core_e_address & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
        temp_reserve = 0x00 & SFF.RESERVED_MASK
        temp_axon_array = axon_array & SFF.AXON_MASK
        temp_time_slot_array = time_slot_array & SFF.TIMESLOT_MASK

        frameinfo = (
            (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
            | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
            | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
            | (temp_core_e_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
            | (temp_reserve << SFF.RESERVED_OFFSET)
            | (temp_axon_array << SFF.AXON_OFFSET)
            | (temp_time_slot_array << SFF.TIMESLOT_OFFSET)
        )

        if save_path is not None:
            np.save(save_path, frameinfo)

        return frameinfo


class RuntimeDecoder:
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
                FrameHeader(
                    (frame >> FrameFormat.GENERAL_HEADER_OFFSET)
                    & FrameFormat.GENERAL_HEADER_MASK
                )
            )

        if all(element == FrameHeader.WORK_TYPE1 for element in header_list):
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
            header = FrameHeader(
                (frame >> FrameFormat.GENERAL_HEADER_OFFSET)
                & FrameFormat.GENERAL_HEADER_MASK
            )

            if header != FrameHeader.WORK_TYPE1:
                raise ValueError(
                    "The header of the frame is not WORK_TYPE1, please check the input frames."
                )

        axons = (frames >> SFF.AXON_OFFSET) & SFF.AXON_MASK
        time_slots = (frames >> SFF.TIMESLOT_OFFSET) & SFF.TIMESLOT_MASK
        data = (frames >> SFF.DATA_OFFSET) & SFF.DATA_MASK
        core_addr = (
            frames >> SFF.GENERAL_CORE_ADDR_OFFSET
        ) & SFF.GENERAL_CORE_ADDR_MASK

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
        out_frame_info = out_frame & ((1 << 64) - 1 - SFF.DATA_MASK)

        same_frame_info = np.in1d(frame_info, out_frame_info)
        idx = np.where(same_frame_info == True)
        out_data = np.zeros((time_step * axon_num), dtype=np.uint64)
        out_data[idx] = out_frame & SFF.DATA_MASK
        out_data = out_data
        return out_data.reshape(time_step, axon_num)

    @staticmethod
    def gen_frameinfo(
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[RId], RId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        chip_coord: Union[List[Coord], Coord] = Coord(0, 0),
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        header = [FrameHeader.WORK_TYPE1]
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
        temp_reserve = 0x00 & SFF.RESERVED_MASK
        temp_axon_array = axon_array & SFF.AXON_MASK
        temp_time_slot_array = time_slot_array & SFF.TIMESLOT_MASK

        frameinfo = (
            (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
            | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
            | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
            | (temp_core_e_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
            | (temp_reserve << SFF.RESERVED_OFFSET)
            | (temp_axon_array << SFF.AXON_OFFSET)
            | (temp_time_slot_array << SFF.TIMESLOT_OFFSET)
        )

        if save_path is not None:
            np.save(save_path, frameinfo)

        return frameinfo
