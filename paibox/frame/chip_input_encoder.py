from typing import List, Optional, Union

import numpy as np
from paibox.frame.frame_gen_parse import OfflineFrameGen
from paibox.frame.params import FrameFormat
from paibox.libpaicore.v2 import Coord, ReplicationId
from paibox.frame.params import WorkFrame1Format, FrameHead
from typing import List, Union, Optional


class ChipInputEncoder:
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
        work1_frames = OfflineFrameGen.gen_work_frame1_fast(
            frameinfo=self.frameinfo, data=self.data
        )
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
        core_e_coord: Union[List[ReplicationId], ReplicationId],
        axon: Union[List[int], int],
        time_slot: Union[List[int], int],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        header = [FrameHead.WORK_TYPE1]
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
