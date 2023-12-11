import numpy as np

from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, overload, Union

from paibox.libpaicore import (
    Coord,
    CoordLike,
    to_coordoffset,
    FrameHeader as FH,
    SpikeFrameFormat as SFF,
    RIdLike,
)

from .libframe.frames import OfflineWorkFrame1
from .libframe.utils import header_check
from .libframe._types import *
from .frame_gen import OfflineFrameGen


__all__ = ["RuntimeEncoder", "RuntimeDecoder"]


class RuntimeEncoder:
    @staticmethod
    def encode(data: DataArrayType, iframe_info: FrameArrayType) -> FrameArrayType:
        return OfflineFrameGen.gen_work_frame1_fast(
            iframe_info, np.asarray(data, dtype=np.uint8)
        )

    @overload
    @staticmethod
    def gen_input_frames_info(
        *, input_proj_info: Dict[str, Any]
    ) -> List[FrameArrayType]:
        ...

    @overload
    @staticmethod
    def gen_input_frames_info(
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
    ) -> FrameArrayType:
        ...

    @staticmethod
    def gen_input_frames_info(
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
        *,
        input_proj_info: Optional[Dict[str, Any]] = None,
    ) -> Union[FrameArrayType, List[FrameArrayType]]:
        """Generate the common part of the input spike frames by given the dictionary  \
            of input projections.
        
        Args:
            - input_proj_info: the dictionary of input projections exported \
                from `paibox.Mapper`.
        """
        if input_proj_info is not None:
            frames = []

            # Traverse the input nodes
            for inode in input_proj_info.values():
                frames_of_inp = OfflineWorkFrame1._frame_dest_reorganized(inode)
                frames.append(frames_of_inp)

            return frames

        assert chip_coord is not None
        assert core_coord is not None
        assert rid is not None
        assert timeslots is not None
        assert axons is not None

        return OfflineWorkFrame1.concat_frame_dest(
            chip_coord, core_coord, rid, timeslots, axons
        )


class RuntimeDecoder:
    @overload
    @staticmethod
    def decode_spike_less1152(
        frames: FrameArrayType, oframe_info: FrameArrayType
    ) -> NDArray[np.uint8]:
        ...

    @overload
    @staticmethod
    def decode_spike_less1152(
        frames: FrameArrayType, oframe_info: List[FrameArrayType]
    ) -> List[NDArray[np.uint8]]:
        ...

    @staticmethod
    def decode_spike_less1152(
        frames: FrameArrayType, oframe_info: Union[FrameArrayType, List[FrameArrayType]]
    ) -> Union[NDArray[np.uint8], List[NDArray[np.uint8]]]:
        header_check(frames, FH.WORK_TYPE1)

        if isinstance(oframe_info, list):
            output = []

            seen_core_coords = (
                frames >> SFF.GENERAL_CORE_ADDR_OFFSET
            ) & SFF.GENERAL_CORE_ADDR_MASK

            for i, oframe in enumerate(oframe_info):
                _cur_coord = Coord(0, 0) + to_coordoffset(i)

                indices = np.where(seen_core_coords == _cur_coord.address)
                frames_on_coord = frames[indices]

                data = np.zeros_like(oframe, dtype=np.uint8)

                axons_on_coord = (frames_on_coord >> SFF.AXON_OFFSET) & SFF.AXON_MASK
                data_on_coord = (frames_on_coord >> SFF.DATA_OFFSET) & SFF.DATA_MASK
                all_axons = (oframe >> SFF.AXON_OFFSET) & SFF.AXON_MASK

                valid_idx = np.isin(all_axons, axons_on_coord)

                data[valid_idx] = data_on_coord
                output.append(data)

            return output

        else:
            data = np.zeros_like(oframe_info, dtype=np.uint8)

            axons_on_coord = (frames >> SFF.AXON_OFFSET) & SFF.AXON_MASK
            data_on_coord = (frames >> SFF.DATA_OFFSET) & SFF.DATA_MASK
            all_axons = (oframe_info >> SFF.AXON_OFFSET) & SFF.AXON_MASK

            valid_idx = np.isin(all_axons, axons_on_coord)

            data[valid_idx] = data_on_coord

            return data

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

    @overload
    @staticmethod
    def gen_output_frames_info(
        *, output_dest_info: Dict[str, Any]
    ) -> List[FrameArrayType]:
        ...

    @overload
    @staticmethod
    def gen_output_frames_info(
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        timeslots: ArrayType,
        axons: ArrayType,
    ) -> FrameArrayType:
        ...

    @staticmethod
    def gen_output_frames_info(
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
        *,
        output_dest_info: Optional[Dict[str, Any]] = None,
    ) -> Union[FrameArrayType, List[FrameArrayType]]:
        """Generate the common part of the expected output spike frames by given    \
            the dictionary of output destinations.
            
        Args:
            - output_dest_info: the dictionary of output destinations exported      \
                from `paibox.Mapper`.
        """
        if output_dest_info is not None:
            frames = []

            for onode in output_dest_info.values():
                # Traverse output destinations of a node
                for dest_on_coord in onode.values():
                    frames_of_dest = OfflineWorkFrame1._frame_dest_reorganized(
                        dest_on_coord
                    )
                    frames.append(frames_of_dest)

            return frames

        assert chip_coord is not None
        assert core_coord is not None
        assert rid is not None
        assert timeslots is not None
        assert axons is not None

        return OfflineWorkFrame1.concat_frame_dest(
            chip_coord, core_coord, rid, timeslots, axons
        )
