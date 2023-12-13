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
from .libframe.utils import header_check, print_frame
from .libframe._types import *
from .frame_gen import OfflineFrameGen


__all__ = ["RuntimeEncoder", "RuntimeDecoder"]


class RuntimeEncoder:
    @staticmethod
    def encode(data: DataArrayType, iframe_info: FrameArrayType) -> FrameArrayType:
        """Encode input data with common information of input frames.

        Args:
            - data: the raw data for one input node. It will be flatten after encoding.
            - iframe_info: the common information of input frames for one input node.

        Returns:
            Return the encoded arrays in spike frame format.
        """
        return OfflineFrameGen.gen_work_frame1_fast(
            iframe_info, np.asarray(data, dtype=np.uint8)
        )

    @overload
    @staticmethod
    def gen_input_frames_info(
        timestep: int, *, input_proj_info: Dict[str, Any]
    ) -> List[FrameArrayType]:
        ...

    @overload
    @staticmethod
    def gen_input_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
    ) -> FrameArrayType:
        ...

    @staticmethod
    def gen_input_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
        *,
        input_proj_info: Optional[Dict[str, Any]] = None,
    ) -> Union[FrameArrayType, List[FrameArrayType]]:
        """Generate the common information of input frames by given the dictionary  \
            of input projections.
        
        Args:
            - input_proj_info: the dictionary of input projections exported from    \
                `paibox.Mapper`.  Or you can specify the following parameters:
            - chip_coord: the destination chip coordinate of the output node.
            - core_coord: the destination coord coordinate of the output node.
            - rid: Always `(0, 0)`.
            - timeslots: the range of timeslots from 0 to T.
            - axons: the range of destination address of axons, from 0 to N.
        
        NOTE: If there are #C input nodes, the total shape of inputs will be: C*T*N.
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
            chip_coord, core_coord, rid, axons, timeslots
        )


class RuntimeDecoder:
    @overload
    @staticmethod
    def decode_spike_less1152(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: FrameArrayType,
        flatten: bool = False,
    ) -> NDArray[np.uint8]:
        ...

    @overload
    @staticmethod
    def decode_spike_less1152(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: List[FrameArrayType],
        flatten: bool = False,
    ) -> List[NDArray[np.uint8]]:
        ...

    @staticmethod
    def decode_spike_less1152(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: Union[FrameArrayType, List[FrameArrayType]],
        flatten: bool = False,
    ) -> Union[NDArray[np.uint8], List[NDArray[np.uint8]]]:
        """Decode output spike frames.

        Args:
            - oframes: the output spike frames.
            - oframe_infos: the expected common information of output frames.
            - flatten: whether flatten the decoded data.

        Returns:
            Return the decoded output data. If `oframe_infos` is a list, the output will    \
            be a list where each element represents the decoded data for each output node.
        """
        header_check(oframes, FH.WORK_TYPE1)

        if isinstance(oframe_infos, list):
            output = []

            # From (0, 0) -> (N, 0)
            seen_core_coords = (
                oframes >> SFF.GENERAL_CORE_ADDR_OFFSET
            ) & SFF.GENERAL_CORE_ADDR_MASK

            for i, oframe_info in enumerate(oframe_infos):
                data = np.zeros_like(oframe_info, dtype=np.uint8)

                _cur_coord = Coord(0, 0) + to_coordoffset(i)
                indices = np.where(_cur_coord.address == seen_core_coords)[0]

                if not np.array_equal(indices, []):
                    # Part of frame on the core coordinate.
                    oframes_on_coord = oframes[indices]
                    oframes_on_coord.sort()
                    data_on_coord = (
                        oframes_on_coord >> SFF.DATA_OFFSET
                    ) & SFF.DATA_MASK

                    valid_idx = np.isin(
                        oframe_info,
                        oframes_on_coord & (SFF.GENERAL_MASK - SFF.DATA_MASK),
                    )
                    data[valid_idx] = data_on_coord

                d_with_shape = data.reshape(timestep, -1)
                if flatten:
                    output.append(d_with_shape.flatten())
                else:
                    output.append(d_with_shape)

            return output

        else:
            data = np.zeros_like(oframe_infos, dtype=np.uint8)

            oframes.sort()
            data_on_coord = (oframes >> SFF.DATA_OFFSET) & SFF.DATA_MASK

            valid_idx = np.isin(
                oframe_infos, oframes & (SFF.GENERAL_MASK - SFF.DATA_MASK)
            )
            data[valid_idx] = data_on_coord
            d_with_shape = data.reshape(timestep, -1)
            #d_with_shape = data.reshape(-1, timestep).T

            if flatten:
                return d_with_shape.flatten()
            else:
                return d_with_shape

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
        timestep: int, *, output_dest_info: Dict[str, Any]
    ) -> List[FrameArrayType]:
        ...

    @overload
    @staticmethod
    def gen_output_frames_info(
        timestep: int,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        axons: ArrayType,
    ) -> FrameArrayType:
        ...

    @staticmethod
    def gen_output_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        axons: Optional[ArrayType] = None,
        *,
        output_dest_info: Optional[Dict[str, Any]] = None,
    ) -> Union[FrameArrayType, List[FrameArrayType]]:
        """Generate the common information of output frames by given the dictionary \
            of output destinations.
            
        Args:
            - output_dest_info: the dictionary of output destinations exported from \
                `paibox.Mapper`. Or you can specify the following parameters:
            - chip_coord: the destination chip coordinate of the output node.
            - core_coord: the destination coord coordinate of the output node.
            - rid: Always `(0, 0)`.
            - axons: the range of destination address of axons, from 0 to N.
        
        NOTE: If there are #C output nodes, the total shape of outputs will be: C*N.
        """
        if output_dest_info is not None:
            frames = []
            ts = []

            for onode in output_dest_info.values():
                # Traverse output destinations of a node
                for dest_on_coord in onode.values():
                    ts.clear()

                    for i in range(timestep):
                        ts.extend([i] * len(dest_on_coord["addr_axon"]))

                    dest_on_coord["tick_relative"] = ts
                    dest_on_coord["addr_axon"] *= timestep

                    frames_of_dest = OfflineWorkFrame1._frame_dest_reorganized(
                        dest_on_coord
                    )
                    frames_of_dest.sort()
                    frames.append(frames_of_dest)

            return frames

        assert chip_coord is not None
        assert core_coord is not None
        assert rid is not None
        assert axons is not None

        ts = []

        for i in range(timestep):
            ts.extend([i] * len(axons))

        oframes_info = OfflineWorkFrame1.concat_frame_dest(
            chip_coord, core_coord, rid, axons * timestep, ts
        )

        oframes_info.sort()
        return oframes_info
