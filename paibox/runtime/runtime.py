"""
The `PAIBoxRuntime` only guarantees that its functional features are consistent with
the PAIBox released with it, and that the hardware platform can implement custom
utilities on it. Make sure you know what you're doing when you make changes.

The runtime dose not depend on any modules of PAIBox.
"""

from typing import Any, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

try:
    import paicorelib
except ImportError:
    raise ImportError(
        "The runtime requires paicorelib. Please install it by running `pip install paicorelib`."
    ) from None

del paicorelib

import sys

from paicorelib import Coord, CoordLike, HwConfig, RIdLike, to_coordoffset
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineWorkFrame1Format as Off_WF1F
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import OfflineWorkFrame1
from paicorelib.framelib.types import ArrayType, DataArrayType, FrameArrayType
from paicorelib.framelib.utils import header_check

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

__all__ = ["PAIBoxRuntime"]

import paicorelib.framelib.types as ftypes

if hasattr(ftypes, "PAYLOAD_DATA_DTYPE"):
    PAYLOAD_DATA_DTYPE = ftypes.PAYLOAD_DATA_DTYPE  # type: ignore
else:
    PAYLOAD_DATA_DTYPE = np.uint8

del ftypes

PayloadDataType = NDArray[PAYLOAD_DATA_DTYPE]

if hasattr(HwConfig, "N_TIMESLOT_MAX"):
    MAX_TIMESLOT = HwConfig.N_TIMESLOT_MAX - 1  # Start from 0
else:
    MAX_TIMESLOT = 255

LENGTH_EX_MULTIPLE_KEY = "tick_relative"  # Use the key to represent the length expansion multiple of the output node.


def max_timeslot_check(timestep: int, raw_ts: ArrayType) -> None:
    if timestep * max(raw_ts) > MAX_TIMESLOT:
        raise ValueError(
            f"{timestep}*{max(raw_ts)} out of max timeslot ({MAX_TIMESLOT})"
        )


def valid_indices_len_check(idx: list[int], received_data: np.ndarray) -> None:
    if len(idx) != len(received_data):
        raise ValueError(
            "Length of valid indices & received data do not match: "
            f"{len(idx)} != {len(received_data)}"
        )


def get_length_ex_onode(onode_attrs: dict[str, Any]) -> int:
    """Retrieve the length expansion multiple of the output node by the given attributes."""
    if not all(LENGTH_EX_MULTIPLE_KEY in dest for dest in onode_attrs.values()):
        raise KeyError(
            f"key '{LENGTH_EX_MULTIPLE_KEY}' not found in output destination attributes"
        )

    return max(max(dest[LENGTH_EX_MULTIPLE_KEY]) for dest in onode_attrs.values()) + 1


class PAIBoxRuntime:
    @staticmethod
    def encode(
        data: DataArrayType, iframe_info: FrameArrayType, repeat: int = 1
    ) -> FrameArrayType:
        """Encode input data with common information of input frames.

        Args:
            - data: the raw data for one input node. It will be flatten after encoding.
            - iframe_info: the common information of input frames for one input node.
            - repeat: used to tile the data. For example, if timestep = 3, the original
                data [0, 1, 2] will be tiled as [0, 1, 2, 0, 1, 2, 0, 1, 2].

        Returns:
            Return the encoded arrays in working frame type I format.
        """
        _data = np.tile(np.asarray(data, dtype=PAYLOAD_DATA_DTYPE), repeat)

        return OfflineFrameGen.gen_work_frame1_fast(iframe_info, _data)

    @overload
    @staticmethod
    def gen_input_frames_info(
        timestep: int, *, input_proj_info: dict[str, Any]
    ) -> list[FrameArrayType]: ...

    @overload
    @staticmethod
    def gen_input_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
    ) -> FrameArrayType: ...

    @staticmethod
    def gen_input_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        timeslots: Optional[ArrayType] = None,
        axons: Optional[ArrayType] = None,
        *,
        input_proj_info: Optional[dict[str, Any]] = None,
    ) -> Union[FrameArrayType, list[FrameArrayType]]:
        """Generate the common information of input frames by given the dictionary  \
            of input projections.

        Args:
            - input_proj_info: the dictionary of input projections exported from    \
                `paibox.Mapper`, or you can specify the following parameters.
            - chip_coord: the destination chip coordinate of the output node.
            - core_coord: the destination coord coordinate of the output node.
            - rid: the replication ID.
            - timeslots: the range of timeslots from 0 to T.
            - axons: the range of destination address of axons, from 0 to N.

        NOTE: If there are #C input nodes, the total shape of inputs will be: C*T*N.
        """
        if input_proj_info is not None:
            frames = []
            ts = []

            # Traverse the input nodes
            for inode in input_proj_info.values():
                raw_ts: list[int] = inode["tick_relative"]
                max_timeslot_check(timestep, raw_ts)

                interval = max(raw_ts) - min(raw_ts) + 1

                ts.clear()
                for i in range(timestep):
                    ts.extend(
                        [addr + (i * interval) for addr in inode["tick_relative"]]
                    )

                inode["tick_relative"] = ts
                # addr_axon: [0-X] -> [0-X]*timestep
                inode["addr_axon"] *= timestep

                frames_of_inp = OfflineWorkFrame1._frame_dest_reorganized(inode)
                frames.append(frames_of_inp)

            return frames

        assert chip_coord is not None
        assert core_coord is not None
        assert rid is not None
        assert timeslots is not None
        assert axons is not None

        max_timeslot_check(timestep, timeslots)

        # For example:
        # [0, 1, 1, 1, 2, 2] with T = 3 ->
        # [0, 1, 1, 1, 2, 2,
        #  3, 4, 4, 4, 5, 5,
        #  6, 7, 7, 7, 8, 8]
        interval = max(timeslots) - min(timeslots) + 1

        ts = []
        for i in range(timestep):
            ts.extend([elem + i * interval for elem in timeslots])

        return OfflineWorkFrame1.concat_frame_dest(
            chip_coord, core_coord, rid, axons * timestep, ts
        )

    @staticmethod
    def decode(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: Union[FrameArrayType, list[FrameArrayType]],
        flatten: bool = False,
    ) -> Union[PayloadDataType, list[PayloadDataType]]:
        """Decode output frames from chips.

        Args:
            - timestep: the number of timesteps.
            - oframes: the output frames.
            - oframe_infos: the expected common information (without payload) of output frames.
            - flatten: whether to return the flattened data.

        Returns:
            Return the decoded output data. If `oframe_infos` is a list, the output will be a list  \
            where each element represents the decoded data for each output node.

        NOTE: This method has real-time requirement.
        """
        header_check(oframes, FH.WORK_TYPE1)

        if isinstance(oframe_infos, list):
            output = []
            # From (0, 0) -> (N, 0)
            seen_core_coords = (
                oframes >> Off_WF1F.GENERAL_CORE_ADDR_OFFSET
            ) & Off_WF1F.GENERAL_CORE_ADDR_MASK

            for i, oframe_info in enumerate(oframe_infos):
                data = np.zeros_like(oframe_info, dtype=PAYLOAD_DATA_DTYPE)
                if len(oframes) > 0:
                    # Traverse the coordinates in a specific order. Must be in the same order as when exporting.
                    # See `paibox.Mapper.export()` for more details.
                    _cur_coord = Coord(0, 0) + to_coordoffset(i)
                    indices = np.where(_cur_coord.address == seen_core_coords)[0]
                    if indices.size > 0:
                        # Part of frame on the core coordinate.
                        oframes_on_coord = oframes[indices]
                        # oframes_on_coord.sort()
                        data_on_coord = (
                            oframes_on_coord >> Off_WF1F.DATA_OFFSET
                        ) & Off_WF1F.DATA_MASK

                        valid_idx = [
                            np.where(oframe_info == value)[0][0]  # may not found
                            for value in oframes_on_coord
                            & (Off_WF1F.GENERAL_MASK - Off_WF1F.DATA_MASK)
                        ]

                        # Match the valid indices with the data.
                        valid_indices_len_check(valid_idx, data_on_coord)
                        data[valid_idx] = data_on_coord

                d_with_shape = data.reshape(timestep, -1)
                output.append(d_with_shape)

            if flatten:
                return [arr.ravel() for arr in output]
            else:
                return output

        else:
            data = np.zeros_like(oframe_infos, dtype=PAYLOAD_DATA_DTYPE)
            if len(oframes) > 0:
                oframes.sort()
                data_on_coord = (oframes >> Off_WF1F.DATA_OFFSET) & Off_WF1F.DATA_MASK

                valid_idx = np.isin(
                    oframe_infos, oframes & (Off_WF1F.GENERAL_MASK - Off_WF1F.DATA_MASK)
                )
                data[valid_idx] = data_on_coord

            d_with_shape = data.reshape(-1, timestep).T
            if flatten:
                return d_with_shape.ravel()
            else:
                return d_with_shape

    # Keep compatible
    @staticmethod
    @deprecated(
        "'decode_spike_less1152' is deprecated and will be removed in the future. "
        "Use 'decode' instead.",
        category=DeprecationWarning,
    )
    def decode_spike_less1152(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: Union[FrameArrayType, list[FrameArrayType]],
        flatten: bool = False,
    ) -> Union[PayloadDataType, list[PayloadDataType]]:
        return PAIBoxRuntime.decode(timestep, oframes, oframe_infos, flatten)

    @overload
    @staticmethod
    def gen_output_frames_info(
        timestep: int, *, output_dest_info: dict[str, Any]
    ) -> list[FrameArrayType]: ...

    @overload
    @staticmethod
    def gen_output_frames_info(
        timestep: int,
        chip_coord: CoordLike,
        core_coord: CoordLike,
        rid: RIdLike,
        axons: ArrayType,
    ) -> FrameArrayType: ...

    @staticmethod
    def gen_output_frames_info(
        timestep: int,
        chip_coord: Optional[CoordLike] = None,
        core_coord: Optional[CoordLike] = None,
        rid: Optional[RIdLike] = None,
        axons: Optional[ArrayType] = None,
        *,
        output_dest_info: Optional[dict[str, Any]] = None,
    ) -> Union[FrameArrayType, list[FrameArrayType]]:
        """Generate the common information of output frames by given the dictionary \
            of output destinations.

        Args:
            - timestep: used to tile the "tick_relative" info of output destinations.
            - output_dest_info: the dictionary of output destinations exported from \
                `paibox.Mapper`, or you can specify the following parameters.
            - chip_coord: the destination chip coordinate of the output node.
            - core_coord: the destination coord coordinate of the output node.
            - rid: the replication ID. (maybe unused)
            - axons: the range of destination address of axons, from 0 to N.
        """
        assert 0 < timestep <= MAX_TIMESLOT

        if output_dest_info is not None:
            frames = []
            ts = []
            frames_of_dest = []

            for onode in output_dest_info.values():
                frames_of_dest.clear()
                temp = []
                # Get the length expansion multiple of the output node
                len_ex_onode = get_length_ex_onode(onode)

                if len_ex_onode * timestep > MAX_TIMESLOT:
                    raise ValueError(
                        f"Length expansion multiple ({len_ex_onode}) * timestep ({timestep})"
                        f"out of max timeslot ({MAX_TIMESLOT})"
                    )

                # Traverse output destinations of a node
                for t in range(timestep):
                    for dest_on_coord in onode.values():
                        # For example:
                        # TR: [0,0,0,0,1,1] with T=3 -> [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,5]
                        if t == 0:
                            temp.extend(
                                OfflineWorkFrame1._frame_dest_reorganized(dest_on_coord)
                            )
                        else:
                            dest_on_coord["tick_relative"] = [
                                x + len_ex_onode for x in dest_on_coord["tick_relative"]
                            ]
                            temp.extend(
                                OfflineWorkFrame1._frame_dest_reorganized(dest_on_coord)
                            )

                frames_of_dest.append(temp)
                frames.append(np.hstack(frames_of_dest))

            return frames

        assert chip_coord is not None
        assert core_coord is not None
        assert rid is not None
        assert axons is not None

        # [i]*len(addr_axon) for i in [0, timestep)
        ts = []
        for i in range(timestep):
            ts.extend([i] * len(axons))

        oframes_info = OfflineWorkFrame1.concat_frame_dest(
            chip_coord, core_coord, rid, axons * timestep, ts
        )

        oframes_info.sort()  # in-place sort to save memory
        return oframes_info
