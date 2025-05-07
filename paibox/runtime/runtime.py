"""
The `PAIBoxRuntime` only guarantees that its functional features are consistent with
the PAIBox released with it, and that the hardware platform can implement custom
utilities on it. Make sure you know what you're doing when you make changes.

The runtime dose not depend on any modules of PAIBox.
"""

from typing import Any, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

try:
    import paicorelib
except ImportError:
    raise ImportError(
        "The runtime requires paicorelib. Please install it by running 'pip install paicorelib'."
    ) from None

del paicorelib

import sys

from paicorelib import ChipCoord, Coord, CoordLike, HwConfig
from paicorelib import ReplicationId as RId
from paicorelib import RIdLike, to_coordoffset
from paicorelib.framelib.frame_defs import FrameFormat as FF
from paicorelib.framelib.frame_defs import FrameHeader as FH
from paicorelib.framelib.frame_defs import OfflineConfigFrame3Format as Off_NRAMF
from paicorelib.framelib.frame_defs import OfflineWorkFrame1Format as Off_WF1F
from paicorelib.framelib.frame_gen import OfflineFrameGen
from paicorelib.framelib.frames import (
    _L_PACKAGE_TYPE_CONF_TESTOUT,
    OfflineTestInFrame3,
    OfflineWorkFrame1,
)
from paicorelib.framelib.types import ArrayType, DataArrayType, FrameArrayType
from paicorelib.framelib.utils import framearray_header_check

from .types import *

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
VOLTAGE_DTYPE = np.int32
VoltageType = NDArray[VOLTAGE_DTYPE]

if hasattr(HwConfig, "N_TIMESLOT_MAX"):
    MAX_TIMESLOT = HwConfig.N_TIMESLOT_MAX - 1  # Start from 0
else:
    MAX_TIMESLOT = 255

# Use the key to represent the length expansion multiple of the output node.
LENGTH_EX_MULTIPLE_KEY = "tick_relative"
_RID_UNSET = RId(0, 0)


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
            for _inode in input_proj_info.values():
                inode = InputProjInfoKeys(_inode)  # cast to typed dict
                raw_ts = inode["tick_relative"]
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

                frames_of_inp = OfflineWorkFrame1._frame_dest_reorganized(
                    cast(dict[str, Any], inode)
                )
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

    @overload
    @staticmethod
    def decode(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: FrameArrayType,
        flatten: bool = False,
    ) -> PayloadDataType: ...

    @overload
    @staticmethod
    def decode(
        timestep: int,
        oframes: FrameArrayType,
        oframe_infos: list[FrameArrayType],
        flatten: bool = False,
    ) -> list[PayloadDataType]: ...

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
            Return the decoded_v output data. If `oframe_infos` is a list, the output will be a list  \
            where each element represents the decoded_v data for each output node.

        NOTE: This method has real-time requirement.
        """
        if oframes.size > 0:
            framearray_header_check(oframes, FH.WORK_TYPE1)

        if isinstance(oframe_infos, list):
            output = []
            # From (0, 0) -> (N, 0)
            seen_core_coords = (
                oframes >> Off_WF1F.GENERAL_CORE_ADDR_OFFSET
            ) & Off_WF1F.GENERAL_CORE_ADDR_MASK

            for i, oframe_info in enumerate(oframe_infos):
                data = np.zeros_like(oframe_info, dtype=PAYLOAD_DATA_DTYPE)
                if oframes.size > 0:
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

                        valid_idx = []
                        for value in oframes_on_coord & (
                            Off_WF1F.GENERAL_MASK - Off_WF1F.DATA_MASK
                        ):
                            matched = np.where(oframe_info == value)[0]
                            if matched.size == 0:  # not matched
                                # TODO Parser the unmatched frame to trace the error.
                                raise ValueError("The output frame is not matched")

                            valid_idx.append(matched[0])

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
            if oframes.size > 0:
                oframes.sort()
                data_on_coord = (oframes >> Off_WF1F.DATA_OFFSET) & Off_WF1F.DATA_MASK

                valid_idx = np.isin(
                    oframe_infos, oframes & (Off_WF1F.GENERAL_MASK - Off_WF1F.DATA_MASK)
                )
                data[valid_idx] = data_on_coord

            d_with_shape = data.reshape(timestep, -1)
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
        """Generate the common information of output frames by given the dictionary of output destinations.

        Args:
            timestep (int): used to tile the 'tick_relative' info of output destinations.
            output_dest_info: the dictionary of output destinations exported from `paibox.Mapper`, or you \
                can specify the following parameters.
            chip_coord: the destination chip coordinate of the output node.
            core_coord: the destination coord coordinate of the output node.
            rid: the replication ID. (maybe unused)
            axons: the range of destination address of axons, from 0 to N.
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

    @staticmethod
    def gen_read_neuron_attrs_frames(
        neu_phy_loc: dict[str, Any],
    ) -> list[FrameArrayType]:
        """Generate test input frame type III for single neuron node to read their attributes.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of a single neuron node. For example:

            >>> d = {
                    "IF_0": {
                        "(0,0)": {
                            "(0,0)": {
                                "n_neuron": 50,
                                "addr_offset": 0,
                                "interval": 8,
                                "idx_offset": 0
                            },
                            "(0,1)": {
                                "n_neuron": 50,
                                "addr_offset": 0,
                                "interval": 8,
                                "idx_offset": 50
                            }
                        }
                    }
                }
            >>> tframes3_if0 = PAIBoxRuntime.gen_read_neuron_attrs_frames(d["IF_0"])

        NOTE: Test output frames will be output to `test_chip_addr` which is configured before. Since the   \
            chip has a hardware flaw that when the SRAM is read continuously, the second neuron will be     \
            missed. To avoid this, we need to read the neuron attributes one by one. It takes less time to  \
            read the neuron attributes individually than sequentially.
        """
        tframe3: list[OfflineTestInFrame3] = []

        for chip_coord, core_locs in neu_phy_loc.items():
            for core_coord, _seg_addr in core_locs.items():
                nseg_addr = NeuSegAddrKeys(_seg_addr)  # cast to typed dict
                for i in range(nseg_addr["n_neuron"]):
                    # NOTE: Mapping between logical neuron indexes, neuron addresses & SRAM addresses:
                    # Logical idx:                    [0]                           [1]
                    #                   |<------- interval=8 -------->|                             |
                    # Neuron address:      [0]      [1]   ...   [7]      [8]        ...       [15]
                    # SRAM address:     [0*4+:4] [1*4+:4] ... [7*4+:4] [8*4+:4]     ...     [15*4+:4]
                    # NOTE: According to our experiments, the attributes of each logical neuron at index `i` is stored
                    # many times repeatedly in the **neuron address** addr_offset+[i],[i+1],...,[i+interval-1].
                    # However, the correct voltage is stored in the **neuron address** addr_offset+[i], or
                    # [(addr_offset+i)*4+:4] in SRAM. Reading [i] is the most efficient way to get all attributes. This
                    # behavior above is not officially documented in the chip manual.
                    tframe3.append(
                        OfflineFrameGen.gen_testin_frame3(
                            ChipCoord(*coordstr_to_tuple(chip_coord)),
                            Coord(*coordstr_to_tuple(core_coord)),
                            _RID_UNSET,
                            # NOTE: Attention! The argument `sram_base_addr` of config/test frame 3 & 4 is incorrectly
                            # named. In fact, it is the **neuron address** as described above.
                            nseg_addr["addr_offset"] + i * nseg_addr["interval"],
                            4,
                        )
                    )

        return [f.value for f in tframe3]

    @staticmethod
    def decode_neuron_voltage(
        neu_phy_loc: dict[str, Any], otframes3: FrameArrayType
    ) -> VoltageType:
        """Decode type III test output frames of a single neuron node for reading the voltage. The physical     \
            locations of the neurons will be aligned with their logical positions.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of a single neuron node.
            otframes3 (FrameArrayType): the test output frames of type III.
            
            >>> decoded_v = PAIBoxRuntime.decode_neuron_voltage(d["IF_0"], otframes3_if0)
            
            where `otframes3_if0` is the output frames of type III for neuron 'IF_0'.

        NOTE: Only single node decoding is supported for now. The test output frames for each neuron may be     \
            unordered.
        """
        if (n_neu := len(neu_phy_loc)) > 1:
            raise ValueError(
                f"Only single node decoding is supported for now, but got {n_neu}."
            )

        neu = list(neu_phy_loc)[0]
        n_neuron_total = 0
        core_locs: dict[Coord, NeuSegAddrKeys] = {}  # records the core coordinates
        for coord_str, _seg_addr in neu_phy_loc[neu].items():
            cur_coord = Coord(*coordstr_to_tuple(coord_str))
            nseg_addr = NeuSegAddrKeys(_seg_addr)  # cast to typed dict
            core_locs[cur_coord] = nseg_addr
            n_neuron_total += nseg_addr["n_neuron"]

        decoded_v = np.zeros((n_neuron_total,), dtype=VOLTAGE_DTYPE)

        def parser(otframe3: FrameArrayType, i: Optional[int] = None) -> None:
            """Parser of test output frame type III."""
            if i is None:
                i = 0

            while i < len(otframe3):
                cur_frame = int(otframe3[i])
                if (  # check current frame only
                    FH.TEST_TYPE3
                    == FH(
                        (cur_frame >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK
                    )
                    and (cur_frame >> FF.GENERAL_PACKAGE_TYPE_OFFSET)
                    & FF.GENERAL_PACKAGE_TYPE_MASK
                    == _L_PACKAGE_TYPE_CONF_TESTOUT
                ):  # here is the start frame
                    core_coord = (
                        cur_frame >> FF.GENERAL_CORE_ADDR_OFFSET
                    ) & FF.GENERAL_CORE_ADDR_MASK
                    neu_addr = (
                        cur_frame >> FF.GENERAL_PACKAGE_SRAM_ADDR_OFFSET
                    ) & FF.GENERAL_PACKAGE_SRAM_ADDR_MASK
                    n_package = (
                        cur_frame >> FF.GENERAL_PACKAGE_NUM_OFFSET
                    ) & FF.GENERAL_PACKAGE_NUM_MASK

                    assert n_package == 4
                    # The voltage is at #1 of 4 frames in package.
                    v = (
                        int(otframe3[i + 1]) >> Off_NRAMF.VJT_PRE_OFFSET
                    ) & Off_NRAMF.VJT_PRE_MASK

                    if (coord := Coord.from_addr(core_coord)) not in core_locs:
                        expected = ", ".join(str(c) for c in core_locs)
                        raise ValueError(
                            f"{coord} is not in expected locations: {expected}."
                        )

                    nseg_addr = core_locs[coord]
                    # See comments in `gen_read_neuron_attrs_frames()` above.
                    logic_idx = nseg_addr["idx_offset"] + (
                        (neu_addr - nseg_addr["addr_offset"]) // nseg_addr["interval"]
                    )
                    decoded_v[logic_idx] = convert_30bit_to_signed(v)

                    i += n_package + 1
                else:
                    raise ValueError("Invalid test output frame type III")

        parser(otframes3)

        return decoded_v


def convert_30bit_to_signed(x: int) -> VOLTAGE_DTYPE:
    """Convert an integer to a 32-bit signed number."""
    x_30b = x & ((1 << 30) - 1)

    if (x_30b >> 29) & 1:
        # if the 30th bit is 1, the number is negative
        x_30b -= 0x4000_0000

    return VOLTAGE_DTYPE(x_30b)
