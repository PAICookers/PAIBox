"""
The `PAIBoxRuntime` only guarantees that its functional features are consistent with
the PAIBox released with it, and that the hardware platform can implement custom
utilities on it. Make sure you know what you're doing when you make changes.

The runtime dose not depend on any modules of PAIBox.
"""

import sys
from typing import Any, Literal, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray
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

from paibox.runtime.types import NeuSegAddrKeys

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
            output_dest_info: the dictionary of output destinations exported from `paibox.Mapper`, or you   \
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
        reading_mode: Literal["onebyone", "contiguous"] = "contiguous",
    ) -> list[tuple[FrameArrayType, int]]:
        """Generate test input frame type III for single neuron node to read their attributes.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of a single neuron node. For example:
            reading_mode ("onebyone", "contiguous"):
                - "onebyone": read the addresses of neurons that contain the correct voltage at intervals.
                - "contiguous" (default): read the addresses of neurons contiguously. Necessary to retrieve \
                    the addresses that store the correct voltage based on the interval.

        Returns:
            A list of tuples of test input frames & the number of packages to read.

        Usage:

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
            >>> tframes3_if0[0]
            >>> (input test frame, #N of packages to read)

        NOTE: Test output frames will be output to `test_chip_addr` which is configured before. The output  \
            frames will be out of order if using the replication id to test multiple cores at the same time.

        NOTE: Since the chip has a hardware flaw that once read the neuron addresses contiguously, the 2nd  \
            address will be missed maybe. According to out experiments, for example, reading the neuron     \
            addresses [0]~[99], the output frame will be of the neurons addresses [0] & [2]-[100].

            This behavior is not officially documented in any chip manuals.
        """
        # [(frame package, n_package)]
        tframe3: list[tuple[OfflineTestInFrame3, int]] = []
        assert reading_mode in ("onebyone", "contiguous")

        for _chip_coord, core_locs in neu_phy_loc.items():
            chip_coord = ChipCoord(*coordstr_to_tuple(_chip_coord))
            for _core_coord, _seg_addr in core_locs.items():
                core_coord = Coord(*coordstr_to_tuple(_core_coord))
                nseg_addr = NeuSegAddrKeys(_seg_addr)  # cast to typed dict
                n_neuron = nseg_addr["n_neuron"]
                addr_offset = nseg_addr["addr_offset"]
                interval = nseg_addr["interval"]

                if reading_mode == "onebyone":
                    for i in range(n_neuron):
                        # NOTE: Mapping between logical neuron indexes, neuron addresses & SRAM addresses:
                        # Logical idx:                    [0]                           [1]
                        #                   |<------- interval=8 -------->|                             |
                        # Neuron address:      [0]      [1]   ...   [7]      [8]        ...       [15]
                        # SRAM address:     [0*4+:4] [1*4+:4] ... [7*4+:4] [8*4+:4]     ...     [15*4+:4]
                        # NOTE: According to our experiments, the attributes of each logical neuron at index `i` is stored
                        # many times repeatedly in the **neuron address** addr_offset+[i],[i+1],...,[i+interval-1].
                        # However, the correct voltage is stored in the **neuron address** addr_offset+[i], or
                        # [(addr_offset+i)*4+:4] in SRAM. Reading [i] is the most efficient way to get all attributes.
                        # This behavior above is not officially documented in any chip manuals.
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    # NOTE: Attention! The argument `sram_base_addr` of config/test frame 3 & 4 is incorrectly
                                    # named. In fact, it is the **neuron start address** as described above.
                                    addr_offset + i * interval,
                                    4,
                                ),
                                4,
                            )
                        )
                else:
                    if interval == 1:
                        # Read two times if #N of neurons > 1, otherwise read once.
                        # 1. Set neuron start addr=addr_offset,   n_package=4*1*(N-1), to read neuron addr_offset+[0] & [2]~[N-1].
                        # 2. Set neuron start addr=1+addr_offset, n_package=4*1*1,     to read neuron addr_offset+[1](if N > 1).
                        n_package = 4 * (n_neuron - 1) if n_neuron > 1 else 4
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    addr_offset,
                                    n_package,
                                ),
                                n_package,
                            )
                        )
                        if n_neuron > 1:
                            tframe3.append(
                                (
                                    OfflineFrameGen.gen_testin_frame3(
                                        chip_coord,
                                        core_coord,
                                        _RID_UNSET,
                                        1 + addr_offset,
                                        4,
                                    ),
                                    4,
                                )
                            )
                    else:
                        # Although the addresses of neurons read contiguously have deviations, the address
                        # containing the correct voltage have still been read accurately.
                        # When interval > 1, for example 4:
                        # In order to read neuron [0], [4], [4*2], ..., [4*N], set test input frame with:
                        #   start addr = addr_offset
                        #   n_package = 4*interval(4)*N
                        # Return addresses: [0], [2], [3], [4], ..., [4*N+1]
                        n_package = 4 * interval * n_neuron
                        tframe3.append(
                            (
                                OfflineFrameGen.gen_testin_frame3(
                                    chip_coord,
                                    core_coord,
                                    _RID_UNSET,
                                    addr_offset,
                                    n_package,
                                ),
                                n_package,
                            )
                        )

        return [(f.value, n_package) for (f, n_package) in tframe3]

    @staticmethod
    def decode_voltage(
        neu_phy_loc: dict[str, Any],
        *otf3_package: FrameArrayType,
        reading_mode: Literal["onebyone", "contiguous"] = "contiguous",
        is_complete: bool = True,
    ) -> VoltageType:
        """Decode type III test output frames of a single neuron node for reading the voltage. The physical \
            locations of the neurons will be aligned with their logical positions.

        Args:
            neu_phy_loc (dict[str, Any]): the physical locations of a single neuron node.
            otf3_package (FrameArrayType): the test output frames of type III.
            reading_mode ("onebyone", "contiguous"): the reading mode of neuron addresses.
            is_complete (bool): whether the current decoding is for all neurons declared in `neu_phy_loc`.  \
                If not, it's necessary to distinguish the decoded voltage returned by multiple calls to this\
                function.

        Usage:

            >>> ...
            >>> reading_mode = "contiguous"
            >>> tframes3_if0 = PAIBoxRuntime.gen_read_neuron_attrs_frames(d["IF_0"], reading_mode)

            >>> # At hardware platform:
            >>> otframes = []
            >>> for (item, n_package) in tframes3_if0:
            >>>     itf = item[0]
            >>>     send(itf) # Send to the chip
            >>>     r = recv(n_package+2)[:1+n_package] # Receive & retrieve: start frame + #N packages
            >>>     otframes.append(r)
            >>> decoded_v = PAIBoxRuntime.decode_voltage(d["IF_0"], *otframes, reading_mode="contiguous)
        """
        n_neuron_total, core_nseg_locs = get_info_neu_phy_loc(neu_phy_loc)
        decoded_v = np.zeros((n_neuron_total,), dtype=VOLTAGE_DTYPE)

        n_neu_proc = 0
        for otf3 in otf3_package:
            assert otf3.ndim == 1
            n_neu_proc += decode_partial_voltage(
                otf3, core_nseg_locs, reading_mode, decoded_v
            )

        if is_complete and (n_neu_proc != n_neuron_total):
            raise ValueError(
                f"The number of total neurons decoded is {n_neu_proc}, but expected {n_neuron_total}"
            )

        return decoded_v


def get_info_neu_phy_loc(
    neu_phy_loc: dict[str, Any],
) -> tuple[int, CoreNeuSegLocType]:
    """Get the information of neuron physical locations.

    Returns:
        A tuple of the total number of neurons & the dictionary of core-neuron segment locations.
    """
    if (n_neu := len(neu_phy_loc)) > 1:
        raise ValueError(
            f"Only single node decoding is supported for now, but got {n_neu}."
        )

    neu = list(neu_phy_loc)[0]
    n_neuron_total = 0
    core_nseg_locs: CoreNeuSegLocType = {}
    for coord_str, _seg_addr in neu_phy_loc[neu].items():
        cur_coord = Coord(*coordstr_to_tuple(coord_str))
        nseg_addr = NeuSegAddrKeys(_seg_addr)  # cast to typed dict
        core_nseg_locs[cur_coord] = nseg_addr
        n_neuron_total += nseg_addr["n_neuron"]

    return n_neuron_total, core_nseg_locs


def decode_partial_voltage(
    otframe3: FrameArrayType,
    core_locs: CoreNeuSegLocType,
    reading_mode: Literal["onebyone", "contiguous"],
    out: VoltageType,
) -> int:
    """According to the test output frames of each core `otframe3` for a single neuron node, decode the \
        corresponding voltage for this part and store it in the array `out`.

    Args:
        otframe3 (FrameArrayType): the test output frame of type III.
        core_locs (dict[Coord, NeuSegAddrKeys]): the dictionary of core-neuron segment locations.
        reading_mode ("onebyone", "contiguous"): the reading mode of neuron addresses.
        out (VoltageType): the array to store the complete decoded voltages.

    Returns:
        n_neuron (int): The number of processed neurons in the current package.

    NOTE: If reading mode is "onebyone", `otframe3` is a frame package of length 4*1*1. Decode 1 neuron \
        for each package.
        If reading mode is "contiguous", `otframe3` is a frame package of length 4*interval*N. Decode N \
        neurons for each package.

        Since the chip has a hardware flaw that once read the neuron addresses contiguously, the 2nd    \
        address will be missed maybe. The method for retrieving the addresses that contain the correct  \
        voltage from the discontiguous neuron addresses, and corresponding them with logical positions, \
        is derived from our experiments.

        This behavior is not officially documented in any chip manuals.
    """
    start_frame = int(otframe3[0])
    if not (
        FH.TEST_TYPE3
        == FH((start_frame >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK)
        and (start_frame >> FF.GENERAL_PACKAGE_TYPE_OFFSET)
        & FF.GENERAL_PACKAGE_TYPE_MASK
        == _L_PACKAGE_TYPE_CONF_TESTOUT
    ):
        raise ValueError("Invalid test output frame type III")

    core_coord = (
        start_frame >> FF.GENERAL_CORE_ADDR_OFFSET
    ) & FF.GENERAL_CORE_ADDR_MASK
    neu_addr = (
        start_frame >> FF.GENERAL_PACKAGE_SRAM_ADDR_OFFSET
    ) & FF.GENERAL_PACKAGE_SRAM_ADDR_MASK
    n_package = (
        start_frame >> FF.GENERAL_PACKAGE_NUM_OFFSET
    ) & FF.GENERAL_PACKAGE_NUM_MASK

    if n_package != otframe3.size - 1:
        raise ValueError(
            f"The number of packages is expected to be {n_package}, but got {otframe3.size-1}"
        )

    if (coord := Coord.from_addr(core_coord)) not in core_locs:
        expected = ", ".join(str(c) for c in core_locs)
        raise ValueError(f"{coord} is not in expected locations: {expected}.")

    nseg_addr = core_locs[coord]
    interval = nseg_addr["interval"]

    if reading_mode == "onebyone":
        if (n_neu_proc := n_package // 4) != 1:
            raise ValueError(
                "When reading neuron addresses one by one, the number of packages is expected to be 4, "
                f"but got {n_package}"
            )
    else:
        if (n_neu_proc := n_package // (4 * interval)) > out.size:
            raise ValueError(
                f"The number of packages exceeds the max size: {n_package} > {4*interval*out.size}"
            )

    # Get the voltage of neuron[0]. Slice starting from 1 to skip the start frame.
    v_array_idx0 = (
        int(otframe3[1]) >> Off_NRAMF.VJT_PRE_OFFSET
    ) & Off_NRAMF.VJT_PRE_MASK

    # See comments in `gen_read_neuron_attrs_frames()` above.
    logic_idx = nseg_addr["idx_offset"] + (
        (neu_addr - nseg_addr["addr_offset"]) // interval
    )
    out[logic_idx] = convert_30bit_to_signed(v_array_idx0)

    if reading_mode == "onebyone" or n_neu_proc == 1:
        return 1

    # In contiguous mode, read the rest frames containing the voltage of neurons[2:N-1]
    start_idx_2nd = 1 + 4 * (interval - 1) if interval > 1 else 1 + 4 * 1
    end_idx_2nd = (-1 * 4) if interval > 1 else None

    # When interval>1, the 2nd~#N neuron address is in order in `otframe3`, while interval=1,
    # the 2nd neuron address is missed, so start from `logic_idx+2`.
    start_logic_idx_2nd = logic_idx + 1 if interval > 1 else logic_idx + 2

    v_array = (
        otframe3[start_idx_2nd : end_idx_2nd : 4 * interval] >> Off_NRAMF.VJT_PRE_OFFSET
    ) & Off_NRAMF.VJT_PRE_MASK

    assert v_array.size == n_neu_proc - 1

    vf_convert = np.vectorize(convert_30bit_to_signed, otypes=[VOLTAGE_DTYPE])
    out[start_logic_idx_2nd : start_logic_idx_2nd + (n_neu_proc - 1)] = vf_convert(
        v_array
    )

    return n_neu_proc


def convert_30bit_to_signed(x: int) -> VOLTAGE_DTYPE:
    """Convert an integer to a 32-bit signed number."""
    x_30b = x & ((1 << 30) - 1)

    if (x_30b >> 29) & 1:
        x_30b -= 0x4000_0000  # Negative integer

    return VOLTAGE_DTYPE(x_30b)
