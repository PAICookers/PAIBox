from typing import Any, List, Optional, Tuple, Union

import numpy as np

from paibox.frame.params import FrameHead
from paibox.libpaicore.v2 import *
from paibox.libpaicore.v2 import Coord, ReplicationId

from .params import *

__all__ = ["Frame", "FramePackage"]


class Frame:
    """frames which contains information.

    single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits

    frames group:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits

    The number of Header is 1,and the number of others is N.

    """

    def __init__(
        self,
        header: Optional[FrameHead] = None,
        chip_coord: Optional[Union[List[Coord], Coord]] = None,
        core_coord: Optional[Union[List[Coord], Coord]] = None,
        core_ex_coord: Optional[Union[List[ReplicationId], ReplicationId]] = None,
        payload: Optional[Union[List[int], int, np.ndarray]] = None,
        value: Optional[np.ndarray] = None,
    ) -> None:
        info = [header, chip_coord, core_coord, core_ex_coord, payload]

        if value is not None and all(var is None for var in info):
            (
                self.header,
                self.chip_coord,
                self.core_coord,
                self.core_ex_coord,
                self.payload,
            ) = self._get_frame_by_value(value)

            self._value = value

        elif value is None and all(var is not None for var in info):
            (
                self.header,
                self.chip_coord,
                self.core_coord,
                self.core_ex_coord,
                self.payload,
            ) = self._type_format(
                header, chip_coord, core_coord, core_ex_coord, payload  # type: ignore
            )

            temp_header = self.header_value & FrameFormat.GENERAL_HEADER_MASK
            temp_chip_address = self.chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
            temp_core_address = self.core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
            temp_core_ex_address = (
                self.core_ex_address & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
            )
            temp_payload = self.payload & FrameFormat.GENERAL_PAYLOAD_MASK

            self._value = np.array(
                (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
                | (temp_chip_address << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
                | (temp_core_address << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
                | (temp_core_ex_address << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
                | (temp_payload << FrameFormat.GENERAL_PAYLOAD_OFFSET)
            ).astype(np.uint64)

        else:
            raise ValueError(
                "The input parameters are not correct, please check the input parameters."
            )

    def _type_format(
        self,
        header=FrameHead,
        chip_coord=Union[List[Coord], Coord],
        core_coord=Union[List[Coord], Coord],
        core_ex_coord=Union[List[ReplicationId], ReplicationId],
        payload=Union[List[int], int, np.ndarray],
    ) -> Tuple:
        # function implementation
        chip_coord = [chip_coord] if not isinstance(chip_coord, list) else chip_coord
        core_coord = [core_coord] if not isinstance(core_coord, list) else core_coord
        core_ex_coord = (
            [core_ex_coord] if not isinstance(core_ex_coord, list) else core_ex_coord
        )
        payload = (
            np.array([payload]).astype(np.uint64)
            if isinstance(payload, int)
            else np.array(payload).astype(np.uint64)
        )

        return header, chip_coord, core_coord, core_ex_coord, payload

    @classmethod
    def get_frame(
        cls,
        header: FrameHead,
        chip_coord: Union[List[Coord], Coord],
        core_coord: Union[List[Coord], Coord],
        core_ex_coord: Union[List[ReplicationId], ReplicationId],
        payload: Union[List[int], int, np.ndarray],
    ) -> "Frame":
        return cls(
            header=header,
            chip_coord=chip_coord,
            core_coord=core_coord,
            core_ex_coord=core_ex_coord,
            payload=payload,
        )

    def _get_frame_by_value(self, value: np.ndarray):
        """从64位的frame中解析出各个字段的值"""

        header_list = (
            value >> FrameFormat.GENERAL_HEADER_OFFSET
        ) & FrameFormat.GENERAL_HEADER_MASK
        chip_coord_list = (
            value >> FrameFormat.GENERAL_CHIP_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CHIP_ADDR_MASK
        core_coord_list = (
            value >> FrameFormat.GENERAL_CORE_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CORE_ADDR_MASK
        core_ex_coord_list = (
            value >> FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
        payload_list = (
            value >> FrameFormat.GENERAL_PAYLOAD_OFFSET
        ) & FrameFormat.GENERAL_PAYLOAD_MASK

        header_list = header_list.tolist()
        if all(header == header_list[0] for header in header_list):
            header = FrameHead(header_list[0])  # type: ignore
        else:
            raise ValueError(
                "The header of the frame is not the same, please check the frames value."
            )

        payload_list = payload_list.tolist()

        chip_coord = [Coord.from_addr(coord) for coord in chip_coord_list.tolist()]
        core_coord = [Coord.from_addr(coord) for coord in core_coord_list.tolist()]
        core_ex_coord = [
            ReplicationId.from_addr(coord) for coord in core_ex_coord_list.tolist()
        ]
        payload = np.array(payload_list).astype(np.uint64)

        return (
            header,
            chip_coord,
            core_coord,
            core_ex_coord,
            payload,
        )

    def __len__(self) -> int:
        if isinstance(self._value, np.ndarray):
            return len(self._value)
        else:
            return 1

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def header_value(self) -> np.ndarray:
        return np.array([self.header.value]).astype(np.uint64)  # type: ignore

    @property
    def chip_address(self) -> np.ndarray:
        return np.array([coord.address for coord in self.chip_coord]).astype(np.uint64)

    @property
    def core_address(self) -> np.ndarray:
        return np.array([coord.address for coord in self.core_coord]).astype(np.uint64)

    @property
    def core_ex_address(self) -> np.ndarray:
        return np.array([coord.address for coord in self.core_ex_coord]).astype(
            np.uint64
        )

    def __repr__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:             {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_EX address:  {self.core_ex_coord}\n"
            f"Payload:          {self.payload}\n"
        )


class FramePackage:
    """Frame package for a length of `N` contents:
    1. [Header(sub type)] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
            4 bits               10 bits            10 bits            10 bits          30 bits
    2. [contents[0]], 64 bits.
    N+1. [contents[N-1]], 64 bits.

    """

    def __init__(
        self,
        header: Optional[FrameHead] = None,
        chip_coord: Optional[Coord] = None,
        core_coord: Optional[Coord] = None,
        core_ex_coord: Optional[ReplicationId] = None,
        payload: Optional[Union[int, np.uint64]] = None,
        data_package: Optional[Union[List, np.ndarray]] = None,
        value: Optional[np.ndarray] = None,
    ) -> None:
        info = [header, chip_coord, core_coord, core_ex_coord, payload, data_package]

        if value is not None and all(var is None for var in info):
            (
                self.header,
                self.chip_coord,
                self.core_coord,
                self.core_ex_coord,
                self.payload,
            ) = self._get_first_frame_by_value(value[0])
            self.data_package = value[1:]
            self._length = len(value)
            self._value = value

        elif value is None and all(var is not None for var in info):
            if isinstance(payload, int):
                payload = np.array(payload).astype(np.uint64)

            if isinstance(data_package, list):
                data_package = np.array(data_package).astype(np.uint64)

            self.header = header
            self.chip_coord = chip_coord
            self.core_coord = core_coord
            self.core_ex_coord = core_ex_coord
            self.payload = payload
            self.data_package = data_package

            self._length = len(self.data_package) + 1  # type: ignore

            temp_header = np.array(self.header.value).astype(np.uint64) & FrameFormat.GENERAL_HEADER_MASK  # type: ignore
            temp_chip_addr = (
                np.array(self.chip_coord.address).astype(np.uint64) & FrameFormat.GENERAL_CHIP_ADDR_MASK  # type: ignore
            )
            temp_core_addr = (
                np.array(self.core_coord.address).astype(np.uint64) & FrameFormat.GENERAL_CORE_ADDR_MASK  # type: ignore
            )
            temp_core_ex_addr = (
                np.array(self.core_ex_coord.address).astype(np.uint64)  # type: ignore
                & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
            )
            temp_payload = self.payload & FrameFormat.GENERAL_PAYLOAD_MASK  # type: ignore

            temp = np.uint64(
                (
                    (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
                    | (temp_chip_addr << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
                    | (temp_core_addr << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
                    | (temp_core_ex_addr << FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET)
                    | (temp_payload << FrameFormat.GENERAL_PAYLOAD_OFFSET)
                )
            )

            frame_package = np.array([]).astype(np.uint64)
            frame_package = np.append(frame_package, temp)
            if self.data_package.size != 0:  # type: ignore
                frame_package = np.append(frame_package, self.data_package)  # type: ignore

            self._value = np.array(frame_package).astype(np.uint64)

        else:
            raise ValueError(
                "The input parameters are not correct, please check the input parameters."
            )

    def _get_first_frame_by_value(self, value1):
        header = (
            value1 >> FrameFormat.GENERAL_HEADER_OFFSET
        ) & FrameFormat.GENERAL_HEADER_MASK
        chip_coord = (
            value1 >> FrameFormat.GENERAL_CHIP_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CHIP_ADDR_MASK
        core_coord = (
            value1 >> FrameFormat.GENERAL_CORE_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CORE_ADDR_MASK
        core_ex_coord = (
            value1 >> FrameFormat.GENERAL_CORE_EX_ADDR_OFFSET
        ) & FrameFormat.GENERAL_CORE_EX_ADDR_MASK
        payload = (
            value1 >> FrameFormat.GENERAL_PAYLOAD_OFFSET
        ) & FrameFormat.GENERAL_PAYLOAD_MASK

        header = FrameHead(header)
        chip_coord = Coord.from_addr(int(chip_coord))
        core_coord = Coord.from_addr(int(core_coord))
        core_ex_coord = ReplicationId.from_addr(int(core_ex_coord))
        payload = np.array(payload).astype(np.uint64)
        return header, chip_coord, core_coord, core_ex_coord, payload

    @property
    def length(self) -> int:
        return self._length

    @property
    def n_package(self) -> int:
        return self.data_package.size  # type: ignore

    @property
    def value(self) -> np.ndarray:
        return self._value

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        _present = f"FramePackage info:\n" f"Chip address:     {self.chip_coord}\n"

        _present = (
            f"FramePackage info:\n"
            f"Header:           {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_EX address:  {self.core_ex_coord}\n"
            f"Payload:          {self.payload}\n"
            f"Data:\n"
        )

        if len(self.data_package) != 0:  # type: ignore
            for i in range(self.length - 1):
                _present += f"#{i}:{self.data_package[i]}\n"  # type: ignore
        return _present
