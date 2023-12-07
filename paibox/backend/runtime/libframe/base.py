from dataclasses import dataclass, field
from typing import Union, Tuple
import numpy as np

from ._types import BasicFrameArray, FRAME_DTYPE, FrameArrayType
from .utils import check_elem_same, header2type
from paibox.libpaicore import (
    Coord,
    FrameFormat as FF,
    FrameHeader as FH,
    FrameType as FT,
    ReplicationId as RId,
)


__all__ = ["FrameFactory"]


@dataclass
class Frame:
    """frames which contains information.

    single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits

    frames group:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits
    """

    header: FH
    chip_coord: Coord
    core_coord: Coord
    rid: RId
    payload: FrameArrayType = field(
        default_factory=lambda: np.empty(0, dtype=FRAME_DTYPE)
    )

    @classmethod
    def _decode(
        cls,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        payload: FrameArrayType,
    ):
        return cls(header, chip_coord, core_coord, rid, payload)

    @property
    def frame_type(self) -> FT:
        return header2type(self.header)

    @property
    def chip_addr(self) -> int:
        return self.chip_coord.address

    @property
    def core_addr(self) -> int:
        return self.core_coord.address

    @property
    def rid_addr(self) -> int:
        return self.rid.address

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the single frame."""
        value = self._frame_common + (self.payload & FF.GENERAL_PAYLOAD_MASK)
        value = np.asarray(value, dtype=FRAME_DTYPE)
        value.setflags(write=False)

        return value

    @property
    def _frame_common(self) -> int:
        header = self.header.value & FF.GENERAL_HEADER_MASK
        chip_addr = self.chip_addr & FF.GENERAL_CHIP_ADDR_MASK
        core_addr = self.core_addr & FF.GENERAL_CORE_ADDR_MASK
        rid_addr = self.rid_addr & FF.GENERAL_CORE_EX_ADDR_MASK

        return (
            (header << FF.GENERAL_HEADER_OFFSET)
            + (chip_addr << FF.GENERAL_CHIP_ADDR_OFFSET)
            + (core_addr << FF.GENERAL_CORE_ADDR_OFFSET)
            + (rid_addr << FF.GENERAL_CORE_EX_ADDR_OFFSET)
        )

    def __len__(self) -> int:
        return self.payload.size

    def __str__(self) -> str:
        return (
            f"Frame info:\n"
            f"Head:                 {self.header}\n"
            f"Chip address:         {self.chip_coord}\n"
            f"Core address:         {self.core_coord}\n"
            f"Replication address:  {self.rid}\n"
            f"Payload:              {self.payload}\n"
        )


@dataclass
class FramePackage(Frame):
    """Frame package for a length of `N` contents:

    1. [Header(sub type)] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
            4 bits               10 bits            10 bits            10 bits          30 bits
    2. [contents[0]], 64 bits.
    N+1. [contents[N-1]], 64 bits.

    """

    payload: FRAME_DTYPE
    packages: FrameArrayType = field(
        default_factory=lambda: np.empty(0, dtype=FRAME_DTYPE)
    )

    @classmethod
    def _decode(
        cls,
        header: FH,
        chip_coord: Coord,
        core_coord: Coord,
        rid: RId,
        payload: FRAME_DTYPE,
        packages: FrameArrayType,
    ):
        assert payload.ndim == 1
        return cls(header, chip_coord, core_coord, rid, payload, packages)

    @property
    def n_package(self) -> int:
        return self.packages.size

    @property
    def value(self) -> FrameArrayType:
        """Get the full frames of the group."""
        value = np.zeros((len(self),), dtype=FRAME_DTYPE)

        value[0] = self._frame_common + (int(self.payload) & FF.GENERAL_PAYLOAD_MASK)
        value[1:] = self.packages.copy()
        value.setflags(write=False)

        return value

    def __len__(self) -> int:
        return 1 + self.n_package

    def __str__(self) -> str:
        _present = (
            f"FramePackage info:\n"
            f"Header:               {self.header}\n"
            f"Chip address:         {self.chip_coord}\n"
            f"Core address:         {self.core_coord}\n"
            f"Replication address:  {self.rid}\n"
            f"Payload:              {self.payload}\n"
            f"Data:\n"
        )

        for i in range(self.n_package):
            _present += f"#{i}: {self.packages[i]}\n"

        return _present


class FrameFactory:
    @staticmethod
    def _is_package_format(header: FH, bit19: int) -> bool:
        """Check whether the frame array is in package format.

        Config type III & IV, or test out type III or IV (bit 19 is 0).
        """
        return (header is FH.CONFIG_TYPE3 or header is FH.CONFIG_TYPE4) or (
            (header is FH.TEST_TYPE3 or header is FH.TEST_TYPE4) and bit19 == 0
        )

    @staticmethod
    def framearray2np(frame_array: BasicFrameArray) -> FrameArrayType:
        if isinstance(frame_array, int):
            nparray = np.asarray([frame_array], dtype=FRAME_DTYPE)
        elif isinstance(frame_array, np.ndarray):
            if frame_array.ndim != 1:
                # TODO
                raise ValueError(f"Shape wrong")

            nparray = frame_array.astype(FRAME_DTYPE)
        elif isinstance(frame_array, (list, tuple)):
            nparray = np.asarray(frame_array, dtype=FRAME_DTYPE)
        else:
            # TODO
            raise TypeError

        return nparray

    @staticmethod
    def _extract_common(
        frame_array: FrameArrayType,
    ) -> Tuple[Coord, Coord, RId, FrameArrayType]:
        chip_coords = (
            frame_array >> FF.GENERAL_CHIP_ADDR_OFFSET
        ) & FF.GENERAL_CHIP_ADDR_MASK
        core_coords = (
            frame_array >> FF.GENERAL_CORE_ADDR_OFFSET
        ) & FF.GENERAL_CORE_ADDR_MASK
        rids = (
            frame_array >> FF.GENERAL_CORE_EX_ADDR_OFFSET
        ) & FF.GENERAL_CORE_EX_ADDR_MASK
        payload = (frame_array >> FF.GENERAL_PAYLOAD_OFFSET) & FF.GENERAL_PAYLOAD_MASK

        if not check_elem_same(chip_coords):
            raise ValueError(
                "The header of the frame is not the same, please check the frames value."
            )

        if not check_elem_same(core_coords):
            raise ValueError(
                "The header of the frame is not the same, please check the frames value."
            )

        if not check_elem_same(rids):
            raise ValueError(
                "The header of the frame is not the same, please check the frames value."
            )

        chip_coord = chip_coords[0]
        core_coord = core_coords[0]
        rid = rids[0]

        return (
            Coord.from_addr(int(chip_coord)),
            Coord.from_addr(int(core_coord)),
            RId.from_addr(int(rid)),
            payload.astype(FRAME_DTYPE),
        )

    @classmethod
    def decode(cls, value: BasicFrameArray) -> Union[Frame, FramePackage]:
        nparray = FrameFactory.framearray2np(value)

        header0 = FH(
            (int(nparray[0]) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK
        )

        bit19 = (
            int(nparray[0]) >> FF.DATA_PACKAGE_TYPE_OFFSET
        ) & FF.DATA_PACKAGE_TYPE_MASK

        if FrameFactory._is_package_format(header0, bit19):
            chip_coord, core_coord, rid, payload = FrameFactory._extract_common(
                nparray[0]
            )
            return FramePackage._decode(
                header0, chip_coord, core_coord, rid, FRAME_DTYPE(payload), nparray[1:]
            )

        headers = (nparray >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK

        if not check_elem_same(headers):
            raise ValueError(
                "The header of the frame is not the same, please check the frames value."
            )
        else:
            return Frame._decode(header0, *FrameFactory._extract_common(nparray))

    # @classmethod
    # def encode(
    #     cls,
    #     header: FH,
    #     chip_coord: CoordLike,
    #     core_coord: CoordLike,
    #     rid: CoordLike,
    #     payload: BasicFrameArray,
    #     packages: Optional[BasicFrameArray] = None,
    # ) -> Union[Frame, FramePackage]:
    #     _chip_coord = to_coord(chip_coord)
    #     _core_coord = to_coord(core_coord)
    #     _rid = RId(*to_coord(rid).to_tuple())
    #     _payload = FrameFactory.framearray2np(payload)

    #     bit19 = (
    #         int(_payload[0]) >> FF.DATA_PACKAGE_TYPE_OFFSET
    #     ) & FF.DATA_PACKAGE_TYPE_MASK

    #     if FrameFactory._is_package_format(header, bit19):
    #         if packages is None:
    #             # TODO
    #             raise ValueError

    #         _packages = FrameFactory.framearray2np(packages)
    #         return FramePackage._decode(
    #             header, _chip_coord, _core_coord, _rid, _payload, _packages
    #         )

    #     else:
    #         return Frame._decode(header, _chip_coord, _core_coord, _rid, _payload)
