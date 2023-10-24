from paibox.frame.params import FrameHead
from paibox.libpaicore.v2 import Coord, ReplicationId
from .params import *
from paibox.libpaicore.v2 import *

from typing import List,Union
import numpy as np

__all__ = ["Frame", "FrameGroup", "FramePackage"]

class Frame:
    """Single frame which contains information.

    Single frame:
        [Header] + [chip coordinate] + [core coordinate] + [replication id] + [payload]
        4 bits         10 bits             10 bits             10 bits         30 bits
    """

    def __init__(self,
                header: FrameHead,
                chip_coord: Coord,
                core_coord: Coord,
                core_e_coord: ReplicationId, 
                payload:int
                ) -> None:
        self.header = header
        self.chip_coord = chip_coord
        self.core_coord = core_coord
        self.core_e_coord = core_e_coord
        self.payload = payload
        
        temp_header = self.header.value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_addr = self.chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_addr = self.core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_e_addr = self.core_e_coord_address & FrameFormat.GENERAL_CORE_E_ADDR_MASK
        temp_payload = self.payload & FrameFormat.GENERAL_PAYLOAD_MASK

        self.frameinfo = np.uint64(
            (
                (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
                | (temp_chip_addr << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
                | (temp_core_addr << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
                | (temp_core_e_addr << FrameFormat.GENERAL_CORE_E_ADDR_OFFSET)
            )
        )
        
        self.frame = np.uint64(
            (
                int(self.frameinfo)
                | (temp_payload << FrameFormat.GENERAL_PAYLOAD_OFFSET)
            )
        )

    @classmethod
    def get_frame(cls, header: FrameHead, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, payload: int) -> "Frame":
        return cls(header, chip_coord, core_coord, core_e_coord, payload)

    def __len__(self) -> int:
        return 1

    @property
    def chip_address(self) -> int:
        return self.chip_coord.address

    @property
    def core_address(self) -> int:
        return self.core_coord.address

    @property
    def core_e_coord_address(self) -> int:
        return self.core_e_coord.address

    @property
    def value(self) -> np.uint64:
        return self.frame

    def __repr__(self) -> str:
        return (
            f"Frame info:\n"
            f"value:            {bin(self.value)[2:].zfill(64)}\n"
            f"Head:             {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_E address:   {self.core_e_coord}\n"
            f"Payload:          {self.payload}\n"
        )
        
    
class FrameGroup(Frame):
    """A group of frames of which the payload is split into `N` parts,
        and the frames have the same header, chip address, core address, core_e address,
        only the payload is different.

        Frame group for a length of `N` payload, the format of each frame is as follows:
        i. [Header(sub type)] + [chip coordinate] + [core coordinate] + [replication id] + [payload[i]]
                4 bits                10 bits             10 bits             10 bits         30 bits


    NOTE: In group of frames, the `payload` is a list of payload in each frame.
    """

    def __init__(self,
                header: FrameHead, 
                chip_coord: Coord,
                core_coord: Coord,
                core_e_coord: ReplicationId,
                payload: List) -> None:
        super().__init__(header, chip_coord, core_coord, core_e_coord, 0)

        self.payload = payload
        self._length = len(payload)

        frame_group = []
        temp_header = self.header.value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_addr = self.chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_addr = self.core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_e_addr = self.core_e_coord_address & FrameFormat.GENERAL_CORE_E_ADDR_MASK
        info = int(
            (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
            | (temp_chip_addr << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
            | (temp_core_addr << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
            | (temp_core_e_addr << FrameFormat.GENERAL_CORE_E_ADDR_OFFSET)
        )

        for load in self.payload:
            temp_payload = load & FrameFormat.GENERAL_PAYLOAD_MASK
            temp = int((info | (temp_payload << FrameFormat.GENERAL_PAYLOAD_OFFSET)))
            frame_group.append(temp) 
        
        self.frame_group = np.array(frame_group).astype(np.uint64)

    @classmethod
    def get_frame_group(cls, header: FrameHead, chip_coord: Coord, core_coord: Coord, rid: ReplicationId, payload: List) -> "FrameGroup":
        return cls(header, chip_coord, core_coord, rid, payload)

    @property
    def value(self):
        return self.frame_group

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item) -> Frame:
        return Frame(
            self.header,
            self.chip_coord,
            self.core_coord,
            self.core_e_coord,
            self.payload[item],
        )

    def __repr__(self):
        _present = (
            f"FrameGroup info:\n"
            f"Header:           {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_E address:   {self.core_e_coord}\n"
            f"Payload:\n"
        )

        for i in range(self._length):
            _present += f"#{i}:"+bin(self.payload[i])[2:].zfill(30)+"\n"

        return _present


class FramePackage(Frame):
    """_summary_

    Args:
        Frame (_type_): _description_
    """

    def __init__(self, header: FrameHead, chip_coord: Coord, core_coord: Coord, core_e_coord: ReplicationId, payload: int, data: List) -> None:
        super().__init__(header, chip_coord, core_coord, core_e_coord, payload)
        self.data = data
        self._length = len(data) + 1
        
        frame_package = []
        temp_header = self.header.value & FrameFormat.GENERAL_HEADER_MASK
        temp_chip_addr = self.chip_address & FrameFormat.GENERAL_CHIP_ADDR_MASK
        temp_core_addr = self.core_address & FrameFormat.GENERAL_CORE_ADDR_MASK
        temp_core_e_addr = self.core_e_coord_address & FrameFormat.GENERAL_CORE_E_ADDR_MASK
        temp_payload = self.payload & FrameFormat.GENERAL_PAYLOAD_MASK

        temp = int(
            (
                (temp_header << FrameFormat.GENERAL_HEADER_OFFSET)
                | (temp_chip_addr << FrameFormat.GENERAL_CHIP_ADDR_OFFSET)
                | (temp_core_addr << FrameFormat.GENERAL_CORE_ADDR_OFFSET)
                | (temp_core_e_addr << FrameFormat.GENERAL_CORE_E_ADDR_OFFSET)
                | (temp_payload << FrameFormat.GENERAL_PAYLOAD_OFFSET)
            )
        )

        frame_package.append(temp)
        if self.data:
            frame_package.extend(self.data)
            
        self.frame_package = np.array(frame_package).astype(np.uint64)

    @property
    def length(self) -> int:
        return self._length

    @property
    def n_package(self) -> int:
        return len(self.data)

    @property
    def value(self) -> np.ndarray:
        return self.frame_package

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        _present = f"FramePackage info:\n" f"Chip address:     {self.chip_coord}\n"

        _present = (
            f"FramePackage info:\n"
            f"Header:           {self.header}\n"
            f"Chip address:     {self.chip_coord}\n"
            f"Core address:     {self.core_coord}\n"
            f"Core_E address:   {self.core_e_coord}\n"
            f"Payload:          {self.payload}\n"
            f"Data:\n"
        )

        if self.data:
            for i in range(self.length - 1):
                _present += f"#{i}:{self.data[i]}\n"
        return _present


class MutiFrame:
    pass