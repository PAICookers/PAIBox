from typing import Optional, Tuple
from .base_frame import *
import numpy as np

from paibox.libpaicore.v2 import Coord
from .util import *
from .params import ParameterRAMFormat as RAMMask
from .params import ParameterRegFormat as RegMask
from .params import *
from .offline_frame import *

class FrameGenOffline:
    """Offline Frame Generator"""

    @staticmethod
    def gen_config_frame1(
        chip_coord: Coord,
        core_coord: Coord,
        core_e_coord: ReplicationId,
        random_seed: int
    ) -> OfflineConfigFrame1:
        return OfflineConfigFrame1(chip_coord,core_coord,core_e_coord,random_seed)
    
    @staticmethod
    def gen_config_frame2(
        chip_coord: Coord,
        core_coord: Coord,
        core_e_coord: ReplicationId,
        parameter_reg: dict
    ) -> FrameGroup:

        return OfflineConfigFrame2(chip_coord,core_coord,core_e_coord,parameter_reg)
    
    @staticmethod
    def gen_config_frame3(
        chip_coord:Coord,
        core_coord:Coord,
        core_e_coord:ReplicationId,
        sram_start_addr: int,
        data_package_num: int,
        neuron_ram:dict
    ) -> OfflineConfigFrame3:
        
        return OfflineConfigFrame3(chip_coord,core_coord,core_e_coord,sram_start_addr,data_package_num,neuron_ram)
    
    @staticmethod
    def gen_config_frame4(
        chip_coord: Coord,
        core_coord: Coord,
        core_e_coord: ReplicationId, 
        sram_start_addr: int,
        data_package_num: int,
        weight_ram: List
    ) -> OfflineConfigFrame4:
        
        return OfflineConfigFrame4(chip_coord,core_coord,core_e_coord,sram_start_addr,data_package_num,weight_ram)
    
    @staticmethod
    def gen_work_frame1(
        chip_coord : Coord,
        core_coord : Coord,
        core_e_coord : ReplicationId,
        axon: int,
        time_slot: int,
        data: int
    ) -> OfflineWorkFrame1:
        return OfflineWorkFrame1(chip_coord,core_coord,core_e_coord,axon,time_slot,data)
    
    @staticmethod
    def gen_work_frame1_frameinfo(
        chip_coord_list : List[Coord],
        core_coord_list : List[Coord],
        core_e_coord_list : List[ReplicationId],
        axon_list: List[int],
        time_slot_list: List[int]
    ) -> np.ndarray:
        
        chip_address_array = np.array([coord.address for coord in chip_coord_list])
        core_address_array = np.array([coord.address for coord in core_coord_list])
        core_e_address_array = np.array([coord.address for coord in core_e_coord_list])
        
        head_array = np.tile(FrameHead.WORK_TYPE1.value,len(chip_address_array))
        axon_array = np.array(axon_list)
        time_slot_array = np.array(time_slot_list)
        
        temp_header_array = head_array & np.uint64(FrameFormat.GENERAL_HEADER_MASK)
        temp_chip_coord_array = chip_address_array & np.uint64(FrameFormat.GENERAL_CHIP_ADDR_MASK)
        temp_core_address_array = core_address_array & np.uint64(FrameFormat.GENERAL_CORE_ADDR_MASK)
        temp_core_e_address_array = core_e_address_array & np.uint64(FrameFormat.GENERAL_CORE_E_ADDR_MASK)
        temp_axon_array = axon_array & np.uint64(WorkFrame1Format.AXON_MASK)
        temp_time_slot_array = time_slot_array & np.uint64(WorkFrame1Format.TIME_SLOT_MASK)
        
        frameinfo = (
            (temp_header_array << np.uint64(FrameFormat.GENERAL_HEADER_OFFSET))
            | (temp_chip_coord_array <<  np.uint64(FrameFormat.GENERAL_CHIP_ADDR_OFFSET))
            | (temp_core_address_array << np.uint64(FrameFormat.GENERAL_CORE_ADDR_OFFSET))
            | (temp_core_e_address_array << np.uint64(FrameFormat.GENERAL_CORE_E_ADDR_OFFSET))
            | (temp_axon_array << np.uint64(WorkFrame1Format.AXON_OFFSET))
            | (temp_time_slot_array << np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET))
        )
        
        return frameinfo
    
    @staticmethod
    def gen_work_frame2(
        chip_coord : Coord,
        time : int
    ) -> OfflineWorkFrame2:

        return OfflineWorkFrame2(chip_coord,time)
    
    @staticmethod
    def gen_work_frame3(
        chip_coord:Coord
    ) -> OfflineWorkFrame3:
        return OfflineWorkFrame3(chip_coord)
    
    @staticmethod
    def gen_work_frame4(
        chip_coord:Coord
    ) -> OfflineWorkFrame4:

        return OfflineWorkFrame4(chip_coord)
    
    
    