"""
    将输出的帧解码为输出数据
"""
from typing import Union
import numpy as np

from paibox.frame.offline_frame import OfflineWorkFrame1
from .params import WorkFrame1Format

class FrameDecoder:
    
    def decode(
        self,
        frames : Union[np.ndarray,OfflineWorkFrame1],
    ) -> dict:
        """将输出数据帧解码，以字典形式返回，按照time_slot排序

        Args:
            frames (Union[np.ndarray,OfflineWorkFrame1]): _description_

        Returns:
            dict: {
                core_addr: {
                    axonid: {
                        time_slot: np.ndarray,
                        data: np.ndarray
                    }
                }
            }
            
            eg: {
                1:{
                    2: {
                        time_slot : [1,3,9],
                        data : [4,5,6]
                    }
                }
            }
        """
        if isinstance(frames,OfflineWorkFrame1):
            frames = frames.value
            
        frames = frames.astype(np.uint64)
        
        axons = (frames >> np.uint64(WorkFrame1Format.AXON_OFFSET)) & np.uint64(WorkFrame1Format.AXON_MASK)
        time_slots = (frames >> np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET)) & np.uint64(WorkFrame1Format.TIME_SLOT_MASK)
        data = (frames >> np.uint64(WorkFrame1Format.DATA_OFFSET)) & np.uint64(WorkFrame1Format.DATA_MASK)
        core_addr = (frames >> np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_OFFSET)) & np.uint64(WorkFrame1Format.GENERAL_CORE_ADDR_MASK)
        
        res = {}
        
        unique_axon = np.unique(axons)
        unique_core_addr = np.unique(core_addr)
        
        for core_value in unique_core_addr:
            axon_positions = {} # 存储所有的axon在frames中的位置
            res[core_value] = {}
            core_addr_positions = np.where(core_addr == core_value)[0] # 获取value在原来的core_addr中的位置
            core_axons = axons[core_addr_positions] #将当前core的frames信息筛选出来
            core_time_slots = time_slots[core_addr_positions]
            core_data = data[core_addr_positions]
            
            for axon_value in unique_axon:
                # print(np.where(axons == value)[0])
                positions = np.where(core_axons == axon_value)[0] # 获取当前core中的当前axon在筛选后的frames信息（core_axons）中的位置
                if len(positions) > 0:
                    axon_positions[axon_value] = positions 

            for axon_value,positions in axon_positions.items():
                res[core_value][axon_value] = {}
                res[core_value][axon_value]["time_slot"] = core_time_slots[positions]
                res[core_value][axon_value]["data"] = core_data[positions]
                
                sorted_indices = np.argsort(res[core_value][axon_value]["time_slot"])
                
                res[core_value][axon_value]["time_slot"] = res[core_value][axon_value]["time_slot"][sorted_indices]
                res[core_value][axon_value]["data"] = res[core_value][axon_value]["data"][sorted_indices]
                
        return res
            

