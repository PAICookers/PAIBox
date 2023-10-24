import numpy as np
from paibox.libpaicore.v2 import *
from typing import List
class DataEncoder:
    @staticmethod
    def encode(
        origin_data : np.ndarray,
        spike_frame_info : np.ndarray
    ) -> np.ndarray:
        data = origin_data.reshape(-1)
        indexes = np.nonzero(data)
        spike_frame_info = spike_frame_info[indexes]
        data = origin_data.reshape(-1)
        dataFrames = spike_frame_info << np.uint64(8) | data
        
        return dataFrames

    @staticmethod
    def compose_frames(
        input_node : List
    ):
        pass
    
    @staticmethod
    def node2frameinfo(
        node : dict,
    ):
        """将输入节点转换为帧前缀

        Args:
            node (dict): _description_
        """
        