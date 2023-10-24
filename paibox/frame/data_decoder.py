"""
    将输入数据进行编码生成输入帧 与 输出的帧解码为输出数据
"""
import numpy as np
import torch
from .params import *

from paibox.libpaicore.v2 import LCN_EX
class FrameDecoder:
    def decode(
        self,
        frames : np.ndarray,
        shape : tuple,
    ):
        time_slot_list = (frames >> np.uint64(WorkFrame1Format.TIME_SLOT_OFFSET)) & np.uint64(WorkFrame1Format.TIME_SLOT_MASK)
        data_list = frames & np.uint64(WorkFrame1Format.DATA_MASK)
        
        
        

