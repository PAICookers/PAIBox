import numpy as np
from typing import Optional, Tuple
from paibox.libpaicore.v2 import *
def print_frame(frames : np.ndarray) -> None:
    for frame in frames:
        print(bin(frame)[2:].zfill(64))
        
def bin_split(x: int, high: int, low: int) -> Tuple[int, int]:
    """用于配置帧2/3型配置各个参数，对需要拆分的配置进行拆分

    Args:
        x (int): 输入待拆分参数
        low (int): 低位长度

    Returns:
        Tuple[int, int]: 返回拆分后的高位和低位
    """
    high_mask = (1 << high) - 1
    highbit = x >> (low) & high_mask

    lowbit_mask = (1 << low) - 1
    lowbit = x & lowbit_mask

    return highbit, lowbit

def Coord2Addr(coord: Coord) -> int:
    return (coord.x << 5) | coord.y

def Addr2Coord(addr: int) -> Coord:
    return Coord(addr >> 5, addr & ((1 << 5) - 1))