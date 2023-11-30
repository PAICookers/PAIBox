import os
from typing import Optional, Tuple, Union

import numpy as np

from paibox.libpaicore.v2 import *


def print_frame(frames: np.ndarray) -> None:
    for frame in frames:
        print(bin(frame)[2:].zfill(64))


def bin_array_split(
    x: Union[np.ndarray, list], high: int, low: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(x).astype(np.uint64)  # type: ignore
    high = np.array([high]).astype(np.uint64)  # type: ignore
    low = np.array([low]).astype(np.uint64)  # type: ignore

    high_mask = np.uint64((np.uint64(1) << high) - np.uint64(1))
    highbit = (x >> low) & high_mask

    lowbit_mask = np.uint64((np.uint64(1) << low) - np.uint64(1))
    lowbit = x & lowbit_mask

    return highbit, lowbit


def bin_split(
    x: Union[np.uint64, int], high: int, low: int
) -> Tuple[np.uint64, np.uint64]:
    """用于配置帧2/3型配置各个参数，对需要拆分的配置进行拆分

    Args:
        x (int): 输入待拆分参数
        low (int): 低位长度

    Returns:
        Tuple[int, int]: 返回拆分后的高位和低位
    """
    x = np.uint64(x)
    high_uint = np.array(high).astype(np.uint64)
    low_uint = np.array(low).astype(np.uint64)

    high_mask = np.uint64((np.uint64(1) << high_uint) - np.uint64(1))
    highbit = np.uint64((x >> low_uint) & high_mask)

    lowbit_mask = np.uint64((np.uint64(1) << low_uint) - np.uint64(1))
    lowbit = x & lowbit_mask

    return highbit, lowbit


def Coord2Addr(coord: Coord) -> int:
    return (coord.x << 5) | coord.y


def Addr2Coord(addr: int) -> Coord:
    return Coord(addr >> 5, addr & ((1 << 5) - 1))


def npFrame2txt(dataPath, inputFrames):
    with open(dataPath, "w") as f:
        for i in range(inputFrames.shape[0]):
            f.write("{:064b}\n".format(inputFrames[i]))


def strFrame2txt(dataPath, inputFrames):
    with open(dataPath, "w") as f:
        for i in range(len(inputFrames)):
            f.write(inputFrames[i] + "\n")


def binFrame2Txt(configPath):
    configFrames = np.fromfile(configPath, dtype="<u8")
    fName, _ = os.path.splitext(configPath)
    configTxtPath = fName + ".txt"
    npFrame2txt(configTxtPath, configFrames)
    print(f"[generate] Generate frames as txt file")


def txtFrame2Bin(configTxtPath):
    config_frames = np.loadtxt(configTxtPath, str)
    config_num = config_frames.size
    config_buffer = np.zeros((config_num,), dtype=np.uint64)
    for i in range(0, config_num):
        config_buffer[i] = int(config_frames[i], 2)
    config_frames = config_buffer
    fName, _ = os.path.splitext(configTxtPath)
    configPath = fName + ".bin"
    config_frames.tofile(configPath)
    print(f"[generate] Generate frames as bin file")


def npFrame2bin(frame, framePath):
    frame.tofile(framePath)
    print(f"Generate frames as bin file at {framePath}")


def save_frames(frames, path):
    with open(path, "w") as f:
        for frame in frames:
            f.write(str(bin(frame))[2:].zfill(64) + "\n")
