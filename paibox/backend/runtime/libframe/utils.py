import os
import numpy as np

from functools import wraps
from pathlib import Path
from pydantic import TypeAdapter
from typing import Any, Dict, Optional, Tuple

from ._types import FrameArrayType
from paibox.libpaicore import FrameHeader as FH, FrameType as FT


# Replace the one from paibox.excpetions
class FrameIllegalError(ValueError):
    """Frame is illegal."""

    pass


def check_elem_same(obj: Any) -> bool:
    if hasattr(obj, "__iter__") or hasattr(obj, "__contains__"):
        return len(set(obj)) == 1

    if isinstance(obj, dict):
        return len(set(obj.values())) == 1

    raise TypeError(f"Unsupported type: {type(obj)}")


def header2type(header: FH) -> FT:
    if header <= FH.CONFIG_TYPE4:
        return FT.FRAME_CONFIG
    elif header <= FH.TEST_TYPE4:
        return FT.FRAME_TEST
    elif header <= FH.WORK_TYPE4:
        return FT.FRAME_WORK

    raise FrameIllegalError(f"Unknown header: {header}")


def print_frame(frames: FrameArrayType) -> None:
    for frame in frames:
        print(bin(frame)[2:].zfill(64))


def np2npy(fp: Path, d: np.ndarray) -> None:
    np.save(fp, d)


def np2bin(fp: Path, d: np.ndarray) -> None:
    d.tofile(fp)


def np2txt(fp: Path, d: np.ndarray) -> None:
    with open(fp, "w") as f:
        for i in range(d.size):
            f.write("{:064b}\n".format(d[i]))


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


# Replace the one from paibox.utils
def bin_split(x: int, pos: int, high_mask: Optional[int] = None) -> Tuple[int, int]:
    """Split an integer, return the high and low part.

    Argument:
        - x: the integer
        - pos: the position (LSB) to split the binary.
        - high_mask: mask for the high part. Optional.

    Example::

        >>> bin_split(0b1100001001, 3)
        97(0b1100001), 1
    """
    low = x & ((1 << pos) - 1)

    if isinstance(high_mask, int):
        high = (x >> pos) & high_mask
    else:
        high = x >> pos

    return high, low


def params_check(checker: TypeAdapter):
    def inner(func):
        @wraps(func)
        def wrapper(params: Dict[Any, Any], *args, **kwargs):
            checked = checker.validate_python(params)
            return func(checked, *args, **kwargs)

        return wrapper

    return inner


def params_check2(checker1: TypeAdapter, checker2: TypeAdapter):
    def inner(func):
        @wraps(func)
        def wrapper(params1: Dict[Any, Any], params2: Dict[Any, Any], *args, **kwargs):
            checked1 = checker1.validate_python(params1)
            checked2 = checker2.validate_python(params2)
            return func(checked1, checked2, *args, **kwargs)

        return wrapper

    return inner
