import os
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import TypeAdapter

from paibox.libpaicore import FrameHeader as FH
from paibox.libpaicore import FrameType as FT
from paibox.libpaicore import FrameFormat as FF

from ._types import BasicFrameArray, FRAME_DTYPE, FrameArrayType


# Replace the one from `paibox.excpetions`
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


def header_check(frames: FrameArrayType, expected_type: FH) -> None:
    """Check the header of frame arrays.

    TODO Is it necessary to deal with the occurrence of illegal frames? Filter & return.
    """
    header0 = FH((int(frames[0]) >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK)

    if header0 is not expected_type:
        raise ValueError(
            f"Expected frame type {expected_type.name}, but got: {header0.name}."
        )

    headers = (frames >> FF.GENERAL_HEADER_OFFSET) & FF.GENERAL_HEADER_MASK

    if not check_elem_same(headers):
        raise ValueError(
            "The header of the frame is not the same, please check the frames value."
        )


def frame_array2np(frame_array: BasicFrameArray) -> FrameArrayType:
    if isinstance(frame_array, int):
        nparray = np.asarray([frame_array], dtype=FRAME_DTYPE)

    elif isinstance(frame_array, np.ndarray):
        if frame_array.ndim != 1:
            warnings.warn(
                f"ndim of frame arrays must be 1, but got {frame_array.ndim}. Flatten anyway.",
                UserWarning,
            )
        nparray = frame_array.flatten().astype(FRAME_DTYPE)

    elif isinstance(frame_array, (list, tuple)):
        nparray = np.asarray(frame_array, dtype=FRAME_DTYPE)

    else:
        raise TypeError(
            f"Expect int, list, tuple or np.ndarray, but got {type(frame_array)}"
        )

    return nparray


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
        def wrapper(params: Dict[str, Any], *args, **kwargs):
            checked = checker.validate_python(params)
            return func(checked, *args, **kwargs)

        return wrapper

    return inner


def params_check2(checker1: TypeAdapter, checker2: TypeAdapter):
    def inner(func):
        @wraps(func)
        def wrapper(params1: Dict[str, Any], params2: Dict[str, Any], *args, **kwargs):
            checked1 = checker1.validate_python(params1)
            checked2 = checker2.validate_python(params2)
            return func(checked1, checked2, *args, **kwargs)

        return wrapper

    return inner
