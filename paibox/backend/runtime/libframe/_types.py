import sys
from typing import List, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

FRAME_DTYPE: TypeAlias = np.uint64
FrameArrayType: TypeAlias = NDArray[FRAME_DTYPE]
ArrayType = TypeVar("ArrayType", List[int], Tuple[int, ...], np.ndarray)
BasicFrameArray = TypeVar(
    "BasicFrameArray", int, List[int], Tuple[int, ...], NDArray[FRAME_DTYPE]
)
IntScalarType = TypeVar("IntScalarType", int, np.integer)
DataType = TypeVar("DataType", int, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.integer, List[int], Tuple[int, ...], np.ndarray
)
