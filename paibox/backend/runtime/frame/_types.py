from typing import List, Tuple, TypeVar
from typing_extensions import TypeAlias
import numpy as np
from numpy.typing import NDArray

FRAME_DTYPE: TypeAlias = np.uint64
FrameArray = TypeVar(
    "FrameArray", int, List[int], Tuple[int, ...], NDArray[FRAME_DTYPE]
)
FrameArrayType: TypeAlias = NDArray[FRAME_DTYPE]
