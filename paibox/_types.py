from typing import List, Tuple, TypeVar

import numpy as np

Shape = TypeVar("Shape", int, Tuple[int, ...], List[int])
Spike = TypeVar("Spike", List[int], np.ndarray)
SpikeArray = TypeVar("SpikeArray", List[List[int]], np.ndarray)

Array = TypeVar("Array", np.ndarray, List[int])
Scalar = TypeVar("Scalar", int, float)
