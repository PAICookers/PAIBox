from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Shape = TypeVar("Shape", int, tuple[int, ...], list[int], np.ndarray)
Scalar = TypeVar("Scalar", int, float, np.generic)
IntScalarType = TypeVar("IntScalarType", int, bool, np.integer)
DataType = TypeVar("DataType", int, bool, np.integer, np.ndarray)

VOLTAGE_DTYPE = np.int32
LEAK_V_DTYPE = VOLTAGE_DTYPE
THRES_V_DTYPE = VOLTAGE_DTYPE
WEIGHT_DTYPE = np.int8
NEUOUT_SPIKE_DTYPE = bool
NEUOUT_U8_DTYPE = np.uint8

LeakVType = NDArray[LEAK_V_DTYPE]
NeuOutSpikeType = NDArray[NEUOUT_SPIKE_DTYPE]
SynOutType = NDArray[VOLTAGE_DTYPE]
VoltageType = NDArray[VOLTAGE_DTYPE]
NeuOutType = NDArray[NEUOUT_U8_DTYPE]
WeightType = NDArray[WEIGHT_DTYPE]
