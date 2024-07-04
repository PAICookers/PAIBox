import sys
from typing import TypeVar, Union

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

Shape = TypeVar("Shape", int, tuple[int, ...], list[int], np.ndarray)
ArrayType = TypeVar("ArrayType", list[int], tuple[int, ...], np.ndarray)
Scalar = TypeVar("Scalar", int, float, np.generic)
IntScalarType = TypeVar("IntScalarType", int, np.bool_, np.integer)
DataType = TypeVar("DataType", int, np.bool_, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.bool_, np.integer, list[int], tuple[int, ...], np.ndarray
)

LEAK_V_DTYPE = np.int32
SPIKE_DTYPE = np.bool_
VOLTAGE_DTYPE = np.int32
NEUOUT_SPIKE_DTYPE = np.bool_
NEUOUT_U8_DTYPE = np.uint8
NEUOUT_DTYPE = Union[NEUOUT_SPIKE_DTYPE, NEUOUT_U8_DTYPE]

LeakVType: TypeAlias = NDArray[LEAK_V_DTYPE]
SpikeType: TypeAlias = NDArray[SPIKE_DTYPE]
SynOutType: TypeAlias = NDArray[VOLTAGE_DTYPE]
VoltageType: TypeAlias = NDArray[VOLTAGE_DTYPE]
NeuOutType: TypeAlias = NDArray[NEUOUT_DTYPE]
WeightType: TypeAlias = NDArray[Union[np.bool_, np.int8]]
