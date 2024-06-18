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
LeakVType: TypeAlias = NDArray[np.int32]
SpikeType: TypeAlias = NDArray[np.bool_]
SynOutType: TypeAlias = NDArray[np.int32]
VoltageType: TypeAlias = NDArray[np.int32]
WeightType: TypeAlias = NDArray[Union[np.bool_, np.int8]]
