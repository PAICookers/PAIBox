from typing import Literal, TypeVar

T = TypeVar("T")

_TupleAnyType = T | tuple[T, ...]
_Tuple1Type = T | tuple[T]
_Tuple2Type = T | tuple[T, T]
_Tuple3Type = T | tuple[T, T, T]
_Tuple4Type = T | tuple[T, T, T, T]

_SizeAnyType = _TupleAnyType[int]
_Size1Type = _Tuple1Type[int]
_Size2Type = _Tuple2Type[int]
_Size3Type = _Tuple3Type[int]
_Size4Type = _Tuple4Type[int]

SizeAnyType = tuple[int, ...]
Size1Type = tuple[int]
Size2Type = tuple[int, int]
Size3Type = tuple[int, int, int]
Size4Type = tuple[int, int, int, int]

_Order2d = Literal["CL", "LC"]  # Feature map order in 2d
_Order3d = Literal["CHW", "HWC"]  # Feature map order in 3d
_KOrder3d = Literal["OIL", "IOL"]  # Kernel order in 1d convolution
_KOrder4d = Literal["OIHW", "IOHW"]  # Kernel order in 2d convolution
