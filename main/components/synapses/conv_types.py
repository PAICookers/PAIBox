from typing import Literal, Tuple, TypeVar, Union

T = TypeVar("T")

_TupleAnyType = Union[T, Tuple[T, ...]]
_Tuple1Type = Union[T, Tuple[T]]
_Tuple2Type = Union[T, Tuple[T, T]]
_Tuple3Type = Union[T, Tuple[T, T, T]]

_SizeAnyType = _TupleAnyType[int]
_Size1Type = _Tuple1Type[int]
_Size2Type = _Tuple2Type[int]
_Size3Type = _Tuple3Type[int]

SizeAnyType = Tuple[int, ...]
Size1Type = Tuple[int]
Size2Type = Tuple[int, int]
Size3Type = Tuple[int, int, int]

_Order2d = Literal["CL", "LC"]  # Feature map order in 2d
_Order3d = Literal["CHW", "HWC"]  # Feature map order in 3d
_KOrder3d = Literal["OIL", "IOL"]  # Kernel order in 1d convolution
_KOrder4d = Literal["OIHW", "IOHW"]  # Kernel order in 2d convolution
