from typing import (
    AbstractSet,
    Any,
    Iterable,
    Iterator,
    List,
    MutableSet,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

Shape = TypeVar("Shape", int, Tuple[int, ...], List[int])
ArrayType = TypeVar("ArrayType", List[int], Tuple[int, ...], np.ndarray)
Scalar = TypeVar("Scalar", int, float, np.generic)
IntScalarType = TypeVar("IntScalarType", int, np.bool_, np.integer)
DataType = TypeVar("DataType", int, np.bool_, np.integer, np.ndarray)
DataArrayType = TypeVar(
    "DataArrayType", int, np.bool_, np.integer, List[int], Tuple[int, ...], np.ndarray
)
SpikeType: TypeAlias = NDArray[np.bool_]
SynOutType: TypeAlias = NDArray[np.int32]
VoltageType: TypeAlias = NDArray[np.int32]
WeightType: TypeAlias = NDArray[Union[np.bool_, np.int8]]

_T = TypeVar("_T")


class FrozenOrderedSet(AbstractSet[_T]):
    """A set that preserves insertion order and is hashable."""

    def __init__(self, data: Optional[Iterable[Any]] = None) -> None:
        if data is None:
            data = []

        self.data = dict.fromkeys(data)

    def __contains__(self, elem: Any) -> bool:
        return elem in self.data

    def __iter__(self) -> Iterator[_T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __hash__(self) -> int:
        return self._hash()

    def clear(self) -> None:
        self.data.clear()


class OrderedSet(FrozenOrderedSet[_T], MutableSet):
    """A set that preserves insertion order and is mutable."""

    def add(self, value: Any) -> None:
        self.data[value] = None

    def discard(self, value: Any) -> None:
        self.data.pop(value, None)

    def update(self, other: Iterable[Any]) -> None:
        self.data.update((value, None) for value in other)

    def difference_update(self, other: Iterable[Any]) -> None:
        self -= set(other)

    def __ior__(self, other: Iterable[Any]) -> "OrderedSet[_T]":
        self.update(other)
        return self

    def __hash__(self) -> NoReturn:
        raise TypeError(
            "'OrderedSet' is not hashable (use 'FrozenOrderedSet' instead)."
        )
