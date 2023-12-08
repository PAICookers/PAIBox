from collections.abc import MutableSet, Set
from typing import Any, List, Optional, Tuple, TypeVar

import numpy as np

Shape = TypeVar("Shape", int, Tuple[int, ...], List[int])
Spike = TypeVar("Spike", List[int], np.ndarray)
ArrayType = TypeVar("ArrayType", List[int], Tuple[int, ...], np.ndarray)
Scalar = TypeVar("Scalar", int, float, np.generic)
IntScalarType = TypeVar("IntScalarType", int, np.integer)
DataArrayType = TypeVar(
    "DataArrayType", int, np.integer, List[int], Tuple[int, ...], np.ndarray
)

T = TypeVar("T")


class FrozenOrderedSet(Set[T]):
    """A set that preserves insertion order and is hashable."""

    def __init__(self, data: Optional[Any] = None) -> None:
        if data is None:
            data = []

        self.data = dict((d, None) for d in data)

    def __contains__(self, elem) -> bool:
        return elem in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __hash__(self) -> int:
        return self._hash()

    def clear(self) -> None:
        self.data.clear()


class OrderedSet(FrozenOrderedSet[T], MutableSet):
    """A set that preserves insertion order and is mutable."""

    def add(self, value: Any) -> None:
        self.data[value] = None

    def discard(self, value: Any) -> None:
        self.data.pop(value, None)

    def update(self, other) -> None:
        self.data.update((value, None) for value in other)

    def difference_update(self, other) -> None:
        self -= other

    def __ior__(self, other):
        self.update(other)
        return self

    def __hash__(self):
        raise TypeError("OrderedSet is not hashable (use FrozenOrderedSet)")
