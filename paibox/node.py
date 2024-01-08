from typing import Any, Generic, Iterable, Sequence, TypeVar

__all__ = ["NodeDict", "NodeList"]


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class NodeList(list, Generic[_T]):
    def __init__(self, seq: Sequence[Any] = ()) -> None:
        super().__init__()
        self.extend(seq)

    def append(self, elem: Any) -> "NodeList[_T]":
        super().append(elem)
        return self

    def extend(self, iterable: Iterable[Any]) -> "NodeList[_T]":
        for elem in iterable:
            self.append(elem)

        return self


class NodeDict(dict, Generic[_KT, _VT]):
    def __setitem__(self, key: _KT, value: _VT) -> "NodeDict[_KT, _VT]":
        super().__setitem__(key, value)
        return self

    def __getitem__(self, key: _KT) -> _VT:
        if key in self:
            return super().__getitem__(key)

        raise KeyError(f"Key '{key}' not found.")
