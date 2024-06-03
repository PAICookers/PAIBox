from collections import UserDict, UserList
from typing import TypeVar

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class NodeList(UserList[_T]):
    pass


class NodeDict(UserDict[_KT, _VT]):
    pass
