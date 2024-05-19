import sys
from typing import TypeVar

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


if sys.version_info >= (3, 9):
    from collections import UserDict, UserList

    class NodeList(UserList[_T]):
        pass

    class NodeDict(UserDict[_KT, _VT]):
        pass

else:
    from typing import Dict, List

    class NodeList(List[_T]):
        pass

    class NodeDict(Dict[_KT, _VT]):
        pass
