from typing import Optional

from paibox.generic import get_unique_name, is_name_unique


class PAIBoxObject:
    def __init__(self, name: Optional[str] = None) -> None:
        self._name: str = self.unique_name(name)

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        return type(self) == type(other) and self._name == other._name

    def __hash__(self):
        return hash((type(self), self._name))

    def unique_name(
        self, name: Optional[str] = None, _type: Optional[str] = None
    ) -> str:
        if name is None:
            if _type is None:
                __type = self.__class__.__name__
            else:
                __type = _type

            return get_unique_name(__type)

        is_name_unique(name, self)
        return name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = self.unique_name(name)


class StatelessObject(PAIBoxObject):
    pass


class DynamicSys(PAIBoxObject):
    pass
