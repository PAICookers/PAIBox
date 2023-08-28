from typing import List, Literal, Optional, Set

from paibox.collector import Collector
from paibox.generic import get_unique_name, is_name_unique
from paibox.node import NodeDict, NodeList


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

    def nodes(
        self,
        method: Literal["absolute", "relative"] = "absolute",
        level: int = -1,
        include_self: bool = True,
    ) -> Collector:
        """Collect all the children nodes."""
        return self._find_nodes(method, level, include_self)

    def _find_nodes(
        self,
        method: Literal["absolute", "relative"] = "absolute",
        level: int = -1,
        include_self: bool = True,
        lid: int = 0,
        _paths: Optional[Set] = None,
    ) -> Collector:
        if _paths is None:
            _paths = set()

        gather = Collector()
        if include_self:
            if method == "absolute":
                gather[self.name] = self
            else:
                gather[""] = self

        if (level > -1) and (lid >= level):
            return gather

        nodes = []
        for k, v in self.__dict__.items():
            if isinstance(v, PAIBoxObject):
                _add_node2(self, v, _paths, gather, nodes)
            elif isinstance(v, NodeList):
                for v2 in v:
                    _add_node2(self, v2, _paths, gather, nodes)
            elif isinstance(v, NodeDict):
                for v2 in v.values():
                    if isinstance(v2, PAIBoxObject):
                        _add_node2(self, v2, _paths, gather, nodes)

        # finding nodes recursively
        for v in nodes:
            gather.update(
                v._find_nodes(
                    level=level,
                    include_self=include_self,
                    lid=lid + 1,
                    _paths=_paths,
                )
            )

        return gather


def _add_node2(
    obj: object, v: PAIBoxObject, _paths: Set, gather: Collector, nodes: List
):
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[v.name] = v
        nodes.append(v)


class StatelessObject(PAIBoxObject):
    pass


class DynamicSys(PAIBoxObject):
    def __call__(self, x):
        raise NotImplementedError

    def update(self, x):
        raise NotImplementedError
