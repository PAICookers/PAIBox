import sys
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from .collector import Collector
from .generic import get_unique_name, is_name_unique
from .mixin import ReceiveInputProj, StatusMemory, TimeRelatedNode
from .node import NodeDict, NodeList

__all__ = []


_IdPathType: TypeAlias = Tuple[int, int]


class PAIBoxObject:
    _excluded_vars = ()

    def __init__(self, name: Optional[str] = None) -> None:
        self._name: str = self.unique_name(name)

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        return type(self) == type(other) and self._name == other._name

    def __hash__(self) -> int:
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
    ) -> Collector[str, "PAIBoxObject"]:
        """Collect all the children nodes."""
        return self._find_nodes(method, level, include_self)

    def _find_nodes(
        self,
        method: Literal["absolute", "relative"] = "absolute",
        level: int = -1,
        include_self: bool = True,
        lid: int = 0,
        _paths: Optional[Set[_IdPathType]] = None,
    ) -> Collector[str, "PAIBoxObject"]:
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

        def _find_nodes_absolute() -> None:
            nonlocal gather, nodes

            for v in self.__dict__.values():
                if isinstance(v, PAIBoxObject):
                    _add_node2(self, v, _paths, gather, nodes)
                elif isinstance(v, NodeList):
                    for v2 in v:
                        if isinstance(v2, PAIBoxObject):
                            _add_node2(self, v2, _paths, gather, nodes)
                elif isinstance(v, NodeDict):
                    for v2 in v.values():
                        if isinstance(v2, PAIBoxObject):
                            _add_node2(self, v2, _paths, gather, nodes)

            # finding nodes recursively
            for v in nodes:
                gather.update(
                    v._find_nodes(
                        method=method,
                        level=level,
                        include_self=include_self,
                        lid=lid + 1,
                        _paths=_paths,
                    )
                )

        def _find_nodes_relative() -> None:
            nonlocal gather, nodes

            for k, v in self.__dict__.items():
                if isinstance(v, PAIBoxObject):
                    _add_node1(self, k, v, _paths, gather, nodes)
                elif isinstance(v, NodeList):
                    for i, v2 in enumerate(v):
                        if isinstance(v2, PAIBoxObject):
                            _add_node1(self, f"{k}-{str(i)}", v2, _paths, gather, nodes)
                elif isinstance(v, NodeDict):
                    for k2, v2 in v.items():
                        if isinstance(v2, PAIBoxObject):
                            _add_node1(self, f"{k}.{k2}", v2, _paths, gather, nodes)

            # finding nodes recursively
            for k1, v1 in nodes:
                for k2, v2 in v1._find_nodes(
                    method=method,
                    level=level,
                    include_self=include_self,
                    lid=lid + 1,
                    _paths=_paths,
                ).items():
                    if k2:
                        gather[f"{k1}.{k2}"] = v2

        nodes = []

        if method == "absolute":
            _find_nodes_absolute()
        else:
            _find_nodes_relative()

        return gather


def _add_node1(
    obj: object,
    k: str,
    v: PAIBoxObject,
    _paths: Set[_IdPathType],
    gather: Collector[str, PAIBoxObject],
    nodes: List[Tuple[str, PAIBoxObject]],
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[k] = v
        nodes.append((k, v))


def _add_node2(
    obj: object,
    v: PAIBoxObject,
    _paths: Set[_IdPathType],
    gather: Collector[str, PAIBoxObject],
    nodes: List[PAIBoxObject],
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[v.name] = v
        nodes.append(v)


class DynamicSys(PAIBoxObject, StatusMemory):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        super(PAIBoxObject, self).__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset_state(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def shape_in(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def shape_out(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def num_in(self) -> int:
        raise NotImplementedError

    @property
    def num_out(self) -> int:
        raise NotImplementedError

    @property
    def output(self):
        """Actual output to the sucessors."""
        raise NotImplementedError

    @property
    def feature_map(self):
        raise NotImplementedError

    @property
    def state(self) -> NodeDict:
        """Return the stateful attirbutes in the system."""
        return self._memories


class NeuDyn(DynamicSys, ReceiveInputProj, TimeRelatedNode):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.master_nodes = NodeDict()

    def export_params(self) -> Dict[str, Any]:
        """Export the parameters into dictionary."""
        params = {}

        for k, v in self.__dict__.items():
            if k in self._excluded_vars:
                continue

            if sys.version_info >= (3, 9):
                params.update({k.removeprefix("_"): v})
            else:
                params.update({k.lstrip("_"): v})

        return params

    @property
    def is_working(self) -> bool:
        return (self.tick_wait_start > 0 and self.timestamp >= 0) and (
            self.tick_wait_end == 0 or self.timestamp + 1 <= self.tick_wait_end
        )

    @property
    def delay_relative(self) -> int:
        return self._delay

    @property
    def tick_wait_start(self) -> int:
        return self._tws

    @property
    def tick_wait_end(self) -> int:
        return self._twe

    @property
    def unrolling_factor(self) -> int:
        return self._unrolling_factor

    @unrolling_factor.setter
    def unrolling_factor(self, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"'unrolling_factor' must be positive, but got {factor}.")

        self._unrolling_factor = factor
