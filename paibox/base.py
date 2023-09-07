from typing import List, Literal, Optional, Set, Tuple

import numpy as np

from ._types import Shape
from .collector import Collector
from .generic import get_unique_name, is_name_unique
from .mixin import ReceiveInputProj
from .node import NodeDict, NodeList
from .utils import as_shape, shape2num


class PAIBoxObject:
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

        def _find_nodes_absolute() -> None:
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
                        method=method,
                        level=level,
                        include_self=include_self,
                        lid=lid + 1,
                        _paths=_paths,
                    )
                )

        def _find_nodes_relative() -> None:
            for k, v in self.__dict__.items():
                if isinstance(v, PAIBoxObject):
                    _add_node1(self, k, v, _paths, gather, nodes)
                elif isinstance(v, NodeList):
                    for i, v2 in enumerate(v):
                        _add_node1(self, k + "-" + str(i), v2, _paths, gather, nodes)
                elif isinstance(v, NodeDict):
                    for k2, v2 in v.items():
                        if isinstance(v2, PAIBoxObject):
                            _add_node1(self, k + "." + k2, v2, _paths, gather, nodes)

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
    obj: object, k: str, v: PAIBoxObject, _paths: Set, gather: Collector, nodes: List
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[k] = v
        nodes.append((k, v))


def _add_node2(
    obj: object, v: PAIBoxObject, _paths: Set, gather: Collector, nodes: List
) -> None:
    path = (id(obj), id(v))

    if path not in _paths:
        _paths.add(path)
        gather[v.name] = v
        nodes.append(v)


class DynamicSys(PAIBoxObject):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError


class NeuDyn(DynamicSys, ReceiveInputProj):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.master_node = NodeDict()

    @property
    def spike(self) -> np.ndarray:
        raise NotImplementedError


class SynSys(DynamicSys):
    @property
    def connectivity(self) -> np.ndarray:
        raise NotImplementedError


class Projection(DynamicSys):
    @property
    def shape_in(self) -> ...:
        raise NotImplementedError

    @property
    def shape_out(self) -> ...:
        raise NotImplementedError

    @property
    def output(self):
        raise NotImplementedError


class Process(DynamicSys):
    def __init__(
        self,
        shape_out: Shape = 1,
        *,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._shape_out = as_shape(shape_out)
        self.num = shape2num(self.shape_out)
        self.keep_size = keep_size

        self.output = np.zeros(self.varshape, dtype=np.bool_)

    def run(self, duration: int, dt: int = 1) -> np.ndarray:
        if duration < 0:
            # TODO
            raise ValueError

        n_steps = int(duration / dt)
        return self.run_steps(n_steps)

    def run_steps(self, n_steps: int) -> np.ndarray:
        output = np.zeros((n_steps,) + self.varshape, dtype=np.bool_)

        for i in range(n_steps):
            self.update()
            output[i] = self.state

        return output

    @property
    def state(self) -> np.ndarray:
        return self.output

    @property
    def varshape(self) -> Tuple[int, ...]:
        return self.shape_out if self.keep_size else (self.num,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape_out
