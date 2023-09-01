from typing import Callable, List, Literal, Optional, Set, Tuple, Union, overload
import numpy as np

from .collector import Collector
from .mixin import ReceiveInputProj
from .generic import get_unique_name, is_name_unique
from .utils import is_shape, shape2num, to_shape
from .node import NodeDict, NodeList
from ._types import Shape


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


class DynamicSys(PAIBoxObject, ReceiveInputProj):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.node_input = NodeDict()

    def __call__(self, x):
        raise NotImplementedError

    def update(self, x):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError


class Projection(PAIBoxObject):
    def __init__(self, shape: Shape, keep_size: bool = False, name: Optional[str] = None) -> None:
        self.keep_size = keep_size
        self.num = shape2num(shape)
        self.shape = to_shape(shape) if keep_size else (self.num,)

        super().__init__(name)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def shape_in(self):
        raise NotImplementedError

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.shape


class InputProj(Projection):
    def __init__(
        self,
        shape: Shape,
        val_or_func: Union[int, np.ndarray, Callable],
        *,
        keep_size: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Input projection to define an output or a generation function.

        Arguments:
            - shape: the output shape.
            - val_or_func: it can be an integer, `np.ndarray`, or a `Callable`.
            - name: the name of the node. Optional.
        """
        super().__init__(shape, keep_size, name)
        
        if isinstance(val_or_func, int):
            _val = np.full(self.shape, val_or_func)
            _func = None
        elif isinstance(val_or_func, np.ndarray):
            if not is_shape(val_or_func, self.shape):
                # TODO Error description
                raise ValueError
            _val = val_or_func
            _func = None
        else:
            _val = np.zeros(self.shape)
            _func = val_or_func

        self.output = _val
        self.output_func = _func

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs) -> np.ndarray:
        if isinstance(self.output_func, Callable):
            self.output = self.output_func(*args, **kwargs)

        return self.output


class OutputNode(Projection):
    pass
