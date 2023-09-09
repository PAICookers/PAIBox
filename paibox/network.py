from typing import Callable, Optional, Tuple, Type, Union

import numpy as np

from ._types import Shape
from .base import DynamicSys, NeuDyn, Process, Projection, SynSys
from .mixin import Container
from .node import NodeDict
from .utils import as_shape


class DynSysGroup(DynamicSys, Container):
    def __init__(
        self,
        *components,
        component_type: Type[DynamicSys] = DynamicSys,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(self.elem_format(component_type, *components))

    def update(self, *args, **kwargs) -> None:
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        for node in nodes.subset(Projection).values():
            node(*args, **kwargs)

        for node in nodes.subset(SynSys).values():
            node()

        for node in nodes.subset(NeuDyn).values():
            node()


class Network(DynSysGroup):
    pass


class Sequential(DynamicSys, Container):
    def __init__(
        self,
        *components,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(self.elem_format(object, *components))

    def update(self, x):
        for child in self.children.values():
            x = child(x)

        return x

    def __getitem__(self, item: Union[str, int, slice]):
        if isinstance(item, str):
            if item in self.children:
                return self.children[item]
            else:
                # TODO
                raise KeyError

        if isinstance(item, int):
            if item > self.__len__():
                raise IndexError

            return tuple(self.children.values())[item]

        if isinstance(item, slice):
            return Sequential(**dict(tuple(self.children.items())[item]))

        raise KeyError

    def __len__(self) -> int:
        return len(self.children)


class InputProj(Projection):
    def __init__(
        self,
        val_or_func: Union[int, np.integer, np.ndarray, Callable],
        *,
        shape: Optional[Shape] = None,
        target: Optional[Union[NeuDyn, SynSys]] = None,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Input projection to define an output or a generation function.

        Arguments:
            - val_or_func: the output value(integer, np.ndarray) or a process.
            - shape: the output shape. If not provided, try to use the shape_in of `target`. Otherwise raise error.
            - target: the output target, `NeuDyn`. Optional.
            - name: the name of the node. Optional.
        """
        super().__init__(name)

        if isinstance(val_or_func, (int, np.integer)):
            self.val = int(val_or_func)
            if not target:
                # TODO
                raise ValueError
            self._shape = target.shape_in
        elif isinstance(val_or_func, np.ndarray):
            if shape:
                if as_shape(shape) != val_or_func.shape:
                    # TODO
                    raise ValueError

                self._shape = as_shape(shape)
            else:
                if not target:
                    # TODO
                    raise ValueError
                self._shape = target.shape_in
        else:
            self.val = val_or_func
            self._shape = self.val.varshape

        self._state = np.zeros(self._shape)

        if isinstance(target, NeuDyn):
            target.register_master(f"{self.name}.output", self)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if isinstance(self.val, Callable):
            self._state = self.val(*args, **kwargs)
        else:
            self._state = np.full(self.shape_out, self.val)

        return self._state

    @property
    def output(self) -> np.ndarray:
        return self._state

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return (0,)

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self._shape


class OutputProj(Projection):
    pass
