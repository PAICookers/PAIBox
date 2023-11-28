from typing import Optional, Type, Union

import numpy as np

from .base import DynamicSys, NeuDyn
from .mixin import Container
from .node import NodeDict
from .projection import Projection
from .synapses import SynSys

__all__ = ["DynSysGroup", "Network"]


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

    def reset_state(self) -> None:
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        for node in nodes.subset(Projection).values():
            node.reset_state()

        for node in nodes.subset(SynSys).values():
            node.reset_state()

        for node in nodes.subset(NeuDyn).values():
            node.reset_state()


Network = DynSysGroup


class Sequential(DynamicSys, Container):
    def __init__(
        self,
        *components,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(self.elem_format(DynamicSys, *components))

    def update(self, x: np.ndarray) -> np.ndarray:
        for child in self.children.values():
            x = child(x)

        return x

    def reset_state(self) -> None:
        for child in self.children.values():
            child.reset_state()

    def __getitem__(self, item: Union[str, int, slice]):
        if isinstance(item, str):
            if item in self.children:
                return self.children[item]
            else:
                raise KeyError(f"Key {item} not found.")

        if isinstance(item, int):
            if item > len(self):
                raise IndexError(f"Index out of range: {item}")

            return tuple(self.children.values())[item]

        if isinstance(item, slice):
            return Sequential(**dict(tuple(self.children.items())[item]))

        raise TypeError(
            f"Expected type str, int or slice, but got {item}, type {type(item)}"
        )

    def __len__(self) -> int:
        return len(self.children)
