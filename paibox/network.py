from typing import Optional

from .base import DynamicSys, PAIBoxObject
from .mixin import Container
from .node import NodeDict


class DynamicGroup(DynamicSys, Container):
    def __init__(
        self,
        *components,
        component_type: type = PAIBoxObject,
        name: Optional[str] = None
    ):
        super().__init__(name)

        self.children = NodeDict(self.elem_format(component_type, *components))


class Network(DynamicGroup):
    pass


class Sequential(DynamicSys, Container):
    def __init__(
        self,
        *components,
        component_type: type[PAIBoxObject] = PAIBoxObject,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name)

        self.children = NodeDict(self.elem_format(component_type, *components))

    def update(self, x):
        for child in self.children.values():
            x = child(x)

        return x

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.children:
                return self.children[item]
            else:
                raise KeyError

        if isinstance(item, int):
            return tuple(self.children.values())[item]

        if isinstance(item, slice):
            return Sequential(**dict(tuple(self.children.items())[item]))

        if isinstance(item, (tuple, list)):
            _all_nodes = tuple(self.children.items())
            return Sequential(**dict(_all_nodes[k] for k in item))

        raise KeyError
