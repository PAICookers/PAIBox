from typing import Optional, Tuple

from .base import DynamicSys, NeuDyn, PAIBoxObject, Process, Projection, SynSys
from .mixin import Container
from .node import NodeDict


class DynSysGroup(DynamicSys, Container):
    def __init__(
        self,
        *components,
        component_type: type[PAIBoxObject] = PAIBoxObject,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(self.elem_format(component_type, *components))

    def update(self, **kwargs) -> None:
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        for node in nodes.subset(Projection).values():
            node(**kwargs)

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
            # FIXME
            _all_nodes = tuple(self.children.items())
            return Sequential(**dict(_all_nodes[k] for k in item))

        raise KeyError


class InputProj(Projection):
    def __init__(
        self,
        process: Process,
        target: Optional[NeuDyn] = None,
        *,
        keep_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Input projection to define an output or a generation function.

        Arguments:
            - shape: the output shape.
            - val_or_proc: it can be an integer, `np.ndarray`, or a `Callable`.
            - name: the name of the node. Optional.
        """
        super().__init__(name)
        self.process = process
        self.shape = self.process.varshape

        if target:
            target.register_master(f"{self.name}.output", self)

    def __call__(self, *args, tick, **kwargs):
        return self.update(tick, **kwargs)

    def update(self, tick, **kwargs):
        self.process.update(tick=tick, **kwargs)

    @property
    def output(self):
        return self.process.state

    @property
    def state(self):
        return self.process.state

    @property
    def shape_in(self) -> int:
        return 0

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.shape


class OutputProj(Projection):
    pass
