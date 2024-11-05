import sys
from typing import Optional, Union

import numpy as np

from .base import DynamicSys, SynSys
from .collector import Collector
from .components import NeuModule, Neuron, Projection
from .components.functional import (
    AvgPool2dSemiFolded,
    Conv2dSemiFolded,
    LinearSemiFolded,
    MaxPool2dSemiFolded,
)
from .components.modules import BuiltComponentType
from .mixin import Container
from .node import NodeDict, NodeList

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


__all__ = ["DynSysGroup", "Network"]


class DynSysGroup(DynamicSys, Container):
    def __init__(
        self,
        *components_as_tuple,
        component_type: type = DynamicSys,
        name: Optional[str] = None,
        **components_as_dict,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(
            self.elem_format(component_type, *components_as_tuple, **components_as_dict)
        )

    def update(self, **kwargs) -> None:
        """Network update.

        XXX: The hierarchy of `NeuModule` requires that its update order is after synapses & before neurons.    \
            For example, a network with topology I1 -> M1 -> S1 -> N1, where the M1 consists of S2, S3 & N2. The\
            right update order is I1 -> S1, S2, S3 -> N1, N2. So the update order inside M1 is S2, S3 -> N2, of \
            which the update order is exactly between the synapses & neurons outside the module.

            It requires that the computing mechanism described inside modules can only be the computing process \
            from synapses (as inputs) to neurons (as outputs).
        """
        nodes = self.components

        for node in nodes.subset(Projection).values():
            node(**kwargs)

        for node in nodes.subset(SynSys).values():
            node()

        for node in nodes.subset(NeuModule).values():
            node()

        for node in nodes.subset(Neuron).values():
            node()

    def reset_state(self) -> None:
        nodes = self.components

        for node in nodes.subset(Projection).values():
            node.reset_state()

        for node in nodes.subset(SynSys).values():
            node.reset_state()

        for node in nodes.subset(NeuModule).values():
            node.reset_state()

        for node in nodes.subset(Neuron).values():
            node.reset_state()

    def __call__(self, **kwargs) -> None:
        return self.update(**kwargs)

    @classmethod
    def build_fmodule(
        cls, network: "DynSysGroup", **build_options
    ) -> dict[NeuModule, BuiltComponentType]:
        generated = dict()
        modules = network.nodes().subset(NeuModule).unique()

        # Valid interval for semi-folded components
        # If the input data is input continuously on the W-axis, the initial
        # valid interval for the first semi-folded component is 1.
        semi_valid_interval = 1
        ts_1st_valid_out = 0

        for module in modules.values():
            if isinstance(
                module, (Conv2dSemiFolded, MaxPool2dSemiFolded, AvgPool2dSemiFolded)
            ):
                generated[module] = module.build(
                    network, semi_valid_interval, ts_1st_valid_out, **build_options
                )
                semi_valid_interval *= module.stride[1]
                ts_1st_valid_out = module.ts_1st_valid_out
            elif isinstance(module, LinearSemiFolded):
                generated[module] = module.build(
                    network, semi_valid_interval, **build_options
                )
            else:
                generated[module] = module.build(network, **build_options)

        network._remove_modules_from_containers(network, modules)

        return generated

    def _add_components(self, *implicit: DynamicSys, **explicit: DynamicSys) -> None:
        """Add new components. When the component is passed in explicitly, its tag name can \
            be specified. When passing in implicitly, its attribute `.name` will be used.

        NOTE: After instantiating components outside `DynSysGroup`, you need to call it to  \
            actually add them to the network.
        """
        for comp in implicit:
            setattr(self, comp.name, comp)

        for tag, comp in explicit.items():
            setattr(self, tag, comp)

    def _remove_components(self, *components: DynamicSys) -> None:
        """Remove components from the network."""
        for cpn in components:
            for tag, obj in self.__dict__.items():
                if cpn is obj:
                    # cpn.__gh_build_ignore__ = False
                    delattr(self, tag)  # remove the cpn from the network
                    break

    def _ignore_components(self, *components: DynamicSys) -> None:
        for cpn in components:
            if cpn in self.__dict__.values():
                cpn.__gh_build_ignore__ = True

    @staticmethod
    def _remove_modules_from_containers(
        network: "DynSysGroup", modules: Collector[str, NeuModule]
    ) -> None:
        """Remove the built modules from the node containers of the network."""
        node_lists = [v for v in network.__dict__.values() if isinstance(v, NodeList)]
        node_dicts = [v for v in network.__dict__.values() if isinstance(v, NodeDict)]

        for module in modules.values():
            for lst in node_lists:
                if module in lst:
                    lst.remove(module)

            for dct in node_dicts:
                if module in dct.values():
                    dct.pop(module)

    @property
    def components(self) -> Collector[str, DynamicSys]:
        """Recursively search for all components within the network."""
        return self.nodes().subset(DynamicSys).unique().not_subset(DynSysGroup)


Network: TypeAlias = DynSysGroup


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
                raise KeyError(f"key '{item}' not found.")

        if isinstance(item, int):
            if item > len(self):
                raise IndexError(f"index out of range {item}.")

            return tuple(self.children.values())[item]

        if isinstance(item, slice):
            return Sequential(**dict(tuple(self.children.items())[item]))

        raise TypeError(
            f"expected type str, int or slice, but got {item}, type {type(item)}."
        )

    def __len__(self) -> int:
        return len(self.children)
