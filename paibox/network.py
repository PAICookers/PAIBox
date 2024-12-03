import sys
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from .base import DynamicSys, SynSys
from .collector import Collector
from .components import NeuModule, Neuron, Projection
from .components._modules import SemiFoldedDataFlowFormat, _SemiFoldedModule
from .components.modules import BuiltComponentType
from .exceptions import NotSupportedError
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

    def build_modules(
        self,
        pred_dg_semi_ops: Optional[dict[str, list[str]]] = None,
        ordered_semi_ops: Optional[list[NeuModule]] = None,
        **build_options,
    ) -> dict[NeuModule, BuiltComponentType]:
        """Build the functional modules in the network.

        Args:
            pred_dg_semi_ops (dict[str, list[str]], None): The predecessor directed graph of semi-folded operators.
            ordered_semi_ops (list[NeuModule], None): The ordered semi-folded operators.

        Returns:
            built_components (dict[NeuModule, BuiltComponentType]): The dictionary of generated basic components after building.
        """
        if pred_dg_semi_ops is not None and ordered_semi_ops is not None:
            # It is the network composed of all semi-folded operators.
            modules = ordered_semi_ops
        else:
            # It is the network composed of general operators.
            modules = list(self.components.subset(NeuModule).unique().values())

        generated = dict()

        # For external input dataflow:
        # 1. The start time is 0.
        # 2. The interval is 1.
        # 3. The #N of data is `INFINITE_DATA_STREAM` since it dosen't effect the subsequent output dataflow.
        # TODO Reserve an interface for setting the properties of external input from `FRONTEND_ENV`?
        last_vld_output_attr = SemiFoldedDataFlowFormat(t_1st_vld=0)

        for m in modules:
            # TODO for the case of the ResBlock, the `pred_dg_semi_ops` will be used.
            if isinstance(m, _SemiFoldedModule):
                generated[m] = m.build(self, last_vld_output_attr, **build_options)
                last_vld_output_attr = m._oflow_format
            else:
                generated[m] = m.build(self, **build_options)

        self._remove_modules(modules)
        return generated

    def is_composed_of_semi_folded_ops(self) -> bool:
        """Check if the network consists entirely or not of semi-folded operators. Return true if all the \
            components are semi-folded operators. Return false if all the components are not semi-folded. \
            In other cases, an exception will be raised.
        """
        if all(isinstance(cpn, _SemiFoldedModule) for cpn in self.components.values()):
            return True
        elif not all(
            isinstance(cpn, _SemiFoldedModule) for cpn in self.components.values()
        ):
            return False
        else:
            # XXX It seems that there will be no network mixed with semi-folded operators at present.
            raise NotSupportedError(
                "mixed semi-folded & normal operators in the network is not supported."
            )

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

    def _remove_modules(self, modules: Sequence[NeuModule]) -> None:
        """Remove the built modules from the network."""
        node_lst = [v for v in self.__dict__.values() if isinstance(v, NodeList)]
        node_dct = [v for v in self.__dict__.values() if isinstance(v, NodeDict)]

        for m in modules:
            for lst in node_lst:
                if m in lst:
                    lst.remove(m)

            for dct in node_dct:
                if m in dct.values():
                    dct.pop(m)

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
