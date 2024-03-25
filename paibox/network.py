import warnings
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
from typing_extensions import TypeAlias

from .base import DynamicSys, NeuDyn, SynSys
from .collector import Collector
from .components import (
    BuildingModule,
    InputProj,
    Neuron,
    Projection,
    RIGISTER_MASTER_KEY_FORMAT,
)
from .exceptions import PAIBoxWarning, RegisterError
from .mixin import Container
from .node import NodeDict

__all__ = ["DynSysGroup", "Network"]

ComponentsType: TypeAlias = Union[InputProj, Neuron, SynSys]
# TODO replace `Neuron` with `NeuDyn`. Need tests.


class DynSysGroup(DynamicSys, Container):

    def __init__(
        self,
        *components_as_tuple,
        component_type: Type = DynamicSys,
        name: Optional[str] = None,
        **components_as_dict,
    ) -> None:
        super().__init__(name)
        self.children = NodeDict(
            self.elem_format(component_type, *components_as_tuple, **components_as_dict)
        )

    def update(self, **kwargs) -> None:
        """Find nodes in the network recursively."""
        nodes = (
            self.nodes(include_self=False, find_recursive=True)
            .subset(DynamicSys)
            .unique()
        )

        for node in nodes.subset(Projection).values():
            node(**kwargs)

        for node in nodes.subset(SynSys).values():
            node()

        for node in nodes.subset(NeuDyn).values():
            node()

    def reset_state(self) -> None:
        nodes = (
            self.nodes(include_self=False, find_recursive=True)
            .subset(DynamicSys)
            .unique()
        )

        for node in nodes.subset(Projection).values():
            node.reset_state()

        for node in nodes.subset(SynSys).values():
            node.reset_state()

        for node in nodes.subset(NeuDyn).values():
            node.reset_state()

    def __call__(self, **kwargs) -> None:
        return self.update(**kwargs)

    def build(self, **build_options) -> None:
        modules = (
            self.nodes(include_self=False, find_recursive=True)
            .subset(BuildingModule)
            .unique()
        )

        for module in modules.values():
            module.build(self, **build_options)

    def add_components(self, *implicit: DynamicSys, **explicit: DynamicSys) -> None:
        """Add new components. When the component is passed in explicitly, its tag name can \
            be specified. When passing in implicitly, its attribute `.name` will be used.

        NOTE: After instantiating components outside `DynSysGroup`, you need to call it to  \
            actually add them to the network.
        """
        for comp in implicit:
            setattr(self, comp.name, comp)

        for tag, comp in explicit.items():
            setattr(self, tag, comp)

    def disconnect_syn(
        self, target_syn: SynSys, exclude_source: bool = False
    ) -> SynSys:
        """Disconnect a synapse in the nwtwork.

        Args:
            - target_syn: target synapse.
            - exclude_source: whether to disconnect the source. If so, remove the synapse   \
                from the network

        Returns: the disconnected synapse.
        """
        ret = target_syn.dest.unregister_master(
            RIGISTER_MASTER_KEY_FORMAT.format(target_syn.name)
        )
        if ret is not target_syn:
            raise RegisterError("unregister failed.")

        if not exclude_source:
            self.remove_component(target_syn)

        return target_syn

    def disconnect_neuron_from(
        self, neuron_a: Neuron, neuron_b: Neuron
    ) -> List[SynSys]:
        """Disconnect synapses between `Neuron` A & B and remove the synapses from the network.

        Args:
            - neuron_a: target neuron A.
            - neuron_b: target neuron B.

        Returns: the disconnected synapses in list.
        """
        return self._disconn_neuron(
            neuron_a,
            lambda syn: syn.source is neuron_a and syn.dest is neuron_b,
            neuron_b,
            remove_syn=True,
        )

    # Not sure about specific needs
    # def diconnect_neuron_succ(self, neuron: Neuron) -> List[SynSys]:
    #     """Disconnect successor synapses of `neuron`.

    #     Args:
    #         - neuron: target neuron.
    #         - remove: whether to remove the original synapses from the network.
    #         - new_source: only valid when `remove` is false.

    #     Returns: the disconnected synapses.
    #     """
    #     return self._disconn_neuron(
    #         neuron, lambda syn: syn.source is neuron, remove_syn=True
    #     )

    # def replace_neuron_succ(self, neuron: Neuron, new_source: Neuron) -> List[SynSys]:
    #     """Replace the source of successor synapses of `neuron` with new one."""
    #     disconn_syns = self._disconn_neuron(
    #         neuron, lambda syn: syn.source is neuron, remove_syn=False
    #     )

    #     for syn in disconn_syns:
    #         syn.source = new_source

    #     return disconn_syns

    # def replace_neuron_pred(self, neuron: Neuron, new_source: Neuron) -> List[SynSys]:
    #     """Replace the destination of predecessor synapses of `neuron` with new one.

    #     Args:
    #         - neuron: target neuron.
    #         - remove: whether to remove the original synapses from the network.

    #     Returns: the disconnected synapses.
    #     """
    #     disconn_syns = self._disconn_neuron(
    #         neuron, lambda syn: syn.dest is neuron, remove_syn=False
    #     )

    #     for syn in disconn_syns:
    #         syn.dest = new_source

    #     return disconn_syns

    def insert_between_neuron(
        self,
        neuron_a: Neuron,
        neuron_b: Neuron,
        cpn_to_insert: Tuple[ComponentsType, ...],
        replace: bool = True,
    ) -> List[SynSys]:
        """Insert new components between `Neuron` A & B.

        Args:
            - neuron_a: target neuron A.
            - neuron_b: target neuron B.
            - cpn_to_insert: components to insert between `neuron_a` & `neuron_b`.
            - replace: whether to disconnect the original synapses. Default is `True`.

        Returns: the disconnected synapses in list.
        """
        if replace:
            removed_syn = self.disconnect_neuron_from(neuron_a, neuron_b)
        else:
            removed_syn = []

        self.add_components(*cpn_to_insert)

        return removed_syn

    def remove_component(self, remove: DynamicSys) -> None:
        """Remove a component from the network."""
        for tag, obj in self.__dict__.items():
            if obj is remove:
                delattr(self, tag)
                break

        return None

    def _disconn_neuron(
        self,
        neuron_a: Neuron,
        condition: Callable[[SynSys], bool],
        neuron_b: Optional[Neuron] = None,
        remove_syn: bool = True,
    ) -> List[SynSys]:
        nodes = (
            self.nodes(include_self=False, find_recursive=True)
            .subset(DynamicSys)
            .unique()
        )

        if neuron_b is None:
            self._assert_neuron(nodes, neuron_a)
        else:
            self._assert_neuron(nodes, neuron_a, neuron_b)

        target_syns = self._find_syn_to_disconn(nodes, condition)

        if target_syns:
            for syn in target_syns:
                self._disconn_syn(syn)

                # The disconnected synapses will not effect the simulation, but will
                # effect the placement in the backend.
                # If the disconnected synapses aren't removed from the network, do cleaning
                # before the compilation in the backend.
                # TODO Add a pre-processing step before the compilation.
                if remove_syn:
                    self.remove_component(syn)

            return target_syns
        else:
            warnings.warn("there is no synapses to disconnect.", PAIBoxWarning)
            return []

    @staticmethod
    def _find_syn_to_disconn(
        nodes: Collector, condition: Callable[[SynSys], bool]
    ) -> List[SynSys]:
        syns = []

        for syn in nodes.subset(SynSys).values():
            if condition(syn):
                syns.append(syn)

        return syns

    @staticmethod
    def _disconn_syn(target_syn: SynSys) -> None:
        ret = target_syn.dest.unregister_master(
            RIGISTER_MASTER_KEY_FORMAT.format(target_syn.name)
        )
        if ret is not target_syn:
            raise RegisterError("unregister failed.")

    @staticmethod
    def _disconn_succ_syn(target_syn: SynSys) -> None:
        ret = target_syn.dest.unregister_master(
            RIGISTER_MASTER_KEY_FORMAT.format(target_syn.name)
        )
        if ret is not target_syn:
            raise RegisterError("unregister failed.")

    @staticmethod
    def _assert_neuron(nodes: Collector, *neurons: Neuron) -> None:
        neu_dyns = nodes.subset(Neuron)

        if any(neuron not in neu_dyns.values() for neuron in neurons):
            raise ValueError("not all neurons found in the network.")


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
