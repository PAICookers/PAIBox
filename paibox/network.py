import warnings
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
from typing_extensions import TypeAlias

from .base import DynamicSys, NeuDyn
from .collector import Collector
from .exceptions import PAIBoxWarning, RegisterError
from .mixin import Container
from .node import NodeDict
from .projection import InputProj, Projection
from .synapses import RIGISTER_MASTER_KEY_FORMAT, SynSys

__all__ = ["DynSysGroup", "Network"]

ComponentsType: TypeAlias = Union[InputProj, NeuDyn, SynSys]


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
        """For a network, the operating nodes within it will be distributed according to the network level  \
            where they are located. For I, S & N, if the network is a two-level nested network, it can be   \
            divided into Ix, Sx, Nx and Iy, Sy, Ny, where x & y are two parts containing many operations.   \

        TODO Prove that the operation sequence I->S->N can be divided into Ix->Sx->Nx->Iy->Sy->Ny & it has  \
            nothing to do with the network topology.
        """
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        for node in nodes.subset(Projection).values():
            node(**kwargs)

        for node in nodes.subset(SynSys).values():
            node()

        for node in nodes.subset(NeuDyn).values():
            node()

        for node in (
            nodes.not_subset(Projection).not_subset(SynSys).not_subset(NeuDyn).values()
        ):
            node()

    def reset_state(self) -> None:
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        for node in nodes.subset(Projection).values():
            node.reset_state()

        for node in nodes.subset(SynSys).values():
            node.reset_state()

        for node in nodes.subset(NeuDyn).values():
            node.reset_state()

        for node in (
            nodes.not_subset(Projection).not_subset(SynSys).not_subset(NeuDyn).values()
        ):
            node.reset_state()

    def __call__(self, **kwargs) -> None:
        return self.update(**kwargs)

    def add_components(self, *implicit: DynamicSys, **explicit: DynamicSys) -> None:
        """Add new components. When a component is passed in explicitly, its tag name \
            can be specified. Otherwise `.name` will be used.

        NOTE: After instantiated the components outside the `DynSysGroup`, you should \
            call `add_components()` to actually add the new components to itself.
        """
        for comp in implicit:
            setattr(self, comp.name, comp)

        for tag, comp in explicit.items():
            setattr(self, tag, comp)

    def _remove_component(self, remove: DynamicSys) -> None:
        """Remove a component in the network."""
        for tag, obj in self.__dict__.items():
            if obj is remove:
                delattr(self, tag)
                break

        return None

    def _disconnect_neudyn(
        self,
        neudyn_a: NeuDyn,
        condition: Callable[[SynSys], bool],
        neudyn_b: Optional[NeuDyn] = None,
        remove_syn: bool = True,
    ) -> List[SynSys]:
        nodes = self.nodes(level=1, include_self=False).subset(DynamicSys).unique()

        if neudyn_b is None:
            self._assert_neudyn(nodes, neudyn_a)
        else:
            self._assert_neudyn(nodes, neudyn_a, neudyn_b)

        target_syns = self._find_syn_to_unregi(nodes, condition)

        if target_syns:
            for syn in target_syns:
                self._disconnect_syn(syn)

                # FIXME The disconnected synapses will not effect the simulation.
                # However, it will effect the placement in the backend.
                if remove_syn:
                    self._remove_component(syn)

            return target_syns
        else:
            warnings.warn("There is no synapse to unregister.", PAIBoxWarning)
            return []

    def disconnect_neudyn_from(
        self, neudyn_a: NeuDyn, neudyn_b: NeuDyn, remove: bool = True
    ) -> List[SynSys]:
        """Disconnect synapses between `NeuDyn` A & B.

        Args:
            - neudyn_a: target `NeuDyn` A.
            - neudyn_b: target `NeuDyn` B.
            - remove: whether to remove the original synapses from the network.

        Returns: the disconnected synapses.
        """
        return self._disconnect_neudyn(
            neudyn_a,
            lambda syn: syn.source is neudyn_a and syn.dest is neudyn_b,
            neudyn_b,
            remove,
        )

    def diconnect_neudyn_succ(
        self, neudyn: NeuDyn, remove: bool = True
    ) -> List[SynSys]:
        """Disconnect successor synapses of `neudyn`.

        Args:
            - neudyn: target `NeuDyn`.
            - remove: whether to remove the original synapses from the network.

        Returns: the disconnected synapses.
        """
        return self._disconnect_neudyn(
            neudyn, lambda syn: syn.source is neudyn, remove_syn=remove
        )

    def diconnect_neudyn_pred(
        self, neudyn: NeuDyn, remove: bool = True
    ) -> List[SynSys]:
        """Disconnect predecessor synapses of `neudyn`.

        Args:
            - neudyn: target `NeuDyn`.
            - remove: whether to remove the original synapses from the network.

        Returns: the disconnected synapses.
        """
        return self._disconnect_neudyn(
            neudyn, lambda syn: syn.dest is neudyn, remove_syn=remove
        )

    def insert_neudyn(
        self,
        neudyn_a: NeuDyn,
        neudyn_b: NeuDyn,
        components_to_insert: Tuple[ComponentsType, ...],
        replace: bool = True,
        remove: bool = True,
    ) -> List[SynSys]:
        """Insert new components between `NeuDyn` A & B.

        Args:
            - neudyn_a: target `NeuDyn` A.
            - neudyn_b: target `NeuDyn` B.
            - components_to_insert: new components to insert between `neudyn_a` & `neudyn_b`.
            - replace: whether to disconnect the original synapses. Default is `True`.
            - remove: whether to remove the original synapses from the network. Valid only when `replace` is `True`.

        Returns: the disconnected synapses.
        """
        if replace:
            removed_syn = self.disconnect_neudyn_from(neudyn_a, neudyn_b, remove=remove)
        else:
            removed_syn = []

        self.add_components(*components_to_insert)

        return removed_syn

    @staticmethod
    def _find_syn_to_unregi(
        nodes: Collector, condition: Callable[[SynSys], bool]
    ) -> List[SynSys]:
        syns = []

        for syn in nodes.subset(SynSys).values():
            if condition(syn):
                syns.append(syn)

        return syns

    @staticmethod
    def _disconnect_syn(target_syn: SynSys) -> None:
        ret = target_syn.dest.unregister_master(
            RIGISTER_MASTER_KEY_FORMAT.format(target_syn.name)
        )
        if ret is not target_syn:
            raise RegisterError("Unregister failed!")

    @staticmethod
    def _assert_neudyn(nodes: Collector, *neudyns: NeuDyn) -> None:
        neu_dyns = nodes.subset(NeuDyn)

        if any(neudyn not in neu_dyns.values() for neudyn in neudyns):
            raise ValueError("Not all NeuDyn found in the network.")


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
