import copy
from functools import wraps
from typing import Any, Type

import numpy as np

import paibox as pb

from .generic import get_unique_name
from .node import NodeDict
from .exceptions import RegisterError

def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def prevent(func):
    """
    Decorate func with this to prevent raising an Exception when \
    an error is encountered.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            return

    return wrapper


def check(attr):
    def decorator(method):
        """
        Decorate method with this to check whether the object \
        has an attribute with the given name.
        """

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, attr):
                return method(self, *args, **kwargs)

            return None

        return wrapper

    return decorator


class MixIn:
    pass


class Container(MixIn):
    children: NodeDict

    def __getitem__(self, item):
        if item in self.children:
            return self.children[item]

        raise KeyError

    def _get_elem_name(self, elem) -> str:
        if isinstance(elem, pb.base.PAIBoxObject):
            return elem._name
        else:
            return get_unique_name("ContainerElem")

    def elem_format(self, child_type: Type, *children):
        elems = dict()

        for child in children:
            if isinstance(child, child_type):
                elems[self._get_elem_name(child)] = child

            elif isinstance(child, (list, tuple)):
                for c in child:
                    if not isinstance(c, child_type):
                        raise ValueError
                    elems[self._get_elem_name((c))] = c

            elif isinstance(child, dict):
                for k, v in child.items():
                    if not isinstance(v, child_type):
                        raise ValueError
                    elems[k] = v
            else:
                raise ValueError

        return elems

    def add_elem(self, **elems) -> None:
        """Add elements as a dictionary"""

        self.children.update(self.elem_format(object, **elems))


class ReceiveInputProj(MixIn):
    master_nodes: NodeDict

    def register_master(self, key: str, master_target) -> None:
        if key in self.master_nodes:
            # TODO
            raise RegisterError(f"Master node with key '{key}' already exists.")

        self.master_nodes[key] = master_target

    def get_master_node(self, key: str):
        return self.master_nodes.get(key, None)

    def sum_inputs(self, *args, init=0, **kwargs) -> np.ndarray:
        # TODO Out is a np.ndarray right now, but it may be more than one type.
        output = init
        for node in self.master_nodes.values():
            output += node.output

        return np.array(output)


class StatusMemory(MixIn):
    """Register memories for stateful variables."""

    def __init__(self) -> None:
        self._memories = NodeDict()
        self._memories_rv = NodeDict()

    def set_memory(self, name: str, value) -> None:
        if hasattr(self, name):
            raise ValueError(f"{name} has been set as a member variable!")

        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self) -> None:
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value) -> None:
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if "_memories" in self.__dict__:
            _memories = self.__dict__.get("_memories")
            if _memories is not None and name in _memories:
                return _memories[name]

        raise AttributeError

    def __setattr__(self, name: str, value: Any) -> None:
        _memories = self.__dict__.get("_memories")
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def memories(self):
        for v in self._memories.values():
            yield v

    def named_memories(self):
        for k, v in self._memories.items():
            yield k, v

    def copy(self) -> NodeDict:
        return copy.deepcopy(self._memories)
