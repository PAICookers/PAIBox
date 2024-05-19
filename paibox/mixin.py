from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Optional, Sequence, Type, TypeVar

import numpy as np

from .context import _FRONTEND_CONTEXT
from .exceptions import RegisterError
from .naming import get_unique_name
from .node import NodeDict
from .types import VoltageType

_T = TypeVar("_T")


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def prevent(func):
    """Decorate function with this to prevent raising an Exception when an error is encountered."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            return

    return wrapper


def check(attr):
    """Decorate function with this to check whether the object has an attribute with the given name."""

    def decorator(method):

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, attr):
                return method(self, *args, **kwargs)

            return None

        return wrapper

    return decorator


class MixIn:
    """Mix-in class."""

    pass


# XXX this class seems useless
class Container(MixIn):
    children: NodeDict[str, Any]

    def __getitem__(self, item: str) -> Any:
        if item in self.children:
            return self.children[item]

        raise KeyError(f"key '{item}' not found.")

    def _get_elem_name(self, elem: Any) -> str:
        from .base import PAIBoxObject

        if isinstance(elem, PAIBoxObject):
            return elem._name
        else:
            return get_unique_name("ContainerElem")

    def elem_format(
        self,
        child_type: Type[_T],
        *children_as_tuple: Sequence[_T],
        **children_as_dict: Dict[Any, _T],
    ) -> Dict[str, _T]:
        elems = dict()

        for child in children_as_tuple:
            if isinstance(child, child_type):
                elems[self._get_elem_name(child)] = child

            elif isinstance(child, (list, tuple)):
                for c in child:
                    if not isinstance(c, child_type):
                        raise ValueError(
                            f"expect type {child_type.__name__}, but got {type(c)}."
                        )
                    elems[self._get_elem_name((c))] = c

            elif isinstance(child, dict):
                for k, v in child.items():
                    if not isinstance(v, child_type):
                        raise ValueError(
                            f"expect type {child_type.__name__}, but got {type(c)}."
                        )
                    elems[k] = v
            else:
                raise TypeError(
                    f"expect elements in dict, list or tuple, but got {type(child)}."
                )

        for k, v in children_as_dict.items():
            if not isinstance(v, child_type):
                raise ValueError(
                    f"expect type {child_type.__name__}, but got {type(v)}."
                )
            elems[k] = v

        return elems

    def add_elem(self, *elems, **elements) -> None:
        """Add elements as a dictionary"""
        self.children.update(self.elem_format(object, *elems, **elements))


class ReceiveInputProj(MixIn):
    master_nodes: NodeDict[str, Any]

    def register_master(self, key: str, master_target) -> None:
        if key in self.master_nodes:
            raise RegisterError(f"master node with key '{key}' already exists.")

        self.master_nodes[key] = master_target

    def unregister_master(self, key: str, strict: bool = True) -> Optional[Any]:
        if key in self.master_nodes:
            return self.master_nodes.pop(key, None)
        elif strict:
            raise KeyError(f"key '{key}' not found in master nodes.")

    def get_master_node(self, key: str) -> Optional[Any]:
        return self.master_nodes.get(key, None)

    def sum_inputs(self, *, init: VoltageType = 0, **kwargs) -> VoltageType:  # type: ignore
        # TODO Out is a np.ndarray right now, but it may be more than one type.
        output = init
        for node in self.master_nodes.values():
            output += node.output.copy()

        return np.array(output).astype(np.int32)


class TimeRelatedNode(MixIn):
    """Add time-related properties for `NeuDyn` & `InputProj`."""

    @property
    def delay_relative(self) -> int:
        """Relative delay, positive."""
        raise NotImplementedError

    @property
    def tick_wait_start(self) -> int:
        """The starting point of the local timeline, non-negative."""
        raise NotImplementedError

    @property
    def tick_wait_end(self) -> int:
        """Duration of the local timeline, non-negative."""
        raise NotImplementedError

    @property
    def timestamp(self) -> int:
        """Local timestamp."""
        return _FRONTEND_CONTEXT["t"] - self.tick_wait_start


class StatusMemory(MixIn):
    """Register memories for stateful variables."""

    def __init__(self) -> None:
        self._memories: NodeDict[str, Any] = NodeDict()
        self._memories_rv: NodeDict[str, Any] = NodeDict()

    def set_memory(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError(f"'{name}' has been set as a member variable.")

        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset_memory(self, name: Optional[str] = None) -> None:
        if isinstance(name, str):
            if name in self._memories:
                self._memories[name] = deepcopy(self._memories_rv[name])
            else:
                raise KeyError(f"key '{name}' not found.")
        else:
            for k in self._memories.keys():
                self._memories[k] = deepcopy(self._memories_rv[k])

    def set_reset_value(self, name: str, init_value: Any) -> None:
        self._memories_rv[name] = deepcopy(init_value)

    def __getattr__(self, name: str) -> Any:
        if "_memories" in self.__dict__:
            _memories = self.__dict__["_memories"]
            if name in _memories:
                return _memories[name]

        raise AttributeError(f"attribute '{name}' not found.")

    def __setattr__(self, name: str, value: Any) -> None:
        _memories = self.__dict__.get("_memories")
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
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

    def __copy__(self) -> NodeDict[str, Any]:
        return deepcopy(self._memories)
