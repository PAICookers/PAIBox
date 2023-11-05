from functools import wraps
from typing import Type

import numpy as np

import paibox as pb

from .generic import get_unique_name
from .node import NodeDict


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


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
            raise ValueError

        self.master_nodes[key] = master_target

    def get_master_node(self, key: str):
        return self.master_nodes.get(key, None)

    def sum_inputs(self, *args, init=0, **kwargs) -> np.ndarray:
        # TODO Out is a np.ndarray right now, but it may be more than one type.
        output = init
        for node in self.master_nodes.values():
            output += node.output

        return np.array(output)
