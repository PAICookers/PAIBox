from functools import wraps
from .base import PAIBoxObject
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

    def _get_elem_name(self, elem):
        if isinstance(elem, PAIBoxObject):
            return elem._name
        else:
            return get_unique_name("ContainerElem")

    def elem_format(self, child_type: type, *children):
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
