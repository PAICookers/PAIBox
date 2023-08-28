import numpy as np
from typing import Any
from ._types import Shape
from .utils import shape2num

global _id_dict, _type_names
_id_dict = dict()
_type_names = dict()

global global_reg_data
global_reg_data = dict()


def is_name_unique(name: str, obj: object) -> None:
    """If the name is unique, record it in the global dictionary. Otherwise raise ValueError."""
    if not name.isidentifier():
        raise ValueError(f"{name} is not a valid identifier")

    if name in _id_dict:
        if _id_dict[name] != id(obj):
            # TODO Define a new type of error
            raise ValueError(
                f"Nme of {obj}({name}) is already used by {_id_dict[name]}"
            )

    else:
        _id_dict[name] = id(obj)


def get_unique_name(_type: str) -> str:
    """Generate a unique name for a given type."""
    if _type not in _type_names:
        _type_names[_type] = 0

    name = f"{_type}_{_type_names[_type]}"
    _type_names[_type] += 1

    return name


def param_alloc(param: Any, shape: Shape):
    if isinstance(param, np.ndarray):
        _param = np.full(shape, param)
    elif np.isscalar(param):
        _param = [param] * shape2num(shape)
    else:
        raise TypeError

    return _param
