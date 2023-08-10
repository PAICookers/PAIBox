from typing import Any, Optional
import numpy as np
from paibox.mixin import MixIn

global _id_dict, _type_names
_id_dict = dict()
_type_names = dict()

global global_reg_data
global_reg_data = dict()


def is_name_unique(name: str, obj: Any) -> None:
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


def get_unique_name(_type: str):
    """Generate a unique name for a given type."""
    if _type not in _type_names:
        _type_names[_type] = 0

    name = f"{_type}_{_type_names[_type]}"
    _type_names[_type] += 1

    return name


def unique_name(
    obj: Any, name: Optional[str] = None, _type: Optional[str] = None
) -> str:
    if name is None:
        if _type is None:
            __type = obj.__class__.__name__
        else:
            __type = _type

        return get_unique_name(__type)

    is_name_unique(name, obj)
    return name


class VarRegister(MixIn):
    local_vars: dict

    def add_register(
        self,
        name: str,
        target: np.ndarray,
        time_step: Optional[int] = None,
        initial_data: Optional[np.ndarray] = None,
    ):
        if name not in global_reg_data:
            global_reg_data[name] = target
        else:
            reg = global_reg_data[name][0]

        return time_step


class ParamRegister(MixIn):
    pass
