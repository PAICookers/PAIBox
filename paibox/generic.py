import warnings

from .exceptions import PAIBoxWarning, RegisterError

global _id_dict, _type_names
_id_dict = dict()
_type_names = dict()


def is_name_unique(name: str, obj: object) -> None:
    """If the name is unique, record it in the global dictionary.
    Otherwise raise an error.
    """
    if not name.isidentifier():
        raise ValueError(f"{name} is not a valid identifier")

    if name in _id_dict:
        if _id_dict[name] != id(obj):
            raise RegisterError(
                f"Name of {obj}({name}) is already used by {_id_dict[name]}"
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


def clear_name_cache(ignore_warn: bool = False) -> None:
    """Clear the name dictionary."""
    _id_dict.clear()
    _type_names.clear()

    if not ignore_warn:
        warnings.warn(f"All named models & ids are cleared.", PAIBoxWarning)
