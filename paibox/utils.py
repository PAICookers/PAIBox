from functools import wraps
from typing import Any, Iterable

"""
    Some handful utilities.
"""


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def check_elem_unique(obj: Any) -> bool:
    """Check whether a object consists of unique elements"""
    if hasattr(obj, "__iter__"):
        return len(obj) == len(set(obj))

    if isinstance(obj, dict):
        return len(obj) == len(set(obj.values()))

    if hasattr(obj, "__contains__"):
        seen = set()
        for elem in obj:
            if elem in seen:
                return False
            seen.add(elem)

        return True

    if hasattr(obj, "__iter__"):
        return len(obj) == len(set(obj))

    raise TypeError(f"Unsupported type to check: {type(obj)}")


def count_unique_elem(obj: Iterable) -> int:
    s = set()
    for item in obj:
        s.add(item)

    return len(s)


def is_nested(obj_on_top: Any) -> bool:
    """Check whether a object is nested"""
    return any(
        isinstance(item, Iterable) and not isinstance(item, str) for item in obj_on_top
    )


if __name__ == "__main__":
    tu = [(2, 3), (3, 4), (3, 4), (5, 6)]

    print(check_elem_unique(tu))

    a = [(1, 2, 3), (3, 4, 5)]

    print(is_nested(a))
