from typing import Any, Iterable, Tuple

import numpy as np

from paibox._types import Shape

"""
    Some handful utilities.
"""


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


def is_nested_obj(obj_on_top: Any) -> bool:
    """Check whether a object is nested"""
    return any(
        isinstance(item, Iterable) and not isinstance(item, str) for item in obj_on_top
    )


def shape2num(shape: Shape) -> int:
    """Convert a shape to a number"""
    if isinstance(shape, int):
        return shape

    if isinstance(shape, (list, tuple)):
        a = 1
        for b in shape:
            a *= b

        return a

    raise ValueError(f"Type of {shape} is not supported: {type(shape)}")


def as_shape(shape, min_dim: int = 0) -> Tuple[int, ...]:
    """Convert a shape to a tuple, like (1,), (10,), or (10, 20)"""
    if is_integer(shape):
        _shape = (shape,)
    elif is_iterable(shape):
        _shape = tuple(shape)
    else:
        raise ValueError(f"Cannot make a shape for {shape}")
    
    if len(_shape) < min_dim:
        _shape = tuple([1] * (min_dim - len(_shape))) + _shape

    return _shape


def is_shape(x, shape: Shape) -> bool:
    if not is_array_like(x):
        raise TypeError(f"Only support an array-like type: {x}")

    _x = np.asarray(x)
    return _x.shape == as_shape(shape)


def is_integer(obj: Any) -> bool:
    return isinstance(obj, (int, np.integer))


def is_number(obj: Any) -> bool:
    return is_integer(obj) or isinstance(obj, (float, np.number))


def is_array(obj: Any) -> bool:
    return isinstance(obj, (np.generic, np.ndarray))


def is_array_like(obj: Any) -> bool:
    return is_array(obj) or is_number(obj) or isinstance(obj, (list, tuple))


def is_iterable(obj: Any) -> bool:
    """Check whether obj is an iterable."""
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0

    return isinstance(obj, Iterable)


def fn_sgn(a, b) -> int:
    """Signal function."""
    return 1 if a > b else -1 if a < b else 0
