from collections.abc import Iterable, Sequence
from typing import Any, Optional, TypeVar

import numpy as np

from paibox.types import Shape

"""Handful utilities."""


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

    raise TypeError(f"unsupported type: {type(obj)}.")


def count_unique_elem(obj: Iterable[Any]) -> int:
    seen = set()
    for item in obj:
        seen.add(item)

    return len(seen)


_T = TypeVar("_T")


def merge_unique_ordered(lst1: list[_T], lst2: list[_T]) -> list[_T]:
    seen = set()
    result = []

    for item in lst1 + lst2:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def check_attr_same(obj: Sequence[Any], attr: str) -> bool:
    return all(getattr(obj[0], attr) == getattr(item, attr) for item in obj)


def check_elem_same(obj: Any) -> bool:
    if hasattr(obj, "__iter__") or hasattr(obj, "__contains__"):
        return len(set(obj)) == 1

    if isinstance(obj, dict):
        return len(set(obj.values())) == 1

    raise TypeError(f"unsupported type: {type(obj)}.")


def is_nested_obj(obj_on_top: Any) -> bool:
    """Check whether a object is nested"""
    return any(
        isinstance(item, Iterable) and not isinstance(item, str) for item in obj_on_top
    )


def shape2num(shape: Shape) -> int:
    """Convert a shape to a number"""
    if isinstance(shape, int):
        return shape
    elif isinstance(shape, np.ndarray):
        return int(np.prod(shape))
    else:
        a = 1
        for b in shape:
            a *= b

        return a


def as_shape(x, min_dim: int = 0) -> tuple[int, ...]:
    """Return a tuple if `x` is iterable or `(x,)` if `x` is integer."""
    if is_integer(x):
        _shape = (int(x),)
    elif is_iterable(x):
        _shape = tuple(int(e) for e in x)
    else:
        raise ValueError(f"{x} cannot be safely converted to a shape.")

    if len(_shape) < min_dim:
        _shape = (1,) * (min_dim - len(_shape)) + _shape

    return _shape


def is_shape(x, shape: Shape) -> bool:
    if not is_array_like(x):
        raise TypeError(f"only support an array-like type: {x}.")

    _x = np.asarray(x)
    return _x.shape == as_shape(shape)


def is_integer(obj: Any) -> bool:
    return isinstance(obj, (int, np.integer))


def is_number(obj: Any) -> bool:
    return is_integer(obj) or isinstance(obj, (float, np.number))


def is_array_like(obj: Any) -> bool:
    return (
        isinstance(obj, np.ndarray) or is_number(obj) or isinstance(obj, (list, tuple))
    )


def is_iterable(obj: Any) -> bool:
    """Check whether obj is an iterable."""
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0

    return isinstance(obj, Iterable)


def fn_sgn(a, b=0) -> int:
    """Signal function."""
    return (a > b) - (a < b)


def typical_round(n: float) -> int:
    if n - int(n) < 0.5:
        return int(n)
    else:
        return int(n) + 1


def bit_reversal(uint: int, n_bit: int = 8) -> int:
    """Reverse the bit order of a N-bit unsigned integer, where N is `n_bit`."""
    reversed = 0
    for i in range(n_bit):
        if (uint >> i) & 1:
            reversed += 1 << (n_bit - 1 - i)

    return reversed


def arg_check_pos(arg: int, desc: Optional[str] = None) -> int:
    _desc = "value" if desc is None else f"{desc}"
    if arg < 1:
        raise ValueError(f"{_desc} must be positive, but got {arg}.")

    return arg


def arg_check_non_pos(arg: int, desc: Optional[str] = None) -> int:
    _desc = "value" if desc is None else f"{desc}"
    if arg > 0:
        raise ValueError(f"{_desc} must be non-positive, but got {arg}.")

    return arg


def arg_check_neg(arg: int, desc: Optional[str] = None) -> int:
    _desc = "value" if desc is None else f"{desc}"
    if arg > -1:
        raise ValueError(f"{_desc} must be negative, but got {arg}.")

    return arg


def arg_check_non_neg(arg: int, desc: Optional[str] = None) -> int:
    _desc = "value" if desc is None else f"{desc}"
    if arg < 0:
        raise ValueError(f"{_desc} must be non-negative, but got {arg}.")

    return arg
