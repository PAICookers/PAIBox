from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

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


def merge_unique_ordered(list1: List[Any], list2: List[Any]) -> List[Any]:
    seen = set()
    result = []

    for item in list1 + list2:
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
    else:
        a = 1
        for b in shape:
            a *= b

        return a


def as_shape(x, min_dim: int = 0) -> Tuple[int, ...]:
    """Return a tuple if `x` is iterable or `(x,)` if `x` is integer."""
    if is_integer(x):
        _shape = (x,)
    elif is_iterable(x):
        if isinstance(x, np.ndarray):
            _shape = tuple(x.astype(int))
        else:
            _shape = tuple(x)
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


def fn_sgn(a, b) -> int:
    """Signal function."""
    return 1 if a > b else -1 if a < b else 0


def bin_split(x: int, pos: int, high_mask: Optional[int] = None) -> Tuple[int, int]:
    """Split an integer, return the high and low part.

    Argument:
        - x: the integer
        - pos: the position (LSB) to split the binary.
        - high_mask: mask for the high part. Optional.

    Example::

        >>> bin_split(0b1100001001, 3)
        97(0b1100001), 1
    """
    low = x & ((1 << pos) - 1)

    if isinstance(high_mask, int):
        high = (x >> pos) & high_mask
    else:
        high = x >> pos

    return high, low


def bin_combine(high: int, low: int, pos: int) -> int:
    """Combine two integers, return the result.

    Argument:
        - high: the integer on the high bit.
        - low: the integer on the low bit.
        - pos: the combination bit if provided. Must be equal or greater than `low.bit_length()`.

    Example::

        >>> bin_combine(0b11000, 0b101, 5)
        773(0b11000_00101)
    """
    if pos < 0:
        raise ValueError("position must be greater than 0")

    if low > 0 and pos < low.bit_length():
        raise ValueError(
            f"Postion of combination must be greater than the bit length of low({low.bit_length()})"
        )

    return (high << pos) + low


def bin_combine_x(*components: int, pos: Union[int, List[int], Tuple[int, ...]]) -> int:
    """Combine more than two integers, return the result.

    Argument:
        - components: the list of integers to be combined.
        - pos: the combination bit(s) if provided. Every bit must be equal or greater than `low.bit_length()`.

    Example::

        >>> bin_combine_x(0b11000, 0b101, 0b1011, pos=[10, 5])
        24747(0b11000_00101_01011)
    """
    if isinstance(pos, (list, tuple)):
        if len(components) != len(pos) + 1:
            raise ValueError(
                f"Length of components and positions illegal: {len(components)}, {len(pos)}"
            )
    else:
        if len(components) != 2:
            raise ValueError(
                f"Length of components must be 2: {len(components)} when position is an integer."
            )

        return bin_combine(*components, pos=pos)

    result = components[-1]

    # Traverse every position from the end to the start
    for i in range(len(pos) - 1, -1, -1):
        result = bin_combine(components[i], result, pos[i])

    return result
