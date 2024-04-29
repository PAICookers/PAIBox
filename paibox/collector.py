from typing import Callable, Dict, Generic, Sequence, Type, TypeVar, Union, overload

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class Collector(dict, Generic[_KT, _VT]):
    def __setitem__(self, key: _KT, value: _VT) -> None:
        if key in self:
            if id(self[key]) != id(value):
                raise ValueError(
                    f"Name '{key}' conflicts: same name for {value} and {self[key]}."
                )

        super().__setitem__(key, value)

    def replace(self, key: _KT, new_value: _VT) -> None:
        self.pop(key)
        self.key = new_value

    @overload
    def update(self, other: Dict[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def update(self, other: Sequence[_T]) -> "Collector[_KT, _T]": ...

    def update(
        self, other: Union[Dict[_KT, _VT], Sequence[_T]]
    ) -> Union["Collector[_KT, _VT]", "Collector[_KT, _T]"]:
        if not isinstance(other, (dict, list, tuple)):
            raise TypeError(
                f"Excepted a dict, list or sequence, but we got {other}, type {type(other)}"
            )

        if isinstance(other, dict):
            for k, v in other.items():
                self[k] = v
        else:
            l = len(self)
            for i, v in enumerate(other):
                self[f"_{l+i}"] = v  # type: ignore

        return self

    @overload
    def __add__(self, other: Dict[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def __add__(self, other: Sequence[_T]) -> "Collector[_KT, _T]": ...

    def __add__(
        self, other: Union[Dict[_KT, _VT], Sequence[_T]]
    ) -> Union["Collector[_KT, _VT]", "Collector[_KT, _T]"]:
        """Merging two dicts.

        Arguments:
            - other: the other dictionary.
        Returns:
            - gather: the new collector.
        """
        gather = type(self)(self)
        gather.update(other)

        return gather

    @overload
    def __sub__(self, other: Dict[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def __sub__(self, other: Sequence[_T]) -> "Collector[str, _T]": ...

    def __sub__(
        self, other: Union[Dict[_KT, _VT], Sequence[_T]]
    ) -> Union["Collector[_KT, _VT]", "Collector[str, _T]"]:
        if not isinstance(other, (dict, list, tuple)):
            raise TypeError(
                f"Excepted a dict, list or sequence, but we got {other}, type {type(other)}"
            )

        gather = type(self)(self)

        if isinstance(other, dict):
            for k, v in other.items():
                if k not in gather.keys():
                    raise ValueError(f"Cannot find '{k}' in {self.keys()}.")

                if id(v) != id(gather[k]):
                    raise ValueError(
                        f"Cannot remove '{k}', since there's two different values:"
                        f"{v} != {gather[k]}"
                    )
                gather.pop(k)
        else:
            id_to_keys = {}
            for k, v in self.items():
                _id = id(v)
                if _id not in id_to_keys:
                    id_to_keys[_id] = []

                id_to_keys[_id].append(k)

            keys_to_remove = []
            for k in other:
                if isinstance(k, str):
                    keys_to_remove.append(k)
                else:
                    keys_to_remove.extend(id_to_keys[id(k)])

            for k in set(keys_to_remove):
                if k not in gather:
                    raise KeyError(f"Key '{k}' not found. Removed failed.")

                gather.pop(k)

        return gather

    def subset(self, obj_type: Type[_T]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if isinstance(v, obj_type):
                gather[k] = v  # type: ignore

        return gather

    def not_subset(self, obj_type: Type[_T]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, obj_type):
                gather[k] = v

        return gather

    def include(self, *types: Type[_T]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if isinstance(v, types):
                gather[k] = v  # type: ignore

        return gather

    def exclude(self, *types: Type[_T]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, types):
                gather[k] = v

        return gather

    def unique(self) -> "Collector":
        gather = type(self)()
        seen = set()

        for k, v in self.items():
            if id(v) not in seen:
                seen.add(id(v))
                gather[k] = v

        return gather

    def key_on_condition(self, condition: Callable[..., bool]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if condition(k):
                gather[k] = v

        return gather

    def value_on_condition(self, condition: Callable[..., bool]) -> "Collector":
        gather = type(self)()

        for k, v in self.items():
            if condition(v):
                gather[k] = v

        return gather
