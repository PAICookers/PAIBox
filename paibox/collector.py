from typing import (
    Callable,
    Dict,
    MutableMapping,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

# XXX: use collections.UserDict[_KT, _VT] in 3.9+


class Collector(Dict[_KT, _VT]):
    def __setitem__(self, key, value) -> None:
        if key in self:
            if id(self[key]) != id(value):
                raise ValueError(
                    f"mame '{key}' conflicts: same name for {value} & {self[key]}."
                )

        super().__setitem__(key, value)

    @overload
    def update(self, other: MutableMapping[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def update(self, other: Sequence[_T]) -> "Collector[_KT, _T]": ...

    def update(
        self, other: Union[MutableMapping[_KT, _VT], Sequence[_T]]
    ) -> Union["Collector[_KT, _VT]", "Collector[_KT, _T]"]:
        if not isinstance(other, (MutableMapping, list, tuple)):
            raise TypeError(
                f"expected a collector, dict, list or sequence, but got {other}, type {type(other)}."
            )

        if isinstance(other, MutableMapping):
            super().update(other)
        else:
            l = len(self)
            for i, v in enumerate(other):
                self[f"_{l+i}"] = v  # type: ignore

        return self

    @overload
    def __add__(self, other: MutableMapping[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def __add__(self, other: Sequence[_T]) -> "Collector[_KT, _T]": ...

    def __add__(
        self, other: Union[MutableMapping[_KT, _VT], Sequence[_T]]
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
    def __sub__(self, other: MutableMapping[_KT, _VT]) -> "Collector[_KT, _VT]": ...

    @overload
    def __sub__(self, other: Sequence[_T]) -> "Collector[str, _T]": ...

    def __sub__(
        self, other: Union[MutableMapping[_KT, _VT], Sequence[_T]]
    ) -> Union["Collector[_KT, _VT]", "Collector[str, _T]"]:
        if not isinstance(other, (MutableMapping, list, tuple)):
            raise TypeError(
                f"expected a collector, dict, list or sequence, but got {other}, type {type(other)}."
            )

        gather = type(self)(self)

        if isinstance(other, MutableMapping):
            for k, v in other.items():
                if k not in gather.keys():
                    raise ValueError(f"cannot find '{k}' in {self.keys()}.")

                if id(v) != id(gather[k]):
                    raise ValueError(
                        f"cannot remove '{k}', since there are two different values: "
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
                    raise KeyError(f"key '{k}' not found. Removed failed.")

                gather.pop(k)

        return gather

    def subset(self, obj_type: Type[_T]) -> "Collector[_KT, _T]":
        gather = Collector()

        for k, v in self.items():
            if isinstance(v, obj_type):
                gather[k] = v

        return gather

    def not_subset(self, obj_type: Type[_T]) -> "Collector[_KT, _VT]":
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, obj_type):
                gather[k] = v

        return gather

    def include(self, *types: Type[_T]) -> "Collector[_KT, _T]":
        gather = Collector()

        for k, v in self.items():
            if isinstance(v, types):
                gather[k] = v

        return gather

    def exclude(self, *types: Type[_T]) -> "Collector[_KT, _VT]":
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, types):
                gather[k] = v

        return gather

    def unique(self) -> "Collector[_KT, _VT]":
        gather = type(self)()
        seen = set()

        for k, v in self.items():
            if id(v) not in seen:
                seen.add(id(v))
                gather[k] = v

        return gather

    def key_on_condition(
        self, condition: Callable[[_KT], bool]
    ) -> "Collector[_KT, _VT]":
        return type(self)({k: v for k, v in self.items() if condition(k)})

    def val_on_condition(
        self, condition: Callable[[_VT], bool]
    ) -> "Collector[_KT, _VT]":
        return type(self)({k: v for k, v in self.items() if condition(v)})
