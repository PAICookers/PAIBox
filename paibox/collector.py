from typing import Any, Dict, Sequence, Type, Union


class Collector(dict):
    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            if id(self[key]) != id(value):
                raise ValueError(
                    f"Name '{key}' conflicts: same name for {value} and {self[key]}."
                )

        dict.__setitem__(self, key, value)

    def replace(self, key, new_value) -> None:
        self.pop(key)
        self.key = new_value

    def update(self, other: Union[Dict, Sequence]):
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
                self[f"_{l+i}"] = v

        return self

    def __add__(self, other: Union[Dict[str, Any], Sequence]):
        """Merging two dicts.

        Arguments:
            - other: the other dictionary.
        Returns:
            - gather: the new collector.
        """
        gather = type(self)(self)
        gather.update(other)

        return gather

    def __sub__(self, other: Union[Dict[str, Any], Sequence]):
        if not isinstance(other, (dict, list, tuple)):
            raise TypeError(
                f"Excepted a dict, list or sequence, but we got {other}, type {type(other)}"
            )

        gather = type(self)(self)

        if isinstance(other, dict):
            for k, v in other.items():
                if k not in gather.keys():
                    raise ValueError(f"Cannot find {k} in {self.keys()}.")

                if id(v) != id(gather[k]):
                    raise ValueError(
                        f"Cannot remove {k}, since there's two different values:"
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
                    raise ValueError
                gather.pop(k)

        return gather

    def subset(self, obj_type: Type):
        gather = type(self)()

        for k, v in self.items():
            if isinstance(v, obj_type):
                gather[k] = v

        return gather

    def not_subset(self, obj_type: Type):
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, obj_type):
                gather[k] = v

        return gather

    def include(self, *types: Type):
        gather = type(self)()

        for k, v in self.items():
            if isinstance(v, types):
                gather[k] = v

        return gather

    def exclude(self, *types: Type):
        gather = type(self)()

        for k, v in self.items():
            if not isinstance(v, types):
                gather[k] = v

        return gather

    def unique(self):
        gather = type(self)()
        seen = set()

        for k, v in self.items():
            if id(v) not in seen:
                seen.add(id(v))
                gather[k] = v

        return gather

    def on_condition(self, condition=lambda node: node):
        gather = type(self)()

        for k, v in self.items():
            if condition(k):
                gather[k] = v

        return gather
