from typing import Any, Dict, Type


class Collector(dict):
    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            if id(self[key]) != id(value):
                raise ValueError(
                    f'Name "{key}" conflicts: same name for {value} and {self[key]}.'
                )

        dict.__setitem__(self, key, value)

    def replace(self, key: Any, new_value: Any) -> None:
        """Replace the old value with a new value of a key."""
        self.pop(key)
        self.update({key: new_value})

    def __add__(self, other: Dict[Any, Any]):
        """Merging two dicts.

        Arguments:
            - other: the other dictionary.
        Returns:
            - gather: the new collector.
        """
        gather = type(self)()
        gather.update(other)

        return gather

    def __sub__(self, other):
        pass

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
            if v.__class__ in types:
                gather[k] = v

        return gather

    def exclude(self, *types: Type):
        gather = type(self)()

        for k, v in self.items():
            if v.__class__ not in types:
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
