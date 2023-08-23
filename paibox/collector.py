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

    # def update(self, other, **kwargs):
    #     assert isinstance(other, (dict, list, tuple))
    #     if isinstance(other, dict):
    #         for k, v in other.items():
    #             self.update({k: v})
    #     elif isinstance(other, (tuple, list)):
    #         num = len(self)
    #         for i, v in enumerate(other):
    #             self[f'_var{i + num}'] = v
    #     else:
    #         raise ValueError(f'Only supports dict/list/tuple, but we got {type(other)}')

    #     for k, v in kwargs.items():
    #         self[k] = v

    #     return self

    @classmethod
    def record(cls):
        pass

    def __add__(self, other: Dict[Any, Any]):
        r"""Merging two dicts.

        Arguments:
            - other: the other dictionary.

        Returns:
            - gather: the new collector.
        """
        gather = type(self)()
        gather.update(other)
        return gather

    def subset(self, _type: Type):
        gather = type(self)()

        for k, v in self.items():
            if isinstance(k, _type):
                gather[k] = v

        return gather
