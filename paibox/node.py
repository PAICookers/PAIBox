from typing import Sequence


class NodeList(list):
    def __init__(self, seq: Sequence = ()) -> None:
        super().__init__()
        self.extend(seq)

    def append(self, elem) -> "NodeList":
        super().append(elem)
        return self

    def extend(self, iterable) -> "NodeList":
        for elem in iterable:
            self.append(elem)

        return self


class NodeDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        return self

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        raise KeyError(f"Key {key} not found.")
