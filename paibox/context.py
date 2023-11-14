from typing import Any


class _Context:
    def __init__(self) -> None:
        self._context = dict()

    def load(self, key, value: Any = None):
        """Load the context by the `key`.

        Args:
            - key: the key to indicate the data.
            - value: the default value when `key` is not defined.
        """
        if key in self._context:
            return self._context[key]

        if value is None:
            raise KeyError("The context of {key} not found.")
        else:
            return value

    def save(self, *args, **kwargs) -> None:
        """Save the context by the key-value pairs."""
        if len(args) % 2 > 0:
            # TODO
            raise ValueError

        for i in range(0, len(args), 2):
            k = args[i]
            v = args[i + 1]
            self._context[k] = v

        for k, v in kwargs.items():
            self._context[k] = v

    def __setitem__(self, key, value) -> None:
        self.save(key, value)

    def __getitem__(self, item):
        return self.load(item)

    def get_contexts(self):
        """Get all contexts."""
        return self._context.copy()

    def clear(self, *args) -> None:
        """Clear contexts."""
        if len(args) > 0:
            for arg in args:
                self._context.pop(arg)
        else:
            self._context.clear()
