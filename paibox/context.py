from typing import Any

__all__ = ["FRONTEND_ENV"]


class _Context:
    def __init__(self) -> None:
        self._context = dict()

    def load(self, key, default: Any = None):
        """Load the context by the `key`.

        Args:
            - key: the key to indicate the data.
            - default: the default value when `key` is not defined.
        """
        if key in self._context:
            return self._context[key]

        if default is None:
            raise KeyError("The context of {key} not found.")

        return default

    def save(self, *args, **kwargs) -> None:
        """Save the context by the key-value pairs."""
        if len(args) % 2 > 0:
            raise TypeError(
                f"Expected even positional arguments but odd given {len(args)}"
            )

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

    def get_ctx(self):
        """Get all contexts."""
        return self._context.copy()

    def clear_ctx(self, *args) -> None:
        """Clear one or some contexts."""
        if len(args) > 0:
            for arg in args:
                self._context.pop(arg)
        else:
            self._context.clear()

    def clear_all(self) -> None:
        """Clear all contexts."""
        return self._context.clear()


class _FrontendContext(_Context):
    def __init__(self, initial_t: int = 0) -> None:
        super().__init__()
        self._context["t"] = initial_t


_FRONTEND_CONTEXT = _FrontendContext()
FRONTEND_ENV = _FRONTEND_CONTEXT
