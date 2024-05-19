from typing import Any, Dict, TypeVar

__all__ = ["FRONTEND_ENV"]


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

# XXX: use collections.UserDict[_KT, _VT] in 3.9+


class _Context(Dict[_KT, _VT]):
    def load(self, key: Any, default: Any = None) -> Any:
        """Load the context by the `key`.

        Args:
            - key: the key to indicate the data.
            - default: the default value when `key` is not defined.
        """
        if key in self:
            return super().__getitem__(key)

        if default is None:
            raise KeyError(f"the context of '{key}' not found.")

        return default

    def save(self, *args, **kwargs) -> None:
        """Save the context by the key-value pairs."""
        if len(args) % 2 > 0:
            raise TypeError(
                f"expected even positional arguments, but odd given ({len(args)})."
            )

        for i in range(0, len(args), 2):
            k = args[i]
            v = args[i + 1]
            super().__setitem__(k, v)

        self.update(**kwargs)  # compatible for py3.8

    def get_ctx(self):
        """Get all contexts."""
        return self.copy()

    def clear_ctx(self, *args) -> None:
        """Clear one or some contexts."""
        if len(args) > 0:
            for arg in args:
                self.pop(arg)
        else:
            self.clear()

    def clear_all(self) -> None:
        """Clear all contexts."""
        return self.clear()


class _FrontendContext(_Context):
    def __init__(self, initial_t: int = 0) -> None:
        super().__init__()
        self["t"] = initial_t  # RO. Changed by simulator.


_FRONTEND_CONTEXT = _FrontendContext()
FRONTEND_ENV = _FRONTEND_CONTEXT
