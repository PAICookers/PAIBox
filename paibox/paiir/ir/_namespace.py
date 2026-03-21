"""IR node namespace management.

Assigns unique names to IR nodes in the format ``{ClassName}_{counter}``.
"""

from typing import Any

__all__ = ["IRNamespace"]


class IRNamespace:
    """IR node namespace ensuring globally unique names."""

    def __init__(self) -> None:
        self._obj_to_name: dict[int, str] = {}
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, obj: Any) -> str:
        """Assign a unique name to *obj*. Returns the same name on repeated calls."""
        obj_id = id(obj)
        if obj_id in self._obj_to_name:
            return self._obj_to_name[obj_id]

        base = type(obj).__name__
        num = self._base_count.get(base, 0)
        candidate = f"{base}_{num}"
        while candidate in self._used_names:
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num + 1
        self._obj_to_name[obj_id] = candidate
        return candidate

    def rename(self, obj: Any, name: str) -> None:
        """Rename a registered object."""
        obj_id = id(obj)
        if obj_id not in self._obj_to_name:
            raise KeyError(f"object {obj!r} not registered in namespace")
        self._obj_to_name[obj_id] = name
        self._used_names.add(name)

    def clear(self) -> None:
        """Clear all registered names."""
        self._obj_to_name.clear()
        self._used_names.clear()
        self._base_count.clear()
