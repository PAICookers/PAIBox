from typing import Any

__all__ = ["_IRNamespace"]


class _IRNamespace:
    def __init__(self) -> None:
        self._obj_to_name: dict[Any, str] = {}
        self._used_names = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, obj: Any | None) -> str:
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        base = obj.__class__.__name__
        num = self._base_count.get(base, 0)
        candidate = f"{base}_{num}"
        while candidate in self._used_names:
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is not None:
            self._obj_to_name[obj] = candidate

        return candidate

    def _rename_object(self, obj: Any, name: str) -> None:
        assert obj in self._obj_to_name
        self._obj_to_name[obj] = name
        self._used_names.add(name)
