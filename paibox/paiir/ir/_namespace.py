"""IR node namespace management.

Assigns unique names to IR nodes in the format ``{BaseName}_{counter}``.
The design stays intentionally small, but follows the useful parts of
``torch.fx.graph._Namespace``:

- object -> name association is stable for the lifetime of the object
- names are unique within the namespace
- callers may provide a preferred candidate base instead of always using the class name
"""

import re
from typing import Any
from weakref import WeakKeyDictionary

__all__ = ["IRNamespace"]

_NAME_REGEX = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*?)(?:_([0-9]+))?$")
_ILLEGAL_CHAR_REGEX = re.compile(r"[^A-Za-z0-9_]")


class IRNamespace:
    """IR node namespace ensuring globally unique names."""

    def __init__(self) -> None:
        self._obj_to_name: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, candidate: str | Any, obj: Any | None = None) -> str:
        """Assign a unique name.

        Supported call forms:
        - ``create_name(obj)``: use ``type(obj).__name__`` as the base
        - ``create_name(candidate, obj)``: use the provided candidate base
        """
        if obj is None and not isinstance(candidate, str):
            obj = candidate
            candidate = type(obj).__name__

        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        normalized = self._normalize_candidate(candidate)
        base, explicit_index = self._split_candidate(normalized)

        if explicit_index is None or normalized in self._used_names:
            num = self._base_count.get(base, 0)
            name = f"{base}_{num}"
        else:
            num = explicit_index
            name = normalized

        while name in self._used_names:
            num += 1
            name = f"{base}_{num}"

        self._used_names.add(name)
        self._base_count[base] = num + 1
        if obj is not None:
            self._obj_to_name[obj] = name
        return name

    def associate_name_with_obj(self, name: str, obj: Any) -> None:
        """Associate an already chosen unique name with an object."""
        normalized = self._normalize_candidate(name)
        if obj in self._obj_to_name:
            raise KeyError(f"object {obj!r} is already registered in namespace")
        if normalized in self._used_names:
            raise ValueError(f"name {normalized!r} is already used in namespace")

        self._used_names.add(normalized)
        base, explicit_index = self._split_candidate(normalized)
        if explicit_index is not None:
            self._base_count[base] = max(
                self._base_count.get(base, 0), explicit_index + 1
            )
        self._obj_to_name[obj] = normalized

    def rename(self, obj: Any, name: str) -> None:
        """Rename a registered object to a new unique name."""
        if obj not in self._obj_to_name:
            raise KeyError(f"object {obj!r} not registered in namespace")

        normalized = self._normalize_candidate(name)
        current_name = self._obj_to_name[obj]
        if normalized != current_name and normalized in self._used_names:
            raise ValueError(f"name {normalized!r} is already used in namespace")

        self._used_names.add(normalized)
        base, explicit_index = self._split_candidate(normalized)
        if explicit_index is not None:
            self._base_count[base] = max(
                self._base_count.get(base, 0), explicit_index + 1
            )
        self._obj_to_name[obj] = normalized

    def clear(self) -> None:
        """Clear all registered names."""
        self._obj_to_name.clear()
        self._used_names.clear()
        self._base_count.clear()

    @staticmethod
    def _normalize_candidate(candidate: str) -> str:
        candidate = _ILLEGAL_CHAR_REGEX.sub("_", candidate)
        if not candidate:
            candidate = "_unnamed"
        if candidate[0].isdigit():
            candidate = f"_{candidate}"
        return candidate

    @staticmethod
    def _split_candidate(candidate: str) -> tuple[str, int | None]:
        match = _NAME_REGEX.match(candidate)
        if match is None:
            return candidate, None
        base, num = match.group(1, 2)
        return base, None if num is None else int(num)
