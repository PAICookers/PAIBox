"""Backendv2 public API.

The compiler entry point is intentionally imported on first access.  Artifact
readers also use backendv2 generated bindings, and importing the mapper there
would eagerly load the compile-only PyTorch/Numba dependency graph.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mapper import Mapper


def __getattr__(name: str) -> Any:
    """Resolve compile-only public symbols lazily."""
    if name == "Mapper":
        from .mapper import Mapper

        return Mapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Mapper"]
