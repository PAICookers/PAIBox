from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mapper import Mapper

__all__ = ["Mapper"]


def __getattr__(name: str):
    if name == "Mapper":
        from .mapper import Mapper

        return Mapper

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
