from typing import Any

from paibox.context import _Context
from paibox.libpaicore import Coord, CoordLike, to_coord

__all__ = ["BACKEND_CONFIG"]


class _BackendContext(_Context):
    def __init__(self) -> None:
        super().__init__()

    def load(self, key, value: Any = None):
        if key == "test_chip_addr":
            return self.test_chip_addr

        return super().load(key, value)

    @property
    def test_chip_addr(self) -> Coord:
        if "test_chip_addr" in self._context:
            return self._context["test_chip_addr"]
        else:
            return Coord(0, 0)

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self._context["test_chip_addr"] = to_coord(addr)

    # TODO Add a configuration that allows users to specify the frame output directory


_BACKEND_CONTEXT = _BackendContext()
BACKEND_CONFIG = _BACKEND_CONTEXT
