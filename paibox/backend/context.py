from typing import TypedDict
from paibox.context import _Context
from paibox.libpaicore import Coord, CoordLike, to_coord

__all__ = ["BACKEND_CONFIG"]


class _BackendContextDict(TypedDict, total=False):
    output_chip_addr: Coord
    output_core_addr: Coord
    test_chip_addr: Coord
    local_chip_addr: Coord


class _BackendContext(_Context):
    def __init__(self) -> None:
        super().__init__()
        self._context["output_chip_addr"] = Coord(1, 0)
        self._context["output_core_addr"] = Coord(0, 0)
        self._context["test_chip_addr"] = self["output_chip_addr"]
        self._context["local_chip_addr"] = Coord(0, 0)

    @property
    def local_chip_addr(self) -> Coord:
        return self["local_chip_addr"]

    @local_chip_addr.setter
    def local_chip_addr(self, addr: CoordLike) -> None:
        self["local_chip_addr"] = to_coord(addr)

    @property
    def test_chip_addr(self) -> Coord:
        return self["test_chip_addr"]

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self["test_chip_addr"] = to_coord(addr)

    # TODO Add a configuration that allows users to specify the frame output directory


_BACKEND_CONTEXT = _BackendContext()
BACKEND_CONFIG = _BACKEND_CONTEXT
