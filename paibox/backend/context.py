from pathlib import Path
from typing import Any, Dict, Union

from paicorelib import Coord, CoordLike, to_coord

from paibox.context import _Context

__all__ = ["BACKEND_CONFIG"]

DEFAULT_OUTPUT_CHIP_ADDR = Coord(1, 0)
DEFAULT_LOCAL_CHIP_ADDR = Coord(0, 0)
DEFAULT_OUTPUT_CORE_ADDR_START = Coord(0, 0)


class _BackendContext(_Context):
    def __init__(self) -> None:
        super().__init__()
        self["output_chip_addr"] = DEFAULT_OUTPUT_CHIP_ADDR  # RO mostly
        self["local_chip_addr"] = DEFAULT_LOCAL_CHIP_ADDR  # RO mostly
        self["build_directory"] = Path.cwd()  # R/W
        self["output_core_addr_start"] = DEFAULT_OUTPUT_CORE_ADDR_START  # RO
        self["cflags"] = dict()  # R/W

    @property
    def local_chip_addr(self) -> Coord:
        return self["local_chip_addr"]

    @local_chip_addr.setter
    def local_chip_addr(self, addr: CoordLike) -> None:
        self["local_chip_addr"] = to_coord(addr)

    @property
    def test_chip_addr(self) -> Coord:
        return self["output_chip_addr"]

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self["output_chip_addr"] = to_coord(addr)

    @property
    def output_dir(self) -> Path:
        return self["build_directory"]

    @output_dir.setter
    def output_dir(self, p: Union[str, Path]) -> None:
        self["build_directory"] = Path(p)

    @property
    def cflags(self) -> Dict[str, Any]:
        """Compilation options."""
        return self["cflags"]


_BACKEND_CONTEXT = _BackendContext()
BACKEND_CONFIG = _BACKEND_CONTEXT


def set_cflag(**kwargs) -> None:
    for k, v in kwargs.items():
        _BACKEND_CONTEXT.cflags[k] = v
