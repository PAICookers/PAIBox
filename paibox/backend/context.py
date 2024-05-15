from pathlib import Path
from typing import Any, Dict, List, Union

from paicorelib import ChipCoord, Coord, CoordLike, to_coord
from paicorelib.coordinate import to_coords

from paibox.context import _Context
from paibox.utils import merge_unique_ordered

__all__ = []

DEFAULT_OUTPUT_CHIP_ADDR = Coord(1, 0)
DEFAULT_LOCAL_CHIP_ADDR = [Coord(0, 0)]  # Support multi-chip.
DEFAULT_OUTPUT_CORE_ADDR_START = Coord(0, 0)
DEFAULT_CORE_PARAMS_CONF_JSON = "core_params.json"
DEFAULT_INPUT_CONF_JSON = "input_proj_info.json"
DEFAULT_OUTPUT_CONF_JSON = "output_dest_info.json"


class _BackendContext(_Context):
    _DefaultContext = {
        "output_chip_addr": DEFAULT_OUTPUT_CHIP_ADDR,  # RO mostly
        "target_chip_addr": DEFAULT_LOCAL_CHIP_ADDR,  # RO mostly
        "build_directory": Path.cwd(),  # R/W
        "output_core_addr_start": DEFAULT_OUTPUT_CORE_ADDR_START,  # RO
        "core_conf_json": DEFAULT_CORE_PARAMS_CONF_JSON,  # RO mostly
        "input_conf_json": DEFAULT_INPUT_CONF_JSON,  # RO mostly
        "output_conf_json": DEFAULT_OUTPUT_CONF_JSON,  # RO mostly
        "cflags": dict(),  # R/W
    }

    def __init__(self) -> None:
        super().__init__()
        self.update(self._DefaultContext)

    @property
    def target_chip_addr(self) -> List[ChipCoord]:
        return self["target_chip_addr"]

    @target_chip_addr.setter
    def target_chip_addr(self, addr: Union[CoordLike, List[CoordLike]]) -> None:
        if isinstance(addr, list):
            self["target_chip_addr"] = to_coords(addr)
        else:
            self["target_chip_addr"] = [to_coord(addr)]

    def add_chip_addr(self, *chip_addrs: CoordLike) -> None:
        # Maintain the order. We may take advantage of the priority
        # of the chip coordinates later.
        self["target_chip_addr"] = merge_unique_ordered(
            self.target_chip_addr, to_coords(chip_addrs)
        )

    @property
    def n_target_chips(self) -> int:
        return len(self.target_chip_addr)

    def _target_chip_addr_repr(self) -> str:
        return ", ".join(str(a) for a in self.target_chip_addr)

    @property
    def output_chip_addr(self) -> ChipCoord:
        return self["output_chip_addr"]

    @output_chip_addr.setter
    def output_chip_addr(self, addr: CoordLike) -> None:
        self["output_chip_addr"] = to_coord(addr)

    @property
    def test_chip_addr(self) -> Coord:
        return self.output_chip_addr

    @test_chip_addr.setter
    def test_chip_addr(self, addr: CoordLike) -> None:
        self.output_chip_addr = to_coord(addr)

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


def set_cflag(**kwargs) -> None:
    for k, v in kwargs.items():
        _BACKEND_CONTEXT.cflags[k] = v
