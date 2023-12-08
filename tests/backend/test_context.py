import pytest

from paibox import BACKEND_CONFIG
from paibox.backend.context import _BACKEND_CONTEXT
from paibox.libpaicore import Coord


def test_backend_context():
    assert BACKEND_CONFIG.test_chip_addr == Coord(1, 0)
    assert BACKEND_CONFIG.local_chip_addr == Coord(0, 0)

    _BACKEND_CONTEXT.test_chip_addr = Coord(3, 4)
    assert BACKEND_CONFIG["test_chip_addr"] == Coord(3, 4)

    BACKEND_CONFIG.local_chip_addr = Coord(10, 10)
    assert BACKEND_CONFIG.local_chip_addr == Coord(10, 10)

    BACKEND_CONFIG["local_chip_addr"] = (10, 10)
    assert not isinstance(BACKEND_CONFIG.local_chip_addr, Coord)
