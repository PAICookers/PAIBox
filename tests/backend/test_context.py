import copy

import pytest
from paicorelib import Coord, CoordOffset

from paibox import BACKEND_CONFIG
from paibox.backend.context import _BACKEND_CONTEXT


def test_backend_context():
    _BACKEND_CONTEXT.test_chip_addr = Coord(3, 4)
    assert BACKEND_CONFIG["output_chip_addr"] == Coord(3, 4)

    BACKEND_CONFIG.local_chip_addr = Coord(10, 10)
    assert BACKEND_CONFIG.local_chip_addr == Coord(10, 10)

    # DO NOT set item in this way!
    # BACKEND_CONFIG["local_chip_addr"] = (10, 10)
    # assert not isinstance(BACKEND_CONFIG.local_chip_addr, Coord)
    BACKEND_CONFIG.local_chip_addr = (10, 10)
    assert isinstance(BACKEND_CONFIG.local_chip_addr, Coord)

    BACKEND_CONFIG.save("strkey", False, 12345, "ABC", a=1, b=2, c=3)
    assert BACKEND_CONFIG["b"] == 2
    assert BACKEND_CONFIG.load("strkey") == False

    cflags = BACKEND_CONFIG.cflags
    cflags["op1"] = True

    assert BACKEND_CONFIG.cflags["op1"] == True

    BACKEND_CONFIG.cflags["op2"] = 999
    assert BACKEND_CONFIG.cflags["op2"] == 999

    ocoord = copy.copy(_BACKEND_CONTEXT["output_core_addr_start"])
    ocoord += CoordOffset(1, 0)

    assert _BACKEND_CONTEXT["output_core_addr_start"] == Coord(0, 0)
