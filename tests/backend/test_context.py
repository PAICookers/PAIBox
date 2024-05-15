import copy
from pathlib import Path

from paicorelib import Coord, CoordOffset

from paibox import BACKEND_CONFIG
from paibox.backend.context import _BACKEND_CONTEXT


def test_backend_context(monkeypatch):
    assert _BACKEND_CONTEXT["output_chip_addr"] == Coord(1, 0)

    monkeypatch.setattr(BACKEND_CONFIG, "test_chip_addr", Coord(3, 4))
    assert BACKEND_CONFIG["output_chip_addr"] == Coord(3, 4)

    monkeypatch.setattr(BACKEND_CONFIG, "target_chip_addr", Coord(10, 10))
    assert BACKEND_CONFIG.target_chip_addr == [Coord(10, 10)]
    assert isinstance(BACKEND_CONFIG.target_chip_addr[0], Coord)

    # Multichip
    clist = [(1, 2), (2, 1), (2, 2)]
    monkeypatch.setattr(BACKEND_CONFIG, "target_chip_addr", clist)
    assert BACKEND_CONFIG.target_chip_addr == clist

    BACKEND_CONFIG.save("strkey", False, 12345, "ABC", a=1, b=2, c=3)
    assert BACKEND_CONFIG["b"] == 2
    assert BACKEND_CONFIG.load("strkey") == False

    cflags = BACKEND_CONFIG.cflags
    cflags["op1"] = True

    assert BACKEND_CONFIG.cflags["op1"] == True

    monkeypatch.setitem(BACKEND_CONFIG.cflags, "op2", 999)
    assert BACKEND_CONFIG.cflags["op2"] == 999

    assert _BACKEND_CONTEXT["output_core_addr_start"] == Coord(0, 0)

    ocoord = copy.copy(_BACKEND_CONTEXT["output_core_addr_start"])
    ocoord += CoordOffset(1, 0)

    opath = Path.cwd() / "output_dest_info.json"
    monkeypatch.setitem(_BACKEND_CONTEXT, "output_conf_json", opath)
    assert _BACKEND_CONTEXT["output_conf_json"] == opath


def test_backend_context_add_chip_addr(monkeypatch):
    monkeypatch.setattr(
        BACKEND_CONFIG, "target_chip_addr", [Coord(1, 0), Coord(10, 10)]
    )

    BACKEND_CONFIG.add_chip_addr((3, 4), (10, 10))
    assert BACKEND_CONFIG["target_chip_addr"][2] == Coord(3, 4)
