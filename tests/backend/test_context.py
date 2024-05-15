import copy
from pathlib import Path

from paicorelib import Coord, CoordOffset

import paibox as pb
from paibox.backend.context import _BACKEND_CONTEXT


def test_backend_context(monkeypatch):
    monkeypatch.setattr(pb.BACKEND_CONFIG, "test_chip_addr", Coord(3, 4))
    assert pb.BACKEND_CONFIG["output_chip_addr"] == Coord(3, 4)

    monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", Coord(10, 10))
    assert pb.BACKEND_CONFIG.target_chip_addr == [Coord(10, 10)]
    assert isinstance(pb.BACKEND_CONFIG.target_chip_addr[0], Coord)

    # Multichip
    clist = [(1, 2), (2, 1), (2, 2)]
    monkeypatch.setattr(pb.BACKEND_CONFIG, "target_chip_addr", clist)
    assert pb.BACKEND_CONFIG.target_chip_addr == clist

    pb.BACKEND_CONFIG.save("strkey", False, 12345, "ABC", a=1, b=2, c=3)
    assert pb.BACKEND_CONFIG["b"] == 2
    assert pb.BACKEND_CONFIG.load("strkey") == False

    cflags = pb.BACKEND_CONFIG.cflags
    cflags["op1"] = True

    assert pb.BACKEND_CONFIG.cflags["op1"] == True

    monkeypatch.setitem(pb.BACKEND_CONFIG.cflags, "op2", 999)
    assert pb.BACKEND_CONFIG.cflags["op2"] == 999

    assert _BACKEND_CONTEXT["output_core_addr_start"] == Coord(0, 0)

    ocoord = copy.copy(_BACKEND_CONTEXT["output_core_addr_start"])
    ocoord += CoordOffset(1, 0)

    opath = Path.cwd() / "output_dest_info.json"
    monkeypatch.setitem(_BACKEND_CONTEXT, "output_conf_json", opath)
    assert _BACKEND_CONTEXT["output_conf_json"] == opath


def test_backend_context_add_chip_addr(monkeypatch):
    monkeypatch.setattr(
        pb.BACKEND_CONFIG, "target_chip_addr", [Coord(1, 0), Coord(10, 10)]
    )

    pb.BACKEND_CONFIG.add_chip_addr((3, 4), (10, 10))
    assert pb.BACKEND_CONFIG["target_chip_addr"][2] == Coord(3, 4)
