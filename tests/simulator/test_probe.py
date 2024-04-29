import pytest

import paibox as pb
from paibox.exceptions import PAIBoxWarning


def test_probe_instanece():
    pbobj = pb.base.PAIBoxObject()
    pb1 = pb.Probe(pbobj, "name", name="pb1")
    pb2 = pb.Probe(pbobj, "name", name="pb2")

    # eq
    assert pb1 != pb2

    # Rename
    with pytest.warns(PAIBoxWarning):
        pb1.name = "pb2"  # Avoid name conflict for `Probe`

    pb3 = pb.Probe(pbobj, "name", name="pb2")
    pb4 = pb.Probe(pbobj, "name", name="pb2")  # Avoid twice.
    pb5 = pb.Probe(pbobj, "name", name="pb2")  # Avoid 3 times.
