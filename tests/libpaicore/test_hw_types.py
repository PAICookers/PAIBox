import pytest

from paibox.libpaicore import NeuronSegment


def test_NeuronSegment():
    n = 100
    ns = NeuronSegment(slice(100, 100 + n, 1), 233, 2)

    assert len(ns.addr_ram) == 2 * n
    print()
