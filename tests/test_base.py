import pytest

import paibox as pb
from paibox.base import PAIBoxObject, DataFlowFormat
from paibox.exceptions import RegisterError


def test_paibox_version():
    assert isinstance(pb.__version__, str) and pb.__version__ >= "0.0.1"


def test_paiboxobject_eq():
    obj1 = PAIBoxObject(name="obj1")
    obj2 = PAIBoxObject(name="obj2")
    obj3 = PAIBoxObject()

    # eq
    assert obj1 != obj2
    assert obj1 != obj3
    assert obj2 != obj3

    # Rename
    with pytest.raises(RegisterError):
        obj1.name = "obj2"

    obj1.name = "obj1_1"


def test_paiboxobject_nodes():
    obj1 = PAIBoxObject(name="obj111")

    nodes1 = obj1.nodes(method="absolute", level=1, include_self=True)
    assert nodes1["obj111"] == obj1

    # The node name of itself is ""
    nodes2 = obj1.nodes(method="relative", level=1, include_self=True)
    assert nodes2[""] == obj1

    nodes3 = obj1.nodes(method="absolute", level=1, include_self=False)
    assert nodes3 == {}

    nodes4 = obj1.nodes(method="absolute", level=-1, include_self=True)
    assert nodes4["obj111"] == obj1


class TestDataFlowFormat:
    def test_dff_infinite_dataflow(self):
        with pytest.raises((AssertionError, ValueError)):
            dff = DataFlowFormat(1, 0, -1)
            _ = dff.t_last_vld

    def test_dff_valid(self):
        # 1. t1 >= tws, t_last > endtick
        dff1 = DataFlowFormat(10, 3, 10, is_local_time=False)
        with pytest.raises(ValueError):
            dff1._check_after_assign(8, 36)

        # 2. t1 >= tws, t_last <= endtick
        dff2 = DataFlowFormat(10, 3, 10, is_local_time=True)
        dff2._check_after_assign(2, 39)
