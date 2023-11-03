import pytest

import paibox as pb


def test_paiboxobject_eq():
    obj1 = pb.base.PAIBoxObject(name="obj1")
    obj2 = pb.base.PAIBoxObject(name="obj2")
    obj3 = pb.base.PAIBoxObject()

    # eq
    assert obj1 != obj2
    assert obj1 != obj3
    assert obj2 != obj3

    # Rename
    with pytest.raises(ValueError):
        obj1.name = "obj2"

    obj1.name = "obj1_1"


def test_paiboxobject_nodes():
    obj1 = pb.base.PAIBoxObject(name="obj111")

    nodes1 = obj1.nodes(method="absolute", level=1, include_self=True)
    assert nodes1["obj111"] == obj1

    # The node name of itself is ""
    nodes2 = obj1.nodes(method="relative", level=1, include_self=True)
    assert nodes2[""] == obj1

    nodes3 = obj1.nodes(method="absolute", level=1, include_self=False)
    assert nodes3 == {}

    nodes4 = obj1.nodes(method="absolute", level=-1, include_self=True)
    print(nodes4)
    assert nodes4["obj111"] == obj1
