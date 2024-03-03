from typing import Set, Tuple

from paibox.collector import Collector
from paibox.types import FrozenOrderedSet, OrderedSet


def test_FrozenOrderedSet_instance():
    frozen: FrozenOrderedSet[int] = FrozenOrderedSet([1, 2, 3, 4])
    assert isinstance(frozen, FrozenOrderedSet)

    t1 = (1, 2, 3, 4)
    t2 = (1, 2)
    frozen2 = FrozenOrderedSet([t1, t2])
    assert len(frozen2) == 2
    frozen2.clear()
    assert len(frozen2) == 0
    assert frozen2.data == {}

    s1 = (1, 2, 3, 4)
    s2 = (1, 2)
    frozen3: FrozenOrderedSet[Set[int]] = FrozenOrderedSet([s1, s2])
    assert len(frozen3) == 2


def test_OrderedSet_instance():
    ordered: OrderedSet[int] = OrderedSet([1, 2, 3, 4])
    assert isinstance(ordered, OrderedSet)

    t1 = (1, 2, 3, 4)
    t2 = (1, 2)

    ordered2: OrderedSet[Tuple[int, ...]] = OrderedSet()
    ordered2.update([t1, t2])
    ordered2.clear()
    assert ordered2.data == {}

    s1 = (1, 2, 3, 4)
    s2 = (1, 2)
    ordered3: OrderedSet[Set[int]] = OrderedSet()
    ordered3.add(s1)
    ordered3.add(s2)
    assert len(ordered3) == 2

    ordered3.difference_update([s1])
    assert len(ordered3) == 1


def test_Collector_operations():
    # Just check the typing hint
    d = {"1": 1, "2": "Tom", "3": False, "4": [1, 2, 3]}

    c1 = Collector(d).subset(str)  # Collector[Any, str]
    c2 = Collector(d).exclude(bool)  # Collector[Any, Any]
