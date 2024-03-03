import pytest

import paibox as pb
from paibox.backend.constrs import GraphNodeConstrs


class TestGraphNodeConstrs:
    @pytest.mark.skip("Not implemented")
    def test_add_node_constr(self):
        constr = GraphNodeConstrs()
        constr.add_node_constr(bounded=["1", "2", "4"])

        constr.add_node_constr(conflicted={"4": {"1"}})

        print()

    def test_tick_wait_attr_constr(self):
        n1 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=0, name="n1")
        n2 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=2, name="n2")
        n3 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n3")
        n4 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n4")
        n5 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n5")
        n6 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n6")

        constr = GraphNodeConstrs.tick_wait_attr_constr([n1, n2, n3, n4, n5, n6])
        assert len(constr) == 4

        constr = GraphNodeConstrs.tick_wait_attr_constr([n4, n5])
        assert len(constr) == 0
