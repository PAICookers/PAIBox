import pytest

import paibox as pb
from paibox.backend.constrs import GraphNodeConstrs


class TestGraphNodeConstrs:
    def test_tick_wait_attr_constr(self):
        n1 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=0, name="n1")
        n2 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=2, name="n2")
        n3 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n3")
        n4 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n4")
        n5 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n5")
        n6 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n6")

        constr = GraphNodeConstrs.apply_constrs([n1, n2, n3, n4, n5, n6])
        assert len(constr) == 4

        constr = GraphNodeConstrs.apply_constrs([n4, n5])
        assert len(constr) == 1

    def test_apply_constraints(self):
        n1 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=0, name="n1")
        n2 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=2, name="n2")
        n3 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n3")
        n4 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n4")
        n5 = pb.LIF(10, 3, tick_wait_start=2, tick_wait_end=0, name="n5")
        n6 = pb.LIF(10, 3, tick_wait_start=1, tick_wait_end=3, name="n6")
        n7 = pb.ANNNeuron(10, 0, tick_wait_start=1, tick_wait_end=0, name="n7")
        n8 = pb.ANNNeuron(
            10, 0, tick_wait_start=1, tick_wait_end=0, pool_max=True, name="n8"
        )
        n9 = pb.ANNNeuron(
            10, 0, tick_wait_start=1, tick_wait_end=2, pool_max=True, name="n9"
        )
        n10 = pb.ANNNeuron(
            20, 1, tick_wait_start=1, tick_wait_end=2, pool_max=True, name="n10"
        )

        constr = GraphNodeConstrs.apply_constrs(
            [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
        )

        assert len(constr) == 6
