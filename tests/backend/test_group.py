import pytest

import paibox as pb
from paibox.backend.group import DataGroup, MergedGroup


class TestDataGroup:
    def test_eq(self):
        """
        n1 -> s1 -> n2
           -> s2 -> n3
        n4 -> s3 -> n3
           -> s4 -> n5
        """
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        n5 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        s4 = pb.FullConn(n4, n5)

        sgrp1 = DataGroup([s1, s2])
        sgrp2 = DataGroup([s2, s1])
        sgrp3 = DataGroup([s3, s4])

        assert sgrp1 == sgrp2
        assert sgrp1 != sgrp3

    def test_reserve_node(self):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n3, n2)
        grp1 = DataGroup([s1, s2])

        # Remove a non-existing node, return self
        new_grp = grp1.reserve_node([n2, n3])
        assert new_grp == grp1

        # Remove an existing node, return a new one
        new_grp = grp1.reserve_node([n3])
        assert set(new_grp.nodes) == {n3}
        assert set(new_grp.edges) == {s2}

    def test_str_format(self, capsys):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        sgrp = DataGroup([s1, s2])

        with capsys.disabled():
            print("\n")
            print(sgrp)


class TestMergedGroup:
    def test_reserve_node(self):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = DataGroup([s1, s2])
        sgrp2 = DataGroup([s3])

        msgrp1 = MergedGroup([sgrp1, sgrp2])
        msgrp2 = msgrp1.reserve_node([n2])

        assert len(msgrp2) == 1
        assert set(msgrp2[0].edges) == {s1}
        assert msgrp2[0] is not sgrp1
        assert set(msgrp2.nodes) == {n2}

    def test_outputs(self):
        """
        n1 -> s1 -> n2
           -> s2 -> n3
        n4 -> s3 -> n3
        """
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = DataGroup([s1, s2])
        sgrp2 = DataGroup([s3])

        msgrp1 = MergedGroup([sgrp1, sgrp2])
        assert msgrp1.outputs.keys() == {n2, n3}
        assert set(msgrp1.outputs[n2]) == {s1}
        assert set(msgrp1.outputs[n3]) == {s2, s3}

    def test_nodes(self):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = DataGroup([s1, s2])
        sgrp2 = DataGroup([s3])

        msgrp1 = MergedGroup([sgrp1, sgrp2])
        assert set(msgrp1.nodes) == {n2, n3}

    def test_str_format(self, capsys):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = DataGroup([s1, s2])
        sgrp2 = DataGroup([s3])

        msgrp = MergedGroup([sgrp1, sgrp2])
        with capsys.disabled():
            print("\n")
            print(msgrp)
