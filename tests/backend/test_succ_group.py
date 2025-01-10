import pytest
import paibox as pb

from paibox.backend.succ_group import MergedSuccGroup, SuccGroup


class TestSuccGroup:
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

        sgrp1 = SuccGroup([s1, s2])
        sgrp2 = SuccGroup([s2, s1])
        sgrp3 = SuccGroup([s3, s4])

        assert sgrp1 == sgrp2
        assert sgrp1 != sgrp3

    def test_remove_node(self):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n3, n2)
        sgrp1 = SuccGroup([s1, s2])

        # Remove a non-existing node, return self
        new_sgrp = sgrp1.remove_node(n4)
        assert new_sgrp is sgrp1

        # Remove an existing node, return a new one
        new_sgrp = sgrp1.remove_node(n2)
        assert set(new_sgrp.nodes) == {n3}
        assert set(new_sgrp) == {s2}

        with pytest.raises(ValueError):
            # Not having the same input
            sgrp2 = SuccGroup([s1, s2, s3])

    def test_str_format(self, capsys):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        sgrp = SuccGroup([s1, s2])

        with capsys.disabled():
            print("\n")
            print(sgrp)


class TestMergedSuccGroup:
    def test_remove_node(self):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = SuccGroup([s1, s2])
        sgrp2 = SuccGroup([s3])

        msgrp1 = MergedSuccGroup([sgrp1, sgrp2])
        removed = msgrp1.remove_node(n3)

        assert set(removed) == {sgrp1, sgrp2}
        assert len(msgrp1) == 1
        assert set(msgrp1[0]) == {s1}
        assert msgrp1[0] is not sgrp1
        assert set(msgrp1.nodes) == {n2}

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
        sgrp1 = SuccGroup([s1, s2])
        sgrp2 = SuccGroup([s3])

        msgrp1 = MergedSuccGroup([sgrp1, sgrp2])
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
        sgrp1 = SuccGroup([s1, s2])
        sgrp2 = SuccGroup([s3])

        msgrp1 = MergedSuccGroup([sgrp1, sgrp2])
        assert set(msgrp1.nodes) == {n2, n3}

    def test_str_format(self, capsys):
        n1 = pb.ANNNeuron(1)
        n2 = pb.ANNNeuron(1)
        n3 = pb.ANNNeuron(1)
        n4 = pb.ANNNeuron(1)
        s1 = pb.FullConn(n1, n2)
        s2 = pb.FullConn(n1, n3)
        s3 = pb.FullConn(n4, n3)
        sgrp1 = SuccGroup([s1, s2])
        sgrp2 = SuccGroup([s3])

        msgrp = MergedSuccGroup([sgrp1, sgrp2])
        with capsys.disabled():
            print("\n")
            print(msgrp)
