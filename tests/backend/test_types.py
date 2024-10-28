import paibox as pb
from paibox.backend.types import MergedSuccGroup, SuccGroup


class TestMergedSuccGroup:

    def test_MergedSuccGroup_inputs(self):
        """
        n1 -> s1 -> n2
           -> s2 -> n3
        n4 -> s3 ->
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

        sgrp1 = SuccGroup(n1, [n2, n3], [s1, s2])
        sgrp2 = SuccGroup(n4, [n3, n5], [s3, s4])

        msgrp = MergedSuccGroup(sgrp1, sgrp2)

        # don't care the order
        assert set(msgrp.outputs.keys()) == set([n2, n3, n5])
