import paibox as pb
from paibox.backend.sub_utils import SubEdge, SubNeuron, sub_node_overlap

from .backend_testcase import _gen_custom_index


class TestPartitionedSlice:
    def test_SubNode_overlap(self):
        n1 = pb.ANNNeuron((10, 32, 32))
        sub_neu1 = SubNeuron(
            n1, custom_index=_gen_custom_index(2 * 32 * 32, 4 * 32 * 32)
        )
        sub_neu2 = SubNeuron(
            n1, custom_index=_gen_custom_index(4 * 32 * 32, 6 * 32 * 32)
        )
        sub_neu3 = SubNeuron(
            n1, custom_index=_gen_custom_index(4 * 32 * 32, 5 * 32 * 32)
        )

        assert sub_node_overlap(sub_neu3, [sub_neu1, sub_neu2])
        assert sub_neu3.custom_index_set.issubset(sub_neu2.custom_index_set)
        assert not sub_neu3.custom_index_set.issubset(sub_neu1.custom_index_set)

    def test_SubNode_str_format(self, capsys):
        n1 = pb.ANNNeuron((10, 32, 32))
        nd_sl1 = SubNeuron(n1, custom_index=_gen_custom_index(2 * 32 * 32, 4 * 32 * 32))

        with capsys.disabled():
            print("\n")
            print(nd_sl1)

    def test_SubEdge_str_format(self, capsys):
        n1 = pb.ANNNeuron((10, 32, 32))
        n2 = pb.ANNNeuron((10, 32, 32))
        e1 = pb.FullConn(n1, n2)
        e_sl1 = SubEdge(e1, out_custom_index=_gen_custom_index(200, 300))

        with capsys.disabled():
            print("\n")
            print(e_sl1)
