import paibox as pb
from paibox.backend._slice import (
    EdgeSlice,
    NodeSlice,
    PrttnSliceType,
    sl_cover,
    sl_overlap,
)
from paibox.backend.types import NeuSegment


def test_sl_overlap():
    sl1 = PrttnSliceType(100, 200)
    sl2 = PrttnSliceType(120, 300)
    assert sl_overlap(sl1, sl2)


def test_sl_cover():
    sl1 = PrttnSliceType(100, 200)
    sl2 = PrttnSliceType(120, 300)
    sl3 = PrttnSliceType(0, 300)
    assert not sl_cover(sl1, sl2)
    assert sl_cover(sl1, sl3)


class TestPartitionedSlice:
    def test_NodeSlice_covered_by_and_overlap(self):
        n1 = pb.ANNNeuron((10, 32, 32))
        n2 = pb.ANNNeuron((100,))
        nd_sl1 = NodeSlice(n1, slice(2 * 32 * 32, 4 * 32 * 32))
        nd_sl2 = NodeSlice(n1, slice(4 * 32 * 32, 6 * 32 * 32))
        nd_sl3 = NodeSlice(n1, slice(6 * 32 * 32, 8 * 32 * 32))
        nd_sl4 = NodeSlice(n1, slice(4 * 32 * 32, 5 * 32 * 32))

        assert nd_sl4.overlap([nd_sl1, nd_sl2])
        assert nd_sl4.covered_by(nd_sl2)
        assert nd_sl4.covered_by([nd_sl1, nd_sl2])
        assert not nd_sl4.covered_by(nd_sl3)

        # Test with `NeuSegment`
        neu_seg1 = NeuSegment(n1, slice(4 * 32 * 32, 6 * 32 * 32), 0)
        neu_seg2 = NeuSegment(n2, slice(0, 100), 100)

        assert nd_sl4.covered_by(neu_seg1)
        assert not nd_sl4.covered_by(neu_seg2)
        assert not nd_sl4.overlap(neu_seg2)

    def test_NodeSlice_str_format(self, capsys):
        n1 = pb.ANNNeuron((10, 32, 32))
        nd_sl1 = NodeSlice(n1, slice(2 * 32 * 32, 4 * 32 * 32))

        with capsys.disabled():
            print("\n")
            print(nd_sl1)

    def test_EdgeSlice_str_format(self, capsys):
        n1 = pb.ANNNeuron((10, 32, 32))
        n2 = pb.ANNNeuron((10, 32, 32))
        e1 = pb.FullConn(n1, n2)
        e_sl1 = EdgeSlice(e1, out_index=slice(200, 300))

        with capsys.disabled():
            print("\n")
            print(e_sl1)
