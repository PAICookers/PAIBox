from paibox.backend._slice import NodeSlice, PrttnSliceType, sl_cover, sl_overlap
import paibox as pb


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
    def test_NodeSlice(self):
        n1 = pb.ANNNeuron((10, 32, 32))
        nd_sl1 = NodeSlice(n1, slice(2 * 32 * 32, 4 * 32 * 32))
        nd_sl2 = NodeSlice(n1, slice(4 * 32 * 32, 6 * 32 * 32))
        nd_sl3 = NodeSlice(n1, slice(6 * 32 * 32, 8 * 32 * 32))
        nd_sl4 = NodeSlice(n1, slice(4 * 32 * 32, 5 * 32 * 32))

        assert nd_sl4.overlap([nd_sl1, nd_sl2])
        assert nd_sl4.covered_by(nd_sl2)
        assert nd_sl4.covered_by([nd_sl1, nd_sl2])
        assert not nd_sl4.covered_by(nd_sl3)
