import pytest

import paibox as pb
from paibox.backend.types import NeuSegment


class TestNeuSegment:
    def test_NeuSegment_getitem(self):
        n1 = pb.ANNNeuron(200)
        neu_seg1 = NeuSegment(n1, slice(0, 120), 0)
        neu_seg2 = NeuSegment(n1, slice(120, 160), 120)
        neu_seg3 = NeuSegment(n1, slice(160, 200), 160)

        # out of range
        with pytest.raises(IndexError):
            result = neu_seg1[50:150]

        with pytest.raises(IndexError):
            result = neu_seg1[130:]

        result = neu_seg2[10:20]
        assert result.index.start == 120 + 10
        assert result.index.stop == 120 + 20
        assert result.offset_nram == 120 + 10

        result = neu_seg2[:30]
        assert result.index.start == 120
        assert result.index.stop == 120 + 30
        assert result.offset_nram == 120

        result = neu_seg3[20:]
        assert result.index.start == 160 + 20
        assert result.index.stop == 200
        assert result.offset_nram == 160 + 20

        # cannot pass an integer
        with pytest.raises(Exception):
            result = neu_seg3[0]  # type: ignore
