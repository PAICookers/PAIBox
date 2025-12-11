import pytest

import paibox as pb
from paibox.backend.types import Custom_Index, DendriteSegment


class TestNeuSegment:
    def test_NeuSegment_getitem(self):
        n1 = pb.ANNNeuron(200)

        def get_custom_index(index_slice: slice) -> list[Custom_Index]:
            return [
                Custom_Index(i, 0) for i in range(index_slice.start, index_slice.stop)
            ]

        neu_seg1 = DendriteSegment(n1, get_custom_index(slice(0, 120)), 0)
        neu_seg2 = DendriteSegment(n1, get_custom_index(slice(120, 160)), 120)
        neu_seg3 = DendriteSegment(n1, get_custom_index(slice(160, 200)), 160)

        # out of range
        with pytest.raises(IndexError):
            result = neu_seg1[50:150]

        with pytest.raises(IndexError):
            result = neu_seg1[130:]

        result = neu_seg2[10:20]
        assert result.index[0].index == 120 + 10
        assert result.index[-1].index == 120 + 20 - 1
        assert result.offset == 120 + 10

        result = neu_seg2[:30]
        assert result.index[0].index == 120
        assert result.index[-1].index == 120 + 30 - 1
        assert result.offset == 120

        result = neu_seg3[20:]
        assert result.index[0].index == 160 + 20
        assert result.index[-1].index == 200 - 1
        assert result.offset == 160 + 20

        # cannot pass an integer
        with pytest.raises(Exception):
            result = neu_seg3[0]  # type: ignore
