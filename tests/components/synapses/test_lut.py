import numpy as np
import pytest

from paibox.components.synapses.lut import LUT, LUT_LEN


class TestLUT:
    def test_lut_getitem(self):
        tb = np.arange(LUT_LEN)
        lut = LUT(tb)

        pick_indices = np.asarray([4, 7, 28])
        assert np.array_equal(lut[pick_indices], lut.offset + pick_indices)

    def test_assign_ltp_ltd(self):
        tb = np.arange(LUT_LEN)
        lut = LUT(tb)

        ltp_lut = np.ones((LUT_LEN - lut.offset + 1,), dtype=np.int8)
        with pytest.raises(ValueError):
            lut.ltp = ltp_lut

        lut.offset -= 10
        lut.ltp = ltp_lut
