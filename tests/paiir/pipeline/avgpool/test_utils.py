from torch import nn

from paibox.paiir.pipeline.avgpool.utils import (
    get_avgpool_divisor,
    get_pool_window_size,
)


class TestAvgPoolUtils:
    def test_avgpool_divisor_can_differ_from_window_size(self):
        pool = nn.AvgPool2d(2, divisor_override=1)
        assert get_pool_window_size(pool) == 4
        assert get_avgpool_divisor(pool) == 1
