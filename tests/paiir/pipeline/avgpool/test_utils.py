import torch
from torch import nn

from paibox.paiir.nn import SumPool1d, SumPool2d
from paibox.paiir.pipeline.avgpool.utils import (
    build_sum_pool,
    get_avgpool_divisor,
    get_pool_window_size,
)


class TestAvgPoolUtils:
    def test_avgpool_divisor_can_differ_from_window_size(self):
        pool = nn.AvgPool2d(2, divisor_override=1)
        assert get_pool_window_size(pool) == 4
        assert get_avgpool_divisor(pool) == 1

    def test_build_sum_pool_preserves_window_geometry(self):
        avg2d = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        sum2d = build_sum_pool(avg2d)
        assert isinstance(sum2d, SumPool2d)
        assert sum2d.kernel_size == (3, 3)
        assert sum2d.stride == (2, 2)
        assert sum2d.padding == (1, 1)
        assert sum2d.dilation == (1, 1)
        assert sum2d.ceil_mode is True

        avg1d = nn.AvgPool1d(kernel_size=5, stride=3, padding=2, ceil_mode=False)
        sum1d = build_sum_pool(avg1d)
        assert isinstance(sum1d, SumPool1d)
        assert sum1d.kernel_size == (5,)
        assert sum1d.stride == (3,)
        assert sum1d.padding == (2,)
        assert sum1d.dilation == (1,)
        assert sum1d.ceil_mode is False

    def test_sumpool2d_supports_dilation(self):
        pool = SumPool2d(kernel_size=2, stride=1, dilation=2)
        x = torch.arange(1, 26, dtype=torch.float32).reshape(1, 1, 5, 5)
        actual = pool(x)
        expected = torch.tensor(
            [[[[28.0, 32.0, 36.0], [48.0, 52.0, 56.0], [68.0, 72.0, 76.0]]]]
        )
        assert torch.equal(actual, expected)

    def test_sumpool1d_supports_dilation(self):
        pool = SumPool1d(kernel_size=3, stride=1, dilation=2)
        x = torch.arange(1, 8, dtype=torch.float32).reshape(1, 1, 7)
        actual = pool(x)
        expected = torch.tensor([[[9.0, 12.0, 15.0]]])
        assert torch.equal(actual, expected)
