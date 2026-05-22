"""Pooling layers for chip deployment."""

from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _pair, _single

from . import functional as F

__all__ = ["SumPool1d", "SumPool2d"]


class _SumPoolNd(Module):
    """Base class for sum pooling modules."""

    __constants__ = ["kernel_size", "stride", "padding", "dilation", "ceil_mode"]
    kernel_size: tuple
    stride: tuple
    padding: tuple
    dilation: tuple
    ceil_mode: bool

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"ceil_mode={self.ceil_mode}"
        )


class SumPool1d(_SumPoolNd):
    r"""Applies 1D sum pooling over an input signal.

    Computes the exact sum of elements in each pooling window,
    avoiding precision loss from AvgPool's division.
    Used for split-core AvgPool+IF deployment.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the window. Default: kernel_size.
        padding: Implicit zero padding on both sides. Default: 0.
        ceil_mode: Use ceil instead of floor for output shape. Default: False.

    Shape:
        - Input: (N, C, L_in) or (C, L_in)
        - Output: (N, C, L_out) or (C, L_out), where

          L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
    """

    kernel_size: tuple[int]
    stride: tuple[int]
    padding: tuple[int]
    dilation: tuple[int]

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t | None = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        return F.sumpool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


class SumPool2d(_SumPoolNd):
    r"""Applies 2D sum pooling over an input signal.

    Computes the exact sum of elements in each pooling window,
    avoiding precision loss from AvgPool's division.
    Used for split-core AvgPool+IF deployment.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the window. Default: kernel_size.
        padding: Implicit zero padding on all sides. Default: 0.
        ceil_mode: Use ceil instead of floor for output shape. Default: False.

    Shape:
        - Input: (N, C, H_in, W_in) or (C, H_in, W_in)
        - Output: (N, C, H_out, W_out) or (C, H_out, W_out), where

          H_out = floor((H_in + 2*padding[0] - kernel_size[0]) / stride[0]) + 1
          W_out = floor((W_in + 2*padding[1] - kernel_size[1]) / stride[1]) + 1
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        return F.sumpool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
