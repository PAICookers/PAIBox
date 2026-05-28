"""Functional interface for custom pooling operations."""

from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t


def sumpool1d(
    input: Tensor,
    kernel_size: _size_1_t,
    stride: _size_1_t | None = None,
    padding: _size_1_t = 0,
    dilation: _size_1_t = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply 1D sum pooling using unfold.

    Computes the exact sum of elements in each pooling window,
    avoiding precision loss from AvgPool's division.

    Args:
        input: Input tensor of shape (N, C, L_in) or (C, L_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Default: kernel_size.
        padding: Implicit zero padding on both sides. Default: 0.
        dilation: Spacing between elements in the pooling window. Default: 1.
        ceil_mode: Use ceil instead of floor for output shape. Default: False.

    Returns:
        Output tensor of shape (N, C, L_out) or (C, L_out).
    """
    if stride is None:
        stride = kernel_size

    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    d = dilation if isinstance(dilation, int) else dilation[0]
    effective_k = d * (k - 1) + 1

    # Apply padding
    if p > 0:
        input = F.pad(input, (p, p))

    if input.ndim == 2:
        input = input.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    _, channels, L_in = input.shape

    # Calculate output size
    if ceil_mode:
        L_out = (L_in - effective_k + s - 1) // s + 1
        # Pad to ensure we get ceil output size
        L_needed = (L_out - 1) * s + effective_k
        if L_needed > L_in:
            input = F.pad(input, (0, L_needed - L_in))
            L_in = input.shape[2]
    else:
        L_out = (L_in - effective_k) // s + 1

    unfolded = F.unfold(input.unsqueeze(2), (1, k), (1, d), (0, 0), (1, s))
    unfolded = unfolded.view(input.shape[0], channels, k, -1)
    summed = unfolded.sum(dim=2).view(input.shape[0], channels, L_out)
    return summed.squeeze(0) if squeeze_batch else summed


def sumpool2d(
    input: Tensor,
    kernel_size: _size_2_t,
    stride: _size_2_t | None = None,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply 2D sum pooling using unfold.

    Computes the exact sum of elements in each pooling window,
    avoiding precision loss from AvgPool's division.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in) or (C, H_in, W_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Default: kernel_size.
        padding: Implicit zero padding on all sides. Default: 0.
        dilation: Spacing between elements in the pooling window. Default: 1.
        ceil_mode: Use ceil instead of floor for output shape. Default: False.

    Returns:
        Output tensor of shape (N, C, H_out, W_out) or (C, H_out, W_out).
    """
    if stride is None:
        stride = kernel_size

    # Normalize to tuples
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size

    if isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride

    if isinstance(padding, int):
        pH = pW = padding
    else:
        pH, pW = padding

    if isinstance(dilation, int):
        dH = dW = dilation
    else:
        dH, dW = dilation

    effective_kH = dH * (kH - 1) + 1
    effective_kW = dW * (kW - 1) + 1

    if input.ndim == 3:
        input = input.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Apply padding (left, right, top, bottom)
    if pH > 0 or pW > 0:
        input = F.pad(input, (pW, pW, pH, pH))

    N, C, H, W = input.shape

    # Calculate output dimensions
    if ceil_mode:
        H_out = (H - effective_kH + sH - 1) // sH + 1
        W_out = (W - effective_kW + sW - 1) // sW + 1
        # Pad to ensure we get ceil output size
        H_needed = (H_out - 1) * sH + effective_kH
        W_needed = (W_out - 1) * sW + effective_kW
        if H_needed > H:
            input = F.pad(input, (0, 0, 0, H_needed - H))
            H = input.shape[2]
        if W_needed > W:
            input = F.pad(input, (0, W_needed - W, 0, 0))
            W = input.shape[3]
    else:
        H_out = (H - effective_kH) // sH + 1
        W_out = (W - effective_kW) // sW + 1

    # Use fold/unfold for efficient computation
    # unfold: (N, C, H, W) -> (N, C*kH*kW, H_out*W_out)
    unfolded = F.unfold(input, (kH, kW), (dH, dW), (0, 0), (sH, sW))

    # Reshape to separate channel and window dimensions
    # (N, C*kH*kW, L) -> (N, C, kH*kW, L)
    unfolded = unfolded.view(N, C, kH * kW, -1)

    # Sum over the window dimension
    # (N, C, kH*kW, L) -> (N, C, L)
    summed = unfolded.sum(dim=2)

    # Reshape to output format: (N, C, H_out, W_out)
    output = summed.view(N, C, H_out, W_out)
    return output.squeeze(0) if squeeze_batch else output
