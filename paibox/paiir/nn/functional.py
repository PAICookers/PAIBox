"""Functional interface for custom pooling operations."""

from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t


def sumpool1d(
    input: Tensor,
    kernel_size: _size_1_t,
    stride: _size_1_t | None = None,
    padding: _size_1_t = 0,
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
        ceil_mode: Use ceil instead of floor for output shape. Default: False.

    Returns:
        Output tensor of shape (N, C, L_out) or (C, L_out).
    """
    if stride is None:
        stride = kernel_size

    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]

    # Apply padding
    if p > 0:
        input = F.pad(input, (p, p))

    L_in = input.shape[2]

    # Calculate output size
    if ceil_mode:
        L_out = (L_in - k + s - 1) // s + 1
        # Pad to ensure we get ceil output size
        L_needed = (L_out - 1) * s + k
        if L_needed > L_in:
            input = F.pad(input, (0, L_needed - L_in))
            L_in = input.shape[2]
    else:
        L_out = (L_in - k) // s + 1

    # Use unfold to extract windows, then sum
    # unfold(dim, size, step) extracts sliding windows
    # (N, C, L_in) -> (N, C, L_out, k)
    unfolded = input.unfold(dimension=2, size=k, step=s)

    return unfolded.sum(dim=-1)


def sumpool2d(
    input: Tensor,
    kernel_size: _size_2_t,
    stride: _size_2_t | None = None,
    padding: _size_2_t = 0,
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

    # Apply padding (left, right, top, bottom)
    if pH > 0 or pW > 0:
        input = F.pad(input, (pW, pW, pH, pH))

    N, C, H, W = input.shape

    # Calculate output dimensions
    if ceil_mode:
        H_out = (H - kH + sH - 1) // sH + 1
        W_out = (W - kW + sW - 1) // sW + 1
        # Pad to ensure we get ceil output size
        H_needed = (H_out - 1) * sH + kH
        W_needed = (W_out - 1) * sW + kW
        if H_needed > H:
            input = F.pad(input, (0, 0, 0, H_needed - H))
            H = input.shape[2]
        if W_needed > W:
            input = F.pad(input, (0, W_needed - W, 0, 0))
            W = input.shape[3]
    else:
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1

    # Use fold/unfold for efficient computation
    # unfold: (N, C, H, W) -> (N, C*kH*kW, H_out*W_out)
    unfolded = F.unfold(input, kernel_size=(kH, kW), stride=(sH, sW))

    # Reshape to separate channel and window dimensions
    # (N, C*kH*kW, L) -> (N, C, kH*kW, L)
    unfolded = unfolded.view(N, C, kH * kW, -1)

    # Sum over the window dimension
    # (N, C, kH*kW, L) -> (N, C, L)
    summed = unfolded.sum(dim=2)

    # Reshape to output format: (N, C, H_out, W_out)
    return summed.view(N, C, H_out, W_out)
