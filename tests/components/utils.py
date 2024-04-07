import numpy as np
from typing import Optional, Tuple

from paibox.types import SpikeType


def maxpool2d_golden(
    x: SpikeType,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    padding: Tuple[int, int],
    fm_order: str,
) -> SpikeType:
    if fm_order == "HWC":
        _x = x.transpose(2, 0, 1)
    else:
        _x = x

    xcin, ih, iw = _x.shape
    kh, kw = kernel_size
    _stride = stride if stride is not None else kernel_size
    oh = (ih - kh + 2 * padding[0]) // _stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // _stride[1] + 1
    cout = xcin

    out = np.zeros((cout, oh, ow), dtype=x.dtype)
    x_padded = np.pad(
        _x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = np.max(
                    x_padded[
                        c,
                        _stride[0] * i : _stride[0] * i + kh,
                        _stride[1] * j : _stride[1] * j + kw,
                    ]
                )

    return out


def avgpool2d_golden(
    x: SpikeType,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    fm_order: str,
) -> SpikeType:
    if fm_order == "HWC":
        _x = x.transpose(2, 0, 1)
    else:
        _x = x

    xcin, ih, iw = _x.shape
    kh, kw = kernel_size
    _stride = stride if stride is not None else kernel_size
    oh = (ih - kh + 2 * padding[0]) // _stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // _stride[1] + 1
    cout = xcin

    out = np.zeros((cout, oh, ow), dtype=np.float16)
    x_padded = np.pad(
        _x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = np.mean(
                    x_padded[
                        c,
                        _stride[0] * i : _stride[0] * i + kh,
                        _stride[1] * j : _stride[1] * j + kw,
                    ]
                )

    return (out > 0.5).astype(np.bool_)
