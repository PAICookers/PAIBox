from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

Size2Type = Tuple[int, int]


"""Faster Conv2d in FP32 format."""


def _conv2d_faster_fp32(
    x_chw: NDArray[Any], kernel: NDArray[Any], stride: Size2Type, padding: Size2Type
) -> NDArray[np.float32]:
    xc, xh, xw = x_chw.shape
    # (O, I, H, W)
    cout, cin, kh, kw = kernel.shape
    assert xc == cin

    x_padded = np.pad(
        x_chw,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    oh = (xh + padding[0] * 2 - kh) // stride[0] + 1
    ow = (xw + padding[1] * 2 - kw) // stride[1] + 1

    # kernel: (cout, cin, kh, kw) -> (cout, cin*kh*kw)
    col_kernel = kernel.reshape(cout, -1)

    # padded: (cin, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (oh*ow, cin*kh*kw)
    col_fm = _2d_im2col_fp32(x_padded, oh, ow, kh, kw, stride)
    # (oh*ow, cin*kh*kw) * (cout, cin*kh*kw)^T = (oh*ow, cout)
    out = col_fm @ col_kernel.T  # + self.bias
    # (oh*ow, cout) -> (cout, oh*ow) -> (cout, oh, ow)
    out = out.astype(np.float32).T.reshape((cout, oh, ow))

    return out


def _2d_im2col_fp32(
    x_padded: NDArray[Any], oh: int, ow: int, kh: int, kw: int, stride: Size2Type
) -> NDArray[np.float32]:
    cols = np.zeros((oh * ow, x_padded.shape[0] * kh * kw), dtype=np.float32)

    _, ph, pw = x_padded.shape

    idx = 0
    for i in range(0, ph - kh + 1, stride[0]):
        for j in range(0, pw - kw + 1, stride[1]):
            cols[idx] = x_padded[:, i : i + kh, j : j + kw].ravel()
            idx += 1

    return cols
