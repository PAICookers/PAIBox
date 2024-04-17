from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Size2Type = Tuple[int, int]


def _conv2d_faster_fp32(
    x_chw: np.ndarray,
    out_shape: Size2Type,
    kernel: NDArray[np.float32],
    stride: Size2Type,
    padding: Size2Type,
) -> np.float32:
    xc, xh, xw = x_chw.shape
    # (O, I, H, W)
    cout, cin, kh, kw = kernel.shape
    assert xc == cin

    x_padded = np.pad(
        x_chw,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    assert (xh + padding[0] * 2 - kh) // stride[0] + 1 == out_shape[0]
    assert (xw + padding[1] * 2 - kw) // stride[1] + 1 == out_shape[1]

    # kernel: (cout, cin, kh, kw) -> (cout, cin*kh*kw)
    col_kernel = kernel.reshape(cout, -1)

    # padded: (cin, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (oh*ow, cin*kh*kw)
    col_fm = _2d_im2col_fp32(x_padded, out_shape[0], out_shape[1], kh, kw, stride)
    # out = np.zeros((cout,) + out_shape, dtype=np.int64)
    # (oh*ow, cin*kh*kw) * (cout, cin*kh*kw)^T = (oh*ow, cout)
    out = col_fm @ col_kernel.T  # + self.bias
    # (oh*ow, cout) -> (cout, oh*ow) -> (cout, oh, ow)
    out = out.astype(np.float32).T.reshape((cout,) + out_shape)

    return out


def _2d_im2col_fp32(
    x_padded: np.ndarray, oh: int, ow: int, kh: int, kw: int, stride: Size2Type
) -> NDArray[np.float32]:
    cols = np.zeros((oh * ow, x_padded.shape[0] * kh * kw), dtype=np.float32)

    _, ph, pw = x_padded.shape

    idx = 0
    for i in range(0, ph - kh + 1, stride[0]):
        for j in range(0, pw - kw + 1, stride[1]):
            cols[idx] = x_padded[:, i : i + kh, j : j + kw].ravel()
            idx += 1

    return cols
