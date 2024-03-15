from functools import partial
from itertools import repeat
from typing import Any, Iterable, Literal, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from paibox.exceptions import ShapeError
from paibox.types import SynOutType, WeightType

T = TypeVar("T")

_TupleAnyType = Union[T, Tuple[T, ...]]
_Tuple1Type = Union[T, Tuple[T]]
_Tuple2Type = Union[T, Tuple[T, T]]
_Tuple3Type = Union[T, Tuple[T, T, T]]
_Tuple4Type = Union[T, Tuple[T, T, T, T]]

_SizeAnyType = _TupleAnyType[int]
_Size1Type = _Tuple1Type[int]
_Size2Type = _Tuple2Type[int]
_Size3Type = _Tuple3Type[int]
_Size4Type = _Tuple4Type[int]

Size2Type = Tuple[int, int]


def _ntuple(x, n: int) -> Tuple[Any, ...]:
    if isinstance(x, Iterable):
        return tuple(x)

    return tuple(repeat(x, n))


_single = partial(_ntuple, n=1)
_pair = partial(_ntuple, n=2)
_triple = partial(_ntuple, n=3)
_quadruple = partial(_ntuple, n=4)


def _fm_ndim1_check(
    fm_shape: _TupleAnyType, fm_order: Literal["CHW", "HWC"]
) -> Size2Type:
    if len(fm_shape) < 1 or len(fm_shape) > 2:
        raise ShapeError()

    if len(fm_shape) == 1:
        channels, l = (1, *fm_shape)
    else:
        if fm_order is "CHW":
            channels, l = fm_shape
        else:
            l, channels = fm_shape

    return channels, l


def _fm_ndim2_check(
    fm_shape: _TupleAnyType, fm_order: Literal["CHW", "HWC"]
) -> Tuple[int, int, int]:
    if len(fm_shape) < 2 or len(fm_shape) > 3:
        raise ShapeError()

    if len(fm_shape) == 2:
        channels, h, w = (1, *fm_shape)
    else:
        if fm_order is "CHW":
            channels, h, w = fm_shape
        else:
            h, w, channels = fm_shape

    return channels, h, w


def _conv2d_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    # padding: Size2Type,
) -> WeightType:
    """Unroll the convolution kernel of 2d convolution into a matrix."""
    cout, cin, kh, kw = kernel.shape

    ih, iw = in_shape
    oh, ow = out_shape
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled = np.zeros((cin * in_size, cout * out_size), dtype=kernel.dtype)
    zeros_image = np.zeros((cin * ih, iw * cout, out_size), dtype=kernel.dtype)

    for i in range(oh):
        for j in range(ow):
            for ch_idx in np.ndindex(kernel.shape[:2]):
                # [0] -> o_ch, [1] -> i_ch
                zeros_image[
                    i * stride[0]
                    + ch_idx[1] * ih : i * stride[0]
                    + ch_idx[1] * ih
                    + kh,
                    j * stride[1]
                    + ch_idx[0] * iw : j * stride[1]
                    + ch_idx[0] * iw
                    + kw,
                    i * ow + j,
                ] = kernel[ch_idx[0], ch_idx[1], :, :]

            t = (
                zeros_image[:, :, i * ow + j]
                .reshape(cin * ih, cout, iw)
                .transpose(1, 0, 2)
            )
            for o_ch in range(cout):
                w_unrolled[:, i * ow + j + o_ch * out_size] = t[o_ch].flatten()

    return w_unrolled


def _conv2d_faster(
    x_chw: np.ndarray,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
) -> SynOutType:
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

    # kernel: (cout, cin, kh, kw) -> (cin*kh*kw, cout)
    col_kernel = kernel.transpose(1, 2, 3, 0).reshape(-1, cout)

    # padded: (cin, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (oh*ow, cin*kh*kw)
    col_fm = _im2col(x_padded, out_shape[0], out_shape[1], kh, kw, stride)

    # out = np.zeros((cout,) + out_shape, dtype=np.int64)
    # (oh*ow, cin*kh*kw) * (cin*kh*kw, cout) = (oh*ow, cout)
    out = col_fm @ col_kernel  # + self.bias

    # (oh*ow, cout) -> (oh, ow, cout) -> (cout, oh, ow)
    out = out.reshape(out_shape + (cout,)).transpose(2, 0, 1)

    return out.astype(np.int32)


def _im2col(
    x_padded: np.ndarray, oh: int, ow: int, kh: int, kw: int, stride: Size2Type
) -> NDArray[np.int64]:
    cols = np.zeros((oh * ow, x_padded.shape[0] * kh * kw), dtype=np.int64)

    _, ph, pw = x_padded.shape

    idx = 0
    for i in range(0, ph - kh + 1, stride[0]):
        for j in range(0, pw - kw + 1, stride[1]):
            cols[idx] = x_padded[:, i : i + kh, j : j + kw].reshape(-1)
            idx += 1

    return cols
