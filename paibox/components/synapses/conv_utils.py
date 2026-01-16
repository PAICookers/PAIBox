import sys
from collections.abc import Iterable
from functools import partial
from itertools import repeat

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from paibox.exceptions import PAIBoxDeprecationWarning, ShapeError
from paibox.types import (
    NEUOUT_U8_DTYPE,
    VOLTAGE_DTYPE,
    WEIGHT_DTYPE,
    NeuOutType,
    SynOutType,
    WeightType,
)

from .conv_types import (
    Size1Type,
    Size2Type,
    Size3Type,
    Size4Type,
    SizeAnyType,
    _Order2d,
    _Order3d,
    _Size1Type,
    _Size2Type,
)

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


def _ntuple(x, n: int):
    if isinstance(x, Iterable):
        return tuple(x)

    return tuple(repeat(x, n))


_single = partial(_ntuple, n=1)
_pair = partial(_ntuple, n=2)
_triple = partial(_ntuple, n=3)
_quadruple = partial(_ntuple, n=4)


INDEX_DTYPE = np.uint32
MAX_INDEX = np.iinfo(INDEX_DTYPE).max


def assert_max_index(*args: int | np.integer) -> None:
    for n in args:
        assert n <= MAX_INDEX, f"Number {n} exceeds max index {MAX_INDEX}"


def group_ch_check(ci: int, co: int, groups: int, ci_in_grp: int) -> None:
    assert (
        ci % groups == 0 and co % groups == 0
    ), f"Input & output channels {ci} & {co} must be divisible by groups {groups}"
    assert ci == ci_in_grp * groups


def fm_ndim1_check(fm_shape: SizeAnyType, fm_order: _Order2d) -> Size2Type:
    if len(fm_shape) < 1 or len(fm_shape) > 2:
        raise ShapeError(f"expected shape of 1 or 2, but got {len(fm_shape)}.")

    if len(fm_shape) == 1:
        channels, fm_len = (1,) + fm_shape
    else:
        if fm_order == "CL":
            channels, fm_len = fm_shape
        else:
            fm_len, channels = fm_shape

    return channels, fm_len


def fm_ndim2_check(fm_shape: SizeAnyType, fm_order: _Order3d) -> Size3Type:
    if len(fm_shape) < 2 or len(fm_shape) > 3:
        raise ShapeError(f"expected shape of 2 or 3, but got {len(fm_shape)}.")

    if len(fm_shape) == 2:
        channels, h, w = (1,) + fm_shape
    else:
        if fm_order == "CHW":
            channels, h, w = fm_shape
        else:
            h, w, channels = fm_shape

    return channels, h, w


def _conv_i2o(in_len: int, k: int, s: int, p: int, d: int) -> int:
    return (in_len + 2 * p - d * (k - 1) - 1) // s + 1


def _conv1d_oshape(
    isize: Size1Type,
    ksize: _Size1Type,
    stride: _Size1Type = 1,
    padding: _Size1Type = 0,
    dilation: _Size1Type = 1,
) -> Size1Type:
    """Compute the output shape of a 1d convolution."""
    assert len(isize) == 1

    in_l = isize[0]
    k = _single(ksize)[0]
    s = _single(stride)[0]
    p = _single(padding)[0]
    d = _single(dilation)[0]

    ol = _conv_i2o(in_l, k, s, p, d)
    return (ol,)


def _conv2d_oshape(
    isize: Size2Type,
    ksize: _Size2Type,
    stride: _Size2Type = 1,
    padding: _Size2Type = 0,
    dilation: _Size2Type = 1,
) -> Size2Type:
    """Compute the output shape of a 2d convolution."""
    assert len(isize) == 2

    k = _pair(ksize)
    s = _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)

    ho, wo = map(lambda args: _conv_i2o(*args), zip(isize, k, s, p, d))
    return (ho, wo)


@deprecated(
    "This is a slower function, use `_conv1d_unroll` instead.",
    category=PAIBoxDeprecationWarning,
)
def _conv1d_unroll_legacy(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    groups: int = 1,
) -> WeightType:
    """Unroll the kernel of 1d convolution into a matrix.

    NOTE: Argument 'groups' is not supported fully in this function. Fixed in `_conv1d_unroll`.
    """
    co, ci_in_grp, kl = kernel.shape
    co_in_grp = co // groups
    kernel = kernel.reshape(groups, co_in_grp, ci_in_grp, kl)
    il = in_shape[0] + 2 * padding[0]
    lo = out_shape[0]

    w_unrolled_np = np.zeros(
        (groups, ci_in_grp * il, co_in_grp * lo), dtype=kernel.dtype
    )
    mat_g = np.zeros((ci_in_grp * il, co_in_grp, lo), dtype=kernel.dtype)

    for g in range(groups):
        for i in range(lo):
            mat_g.fill(0)
            for oc_idx, ic_idx in np.ndindex(kernel.shape[1:3]):
                mat_g[
                    i * stride[0] + ic_idx * il : i * stride[0] + ic_idx * il + kl,
                    oc_idx,
                    i,
                ] = kernel[g, oc_idx, ic_idx, :]

            temp = mat_g[:, :, i].T

            for o_ch in range(co_in_grp):
                w_unrolled_np[g, :, i + o_ch * lo] = temp[o_ch].ravel()

    if padding == (0,):
        return w_unrolled_np.reshape(ci_in_grp * il, co * lo)

    # Remove the part of the padding in the w_unrolled_no_padding
    nil = in_shape[0]
    w_unrolled = np.zeros((groups, ci_in_grp * nil, co_in_grp * lo), dtype=kernel.dtype)

    for i in range(ci_in_grp):
        w_unrolled[:, i * nil : i * nil + nil, :] = w_unrolled_np[
            :, i * il + padding[0] : i * il + il - padding[0], :
        ]

    return w_unrolled.reshape(ci_in_grp * nil, co * lo)


def _conv1d_unroll(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    groups: int = 1,
) -> WeightType:
    p = padding[0]
    return _conv1d_unroll_asymmetric_padding(
        in_shape, out_shape, kernel, stride, _pair(p), groups
    )


def _conv1d_unroll_asymmetric_padding(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size2Type,
    groups: int = 1,
) -> WeightType:
    """Optimized version of conv1d kernel unrolling using vectorization & indexing.

    NOTE: the padding argument is a tuple of 2 values, (pl, pr) specifying the padding for the left & right \
        sides of the input.
    """
    li = in_shape[0]
    lo = out_shape[0]
    co, ci_in_grp, kl = kernel.shape
    co_in_grp = co // groups

    s = stride[0]
    pl, pr = padding
    li_padded = li + pl + pr

    x_grp_idx_shape = (ci_in_grp, li_padded)
    x_grp_idx_n = np.prod(x_grp_idx_shape)
    assert_max_index(x_grp_idx_n)

    x_grp_idx = np.arange(x_grp_idx_n, dtype=np.int16).reshape(x_grp_idx_shape)
    k_ur = np.zeros((groups, ci_in_grp * li_padded, co * lo), dtype=kernel.dtype)

    grp_windows = sliding_window_view(x_grp_idx, (kl,), axis=(1,))[:, ::s, :]  # type: ignore
    mask = grp_windows.transpose(0, 2, 1).reshape(-1, lo)

    for g in range(groups):
        for c in range(co_in_grp):
            cur_c = g * co_in_grp + c
            k = kernel[cur_c].ravel()
            cols = cur_c * lo + np.arange(lo)

            k_ur[g, mask, cols] = k[:, np.newaxis]

    # Handle padding removal
    if pl > 0 or pr > 0:
        k_ur = k_ur.reshape(groups, ci_in_grp, li_padded, -1)
        k_ur = k_ur[:, :, pl : -pr or None, :]

    return k_ur.reshape(-1, co * lo)


@deprecated(
    "This is a slower function, use `_conv2d_unroll` instead.",
    category=PAIBoxDeprecationWarning,
)
def _conv2d_unroll_legacy(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
) -> WeightType:
    """Unroll the kernel of 2d convolution into a matrix.

    NOTE: Argument 'groups' is not supported fully in this function. Fixed in `_conv2d_unroll`.
    """
    co, ci_in_grp, kh, kw = kernel.shape
    co_in_grp = co // groups
    kernel = kernel.reshape(groups, co_in_grp, ci_in_grp, kh, kw)
    hi = in_shape[0] + 2 * padding[0]
    wi = in_shape[1] + 2 * padding[1]
    ho, wo = out_shape
    in_size = hi * wi
    osize = ho * wo

    w_unrolled_np = np.zeros(
        (groups, ci_in_grp * in_size, co_in_grp * osize), dtype=kernel.dtype
    )
    mat_g = np.zeros((ci_in_grp * hi, wi * co_in_grp, osize), dtype=kernel.dtype)

    for g in range(groups):
        for i in range(ho):
            for j in range(wo):
                mat_g.fill(0)
                for oc_idx, ic_idx in np.ndindex(kernel.shape[1:3]):
                    mat_g[
                        i * stride[0] + ic_idx * hi : i * stride[0] + ic_idx * hi + kh,
                        j * stride[1] + oc_idx * wi : j * stride[1] + oc_idx * wi + kw,
                        i * wo + j,
                    ] = kernel[g, oc_idx, ic_idx, :, :]

                temp = (
                    mat_g[:, :, i * wo + j]
                    .reshape(ci_in_grp * hi, co_in_grp, wi)
                    .transpose(1, 0, 2)
                )

                for o_ch in range(co_in_grp):
                    w_unrolled_np[g, :, i * wo + j + o_ch * osize] = temp[o_ch].ravel()

    if padding == (0, 0):
        return w_unrolled_np.reshape(ci_in_grp * in_size, co * osize)

    # Remove the part of the padding in the w_unrolled_np
    nih, niw = in_shape
    nin_size = nih * niw
    w_unrolled = np.zeros(
        (groups, ci_in_grp * nin_size, co_in_grp * osize), dtype=kernel.dtype
    )

    for i in range(ci_in_grp):
        for j in range(nih):
            w_unrolled[:, i * nin_size + j * niw : i * nin_size + j * niw + niw, :] = (
                w_unrolled_np[
                    :,
                    i * in_size
                    + (padding[0] + j) * wi
                    + padding[1] : i * in_size
                    + (padding[0] + j) * wi
                    + padding[1]
                    + niw,
                    :,
                ]
            )

    return w_unrolled.reshape(ci_in_grp * nin_size, co * osize)


def _conv2d_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
) -> WeightType:
    ph, pw = padding
    return _conv2d_unroll_asymmetric_padding(
        in_shape, out_shape, kernel, stride, _pair(ph) + _pair(pw), groups
    )


def _conv2d_unroll_asymmetric_padding(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size4Type,
    groups: int = 1,
) -> WeightType:
    """Optimized version of conv2d kernel unrolling using sliding window view & indexing.

    NOTE: the padding argument is a tuple of 4 values, (ph, pd, pl, pr) specifying the padding for the top, \
        bottom, left & right sides of the input.
    """
    hi, wi = in_shape
    ho, wo = out_shape
    co, ci_in_grp, kh, kw = kernel.shape
    co_in_grp = co // groups

    sh, sw = stride
    pt, pd, pl, pr = padding
    hi_padded = hi + pt + pd
    wi_padded = wi + pl + pr
    osize = ho * wo

    x_grp_idx_shape = (ci_in_grp, hi_padded, wi_padded)
    x_grp_idx_n = np.prod(x_grp_idx_shape)
    assert_max_index(x_grp_idx_n)

    x_grp_idx = np.arange(x_grp_idx_n, dtype=INDEX_DTYPE).reshape(x_grp_idx_shape)
    k_ur = np.zeros((groups, x_grp_idx_n, co * osize), dtype=kernel.dtype)

    grp_windows = sliding_window_view(x_grp_idx, (kh, kw), axis=(1, 2))[  # type: ignore
        :, ::sh, ::sw, :, :
    ]
    mask = grp_windows.transpose(0, 3, 4, 1, 2).reshape(-1, osize)

    for g in range(groups):
        for c in range(co_in_grp):
            cur_c = g * co_in_grp + c
            k = kernel[cur_c].ravel()
            cols = cur_c * osize + np.arange(osize)

            k_ur[g, mask, cols] = k[:, np.newaxis]

    if pt > 0 or pd > 0 or pl > 0 or pr > 0:
        k_ur = k_ur.reshape(groups * ci_in_grp, hi_padded, wi_padded, -1)
        k_ur = k_ur[:, pt : -pd or None, pl : -pr or None, :]

    return k_ur.reshape(-1, co * osize)


def _conv2d_semifolded_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
) -> WeightType:
    ci, hi = in_shape
    co, ho = out_shape
    _, ci_in_grp, kh = kernel.shape

    _, sw = stride
    ph, _ = padding

    hi_pad = hi + 2 * ph
    w_np = np.zeros((ci * hi, co * ho), dtype=kernel.dtype)

    co_in_grp = co // groups
    for g in range(groups):
        for i in range(co_in_grp):
            for j in range(ci_in_grp):
                # Must recreate `w_block` every time because some rows will be deleted.
                w_block = np.zeros((hi_pad, ho), dtype=kernel.dtype)
                for k in range(ho):
                    w_block[k * sw : k * sw + kh, k] = kernel[g * co_in_grp + i, j, :]

                if ph > 0:
                    w_block = np.delete(
                        w_block,
                        np.hstack((np.arange(ph), np.arange(hi_pad - ph, hi_pad))),
                        axis=0,
                    )

                w_np[
                    g * ci_in_grp * hi + j * hi : g * ci_in_grp * hi + (j + 1) * hi,
                    g * ho * co_in_grp + i * ho : g * ho * co_in_grp + (i + 1) * ho,
                ] = w_block

    return w_np


@deprecated(
    "This is a slower function, use `conv1d_faster` instead.",
    category=PAIBoxDeprecationWarning,
)
def conv1d_faster_legacy(
    x: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: _Size1Type = 1,
    padding: _Size1Type = 0,
    dilation: _Size1Type = 1,
    groups: int = 1,
    bias: WeightType | None = None,
) -> SynOutType:
    """Faster 1d convolution."""
    ci = x.shape[0]
    co, ci_in_grp, kl = kernel.shape
    s = _single(stride)
    p = _single(padding)[0]
    d = _single(dilation)

    group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if p > 0:
        x_padded = np.pad(x, ((0, 0), (p, p)))
    else:
        x_padded = x

    out = np.zeros((co,) + out_shape, dtype=np.int64)
    for g in range(groups):
        ci_start = g * ci_in_grp
        ci_end = (g + 1) * ci_in_grp
        co_start = g * co_in_grp
        co_end = (g + 1) * co_in_grp

        x_grp = x_padded[ci_start:ci_end, :]
        kernel_grp = kernel[co_start:co_end, :, :]

        # kernel: (co_in_grp, ci_in_grp, kl) -> (co_in_grp, ci_in_grp*kl)
        col_kernel = kernel_grp.reshape(co_in_grp, -1)

        # padded: (ci_in_grp, xl+2*p[0]-kl) -> (lo, ci_in_grp*kl)
        col_fm = _1d_im2col(x_grp, out_shape[0], kl, s, d)

        # (co_in_grp, ci_in_grp*kl) * (lo, ci_in_grp*kl)^T = (co_in_grp, lo)
        out[co_start:co_end, :] = (col_kernel @ col_fm.T).reshape(co_in_grp, *out_shape)

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (co,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def conv1d_faster(
    x: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: _Size1Type = 1,
    padding: _Size1Type = 0,
    dilation: _Size1Type = 1,
    groups: int = 1,
    bias: WeightType | None = None,
) -> SynOutType:
    """Faster 2d convolution using im2col."""
    ci = x.shape[0]
    co, ci_in_grp, kl = kernel.shape
    s = _single(stride)
    p = _single(padding)
    d = _single(dilation)

    group_ch_check(ci, co, groups, ci_in_grp)

    x_cols = im2col_indices_1d(x, kl, s, p, d, groups, out_shape)
    co_in_grp = co // groups
    kernel_grp = kernel.reshape(groups, co_in_grp, -1)

    out = np.zeros((co,) + out_shape, dtype=np.int64)

    if bias is not None:
        _bias = bias.reshape(-1, 1)
        assert _bias.shape == (co, 1)
    else:
        _bias = 0

    for g in range(groups):
        co_start = g * co_in_grp
        co_end = (g + 1) * co_in_grp
        out_grp = kernel_grp[g] @ x_cols[g].astype(np.int64) + _bias
        out[co_start:co_end, :] = out_grp.reshape(co_in_grp, *out_shape)

    return out.astype(VOLTAGE_DTYPE)


@deprecated(
    "This is a slower function, use `conv2d_faster` instead.",
    category=PAIBoxDeprecationWarning,
)
def conv2d_faster_legacy(
    x: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: _Size2Type = 1,
    padding: _Size2Type = 0,
    dilation: _Size2Type = 1,
    groups: int = 1,
    bias: WeightType | None = None,
) -> SynOutType:
    """Faster 2d convolution.

    NOTE: This implementation is 10x slower than `conv2d_faster`.
    """
    ci = x.shape[0]
    co, ci_in_grp, kh, kw = kernel.shape
    s = _pair(stride)
    ph, pw = _pair(padding)
    d = _pair(dilation)

    group_ch_check(ci, co, groups, ci_in_grp)
    co_in_grp = co // groups

    if ph > 0 or pw > 0:
        x_padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    else:
        x_padded = x

    out = np.zeros((co, *out_shape), dtype=np.int64)

    for g in range(groups):
        x_grp = x_padded[g * ci_in_grp : (g + 1) * ci_in_grp, :, :]
        kernel_grp = kernel[g * co_in_grp : (g + 1) * co_in_grp, :, :, :]
        # kernel: (co_in_grp, ci, kh, kw) -> (co_in_grp, ci*kh*kw)
        col_kernel = kernel_grp.reshape(co_in_grp, -1)
        # padded: (ci, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (ho*wo, ci*kh*kw)
        col_fm = _2d_im2col(x_grp, out_shape[0], out_shape[1], kh, kw, s, d)
        # (ho*wo, ci*kh*kw) * (co, ci*kh*kw)^T = (ho*wo, co_in_grp)
        out_grp = col_fm @ col_kernel.T

        out[g * co_in_grp : (g + 1) * co_in_grp, :] = out_grp.T.reshape(
            (co_in_grp, *out_shape)
        )

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (co,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def conv2d_faster(
    x: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: _Size2Type = 1,
    padding: _Size2Type = 0,
    dilation: _Size2Type = 1,
    groups: int = 1,
    bias: WeightType | None = None,
) -> SynOutType:
    """Faster 2d convolution using im2col."""
    ci = x.shape[0]
    co, ci_in_grp, kh, kw = kernel.shape
    s = _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)

    group_ch_check(ci, co, groups, ci_in_grp)

    x_cols = im2col_indices_2d(x, kh, kw, s, p, d, groups, out_shape)
    co_in_grp = co // groups
    kernel_grp = kernel.reshape(groups, co_in_grp, -1)

    out = np.zeros((co,) + out_shape, dtype=np.int64)

    if bias is not None:
        _bias = bias.reshape(-1, 1)
        assert _bias.shape == (co, 1)
    else:
        _bias = 0

    for g in range(groups):
        co_start = g * co_in_grp
        co_end = (g + 1) * co_in_grp
        out_grp = kernel_grp[g] @ x_cols[g].astype(np.int64) + _bias
        out[co_start:co_end, :] = out_grp.reshape(co_in_grp, *out_shape)

    return out.astype(VOLTAGE_DTYPE)


def _convtranspose1d_unroll(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    output_padding: Size1Type,
) -> WeightType:
    """Unroll the kernel of 1d transposed convolution into a matrix.

    XXX: The case where the input feature map is in 'LC' order is not considered for the time being.
    """
    kernel_flip = np.flip(kernel, axis=2)

    co, ci, kl = kernel_flip.shape
    il = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1) + (kl - 1) * 2
    lo = out_shape[0] + 2 * padding[0] - output_padding[0]

    w_unrolled_np = np.zeros((ci * il, co * lo), dtype=kernel.dtype)
    zeros_image = np.zeros((ci * il, co, lo), dtype=kernel.dtype)

    # stride has been processed in the input matrix
    stride_transpose = 1
    for i in range(lo):
        zeros_image.fill(0)
        for ch_idx in np.ndindex(kernel_flip.shape[:2]):
            # [0] -> o_ch, [1] -> i_ch
            zeros_image[
                i * stride_transpose
                + ch_idx[1] * il : i * stride_transpose
                + ch_idx[1] * il
                + kl,
                ch_idx[0],
                i,
            ] = kernel_flip[ch_idx[0], ch_idx[1], :]

        t = zeros_image[:, :, i].T
        for o_ch in range(co):
            w_unrolled_np[:, i + o_ch * lo] = t[o_ch].ravel()

    # Remove the part of the transpose padding in the w_unrolled_no_padding
    # w_unrolled : (ci*il, co*lo) -> (ci*nil, co*lo), remove (kl - 1) padding
    nil = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1)
    w_unrolled_nk = np.zeros((ci * nil, co * lo), dtype=kernel.dtype)
    for i in range(ci):
        w_unrolled_nk[i * nil : i * nil + nil, :] = w_unrolled_np[
            i * il + kl - 1 : i * il + kl - 1 + nil, :
        ]

    # stripe
    w_reshaped = w_unrolled_nk.reshape((ci, nil, co, lo))
    # w_unrolled_ns = np.zeros((ci, nil, co, lo), dtype=w_unrolled_np.dtype)
    # w_unrolled_ns : (ci, in_shape[0], co ,lo)
    w_unrolled_ns = w_reshaped[::1, :: stride[0], ::1, ::1]

    # padding
    # w_unrolled : (ci, in_shape[0], co, lo - output_padding[0])
    w_unrolled = (
        w_unrolled_ns[:, :, :, padding[0] : (-1 * padding[0])]
        if padding[0] > 0
        else w_unrolled_ns
    )

    # output_padding
    w_unrolled = np.pad(w_unrolled, ((0, 0), (0, 0), (0, 0), (0, output_padding[0])))
    w_unrolled = w_unrolled.reshape(ci * in_shape[0], co * out_shape[0])

    return w_unrolled


def _convtranspose2d_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    output_padding: Size2Type,
) -> WeightType:
    """Unroll the kernel of 2d transposed convolution into a matrix."""
    kernel_flip = np.flip(kernel, axis=(2, 3))
    co, ci, kh, kw = kernel_flip.shape

    hi = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1) + (kh - 1) * 2
    wi = in_shape[1] + (in_shape[1] - 1) * (stride[1] - 1) + (kw - 1) * 2
    ho = out_shape[0] + 2 * padding[0] - output_padding[0]
    wo = out_shape[1] + 2 * padding[1] - output_padding[1]
    in_size = hi * wi
    osize = ho * wo

    w_unrolled_np = np.zeros((ci * in_size, co * osize), dtype=kernel.dtype)
    zeros_image = np.zeros((ci * hi, wi * co, osize), dtype=kernel.dtype)

    stride_transpose = (1, 1)
    for i in range(ho):
        for j in range(wo):
            zeros_image.fill(0)
            for ch_idx in np.ndindex(kernel_flip.shape[:2]):
                # [0] -> o_ch, [1] -> i_ch
                zeros_image[
                    i * stride_transpose[0]
                    + ch_idx[1] * hi : i * stride_transpose[0]
                    + ch_idx[1] * hi
                    + kh,
                    j * stride_transpose[1]
                    + ch_idx[0] * wi : j * stride_transpose[1]
                    + ch_idx[0] * wi
                    + kw,
                    i * wo + j,
                ] = kernel_flip[ch_idx[0], ch_idx[1], :, :]

            t = (
                zeros_image[:, :, i * wo + j]
                .reshape(ci * hi, co, wi)
                .transpose(1, 0, 2)
            )
            for o_ch in range(co):
                w_unrolled_np[:, i * wo + j + o_ch * osize] = t[o_ch].ravel()

    w_unrolled_np = w_unrolled_np.reshape((ci, hi, wi, co, ho, wo))

    # Remove the part of the transpose padding in the w_unrolled_no_padding
    # w_unrolled : (ci*in_size, co*osize) -> (ci*nin_size, co*nout_size), remove (kl - 1) padding
    nih = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1)
    niw = in_shape[1] + (in_shape[1] - 1) * (stride[1] - 1)
    nin_size = nih * niw
    w_unrolled = np.zeros((ci * nin_size, co * osize), dtype=kernel.dtype)
    w_unrolled = w_unrolled.reshape((ci, nih, niw, co, ho, wo))

    # transpose (kernel - 1) padding
    kh_start = (kh - 1) if kh > 1 else None
    kh_end = (1 - kh) if kh > 1 else None
    kw_start = (kw - 1) if kw > 1 else None
    kw_end = (1 - kw) if kw > 1 else None
    w_unrolled = w_unrolled_np[:, kh_start:kh_end, kw_start:kw_end, :, :, :]

    # stripe
    w_unrolled = w_unrolled[::1, :: stride[0], :: stride[1], ::1, ::1, ::1]

    # padding
    ph_start = padding[0] if padding[0] > 0 else None
    ph_end = (-1 * padding[0]) if padding[0] > 0 else None
    pw_start = padding[1] if padding[1] > 0 else None
    pw_end = (-1 * padding[1]) if padding[1] > 0 else None
    w_unrolled = w_unrolled[:, :, :, :, ph_start:ph_end, pw_start:pw_end]

    # output_padding
    w_unrolled = np.pad(
        w_unrolled,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, output_padding[0]),
            (0, output_padding[1]),
        ),
    )
    w_unrolled = w_unrolled.reshape(
        ci * in_shape[0] * in_shape[1], co * out_shape[0] * out_shape[1]
    )

    return w_unrolled


def _convtranspose1d_faster(
    x: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    output_padding: Size1Type,
    bias: WeightType | None = None,
) -> SynOutType:
    # (C, L)
    xc, xl = x.shape

    # (O, I, L)
    co, ci, kl = kernel.shape
    assert xc == ci, "Input channels must match kernel channels."
    assert (xl - 1) * stride[0] - 2 * padding[0] + kl + output_padding[0] == out_shape[
        0
    ]

    # generate new input array
    # inverse stride : Insert 0 between rows and columns
    xc_t = xc
    xl_t = xl + (xl - 1) * (stride[0] - 1)
    x_transpose = np.zeros((xc_t, xl_t), dtype=x.dtype)
    x_transpose[::1, :: stride[0]] = x

    # inverse padding
    # x_transpose : (ci, (xl-1)*(stride-1)+2*(kl-1))
    x_transpose = np.pad(x_transpose, ((0, 0), (kl - 1, kl - 1)))

    # convolution kernel rotated 180 degrees
    kernel_flip = np.flip(kernel, axis=2)
    # kernel: (co, ci, kl) -> (ci*kl, co)
    kernel_col = kernel_flip.reshape(co, -1)

    # col_fm: (ci, nol) -> (nol, ci*kl)
    nol = out_shape[0] - output_padding[0] + 2 * padding[0]
    stride_transpose = (1,)
    col_fm = _1d_im2col(x_transpose, nol, kl, stride_transpose, (1,))

    # (nol, ci*kl) * (ci*kl, co) = (nol, co)
    out = col_fm @ kernel_col.T  # + self.bias
    # (nol, co) -> (co, nol)
    out = out.T

    # inverse padding : (co, (xl-1)*stride+kernel) -> (co, (xl-1)*stride+kernel-2*padding)
    out = out[:, padding[0] : (-1 * padding[0])] if padding[0] > 0 else out

    # output_padding
    out = np.pad(out, ((0, 0), (0, output_padding[0])))

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (co,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def _convtranspose2d_faster(
    x: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    output_padding: Size2Type,
    bias: WeightType | None = None,
) -> SynOutType:
    # (C, H, W)
    xc, xh, xw = x.shape

    # (O, I, H, W)
    co, ci, kh, kw = kernel.shape
    assert xc == ci, "Input channels must match kernel channels."

    # Calculate the shape of the padded input (considering stride)
    ho, wo = out_shape
    assert (xh - 1) * stride[0] - 2 * padding[0] + kh + output_padding[0] == ho
    assert (xw - 1) * stride[1] - 2 * padding[1] + kw + output_padding[1] == wo

    # By modifying the input matrix and convolution kernel
    # we can change the transpose convolution to the form of an ordinary convolution

    # Generate the transpose input array : transpose padding 0 & stride 0
    xc_t = xc
    xh_t = xh + (xh - 1) * (stride[0] - 1)
    xw_t = xw + (xw - 1) * (stride[1] - 1)
    x_transpose = np.zeros((xc_t, xh_t, xw_t), dtype=x.dtype)
    x_transpose[::1, :: stride[0], :: stride[1]] = x
    # padding 0 for transpose not for parameter padding, get new input array x_transpose
    x_transpose = np.pad(x_transpose, ((0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)))

    # kernel: (co, ci, kh, kw) -> (co, ci*kh*kw)
    kernel_flip = np.flip(kernel, axis=(2, 3))  # convolution kernel rotated 180 degrees
    kernel_col = kernel_flip.reshape(co, -1)

    # conv
    stride_transpose = (1, 1)
    noh = ho - output_padding[0] + 2 * padding[0]
    now = wo - output_padding[1] + 2 * padding[1]
    col_fm = _2d_im2col(x_transpose, noh, now, kh, kw, stride_transpose, (1, 1))

    # (ho*wo, ci*kh*kw) * (ci*kh*kw, co) = (ho*wo, co)
    out_col = col_fm @ kernel_col.T
    # (ho*wo, co) -> (ho, wo, co) -> (co, ho, wo)
    out = out_col.astype(VOLTAGE_DTYPE).T.reshape((co,) + (noh, now))

    # padding & output_padding
    # inverse padding
    out = out[
        :,
        padding[0] : (-1 * padding[0]) if padding[0] > 0 else None,
        padding[1] : (-1 * padding[1]) if padding[1] > 0 else None,
    ]
    # output_padding
    out = np.pad(out, ((0, 0), (0, output_padding[0]), (0, output_padding[1])))

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (co,)

        out += _bias

    return out


def _1d_im2col(
    x_padded: NeuOutType,
    lo: int,
    kl: int,
    stride: Size1Type,
    dilation: Size1Type,
) -> NDArray[np.int64]:
    ci, pl = x_padded.shape
    cols = np.zeros((lo, ci * kl), dtype=np.int64)
    s = stride[0]
    d = dilation[0]

    for i in range(lo):
        # Generate the indices for the dilated kernel
        indices = [i * s + k * d for k in range(kl)]
        # Check if all indices are within bounds
        if max(indices) >= pl:
            raise ValueError(
                f"Dilated kernel exceeds input bounds at position ({i}). "
                f"Max indices: h={max(indices)} (input_l={pl})"
            )

        # Extract & flatten the window
        window = x_padded[:, indices].ravel()
        cols[i] = window

    return cols


def _2d_im2col(
    x_padded: NeuOutType,
    ho: int,
    wo: int,
    kh: int,
    kw: int,
    stride: Size2Type,
    dilation: Size2Type,
) -> NDArray[np.int64]:
    co, ph, pw = x_padded.shape
    cols = np.zeros((ho * wo, co * kh * kw), dtype=np.int64)
    sh, sw = stride
    dh, dw = dilation

    idx = 0
    for i in range(ho):
        for j in range(wo):
            # Generate the indices for the dilated kernel
            h_indices = [i * sh + m * dh for m in range(kh)]
            w_indices = [j * sw + n * dw for n in range(kw)]

            # Check bounds
            if max(h_indices) >= ph or max(w_indices) >= pw:
                raise ValueError(
                    f"Dilated kernel exceeds input bounds at position ({i},{j}). "
                    f"Max indices: h={max(h_indices)} (input_h={ph}), "
                    f"w={max(w_indices)} (input_w={pw})"
                )

            # Extract & flatten the window
            window = x_padded[:, h_indices, :][:, :, w_indices].ravel()
            cols[idx] = window
            idx += 1

    return cols


def get_im2col_indices_1d(
    x_shape: Size2Type,
    out_shape: Size1Type,
    kl: int,
    stride: Size1Type = (1,),
    dilation: Size1Type = (1,),
    groups: int = 1,
):
    ci, _ = x_shape
    lo = out_shape[0]
    sl = stride[0]
    dl = dilation[0]

    assert ci % groups == 0, f"Input channels {ci} must be divisible by groups {groups}"
    ci_in_grp = ci // groups

    assert_max_index(kl, lo, ci_in_grp)

    i0 = np.arange(kl, dtype=INDEX_DTYPE) * dl
    i0 = np.tile(i0, ci_in_grp)
    i1 = sl * np.arange(lo, dtype=INDEX_DTYPE)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    k = np.repeat(np.arange(ci_in_grp, dtype=INDEX_DTYPE), kl).reshape(-1, 1)

    return k, i, ci_in_grp


def get_im2col_indices_2d(
    x_shape: Size3Type,
    out_shape: Size2Type,
    kh: int,
    kw: int,
    stride: Size2Type = (1, 1),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
):
    ci, _, _ = x_shape
    ho, wo = out_shape
    sh, sw = stride
    dh, dw = dilation

    assert ci % groups == 0, f"Input channels {ci} must be divisible by groups {groups}"
    ci_in_grp = ci // groups

    assert_max_index(kh, kw, ho, wo, ci_in_grp)

    i0 = np.repeat(np.arange(kh, dtype=INDEX_DTYPE) * dh, kw)
    i0 = np.tile(i0, ci_in_grp)
    i1 = sh * np.repeat(np.arange(ho, dtype=INDEX_DTYPE), wo)
    j0 = np.tile(np.arange(kw, dtype=INDEX_DTYPE) * dw, kh * ci_in_grp)
    j1 = sw * np.tile(np.arange(wo, dtype=INDEX_DTYPE), ho)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(ci_in_grp, dtype=INDEX_DTYPE), kh * kw).reshape(-1, 1)

    return k, i, j, ci_in_grp


def _im2col_indices_groupwise(
    x_padded: np.ndarray, indices: tuple[np.ndarray, ...], ci_in_grp: int, groups: int
) -> np.ndarray:
    reshape_shape = indices[-1].shape  # Get i(1d) or j(2d) shape
    cols = np.zeros((groups,) + reshape_shape, dtype=x_padded.dtype)

    for g in range(groups):
        ci_start = g * ci_in_grp
        ci_end = (g + 1) * ci_in_grp

        x_grp = x_padded[ci_start:ci_end]
        cols_grp = x_grp[indices]

        cols_grp = cols_grp.reshape(reshape_shape)
        cols[g] = cols_grp

    return cols


def im2col_indices_1d(
    x: np.ndarray,
    kl: int,
    stride: Size1Type = (1,),
    padding: Size1Type = (0,),
    dilation: Size1Type = (1,),
    groups: int = 1,
    out_shape: Size1Type | None = None,
):
    x_padded = np.pad(x, ((0, 0), (padding[0], padding[0])), mode="constant")
    if out_shape is None:
        out_shape = _conv1d_oshape(x.shape, kl, stride, padding, dilation)

    k, i, ci_in_grp = get_im2col_indices_1d(
        x.shape, out_shape, kl, stride, dilation, groups
    )

    return _im2col_indices_groupwise(x_padded, (k, i), ci_in_grp, groups)


def im2col_indices_2d(
    x: np.ndarray,
    kh: int,
    kw: int,
    stride: Size2Type = (1, 1),
    padding: Size2Type = (0, 0),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
    out_shape: Size2Type | None = None,
):
    """An implementation of im2col based on some fancy indexing"""
    x_padded = np.pad(
        x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    if out_shape is None:
        out_shape = _conv2d_oshape(x.shape, (kh, kw), stride, padding, dilation)

    k, i, j, ci_in_grp = get_im2col_indices_2d(
        x.shape, out_shape, kh, kw, stride, dilation, groups
    )

    return _im2col_indices_groupwise(x_padded, (k, i, j), ci_in_grp, groups)


def _pool1d_kernel_unroll(
    channels: int,
    in_shape: Size1Type,
    out_shape: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
) -> WeightType:
    li = in_shape[0]
    lo = out_shape[0]
    kl = ksize[0]
    s = stride[0]
    p = padding[0]

    li_padded = li + 2 * p

    x_ch_idx_shape = (li_padded,)
    x_ch_idx_n = np.prod(x_ch_idx_shape)
    assert_max_index(x_ch_idx_n)

    x_ch_idx = np.arange(x_ch_idx_n, dtype=INDEX_DTYPE).reshape(x_ch_idx_shape)

    ch_windows = sliding_window_view(x_ch_idx, (kl,))[::s, :]
    mask = ch_windows.transpose(1, 0).reshape(-1, lo)

    k_ch_ur = np.zeros((li_padded, lo), dtype=WEIGHT_DTYPE)
    k_ch_ur[mask, np.arange(lo)] = 1

    if p > 0:
        k_ch_ur = k_ch_ur.reshape(li_padded, -1)
        k_ch_ur = k_ch_ur[p : -p or None, :]
        k_ch_ur = k_ch_ur.reshape(-1, lo)

    # Extend to all channels
    return np.kron(np.eye(channels), k_ch_ur).astype(WEIGHT_DTYPE)


def _pool2d_kernel_unroll(
    channels: int,
    in_shape: Size2Type,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
) -> WeightType:
    hi, wi = in_shape
    ho, wo = out_shape
    sh, sw = stride
    ph, pw = padding

    hi_padded = hi + 2 * ph
    wi_padded = wi + 2 * pw
    osize = ho * wo

    x_ch_idx_shape = (hi_padded, wi_padded)
    x_ch_idx_n = np.prod(x_ch_idx_shape)
    assert_max_index(x_ch_idx_n)

    x_ch_idx = np.arange(x_ch_idx_n, dtype=INDEX_DTYPE).reshape(x_ch_idx_shape)

    ch_windows = sliding_window_view(x_ch_idx, ksize)[::sh, ::sw, :, :]
    mask = ch_windows.transpose(2, 3, 0, 1).reshape(-1, osize)

    k_ch_ur = np.zeros((hi_padded * wi_padded, osize), dtype=WEIGHT_DTYPE)
    k_ch_ur[mask, np.arange(osize)] = 1

    if ph > 0 or pw > 0:
        k_ch_ur = k_ch_ur.reshape(hi_padded, wi_padded, -1)
        k_ch_ur = k_ch_ur[ph : -ph or None, pw : -pw or None, :]
        k_ch_ur = k_ch_ur.reshape(-1, osize)

    # Extend to all channels
    return np.kron(np.eye(channels), k_ch_ur).astype(WEIGHT_DTYPE)


def _func_pool1d(
    x: NeuOutType,
    out_shape: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
    type: str,
    threshold: int,
) -> NeuOutType:
    xcin, xl = x.shape
    kl = ksize[0]
    lo = out_shape[0]
    co = xcin
    s = stride[0]
    p = padding[0]

    assert (xl + p * 2 - kl) // s + 1 == lo

    out = np.zeros((co, lo), dtype=np.int32)

    if p > 0:
        x_padded = np.pad(x, ((0, 0), (p, p)))
    else:
        x_padded = x

    for c in range(co):
        for i in range(lo):
            if type == "avg":
                out[c, i] = np.sum(x_padded[c, s * i : s * i + kl])
            else:
                out[c, i] = np.max(x_padded[c, s * i : s * i + kl])

    if type == "avg":
        result = out >= threshold
    else:
        result = out

    return result.astype(NEUOUT_U8_DTYPE)


def _func_pool2d(
    x: NeuOutType,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    type: str,
    threshold: int,
) -> NeuOutType:
    xcin, xh, xw = x.shape
    kh, kw = ksize
    ho, wo = out_shape
    co = xcin
    sh, sw = stride
    ph, pw = padding

    assert (xh + ph * 2 - kh) // sh + 1 == ho
    assert (xw + pw * 2 - kw) // sw + 1 == wo

    out = np.zeros((co, ho, wo), dtype=np.int32)

    if ph > 0 or pw > 0:
        x_padded = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    else:
        x_padded = x

    for c in range(co):
        for i in range(ho):
            for j in range(wo):
                if type == "avg":
                    out[c, i, j] = np.sum(
                        x_padded[c, sh * i : sh * i + kh, sw * j : sw * j + kw]
                    )
                else:
                    out[c, i, j] = np.max(
                        x_padded[c, sh * i : sh * i + kh, sw * j : sw * j + kw]
                    )

    if type == "avg":
        result = out >= threshold
    else:
        result = out

    return result.astype(NEUOUT_U8_DTYPE)
