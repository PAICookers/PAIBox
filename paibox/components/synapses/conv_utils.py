import sys
from collections.abc import Iterable
from functools import partial
from itertools import repeat
from typing import Optional, Union

import numpy as np
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

from .conv_types import Size1Type, Size2Type, Size3Type, SizeAnyType, _Order2d, _Order3d

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


def _fm_ndim1_check(fm_shape: SizeAnyType, fm_order: _Order2d) -> Size2Type:
    if len(fm_shape) < 1 or len(fm_shape) > 2:
        raise ShapeError(f"expected shape of 1 or 2, but got {len(fm_shape)}.")

    if len(fm_shape) == 1:
        channels, l = (1,) + fm_shape
    else:
        if fm_order == "CL":
            channels, l = fm_shape
        else:
            l, channels = fm_shape

    return channels, l


def _fm_ndim2_check(fm_shape: SizeAnyType, fm_order: _Order3d) -> Size3Type:
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


def _conv1d_unroll(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    groups: int = 1,
) -> WeightType:
    """Unroll the kernel of 1d convolution into a matrix."""
    cout, cin_per_grp, kl = kernel.shape
    cout_per_grp = cout // groups
    kernel = kernel.reshape(groups, cout_per_grp, cin_per_grp, kl)
    il = in_shape[0] + 2 * padding[0]
    ol = out_shape[0]

    w_unrolled_np = np.zeros(
        (groups, cin_per_grp * il, cout_per_grp * ol), dtype=kernel.dtype
    )
    mat_g = np.zeros((cin_per_grp * il, cout_per_grp, ol), dtype=kernel.dtype)

    for g in range(groups):
        for i in range(ol):
            mat_g.fill(0)
            for oc_idx, ic_idx in np.ndindex(kernel.shape[1:3]):
                mat_g[
                    i * stride[0] + ic_idx * il : i * stride[0] + ic_idx * il + kl,
                    oc_idx,
                    i,
                ] = kernel[g, oc_idx, ic_idx, :]

            temp = mat_g[:, :, i].T

            for o_ch in range(cout_per_grp):
                w_unrolled_np[g, :, i + o_ch * ol] = temp[o_ch].ravel()

    if padding == (0,):
        return w_unrolled_np.reshape(cin_per_grp * il, cout * ol)

    # Remove the part of the padding in the w_unrolled_no_padding
    nil = in_shape[0]
    w_unrolled = np.zeros(
        (groups, cin_per_grp * nil, cout_per_grp * ol), dtype=kernel.dtype
    )

    for i in range(cin_per_grp):
        w_unrolled[:, i * nil : i * nil + nil, :] = w_unrolled_np[
            :, i * il + padding[0] : i * il + il - padding[0], :
        ]

    return w_unrolled.reshape(cin_per_grp * nil, cout * ol)


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
    """Unroll the kernel of 2d convolution into a matrix."""
    cout, cin_per_grp, kh, kw = kernel.shape
    cout_per_grp = cout // groups
    kernel = kernel.reshape(groups, cout_per_grp, cin_per_grp, kh, kw)
    ih = in_shape[0] + 2 * padding[0]
    iw = in_shape[1] + 2 * padding[1]
    oh, ow = out_shape
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled_np = np.zeros(
        (groups, cin_per_grp * in_size, cout_per_grp * out_size), dtype=kernel.dtype
    )
    mat_g = np.zeros(
        (cin_per_grp * ih, iw * cout_per_grp, out_size), dtype=kernel.dtype
    )

    for g in range(groups):
        for i in range(oh):
            for j in range(ow):
                mat_g.fill(0)
                for oc_idx, ic_idx in np.ndindex(kernel.shape[1:3]):
                    mat_g[
                        i * stride[0] + ic_idx * ih : i * stride[0] + ic_idx * ih + kh,
                        j * stride[1] + oc_idx * iw : j * stride[1] + oc_idx * iw + kw,
                        i * ow + j,
                    ] = kernel[g, oc_idx, ic_idx, :, :]

                temp = (
                    mat_g[:, :, i * ow + j]
                    .reshape(cin_per_grp * ih, cout_per_grp, iw)
                    .transpose(1, 0, 2)
                )

                for o_ch in range(cout_per_grp):
                    w_unrolled_np[g, :, i * ow + j + o_ch * out_size] = temp[
                        o_ch
                    ].ravel()

    if padding == (0, 0):
        return w_unrolled_np.reshape(cin_per_grp * in_size, cout * out_size)

    # Remove the part of the padding in the w_unrolled_np
    nih, niw = in_shape
    nin_size = nih * niw
    w_unrolled = np.zeros(
        (groups, cin_per_grp * nin_size, cout_per_grp * out_size), dtype=kernel.dtype
    )

    for i in range(cin_per_grp):
        for j in range(nih):
            w_unrolled[:, i * nin_size + j * niw : i * nin_size + j * niw + niw, :] = (
                w_unrolled_np[
                    :,
                    i * in_size
                    + (padding[0] + j) * iw
                    + padding[1] : i * in_size
                    + (padding[0] + j) * iw
                    + padding[1]
                    + niw,
                    :,
                ]
            )

    return w_unrolled.reshape(cin_per_grp * nin_size, cout * out_size)


def _conv2d_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
) -> np.ndarray:
    """Optimized version of conv2d kernel unrolling using vectorization."""
    cout, cin_per_grp, kh, kw = kernel.shape
    cout_per_grp = cout // groups
    kernel = kernel.reshape(groups, cout_per_grp, cin_per_grp, kh, kw)
    ih_padded = in_shape[0] + 2 * padding[0]
    iw_padded = in_shape[1] + 2 * padding[1]
    oh, ow = out_shape
    in_size_padded = ih_padded * iw_padded
    out_size = oh * ow

    # Precompute window template indices
    window_template = np.arange(kh)[:, np.newaxis] * iw_padded + np.arange(kw)
    window_flat = window_template.ravel()

    # Initialize unrolled weights array
    w_unrolled_np = np.zeros(
        (groups, cin_per_grp * in_size_padded, cout_per_grp * out_size),
        dtype=kernel.dtype,
    )

    for g in range(groups):
        kernel_grp = kernel[g]

        for i in range(oh):
            for j in range(ow):
                top = i * stride[0]
                left = j * stride[1]
                start_idx = top * iw_padded + left

                # Compute indices for all input channels
                channel_bases = np.arange(cin_per_grp)[:, np.newaxis] * in_size_padded
                indices = (channel_bases + start_idx + window_flat).ravel()

                # Prepare column indices for all output channels
                col_base_idx = i * ow + j
                col_indices = np.arange(cout_per_grp) * out_size + col_base_idx

                # (cout_per_grp, cin_per_grp*kh*kw)
                kernel_flat = kernel_grp.reshape(cout_per_grp, -1)

                # Assign to output matrix
                w_unrolled_np[g, indices[:, np.newaxis], col_indices] = kernel_flat.T

    # Handle padding removal
    if padding == (0, 0):
        return w_unrolled_np.reshape(cin_per_grp * in_size_padded, cout * out_size)

    # Remove padding
    nih, niw = in_shape
    nin_size = nih * niw
    w_unrolled = np.zeros(
        (groups, cin_per_grp * nin_size, cout_per_grp * out_size), dtype=kernel.dtype
    )

    for ci in range(cin_per_grp):
        for j in range(nih):
            src_start = ci * in_size_padded + (padding[0] + j) * iw_padded + padding[1]
            src_end = src_start + niw
            dest_start = ci * nin_size + j * niw

            w_unrolled[:, dest_start : dest_start + niw, :] = w_unrolled_np[
                :, src_start:src_end, :
            ]

    return w_unrolled.reshape(cin_per_grp * nin_size, cout * out_size)


def _conv2d_semifolded_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    groups: int = 1,
) -> WeightType:
    cout, ck, kh = kernel.shape
    cin = groups * ck
    ih = in_shape[1] + 2 * padding[0]
    _, oh = out_shape
    w_np = np.zeros((cin * in_shape[1], cout * oh), dtype=kernel.dtype)

    cout_per_grp = cout // groups
    for g in range(groups):
        for i in range(cout_per_grp):
            for j in range(ck):
                # Must recreate `w_block` every time because some rows will be deleted.
                w_block = np.zeros((ih, oh), dtype=kernel.dtype)
                for k in range(oh):
                    w_block[k * stride[1] : k * stride[1] + kh, k] = kernel[
                        g * cout_per_grp + i, j, :
                    ]

                if padding[0] > 0:  # H direction
                    w_block = np.delete(
                        w_block,
                        np.hstack(
                            (np.arange(padding[0]), np.arange(ih - padding[0], ih))
                        ),
                        axis=0,
                    )

                w_np[
                    g * ck * in_shape[1]
                    + j * in_shape[1] : g * ck * in_shape[1]
                    + (j + 1) * in_shape[1],
                    g * oh * cout_per_grp
                    + i * oh : g * oh * cout_per_grp
                    + (i + 1) * oh,
                ] = w_block

    return w_np


@deprecated(
    "This is a slower function, use `conv1d_faster` instead.",
    category=PAIBoxDeprecationWarning,
)
def conv1d_faster_legacy(
    x_cl: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type = (1,),
    padding: Size1Type = (0,),
    dilation: Size1Type = (1,),
    groups: int = 1,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    """Faster 1d convolution."""
    cin = x_cl.shape[0]
    cout, cin_per_grp, kl = kernel.shape

    assert cin == cin_per_grp * groups
    assert (
        cin % groups == 0 and cout % groups == 0
    ), f"Input & output channels {cin} & {cout} must be divisible by groups {groups}"

    cout_per_grp = cout // groups

    if padding > (0,):
        x_padded = np.pad(x_cl, ((0, 0), (padding[0], padding[0])))
    else:
        x_padded = x_cl

    out = np.zeros((cout,) + out_shape, dtype=np.int64)
    for g in range(groups):
        cin_start = g * cin_per_grp
        cin_end = (g + 1) * cin_per_grp
        cout_start = g * cout_per_grp
        cout_end = (g + 1) * cout_per_grp

        x_grp = x_padded[cin_start:cin_end, :]
        kernel_grp = kernel[cout_start:cout_end, :, :]

        # kernel: (cout_per_grp, cin_per_grp, kl) -> (cout_per_grp, cin_per_grp*kl)
        col_kernel = kernel_grp.reshape(cout_per_grp, -1)

        # padded: (cin_per_grp, xl+2*p[0]-kl) -> (ol, cin_per_grp*kl)
        col_fm = _1d_im2col(x_grp, out_shape[0], kl, stride, dilation)

        # (cout_per_grp, cin_per_grp*kl) * (ol, cin_per_grp*kl)^T = (cout_per_grp, ol)
        out[cout_start:cout_end, :] = (col_kernel @ col_fm.T).reshape(
            cout_per_grp, *out_shape
        )

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (cout,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def conv1d_faster(
    x_cl: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type = (1,),
    padding: Size1Type = (0,),
    dilation: Size1Type = (1,),
    groups: int = 1,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    """Faster 2d convolution using im2col."""
    cin = x_cl.shape[0]
    cout, cin_per_grp, kl = kernel.shape

    assert cin == cin_per_grp * groups
    assert (
        cin % groups == 0 and cout % groups == 0
    ), f"Input & output channels {cin} & {cout} must be divisible by groups {groups}"

    x_cols = im2col_indices_1d(x_cl, kl, stride, padding, dilation, groups, out_shape)
    cout_per_grp = cout // groups
    kernel_grp = kernel.reshape(groups, cout_per_grp, -1)

    out = np.zeros((cout,) + out_shape, dtype=np.int64)

    if bias is not None:
        _bias = bias.reshape(-1, 1)
        assert _bias.shape == (cout, 1)
    else:
        _bias = 0

    for g in range(groups):
        cout_start = g * cout_per_grp
        cout_end = (g + 1) * cout_per_grp
        out_grp = kernel_grp[g] @ x_cols[g].astype(np.int64) + _bias
        out[cout_start:cout_end, :] = out_grp.reshape(cout_per_grp, *out_shape)

    return out.astype(VOLTAGE_DTYPE)


@deprecated(
    "This is a slower function, use `conv2d_faster` instead.",
    category=PAIBoxDeprecationWarning,
)
def conv2d_faster_legacy(
    x_chw: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type = (1, 1),
    padding: Size2Type = (0, 0),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    """Faster 2d convolution.

    NOTE: This implementation is 10x slower than `conv2d_faster`.
    """
    cout, cin_per_grp, kh, kw = kernel.shape  # (O, I, H, W)

    assert x_chw.shape[0] == cin_per_grp * groups
    assert cout % groups == 0

    cout_per_grp = cout // groups

    x_padded = np.pad(
        x_chw,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
    )
    out = np.zeros((cout, *out_shape), dtype=np.int64)

    for g in range(groups):
        x_grp = x_padded[g * cin_per_grp : (g + 1) * cin_per_grp, :, :]
        kernel_grp = kernel[g * cout_per_grp : (g + 1) * cout_per_grp, :, :, :]
        # kernel: (cout_per_grp, cin, kh, kw) -> (cout_per_grp, cin*kh*kw)
        col_kernel = kernel_grp.reshape(cout_per_grp, -1)
        # padded: (cin, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (oh*ow, cin*kh*kw)
        col_fm = _2d_im2col(x_grp, out_shape[0], out_shape[1], kh, kw, stride, dilation)
        # (oh*ow, cin*kh*kw) * (cout, cin*kh*kw)^T = (oh*ow, cout_per_grp)
        out_grp = col_fm @ col_kernel.T

        out[g * cout_per_grp : (g + 1) * cout_per_grp, :] = out_grp.T.reshape(
            (cout_per_grp, *out_shape)
        )

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (cout,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def conv2d_faster(
    x_chw: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type = (1, 1),
    padding: Size2Type = (0, 0),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    """Faster 2d convolution using im2col."""
    cin = x_chw.shape[0]
    cout, cin_per_grp, kh, kw = kernel.shape  # (O, I, H, W)

    assert cin == cin_per_grp * groups
    assert (
        cin % groups == 0 and cout % groups == 0
    ), f"Input & output channels {cin} & {cout} must be divisible by groups {groups}"

    x_cols = im2col_indices_2d(
        x_chw, kh, kw, stride, padding, dilation, groups, out_shape
    )
    cout_per_grp = cout // groups
    kernel_grp = kernel.reshape(groups, cout_per_grp, -1)

    out = np.zeros((cout,) + out_shape, dtype=np.int64)

    if bias is not None:
        _bias = bias.reshape(-1, 1)
        assert _bias.shape == (cout, 1)
    else:
        _bias = 0

    for g in range(groups):
        cout_start = g * cout_per_grp
        cout_end = (g + 1) * cout_per_grp
        out_grp = kernel_grp[g] @ x_cols[g].astype(np.int64) + _bias
        out[cout_start:cout_end, :] = out_grp.reshape(cout_per_grp, *out_shape)

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

    cout, cin, kl = kernel_flip.shape
    il = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1) + (kl - 1) * 2
    ol = out_shape[0] + 2 * padding[0] - output_padding[0]

    w_unrolled_np = np.zeros((cin * il, cout * ol), dtype=kernel.dtype)
    zeros_image = np.zeros((cin * il, cout, ol), dtype=kernel.dtype)

    # stride has been processed in the input matrix
    stride_transpose = 1
    for i in range(ol):
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
        for o_ch in range(cout):
            w_unrolled_np[:, i + o_ch * ol] = t[o_ch].ravel()

    # Remove the part of the transpose padding in the w_unrolled_no_padding
    # w_unrolled : (cin*il, cout*ol) -> (cin*nil, cout*ol), remove (kl - 1) padding
    nil = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1)
    w_unrolled_nk = np.zeros((cin * nil, cout * ol), dtype=kernel.dtype)
    for i in range(cin):
        w_unrolled_nk[i * nil : i * nil + nil, :] = w_unrolled_np[
            i * il + kl - 1 : i * il + kl - 1 + nil, :
        ]

    # stripe
    w_reshaped = w_unrolled_nk.reshape((cin, nil, cout, ol))
    # w_unrolled_ns = np.zeros((cin, nil, cout, ol), dtype=w_unrolled_np.dtype)
    # w_unrolled_ns : (cin, in_shape[0], cout ,ol)
    w_unrolled_ns = w_reshaped[::1, :: stride[0], ::1, ::1]

    # padding
    # w_unrolled : (cin, in_shape[0], cout, ol - output_padding[0])
    w_unrolled = (
        w_unrolled_ns[:, :, :, padding[0] : (-1 * padding[0])]
        if padding[0] > 0
        else w_unrolled_ns
    )

    # output_padding
    w_unrolled = np.pad(w_unrolled, ((0, 0), (0, 0), (0, 0), (0, output_padding[0])))
    w_unrolled = w_unrolled.reshape(cin * in_shape[0], cout * out_shape[0])

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
    cout, cin, kh, kw = kernel_flip.shape

    ih = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1) + (kh - 1) * 2
    iw = in_shape[1] + (in_shape[1] - 1) * (stride[1] - 1) + (kw - 1) * 2
    oh = out_shape[0] + 2 * padding[0] - output_padding[0]
    ow = out_shape[1] + 2 * padding[1] - output_padding[1]
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled_np = np.zeros((cin * in_size, cout * out_size), dtype=kernel.dtype)
    zeros_image = np.zeros((cin * ih, iw * cout, out_size), dtype=kernel.dtype)

    stride_transpose = (1, 1)
    for i in range(oh):
        for j in range(ow):
            zeros_image.fill(0)
            for ch_idx in np.ndindex(kernel_flip.shape[:2]):
                # [0] -> o_ch, [1] -> i_ch
                zeros_image[
                    i * stride_transpose[0]
                    + ch_idx[1] * ih : i * stride_transpose[0]
                    + ch_idx[1] * ih
                    + kh,
                    j * stride_transpose[1]
                    + ch_idx[0] * iw : j * stride_transpose[1]
                    + ch_idx[0] * iw
                    + kw,
                    i * ow + j,
                ] = kernel_flip[ch_idx[0], ch_idx[1], :, :]

            t = (
                zeros_image[:, :, i * ow + j]
                .reshape(cin * ih, cout, iw)
                .transpose(1, 0, 2)
            )
            for o_ch in range(cout):
                w_unrolled_np[:, i * ow + j + o_ch * out_size] = t[o_ch].ravel()

    w_unrolled_np = w_unrolled_np.reshape((cin, ih, iw, cout, oh, ow))

    # Remove the part of the transpose padding in the w_unrolled_no_padding
    # w_unrolled : (cin*in_size, cout*out_size) -> (cin*nin_size, cout*nout_size), remove (kl - 1) padding
    nih = in_shape[0] + (in_shape[0] - 1) * (stride[0] - 1)
    niw = in_shape[1] + (in_shape[1] - 1) * (stride[1] - 1)
    nin_size = nih * niw
    w_unrolled = np.zeros((cin * nin_size, cout * out_size), dtype=kernel.dtype)
    w_unrolled = w_unrolled.reshape((cin, nih, niw, cout, oh, ow))

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
        cin * in_shape[0] * in_shape[1], cout * out_shape[0] * out_shape[1]
    )

    return w_unrolled


def _convtranspose1d_faster(
    x_cl: NeuOutType,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
    output_padding: Size1Type,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    # (C, L)
    xc, xl = x_cl.shape

    # (O, I, L)
    cout, cin, kl = kernel.shape
    assert xc == cin, "Input channels must match kernel channels."
    assert (xl - 1) * stride[0] - 2 * padding[0] + kl + output_padding[0] == out_shape[
        0
    ]

    # generate new input array
    # inverse stride : Insert 0 between rows and columns
    xc_t = xc
    xl_t = xl + (xl - 1) * (stride[0] - 1)
    x_transpose = np.zeros((xc_t, xl_t), dtype=x_cl.dtype)
    x_transpose[::1, :: stride[0]] = x_cl

    # inverse padding
    # x_transpose : (cin, (xl-1)*(stride-1)+2*(kl-1))
    x_transpose = np.pad(x_transpose, ((0, 0), (kl - 1, kl - 1)))

    # convolution kernel rotated 180 degrees
    kernel_flip = np.flip(kernel, axis=2)
    # kernel: (cout, cin, kl) -> (cin*kl, cout)
    kernel_col = kernel_flip.reshape(cout, -1)

    # col_fm: (cin, nol) -> (nol, cin*kl)
    nol = out_shape[0] - output_padding[0] + 2 * padding[0]
    stride_transpose = (1,)
    col_fm = _1d_im2col(x_transpose, nol, kl, stride_transpose, (1,))

    # (nol, cin*kl) * (cin*kl, cout) = (nol, cout)
    out = col_fm @ kernel_col.T  # + self.bias
    # (nol, cout) -> (cout, nol)
    out = out.T

    # inverse padding : (cout, (xl-1)*stride+kernel) -> (cout, (xl-1)*stride+kernel-2*padding)
    out = out[:, padding[0] : (-1 * padding[0])] if padding[0] > 0 else out

    # output_padding
    out = np.pad(out, ((0, 0), (0, output_padding[0])))

    if bias is not None:
        _bias = bias.squeeze()
        assert _bias.shape == (cout,)

        out += _bias

    return out.astype(VOLTAGE_DTYPE)


def _convtranspose2d_faster(
    x_chw: NeuOutType,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    output_padding: Size2Type,
    bias: Optional[WeightType] = None,
) -> SynOutType:
    # (C, H, W)
    xc, xh, xw = x_chw.shape

    # (O, I, H, W)
    cout, cin, kh, kw = kernel.shape
    assert xc == cin, "Input channels must match kernel channels."

    # Calculate the shape of the padded input (considering stride)
    oh, ow = out_shape
    assert (xh - 1) * stride[0] - 2 * padding[0] + kh + output_padding[0] == oh
    assert (xw - 1) * stride[1] - 2 * padding[1] + kw + output_padding[1] == ow

    # By modifying the input matrix and convolution kernel
    # we can change the transpose convolution to the form of an ordinary convolution

    # Generate the transpose input array : transpose padding 0 & stride 0
    xc_t = xc
    xh_t = xh + (xh - 1) * (stride[0] - 1)
    xw_t = xw + (xw - 1) * (stride[1] - 1)
    x_transpose = np.zeros((xc_t, xh_t, xw_t), dtype=x_chw.dtype)
    x_transpose[::1, :: stride[0], :: stride[1]] = x_chw
    # padding 0 for transpose not for parameter padding, get new input array x_transpose
    x_transpose = np.pad(x_transpose, ((0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)))

    # kernel: (cout, cin, kh, kw) -> (cout, cin*kh*kw)
    kernel_flip = np.flip(kernel, axis=(2, 3))  # convolution kernel rotated 180 degrees
    kernel_col = kernel_flip.reshape(cout, -1)

    # conv
    stride_transpose = (1, 1)
    noh = oh - output_padding[0] + 2 * padding[0]
    now = ow - output_padding[1] + 2 * padding[1]
    col_fm = _2d_im2col(x_transpose, noh, now, kh, kw, stride_transpose, (1, 1))

    # (oh*ow, cin*kh*kw) * (cin*kh*kw, cout) = (oh*ow, cout)
    out_col = col_fm @ kernel_col.T
    # (oh*ow, cout) -> (oh, ow, cout) -> (cout, oh, ow)
    out = out_col.astype(VOLTAGE_DTYPE).T.reshape((cout,) + (noh, now))

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
        assert _bias.shape == (cout,)

        out += _bias

    return out


def _1d_im2col(
    x_padded: NeuOutType,
    ol: int,
    kl: int,
    stride: Size1Type,
    dilation: Size1Type,
) -> NDArray[np.int64]:
    cin, pl = x_padded.shape
    cols = np.zeros((ol, cin * kl), dtype=np.int64)

    for i in range(ol):
        # Generate the indices for the dilated kernel
        indices = [i * stride[0] + k * dilation[0] for k in range(kl)]
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
    oh: int,
    ow: int,
    kh: int,
    kw: int,
    stride: Size2Type,
    dilation: Size2Type,
) -> NDArray[np.int64]:
    cout, ph, pw = x_padded.shape
    cols = np.zeros((oh * ow, cout * kh * kw), dtype=np.int64)

    idx = 0
    for i in range(oh):
        for j in range(ow):
            # Generate the indices for the dilated kernel
            h_indices = [i * stride[0] + m * dilation[0] for m in range(kh)]
            w_indices = [j * stride[1] + n * dilation[1] for n in range(kw)]

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
    c, _ = x_shape
    ol = out_shape[0]
    sl = stride[0]
    dl = dilation[0]

    assert c % groups == 0, f"Input channels {c} must be divisible by groups {groups}"
    c_per_grp = c // groups

    i0 = np.arange(kl) * dl
    i0 = np.tile(i0, c_per_grp)
    i1 = sl * np.arange(ol)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    k = np.repeat(np.arange(c_per_grp), kl).reshape(-1, 1)

    return k, i, c_per_grp


def get_im2col_indices_2d(
    x_shape: Size3Type,
    out_shape: Size2Type,
    kh: int,
    kw: int,
    stride: Size2Type = (1, 1),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
):
    c, _, _ = x_shape
    ho, wo = out_shape
    sh, sw = stride
    dh, dw = dilation

    assert c % groups == 0, f"Input channels {c} must be divisible by groups {groups}"
    c_per_grp = c // groups

    i0 = np.repeat(np.arange(kh) * dh, kw)
    i0 = np.tile(i0, c_per_grp)
    i1 = sh * np.repeat(np.arange(ho), wo)
    j0 = np.tile(np.arange(kw) * dw, kh * c_per_grp)
    j1 = sw * np.tile(np.arange(wo), ho)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c_per_grp), kh * kw).reshape(-1, 1)

    return k, i, j, c_per_grp


def _im2col_indices_groupwise(
    x_padded: np.ndarray,
    indices: Union[Size2Type, Size3Type],
    c_per_grp: int,
    groups: int,
    reshape_shape: Size2Type,
    slice_dims: int,
):
    cols_list = []

    for g in range(groups):
        ci_start = g * c_per_grp
        ci_end = (g + 1) * c_per_grp

        if slice_dims == 2:
            x_grp = x_padded[ci_start:ci_end, :]
            k, i = indices
            cols_grp = x_grp[k, i]
        elif slice_dims == 3:
            x_grp = x_padded[ci_start:ci_end, :, :]
            k, i, j = indices
            cols_grp = x_grp[k, i, j]
        else:
            raise ValueError(f"Unsupported slice_dims {slice_dims} for im2col helper.")

        cols_grp = cols_grp.reshape(reshape_shape)
        cols_list.append(cols_grp)

    return np.stack(cols_list)


def im2col_indices_1d(
    x: np.ndarray,
    kl: int,
    stride: Size1Type = (1,),
    padding: Size1Type = (0,),
    dilation: Size1Type = (1,),
    groups: int = 1,
    out_shape: Optional[Size1Type] = None,
):
    x_padded = np.pad(x, ((0, 0), (padding[0], padding[0])), mode="constant")
    if out_shape is None:
        ol = (x_padded.shape[1] - dilation[0] * (kl - 1) - 1) // stride[0] + 1
        out_shape = (ol,)

    k, i, c_per_grp = get_im2col_indices_1d(
        x.shape, out_shape, kl, stride, dilation, groups
    )

    return _im2col_indices_groupwise(
        x_padded, (k, i), c_per_grp, groups, (kl * c_per_grp, -1), slice_dims=2
    )


def im2col_indices_2d(
    x: np.ndarray,
    kh: int,
    kw: int,
    stride: Size2Type = (1, 1),
    padding: Size2Type = (0, 0),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
    out_shape: Optional[Size2Type] = None,
):
    """An implementation of im2col based on some fancy indexing"""
    x_padded = np.pad(
        x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    if out_shape is None:
        ho = (x_padded.shape[1] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        wo = (x_padded.shape[2] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        out_shape = (ho, wo)

    k, i, j, c_per_grp = get_im2col_indices_2d(
        x.shape, out_shape, kh, kw, stride, dilation, groups
    )

    return _im2col_indices_groupwise(
        x_padded, (k, i, j), c_per_grp, groups, (kh * kw * c_per_grp, -1), slice_dims=3
    )


def _pool1d_kernel_unroll(
    channels: int,
    in_shape: Size1Type,
    out_shape: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
) -> WeightType:
    kl = ksize[0]
    il = in_shape[0] + 2 * padding[0]
    ol = out_shape[0]

    w_unrolled_np = np.zeros((channels * il, channels * ol), dtype=WEIGHT_DTYPE)
    zeros_image = np.zeros((channels * il, channels), dtype=WEIGHT_DTYPE)

    for i in range(ol):
        zeros_image.fill(0)
        for i_ch in range(channels):
            zeros_image[
                i * stride[0] + i_ch * il : i * stride[0] + i_ch * il + kl, i_ch
            ] = 1

        temp = zeros_image.T

        for o_ch in range(channels):
            w_unrolled_np[:, i + o_ch * ol] = temp[o_ch].ravel()

    if padding == (0,):
        return w_unrolled_np

    nil = in_shape[0]
    w_unrolled = np.zeros((channels * nil, channels * ol), dtype=WEIGHT_DTYPE)

    for i in range(channels):
        w_unrolled[i * nil : i * nil + nil, :] = w_unrolled_np[
            i * il + padding[0] : i * il - padding[0] + il, :
        ]

    return w_unrolled


def _pool2d_kernel_unroll(
    channels: int,
    in_shape: Size2Type,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
) -> WeightType:
    kh, kw = ksize
    ih = in_shape[0] + 2 * padding[0]
    iw = in_shape[1] + 2 * padding[1]
    oh, ow = out_shape
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled_np = np.zeros(
        (channels * in_size, channels * out_size), dtype=WEIGHT_DTYPE
    )
    zeros_image = np.zeros((channels * ih, iw * channels), dtype=WEIGHT_DTYPE)

    for i in range(oh):
        for j in range(ow):
            zeros_image.fill(0)
            for i_ch in range(channels):
                zeros_image[
                    i * stride[0] + i_ch * ih : i * stride[0] + i_ch * ih + kh,
                    j * stride[1] + i_ch * iw : j * stride[1] + i_ch * iw + kw,
                ] = 1

            temp = zeros_image.reshape((channels * ih, channels, iw)).transpose(1, 0, 2)

            for o_ch in range(channels):
                w_unrolled_np[:, i * ow + j + o_ch * out_size] = temp[o_ch].ravel()

    if padding == (0, 0):
        return w_unrolled_np

    nih, niw = in_shape
    nin_size = nih * niw
    w_unrolled = np.zeros(
        (channels * nin_size, channels * out_size), dtype=WEIGHT_DTYPE
    )

    for i in range(channels):
        for j in range(nih):
            w_unrolled[i * nin_size + j * niw : i * nin_size + j * niw + niw, :] = (
                w_unrolled_np[
                    i * in_size
                    + (padding[0] + j) * iw
                    + padding[1] : i * in_size
                    + (padding[0] + j) * iw
                    + padding[1]
                    + niw,
                    :,
                ]
            )

    return w_unrolled


def _func_pool1d(
    x_cl: NeuOutType,
    out_shape: Size1Type,
    ksize: Size1Type,
    stride: Size1Type,
    padding: Size1Type,
    type: str,
    threshold: int,
) -> NeuOutType:
    xcin, xl = x_cl.shape
    kl = ksize[0]
    ol = out_shape[0]
    cout = xcin

    assert (xl + padding[0] * 2 - kl) // stride[0] + 1 == ol

    out = np.zeros((cout, ol), dtype=np.int32)

    if padding > (0,):
        x_padded = np.pad(x_cl, ((0, 0), (padding[0], padding[0])))
    else:
        x_padded = x_cl

    for c in range(cout):
        for i in range(ol):
            if type == "avg":
                out[c, i] = np.sum(x_padded[c, stride[0] * i : stride[0] * i + kl])
            else:
                out[c, i] = np.max(x_padded[c, stride[0] * i : stride[0] * i + kl])

    if type == "avg":
        result = out >= threshold
    else:
        result = out

    return result.astype(NEUOUT_U8_DTYPE)


def _func_pool2d(
    x_chw: NeuOutType,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    type: str,
    threshold: int,
) -> NeuOutType:
    xcin, xh, xw = x_chw.shape
    kh, kw = ksize
    oh, ow = out_shape
    cout = xcin

    assert (xh + padding[0] * 2 - kh) // stride[0] + 1 == oh
    assert (xw + padding[1] * 2 - kw) // stride[1] + 1 == ow

    out = np.zeros((cout, oh, ow), dtype=np.int32)

    if padding > (0, 0):
        x_padded = np.pad(
            x_chw,
            ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        )
    else:
        x_padded = x_chw

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                if type == "avg":
                    out[c, i, j] = np.sum(
                        x_padded[
                            c,
                            stride[0] * i : stride[0] * i + kh,
                            stride[1] * j : stride[1] * j + kw,
                        ]
                    )
                else:
                    out[c, i, j] = np.max(
                        x_padded[
                            c,
                            stride[0] * i : stride[0] * i + kh,
                            stride[1] * j : stride[1] * j + kw,
                        ]
                    )

    if type == "avg":
        result = out >= threshold
    else:
        result = out

    return result.astype(NEUOUT_U8_DTYPE)
