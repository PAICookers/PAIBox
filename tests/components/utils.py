from typing import Optional, Union, overload

import numpy as np

from paibox.components.neuron.base import MetaNeuron
from paibox.components.synapses.conv_types import (
    _Size1Type,
    _Size2Type,
    Size1Type,
    Size2Type,
)
from paibox.components.synapses.conv_utils import _single, _pair
from paibox.types import (
    NEUOUT_U8_DTYPE,
    SPIKE_DTYPE,
    VOLTAGE_DTYPE,
    NeuOutType,
    SpikeType,
    SynOutType,
    VoltageType,
)

__all__ = [
    "ann_bit_trunc",
    "conv1d_golden",
    "conv2d_golden",
    "convtranspose1d_golden",
    "convtranspose2d_golden",
    "maxpool1d_golden",
    "maxpool2d_golden",
    "avgpool1d_golden",
    "avgpool2d_golden",
]


def ann_bit_trunc(v_array: VoltageType, bit_trunc: int = 8) -> NeuOutType:
    return np.where(v_array <= 0, 0, MetaNeuron._truncate(v_array, bit_trunc)).astype(
        NEUOUT_U8_DTYPE
    )


def conv1d_golden(
    x: np.ndarray,
    out_shape: Size1Type,
    kernel: np.ndarray,
    stride: _Size1Type = 1,
    padding: _Size1Type = 0,
    dilation: _Size1Type = 1,
    groups: int = 1,
) -> np.ndarray:
    cout, cin_per_grp, kl = kernel.shape
    cin, il = x.shape

    assert cin == cin_per_grp * groups
    assert (
        cin % groups == 0 and cout % groups == 0
    ), f"Input & output channels {cin} & {cout} must be divisible by groups {groups}"
    cout_per_grp = cout // groups

    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    ol = (il + 2 * padding[0] - dilation[0] * (kl - 1) - 1) // stride[0] + 1
    assert ol == out_shape[0]

    if padding > (0,):
        x_padded = np.pad(x, ((0, 0), (padding[0], padding[0])))
    else:
        x_padded = x

    out = np.zeros((cout,) + out_shape, dtype=np.int64)
    conv_result = np.zeros((cout_per_grp, ol), dtype=np.int64)

    for g in range(groups):
        # Get the input & output channels for this group
        cin_start = g * cin_per_grp
        cout_start = g * cout_per_grp
        cout_end = (g + 1) * cout_per_grp

        conv_result.fill(0)
        for o in range(cout_per_grp):
            for i in range(cin_per_grp):
                for l in range(ol):
                    # Calculate input positions with dilation
                    l_pos = [l * stride[0] + k * dilation[0] for k in range(kl)]

                    # Extract window
                    window = x_padded[cin_start + i, l_pos].astype(np.int64)
                    conv_result[o, l] += np.sum(window * kernel[cout_start + o, i, :])

        out[cout_start:cout_end] = conv_result

    return out


def conv2d_golden(
    x: np.ndarray,
    out_shape: Size2Type,
    kernel: np.ndarray,
    stride: _Size2Type = 1,
    padding: _Size2Type = 0,
    dilation: _Size2Type = 1,
    groups: int = 1,
):
    cout, cin_per_grp, kh, kw = kernel.shape
    cin, ih, iw = x.shape

    assert cin == cin_per_grp * groups
    assert (
        cout % groups == 0 and cin % groups == 0
    ), f"Number of channels {cin} & {cout} must be divisible by groups {groups}"
    cout_per_grp = cout // groups

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    oh = (ih + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (iw + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    assert (oh, ow) == out_shape

    if padding > (0, 0):
        x_padded = np.pad(
            x, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        )
    else:
        x_padded = x

    out = np.zeros((cout,) + out_shape, dtype=np.int64)
    conv_result = np.zeros((cout_per_grp, oh, ow), dtype=np.int64)

    for g in range(groups):
        # Get the input & output channels for this group
        cin_start = g * cin_per_grp
        cout_start = g * cout_per_grp
        cout_end = (g + 1) * cout_per_grp

        conv_result.fill(0)
        for o in range(cout_per_grp):
            for i in range(cin_per_grp):
                for h in range(oh):
                    for w in range(ow):
                        # Calculate input positions with dilation
                        h_pos = [h * stride[0] + m * dilation[0] for m in range(kh)]
                        w_pos = [w * stride[1] + n * dilation[1] for n in range(kw)]

                        # Extract window
                        window = x_padded[cin_start + i, h_pos, :][:, w_pos].astype(
                            np.int64
                        )
                        conv_result[o, h, w] += np.sum(
                            window * kernel[cout_start + o, i, :, :]
                        )

        out[cout_start:cout_end] = conv_result

    return out


def convtranspose1d_golden(
    x: np.ndarray,
    out_shape: Size1Type,
    kernel: np.ndarray,
    stride: Size1Type = (1,),
    padding: Size1Type = (0,),
    output_padding: Size1Type = (0,),
    dilation: Size1Type = (1,),
    groups: int = 1,
):
    cout, cin, kl = kernel.shape
    xcin, il = x.shape
    opl = output_padding[0]

    assert cin == xcin

    ol = (il - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kl - 1) + 1 + opl

    assert ol == out_shape[0]

    nol = ol - opl + 2 * padding[0]

    out = np.zeros((cout,) + (nol,), dtype=np.int64)

    # generate new input array : transpose padding 0 & stride 0
    # Insert 0 between rows and columns (for stride)
    xc_t = xcin
    xl_t = il + (il - 1) * (stride[0] - 1)
    x_transpose = np.zeros((xc_t, xl_t), dtype=x.dtype)
    x_transpose[::1, :: stride[0]] = x
    # padding 0 for transpose not for parameter padding, get new input array x_transpose
    if kl > 1:
        x_transpose = np.pad(x_transpose, ((0, 0), (kl - 1, kl - 1)), mode="constant")

    kernel_flip = np.flip(kernel, axis=2)
    stride_transpose = 1
    conv_result = np.zeros((nol,), dtype=np.int64)

    for o in range(cout):
        for i in range(cin):
            conv_result.fill(0)
            for l in range(nol):
                window = x_transpose[
                    i, l * stride_transpose : l * stride_transpose + kl
                ].astype(np.int64)
                conv_result[l] = np.sum(window * kernel_flip[o, i, :])

            out[o] += conv_result

    # inverse padding : (cout, (xl-1)*stride+kernel) -> (cout, (xl-1)*stride+kernel-2*padding)
    out = out[:, padding[0] : (-1 * padding[0])] if padding[0] > 0 else out

    # output_padding
    if opl > 0:
        out = np.pad(out, ((0, 0), (0, opl)), mode="constant")

    return out


def convtranspose2d_golden(
    x: np.ndarray,
    out_shape: Size2Type,
    kernel: np.ndarray,
    stride: Size2Type = (1, 1),
    padding: Size2Type = (0, 0),
    output_padding: Size2Type = (0, 0),
    dilation: Size2Type = (1, 1),
    groups: int = 1,
):
    cout, cin, kh, kw = kernel.shape
    xcin, ih, iw = x.shape

    assert cin == xcin

    oh = (
        (ih - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kh - 1)
        + 1
        + output_padding[0]
    )
    ow = (
        (iw - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kw - 1)
        + 1
        + output_padding[1]
    )

    assert oh, ow == out_shape

    noh = oh - output_padding[0] + 2 * padding[0]
    now = ow - output_padding[1] + 2 * padding[1]

    out = np.zeros((cout,) + (noh, now), dtype=np.int64)

    # Generate the transpose input arrary : transpose padding 0 & stride 0
    xc_t = xcin
    xh_t = ih + (ih - 1) * (stride[0] - 1)
    xw_t = iw + (iw - 1) * (stride[1] - 1)
    x_transpose = np.zeros((xc_t, xh_t, xw_t), dtype=x.dtype)
    x_transpose[::1, :: stride[0], :: stride[1]] = x
    # padding 0 for transpose not for parameter padding, get new input array x_transpose
    if kh > 1 or kw > 1:
        x_transpose = np.pad(
            x_transpose, ((0, 0), (kh - 1, kh - 1), (kw - 1, kw - 1)), mode="constant"
        )

    kernel_flip = np.flip(kernel, axis=(2, 3))
    stride_transpose = (1, 1)
    conv_result = np.zeros((noh, now), dtype=np.int64)

    for o in range(cout):
        for i in range(cin):
            conv_result.fill(0)
            for h in range(noh):
                for w in range(now):
                    window = x_transpose[
                        i,
                        h * stride_transpose[0] : h * stride_transpose[0] + kh,
                        w * stride_transpose[1] : w * stride_transpose[1] + kw,
                    ].astype(np.int64)
                    conv_result[h, w] = np.sum(window * kernel_flip[o, i, :, :])

            out[o] += conv_result

    # inverse padding
    ph_start = padding[0] if padding[0] > 0 else None
    ph_end = (-1 * padding[0]) if padding[0] > 0 else None
    pw_start = padding[1] if padding[1] > 0 else None
    pw_end = (-1 * padding[1]) if padding[1] > 0 else None
    out = out[:, ph_start:ph_end, pw_start:pw_end]

    # output_padding
    if output_padding > (0, 0):
        out = np.pad(
            out,
            ((0, 0), (0, output_padding[0]), (0, output_padding[1])),
            mode="constant",
        )

    return out


@overload
def maxpool1d_golden(
    x: SpikeType,
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    fm_order: str = "CL",
) -> SpikeType: ...


@overload
def maxpool1d_golden(
    x: NeuOutType,
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    fm_order: str = "CL",
) -> SynOutType: ...


def maxpool1d_golden(
    x: Union[NeuOutType, SpikeType],
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    fm_order: str = "CL",
) -> Union[SynOutType, SpikeType]:
    if fm_order == "LC":
        _x = x.T
    else:
        _x = x

    cin, il = _x.shape
    ksize = _single(ksize)
    stride = _single(stride) if stride is not None else ksize
    padding = _single(padding)

    kl = ksize[0]
    ol = (il - kl + 2 * padding[0]) // stride[0] + 1
    cout = cin

    if x.dtype == NEUOUT_U8_DTYPE:
        # Treat the result as voltage since it will be turncated later.
        out = np.zeros((cout, ol), dtype=VOLTAGE_DTYPE)
    else:
        out = np.zeros((cout, ol), dtype=SPIKE_DTYPE)

    if padding > (0,):
        x_padded = np.pad(_x, ((0, 0), (padding[0], padding[0])))
    else:
        x_padded = _x

    for c in range(cout):
        for i in range(ol):
            out[c, i] = np.max(x_padded[c, stride[0] * i : stride[0] * i + kl])

    return out


@overload
def maxpool2d_golden(
    x: SpikeType,
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    fm_order: str = "CHW",
) -> SpikeType: ...


@overload
def maxpool2d_golden(
    x: NeuOutType,
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    fm_order: str = "CHW",
) -> SynOutType: ...


def maxpool2d_golden(
    x: Union[NeuOutType, SpikeType],
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    fm_order: str = "CHW",
) -> Union[SynOutType, SpikeType]:
    if fm_order == "HWC":
        _x = x.transpose(2, 0, 1)
    else:
        _x = x

    cin, ih, iw = _x.shape
    ksize = _pair(ksize)
    stride = _pair(stride) if stride is not None else ksize
    padding = _pair(padding)

    kh, kw = ksize
    oh = (ih - kh + 2 * padding[0]) // stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // stride[1] + 1
    cout = cin

    if x.dtype == NEUOUT_U8_DTYPE:
        # Treat the result as voltage since it will be turncated later.
        out = np.zeros((cout, oh, ow), dtype=VOLTAGE_DTYPE)
    else:
        out = np.zeros((cout, oh, ow), dtype=SPIKE_DTYPE)

    if padding > (0, 0):
        x_padded = np.pad(
            _x, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        )
    else:
        x_padded = _x

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = np.max(
                    x_padded[
                        c,
                        stride[0] * i : stride[0] * i + kh,
                        stride[1] * j : stride[1] * j + kw,
                    ]
                )

    return out


@overload
def avgpool1d_golden(
    x: SpikeType,
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    threshold: int,
    fm_order: str = "CL",
) -> SpikeType: ...


@overload
def avgpool1d_golden(
    x: NeuOutType,
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    threshold=None,
    fm_order: str = "CL",
) -> SynOutType: ...


def avgpool1d_golden(
    x: Union[NeuOutType, SpikeType],
    ksize: _Size1Type,
    stride: Optional[_Size1Type],
    padding: _Size1Type,
    threshold: Optional[int] = None,
    fm_order: str = "CL",
) -> Union[SynOutType, SpikeType]:
    if fm_order == "LC":
        _x = x.T
    else:
        _x = x

    cin, il = _x.shape
    ksize = _single(ksize)
    stride = _single(stride) if stride is not None else ksize
    padding = _single(padding)

    kl = ksize[0]
    ol = (il - kl + 2 * padding[0]) // stride[0] + 1
    cout = cin

    # Treat the result as voltage since it will be turncated or compared later.
    out = np.zeros((cout, ol), dtype=VOLTAGE_DTYPE)

    if padding > (0,):
        x_padded = np.pad(_x, ((0, 0), (padding[0], padding[0])))
    else:
        x_padded = _x

    for c in range(cout):
        for i in range(ol):
            out[c, i] = np.sum(x_padded[c, stride[0] * i : stride[0] * i + kl])

    if threshold:
        assert x.dtype == SPIKE_DTYPE
        out_aft_thres = out >= threshold
    else:
        # Use the bit truncation method to simulate the behavior of the hardware.
        out_aft_thres = out >> (kl.bit_length() - 1)

    if x.dtype == NEUOUT_U8_DTYPE:
        return out_aft_thres.astype(VOLTAGE_DTYPE)
    else:
        return out_aft_thres.astype(SPIKE_DTYPE)


@overload
def avgpool2d_golden(
    x: SpikeType,
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    threshold: int,
    fm_order: str = "CHW",
) -> SpikeType: ...


@overload
def avgpool2d_golden(
    x: NeuOutType,
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    threshold=None,
    fm_order: str = "CHW",
) -> SynOutType: ...


def avgpool2d_golden(
    x: Union[NeuOutType, SpikeType],
    ksize: _Size2Type,
    stride: Optional[_Size2Type],
    padding: _Size2Type,
    threshold: Optional[int] = None,
    fm_order: str = "CHW",
) -> Union[SynOutType, SpikeType]:
    if fm_order == "HWC":
        _x = x.transpose(2, 0, 1)
    else:
        _x = x

    cin, ih, iw = _x.shape
    ksize = _pair(ksize)
    stride = _pair(stride) if stride is not None else ksize
    padding = _pair(padding)

    kh, kw = ksize
    oh = (ih - kh + 2 * padding[0]) // stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // stride[1] + 1
    cout = cin

    # Treat the result as voltage since it will be turncated or compared later.
    out = np.zeros((cout, oh, ow), dtype=VOLTAGE_DTYPE)
    x_padded = np.pad(_x, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])))

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = np.sum(
                    x_padded[
                        c,
                        stride[0] * i : stride[0] * i + kh,
                        stride[1] * j : stride[1] * j + kw,
                    ]
                )

    if threshold:
        assert x.dtype == SPIKE_DTYPE
        out_aft_thres = out >= threshold
    else:
        # Use the bit truncation method to simulate the behavior of the hardware.
        out_aft_thres = out >> ((kh * kw).bit_length() - 1)

    if x.dtype == NEUOUT_U8_DTYPE:
        return out_aft_thres.astype(VOLTAGE_DTYPE)
    else:
        return out_aft_thres.astype(SPIKE_DTYPE)
