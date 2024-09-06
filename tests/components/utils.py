from typing import Optional, Union, overload

import numpy as np

from paibox.types import (
    NEUOUT_U8_DTYPE,
    SPIKE_DTYPE,
    VOLTAGE_DTYPE,
    WEIGHT_DTYPE,
    NeuOutType,
    SpikeType,
    SynOutType,
)


def conv1d_golden(
    x: np.ndarray,
    out_shape: tuple[int],
    kernel: np.ndarray,
    stride: tuple[int],
    padding: tuple[int],
):
    cout, cin, kl = kernel.shape
    xcin, il = x.shape

    assert cin == xcin

    ol = (il - kl + 2 * padding[0]) // stride[0] + 1

    assert ol == out_shape[0]

    out = np.zeros((cout,) + out_shape, dtype=np.int64)

    x_padded = np.pad(x, ((0, 0), (padding[0], padding[0])), mode="constant")
    conv_result = np.zeros((ol,), dtype=np.int64)

    for o in range(cout):
        for i in range(cin):
            conv_result.fill(0)
            for l in range(ol):
                window = x_padded[i, l * stride[0] : l * stride[0] + kl].astype(
                    np.int64
                )
                conv_result[l] = np.sum(window * kernel[o, i, :])

            out[o] += conv_result

    return out


def conv2d_golden(
    x: np.ndarray,
    out_shape: tuple[int, int],
    kernel: np.ndarray,
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    cout, cin, kh, kw = kernel.shape
    xcin, ih, iw = x.shape

    assert cin == xcin

    oh = (ih - kh + 2 * padding[0]) // stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // stride[1] + 1

    assert oh, ow == out_shape

    out = np.zeros((cout,) + out_shape, dtype=np.int64)

    x_padded = np.pad(
        x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )
    conv_result = np.zeros((oh, ow), dtype=np.int64)

    for o in range(cout):
        for i in range(cin):
            conv_result.fill(0)
            for h in range(oh):
                for w in range(ow):
                    window = x_padded[
                        i,
                        h * stride[0] : h * stride[0] + kh,
                        w * stride[1] : w * stride[1] + kw,
                    ].astype(np.int64)
                    conv_result[h, w] = np.sum(window * kernel[o, i, :, :])

            out[o] += conv_result

    return out


def maxpool1d_golden(
    x: SpikeType,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    fm_order: str = "CL",
) -> SpikeType:
    if fm_order == "LC":
        _x = x.T
    else:
        _x = x

    xcin, il = _x.shape
    kl = kernel_size[0]
    _stride = stride if stride is not None else kernel_size
    ol = (il - kl + 2 * padding[0]) // _stride[0] + 1
    cout = xcin

    out = np.zeros((cout, ol), dtype=x.dtype)
    x_padded = np.pad(
        _x,
        ((0, 0), (padding[0], padding[0])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(ol):
            out[c, i] = np.max(x_padded[c, _stride[0] * i : _stride[0] * i + kl])

    return out


@overload
def maxpool2d_golden(
    x: SpikeType,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    fm_order: str = "CHW",
) -> SpikeType: ...


@overload
def maxpool2d_golden(
    x: NeuOutType,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    fm_order: str = "CHW",
) -> SynOutType: ...


def maxpool2d_golden(
    x: Union[NeuOutType, SpikeType],
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    fm_order: str = "CHW",
) -> Union[SynOutType, SpikeType]:
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

    if x.dtype == NEUOUT_U8_DTYPE:
        # Treat the result as voltage since it will be turncated later.
        out = np.zeros((cout, oh, ow), dtype=VOLTAGE_DTYPE)
    else:
        out = np.zeros((cout, oh, ow), dtype=SPIKE_DTYPE)

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


def avgpool1d_golden(
    x: SpikeType,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    threshold: int,
    fm_order: str = "CL",
) -> SpikeType:
    if fm_order == "LC":
        _x = x.T
    else:
        _x = x

    xcin, il = _x.shape
    kl = kernel_size[0]
    _stride = stride if stride is not None else kernel_size
    ol = (il - kl + 2 * padding[0]) // _stride[0] + 1
    cout = xcin

    out = np.zeros((cout, ol), dtype=WEIGHT_DTYPE)
    x_padded = np.pad(
        _x,
        ((0, 0), (padding[0], padding[0])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(ol):
            out[c, i] = np.sum(x_padded[c, _stride[0] * i : _stride[0] * i + kl])

    return out >= threshold


@overload
def avgpool2d_golden(
    x: SpikeType,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    threshold: int,
    fm_order: str = "CHW",
) -> SpikeType: ...


@overload
def avgpool2d_golden(
    x: NeuOutType,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    threshold: None = None,
    fm_order: str = "CHW",
) -> SynOutType: ...


def avgpool2d_golden(
    x: Union[NeuOutType, SpikeType],
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    threshold: Optional[int] = None,
    fm_order: str = "CHW",
) -> Union[SynOutType, SpikeType]:
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

    # Treat the result as voltage since it will be turncated or compared later.
    out = np.zeros((cout, oh, ow), dtype=VOLTAGE_DTYPE)
    x_padded = np.pad(
        _x,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                out[c, i, j] = np.sum(
                    x_padded[
                        c,
                        _stride[0] * i : _stride[0] * i + kh,
                        _stride[1] * j : _stride[1] * j + kw,
                    ]
                )

    if threshold:
        return out >= threshold
    else:
        return out >> ((kh * kw).bit_length() - 1)
