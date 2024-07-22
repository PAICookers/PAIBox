from typing import Optional

import numpy as np

from paibox.types import NeuOutType, SpikeType


def maxpool2d_golden(
    x: SpikeType,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
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
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    fm_order: str,
    threshold: int,
) -> SpikeType:
    if fm_order == "HWC":
        _x = x.transpose(2, 0, 1)
    else:
        _x = x

    xcin, ih, iw = _x.shape
    kh, kw = kernel_size
    oh = (ih - kh + 2 * padding[0]) // stride[0] + 1
    ow = (iw - kw + 2 * padding[1]) // stride[1] + 1
    cout = xcin

    out = np.zeros((cout, oh, ow), dtype=np.int8)
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
                        stride[0] * i : stride[0] * i + kh,
                        stride[1] * j : stride[1] * j + kw,
                    ]
                )

    return out >= threshold


def max_pooling(
    input_data,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> NeuOutType:
    """
    实现最大池化层

    参数:
    input_data (numpy.ndarray): 输入数据,形状为(channels, height, width)
    kernel_size (int): 池化核大小
    stride (int): 步长

    返回:
    numpy.ndarray: 池化后的输出数据,形状为(channels, new_height, new_width)
    """
    channels, height, width = input_data.shape
    new_height = (height - kernel_size[0]) // stride[0] + 1
    new_width = (width - kernel_size[1]) // stride[1] + 1

    output_data = np.zeros((channels, new_height, new_width))

    for c in range(channels):
        for i in range(new_height):
            for j in range(new_width):
                x1 = i * stride[0]
                y1 = j * stride[1]
                x2 = x1 + kernel_size[0]
                y2 = y1 + kernel_size[1]
                output_data[c, i, j] = np.max(input_data[c, x1:x2, y1:y2])

    return output_data


def avg_pooling(
    input_data,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> NeuOutType:
    """
    实现平均池化层

    参数:
    input_data (numpy.ndarray): 输入数据,形状为(batch_size, channels, height, width)
    kernel_size (int): 池化核大小
    stride (int): 步长

    返回:
    numpy.ndarray: 池化后的输出数据,形状为(batch_size, channels, new_height, new_width)
    """
    channels, height, width = input_data.shape
    kernel_height, kernel_width = kernel_size
    new_height = (height - kernel_size[0]) // stride[0] + 1
    new_width = (width - kernel_size[1]) // stride[1] + 1

    output_data = np.zeros((channels, new_height, new_width), dtype=np.int32)

    for c in range(channels):
        for i in range(new_height):
            for j in range(new_width):
                x1 = i * stride[0]
                y1 = j * stride[1]
                x2 = x1 + kernel_size[0]
                y2 = y1 + kernel_size[1]
                output_data[c, i, j] = np.sum(input_data[c, x1:x2, y1:y2]) >> (
                    (kernel_height * kernel_width).bit_length() - 1
                )

    return output_data
