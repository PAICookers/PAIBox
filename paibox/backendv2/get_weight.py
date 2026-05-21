from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from numba import njit
from paicorelib import AddPotentialMode, DataWidth, WeightCompressType
from rich.progress import track
from torch import Tensor, nn
from torch.nn import functional as F

from ..paiir.ir import (
    AccumulateOp,
    PotentialAddOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from ..paiir.nn.pool import SumPool1d, SumPool2d
from .op_node import CoreOpNode, Neuron, SourceElem, SourceNode
from .weight import Weight


def feature_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    # Backend weight extraction works on feature dimensions only.
    # The leading batch dimension is stripped and must stay equal to 1.
    feature_shape = tuple(shape)
    if len(feature_shape) > 1:
        batch = feature_shape[0]
        if batch != 1:
            raise NotImplementedError(
                f"Only batch size 1 is supported, but got shape {feature_shape}."
            )
        feature_shape = feature_shape[1:]

    return feature_shape


def to_nd_tuple(value: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        if len(value) != ndim:
            raise ValueError(f"Expected a tuple of length {ndim}, but got {value}.")
        return value

    return (value,) * ndim


def ensure_target_components(target: CoreOpNode) -> None:
    # Lazily reconstruct per-predecessor comps/weights so routing can
    # query them even if node initialization did not populate them yet.
    if not target.predecessors:
        return

    if len(target.comps) == len(target.predecessors) and len(target.weights) == len(
        target.predecessors
    ):
        return

    raw_node = target.raw_node
    if isinstance(raw_node, SequentialOp):
        target.comps = [raw_node.comp]
    elif isinstance(raw_node, AccumulateOp):
        target.comps = list(raw_node.comps)
    elif isinstance(raw_node, StandaloneCompOp):
        target.comps = [raw_node.comp]
    elif isinstance(raw_node, StandaloneActOp):
        target.comps = [None]
    elif isinstance(raw_node, PotentialAddOp):
        target.comps = [None] * len(raw_node.signs)
    else:
        raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")

    raw_weights = raw_node.weights
    if raw_weights is None:
        target.weights = [None] * len(target.comps)
    else:
        target.weights = list(raw_weights)


def path_signs(target: CoreOpNode) -> list[int]:
    # Accumulate/Add nodes may encode subtraction through per-path signs.
    raw_node = target.raw_node
    if isinstance(raw_node, (AccumulateOp, PotentialAddOp)):
        return list(raw_node.signs)

    return [1] * len(target.predecessors)


def direct_weight_matrix(
    weight: Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    sign: int,
) -> np.ndarray:
    # Linear/identity-like paths already expose a dense [out, in] matrix.
    matrix = np.asarray(weight.detach().cpu(), dtype=np.int16)
    n_input = int(np.prod(input_shape))
    n_output = int(np.prod(output_shape))

    if matrix.shape != (n_output, n_input):
        raise ValueError(
            f"Direct weight shape mismatch: expected {(n_output, n_input)}, got {matrix.shape}."
        )

    if sign != 1:
        matrix = sign * matrix

    return matrix


def identity_weight_matrix(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], sign: int
) -> np.ndarray:
    # Activation-only / add-only paths do not own raw parameter tensors, but
    # backend routing still needs their implicit identity connectivity.
    n_input = int(np.prod(input_shape))
    n_output = int(np.prod(output_shape))
    if n_input != n_output:
        raise ValueError(
            f"Identity path shape mismatch: input has {n_input} elems, output has {n_output}."
        )

    matrix = np.eye(n_output, dtype=np.int16)
    if sign != 1:
        matrix *= sign

    return matrix


def unfold_input_indices_2d(
    channels: int,
    spatial_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    # Build a lookup table from each sliding-window position back to the
    # flattened input indices. Zero means "came from padding".
    height, width = spatial_shape
    n_input = channels * height * width
    input_ids = torch.arange(1, n_input + 1, dtype=torch.float64).reshape(
        1, channels, height, width
    )
    patches = F.unfold(input_ids, kernel_size, dilation, padding, stride)
    return patches.to(torch.int64).squeeze(0)


def conv2d_weight_matrix(
    weight: Tensor,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    sign: int,
) -> np.ndarray:
    # Expand a Conv2d kernel into a full dense matrix with shape [out, in],
    # where rows are flattened output neurons and columns are flattened inputs.
    in_channels, height, width = input_shape
    out_channels, out_height, out_width = output_shape
    kernel = np.asarray(weight.detach().cpu(), dtype=np.int16)
    _, in_channels_per_group, kernel_height, kernel_width = kernel.shape

    if in_channels != in_channels_per_group * groups:
        raise ValueError(
            f"Conv groups mismatch: input channels {in_channels}, kernel channels {in_channels_per_group}, groups {groups}."
        )

    out_channels_per_group = out_channels // groups
    out_size = out_height * out_width
    n_input = in_channels * height * width
    matrix = np.zeros((out_channels * out_size, n_input), dtype=np.int16)

    patches = (
        unfold_input_indices_2d(
            in_channels,
            (height, width),
            (kernel_height, kernel_width),
            stride,
            padding,
            dilation,
        )
        .cpu()
        .numpy()
    )

    if patches.shape[1] != out_size:
        raise ValueError(
            f"Conv unfold mismatch: expected {out_size} output positions, got {patches.shape[1]}."
        )

    row_offsets = np.tile(
        np.arange(out_size, dtype=np.int64),
        in_channels_per_group * kernel_height * kernel_width,
    )

    for out_channel in range(out_channels):
        group_idx = out_channel // out_channels_per_group
        patch_start = group_idx * in_channels_per_group * kernel_height * kernel_width
        patch_end = patch_start + in_channels_per_group * kernel_height * kernel_width

        # Scatter each kernel coefficient to the input positions that feed the
        # corresponding flattened output locations.
        flat_cols = patches[patch_start:patch_end].reshape(-1)
        valid = flat_cols > 0
        flat_rows = row_offsets[valid]
        flat_values = np.repeat(kernel[out_channel].reshape(-1), out_size)[valid]
        matrix[out_channel * out_size + flat_rows, flat_cols[valid] - 1] = flat_values

    if sign != 1:
        matrix *= sign

    return matrix


def pool2d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    sign: int,
) -> np.ndarray:
    # Pooling has no explicit weight tensor, but connectivity still forms a
    # dense [out, in] matrix. Every valid input in the pooling window gets 1.
    in_channels, height, width = input_shape
    out_channels, out_height, out_width = output_shape
    if in_channels != channels or out_channels != channels:
        raise ValueError(
            f"Pooling channel mismatch: input={in_channels}, output={out_channels}, channels={channels}."
        )

    out_size = out_height * out_width
    n_input = in_channels * height * width
    matrix = np.zeros((out_channels * out_size, n_input), dtype=np.int16)

    patches = (
        unfold_input_indices_2d(
            in_channels, (height, width), kernel_size, stride, padding, dilation
        )
        .cpu()
        .numpy()
    )

    if patches.shape[1] != out_size:
        raise ValueError(
            f"Pooling unfold mismatch: expected {out_size} output positions, got {patches.shape[1]}."
        )

    kernel_elems = int(np.prod(kernel_size))
    row_offsets = np.tile(np.arange(out_size, dtype=np.int64), kernel_elems)

    for channel in range(channels):
        patch_start = channel * kernel_elems
        patch_end = patch_start + kernel_elems

        flat_cols = patches[patch_start:patch_end].reshape(-1)
        valid = flat_cols > 0
        matrix[channel * out_size + row_offsets[valid], flat_cols[valid] - 1] = sign

    return matrix


def conv1d_weight_matrix(
    weight: Tensor,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    groups: int,
    sign: int,
) -> np.ndarray:
    # Reuse the 2D expansion path by treating 1D signals as H=1 feature maps.
    in_channels, in_length = input_shape
    out_channels, out_length = output_shape
    kernel = weight.reshape(weight.shape[0], weight.shape[1], 1, weight.shape[2])
    return conv2d_weight_matrix(
        kernel,
        (in_channels, 1, in_length),
        (out_channels, 1, out_length),
        (1, stride[0]),
        (0, padding[0]),
        (1, dilation[0]),
        groups,
        sign,
    )


def pool1d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    kernel_size: tuple[int],
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    sign: int,
) -> np.ndarray:
    # Reuse the 2D pooling expansion path by treating 1D signals as H=1 maps.
    in_channels, in_length = input_shape
    out_channels, out_length = output_shape
    return pool2d_weight_matrix(
        channels,
        (in_channels, 1, in_length),
        (out_channels, 1, out_length),
        (1, kernel_size[0]),
        (1, stride[0]),
        (0, padding[0]),
        (1, dilation[0]),
        sign,
    )


def _adaptive_pool_window_bounds(
    output_pos: int, input_size: int, output_size: int
) -> tuple[int, int]:
    if output_size <= 0:
        raise ValueError(
            f"Adaptive pooling output size must be positive, got {output_size}."
        )
    if input_size <= 0:
        raise ValueError(
            f"Adaptive pooling input size must be positive, got {input_size}."
        )
    if output_pos < 0 or output_pos >= output_size:
        raise ValueError(
            f"Adaptive pooling output position {output_pos} is outside [0, {output_size})."
        )

    start = output_pos * input_size // output_size
    end = ((output_pos + 1) * input_size + output_size - 1) // output_size
    return start, end


def adaptive_maxpool1d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    sign: int,
) -> np.ndarray:
    in_channels, in_length = input_shape
    out_channels, out_length = output_shape
    if in_channels != channels or out_channels != channels:
        raise ValueError(
            f"AdaptiveMaxPool1d channel mismatch: input={in_channels}, output={out_channels}, channels={channels}."
        )

    n_input = in_channels * in_length
    matrix = np.zeros((out_channels * out_length, n_input), dtype=np.int16)

    for channel in range(channels):
        for out_pos in range(out_length):
            start, end = _adaptive_pool_window_bounds(out_pos, in_length, out_length)
            row = channel * out_length + out_pos
            for in_pos in range(start, end):
                col = channel * in_length + in_pos
                matrix[row, col] = sign

    return matrix


def adaptive_maxpool2d_weight_matrix(
    channels: int,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    sign: int,
) -> np.ndarray:
    in_channels, in_height, in_width = input_shape
    out_channels, out_height, out_width = output_shape
    if in_channels != channels or out_channels != channels:
        raise ValueError(
            f"AdaptiveMaxPool2d channel mismatch: input={in_channels}, output={out_channels}, channels={channels}."
        )

    out_size = out_height * out_width
    n_input = in_channels * in_height * in_width
    matrix = np.zeros((out_channels * out_size, n_input), dtype=np.int16)

    for channel in range(channels):
        for out_h in range(out_height):
            h_start, h_end = _adaptive_pool_window_bounds(out_h, in_height, out_height)
            for out_w in range(out_width):
                w_start, w_end = _adaptive_pool_window_bounds(
                    out_w, in_width, out_width
                )
                row = channel * out_size + out_h * out_width + out_w
                for in_h in range(h_start, h_end):
                    for in_w in range(w_start, w_end):
                        col = channel * in_height * in_width + in_h * in_width + in_w
                        matrix[row, col] = sign

    return matrix


def expanded_path_weight_matrix(
    predecessor: SourceNode,
    target: CoreOpNode,
    comp: nn.Module | None,
    weight: Tensor | None,
    sign: int,
) -> np.ndarray:
    # Convert one predecessor path into a dense [out, in] matrix, choosing
    # the correct expansion strategy from the comp type.
    input_shape = feature_shape(predecessor.shape)
    output_shape = feature_shape(target.shape)

    if comp is None:
        return identity_weight_matrix(input_shape, output_shape, sign)

    if weight is not None and (isinstance(comp, nn.Linear) or weight.ndim == 2):
        return direct_weight_matrix(weight, input_shape, output_shape, sign)

    if isinstance(comp, nn.Conv1d):
        print(
            f"\tExpanding Conv1d weight from {input_shape} to {output_shape} with stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}, groups={comp.groups}."
        )
        assert weight is not None
        assert not isinstance(
            comp.padding, str
        ), "Unsupported padding mode for weight extraction."
        stride = to_nd_tuple(comp.stride, 1)
        padding = to_nd_tuple(comp.padding, 1)
        dilation = to_nd_tuple(comp.dilation, 1)
        assert len(input_shape) == 2
        assert len(output_shape) == 2
        assert len(stride) == 1
        assert len(padding) == 1
        assert len(dilation) == 1
        return conv1d_weight_matrix(
            weight,
            input_shape,
            output_shape,
            stride,
            padding,
            dilation,
            comp.groups,
            sign,
        )

    if isinstance(comp, nn.Conv2d):
        print(
            f"\tExpanding Conv2d weight from {input_shape} to {output_shape} with stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}, groups={comp.groups}."
        )
        assert weight is not None
        assert not isinstance(
            comp.padding, str
        ), "Unsupported padding mode for weight extraction."
        stride = to_nd_tuple(comp.stride, 2)
        padding = to_nd_tuple(comp.padding, 2)
        dilation = to_nd_tuple(comp.dilation, 2)
        assert len(input_shape) == 3
        assert len(output_shape) == 3
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2
        return conv2d_weight_matrix(
            weight,
            input_shape,
            output_shape,
            stride,
            padding,
            dilation,
            comp.groups,
            sign,
        )

    if isinstance(comp, nn.AdaptiveMaxPool1d):
        print(
            f"\tExpanding AdaptiveMaxPool1d from {input_shape} to {output_shape} with output_size={comp.output_size}."
        )
        assert len(input_shape) == 2
        assert len(output_shape) == 2
        return adaptive_maxpool1d_weight_matrix(
            input_shape[0], input_shape, output_shape, sign
        )

    if isinstance(comp, nn.AdaptiveMaxPool2d):
        print(
            f"\tExpanding AdaptiveMaxPool2d from {input_shape} to {output_shape} with output_size={comp.output_size}."
        )
        assert len(input_shape) == 3
        assert len(output_shape) == 3
        return adaptive_maxpool2d_weight_matrix(
            input_shape[0], input_shape, output_shape, sign
        )

    if isinstance(comp, nn.MaxPool1d):
        print(
            f"\tExpanding MaxPool1d from {input_shape} to {output_shape} with kernel_size={comp.kernel_size}, stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}."
        )
        if comp.ceil_mode:
            raise NotImplementedError("MaxPool1d with ceil_mode=True is not supported.")

        kernel_size = to_nd_tuple(comp.kernel_size, 1)
        stride = to_nd_tuple(comp.stride or comp.kernel_size, 1)
        padding = to_nd_tuple(comp.padding, 1)
        dilation = to_nd_tuple(comp.dilation, 1)
        assert len(input_shape) == 2
        assert len(output_shape) == 2
        assert len(kernel_size) == 1
        assert len(stride) == 1
        assert len(padding) == 1
        assert len(dilation) == 1

        return pool1d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            sign,
        )

    if isinstance(comp, SumPool1d) or isinstance(comp, nn.AvgPool1d):
        print(
            f"\tExpanding AvgPool1d from {input_shape} to {output_shape} with kernel_size={comp.kernel_size}, stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}."
        )
        if comp.ceil_mode:
            raise NotImplementedError("AvgPool1d with ceil_mode=True is not supported.")
        kernel_size = to_nd_tuple(comp.kernel_size, 1)
        stride = to_nd_tuple(comp.stride or comp.kernel_size, 1)
        padding = to_nd_tuple(comp.padding, 1)
        assert len(input_shape) == 2
        assert len(output_shape) == 2
        assert len(kernel_size) == 1
        assert len(stride) == 1
        assert len(padding) == 1

        return pool1d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding,
            (1,),
            sign,
        )

    if isinstance(comp, nn.MaxPool2d):
        print(
            f"\tExpanding MaxPool2d from {input_shape} to {output_shape} with kernel_size={comp.kernel_size}, stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}."
        )
        if comp.ceil_mode:
            raise NotImplementedError("MaxPool2d with ceil_mode=True is not supported.")
        kernel_size = to_nd_tuple(comp.kernel_size, 2)
        stride = to_nd_tuple(comp.stride or comp.kernel_size, 2)
        padding = to_nd_tuple(comp.padding, 2)
        dilation = to_nd_tuple(comp.dilation, 2)
        assert len(input_shape) == 3
        assert len(output_shape) == 3
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2

        return pool2d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            sign,
        )

    if isinstance(comp, nn.AvgPool2d) or isinstance(comp, SumPool2d):
        print(
            f"\tExpanding AvgPool2d from {input_shape} to {output_shape} with kernel_size={comp.kernel_size}, stride={comp.stride}, padding={comp.padding}, dilation={comp.dilation}."
        )
        if comp.ceil_mode:
            raise NotImplementedError("AvgPool2d with ceil_mode=True is not supported.")
        kernel_size = to_nd_tuple(comp.kernel_size, 2)
        stride = to_nd_tuple(comp.stride or comp.kernel_size, 2)
        padding = to_nd_tuple(comp.padding, 2)
        assert len(input_shape) == 3
        assert len(output_shape) == 3
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2

        return pool2d_weight_matrix(
            input_shape[0],
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding,
            (1, 1),
            sign,
        )

    raise NotImplementedError(
        f"Unsupported weight expansion for comp {type(comp)} with weight {type(weight)}."
    )


def build_weights(
    raw_neus: list[Neuron],
    input_neus: list[SourceElem],
    target_cache: dict[CoreOpNode, dict[SourceNode, np.ndarray]],
    weights: np.ndarray,
) -> None:
    """
    根据 raw_neus 和 input_neus，从 target_cache 中提取权重矩阵，填充到 weights。

    参数：
        raw_neus: List[Neuron]
        input_neus: List[Neuron]
        target_cache: Dict[target -> Dict[target -> np.ndarray]]
        weights: np.ndarray (shape: [len(raw_neus), len(input_neus)])
    """
    input_group = defaultdict(list)

    for j, elem in enumerate(input_neus):
        input_group[elem.target].append((j, elem.index.idx))

    input_map = {}

    for tgt, elems in input_group.items():
        js = np.fromiter((j for j, _ in elems), dtype=np.int64)
        idxs = np.fromiter((idx for _, idx in elems), dtype=np.int64)
        input_map[tgt] = (js, idxs)

    for i, neu in enumerate(raw_neus):
        path_matrices = target_cache.get(neu.target)
        if path_matrices is None:
            continue

        neu_idx = neu.index.idx

        for tgt, (js, idxs) in input_map.items():
            matrix = path_matrices.get(tgt)
            if matrix is None:
                continue

            weights[i, js] = matrix[neu_idx, idxs]


@njit
def _fill_weights_numba(
    weights,
    matrices,
    out_i_arrays,
    out_idx_arrays,
    in_j_arrays,
    in_idx_arrays,
):
    n_tasks = len(matrices)

    for t in range(n_tasks):
        matrix = matrices[t]

        out_is = out_i_arrays[t]
        out_idxs = out_idx_arrays[t]

        in_js = in_j_arrays[t]
        in_idxs = in_idx_arrays[t]

        # 双循环（Numba会编译成高效代码）
        for oi in range(out_is.shape[0]):
            i = out_is[oi]
            mi = out_idxs[oi]

            for ij in range(in_js.shape[0]):
                j = in_js[ij]
                mj = in_idxs[ij]

                weights[i, j] = matrix[mi, mj]


def build_weights_numba(
    raw_neus: list[Neuron],
    input_neus: list[SourceElem],
    target_cache: dict[CoreOpNode, dict[SourceNode, np.ndarray]],
    weights: np.ndarray,
):
    # -------- 1. input grouping --------
    input_group = defaultdict(list)
    for j, elem in enumerate(input_neus):
        input_group[elem.target].append((j, elem.index.idx))

    input_map = {}
    for tgt, elems in input_group.items():
        js = np.fromiter((j for j, _ in elems), dtype=np.int32)
        idxs = np.fromiter((idx for _, idx in elems), dtype=np.int32)
        input_map[tgt] = (js, idxs)

    # -------- 2. output grouping --------
    output_group = defaultdict(list)
    for i, neu in enumerate(raw_neus):
        output_group[neu.target].append((i, neu.index.idx))

    output_map = {}
    for tgt, elems in output_group.items():
        is_ = np.fromiter((i for i, _ in elems), dtype=np.int32)
        idxs = np.fromiter((idx for _, idx in elems), dtype=np.int32)
        output_map[tgt] = (is_, idxs)

    # -------- 3. flatten cache + 构建任务 --------
    matrices = []
    out_i_arrays = []
    out_idx_arrays = []
    in_j_arrays = []
    in_idx_arrays = []

    for out_tgt, (out_is, out_idxs) in output_map.items():
        path_matrices = target_cache.get(out_tgt)
        if path_matrices is None:
            continue

        for in_tgt, (in_js, in_idxs) in input_map.items():
            matrix = path_matrices.get(in_tgt)
            if matrix is None:
                continue

            matrices.append(matrix)
            out_i_arrays.append(out_is)
            out_idx_arrays.append(out_idxs)
            in_j_arrays.append(in_js)
            in_idx_arrays.append(in_idxs)

    if not matrices:
        return

    # -------- 4. 转为 numba 可用结构 --------
    matrices = list(matrices)
    out_i_arrays = list(out_i_arrays)
    out_idx_arrays = list(out_idx_arrays)
    in_j_arrays = list(in_j_arrays)
    in_idx_arrays = list(in_idx_arrays)

    # -------- 5. 调用 numba --------
    _fill_weights_numba(
        weights,
        matrices,
        out_i_arrays,
        out_idx_arrays,
        in_j_arrays,
        in_idx_arrays,
    )


def get_raw_weights(raw_neus: list[Neuron], input_neus: list[SourceElem]) -> np.ndarray:
    # Return a dense routing-group weight matrix with shape
    # [number of output neurons, number of input elements].
    n_output = len(raw_neus)
    n_input = len(input_neus)
    weights = np.zeros((n_output, n_input), dtype=np.int16)

    # Cache expanded matrices per target node and predecessor node because
    # many raw neurons share the same source/target pair.
    target_cache: dict[CoreOpNode, dict[SourceNode, np.ndarray]] = {}

    for neu in raw_neus:
        target = neu.target
        if target in target_cache:
            continue

        ensure_target_components(target)
        signs = path_signs(target)
        path_matrices: dict[SourceNode, np.ndarray] = {}

        for predecessor, comp, weight, sign in zip(
            target.predecessors, target.comps, target.weights, signs
        ):
            # Each predecessor contributes one dense [target_out, pred_out] block.
            matrix = expanded_path_weight_matrix(
                predecessor, target, comp, weight, sign
            )
            if predecessor in path_matrices:
                path_matrices[predecessor] = path_matrices[predecessor] + matrix
            else:
                path_matrices[predecessor] = matrix

        target_cache[target] = path_matrices

    build_weights_numba(
        raw_neus,
        input_neus,
        target_cache,
        weights,
    )
    return weights


def choose_weight_strategy(
    weight_of_neu: np.ndarray,
    weight_width: DataWidth,
    input_width: DataWidth,
    add_potential: AddPotentialMode,
) -> tuple[Weight, WeightCompressType]:
    w_dense = Weight(
        weight_of_neu,
        WeightCompressType.DENSE,
        weight_width,
        input_width,
        add_potential,
    )
    w_sparse = Weight(
        weight_of_neu,
        WeightCompressType.SPARSE,
        weight_width,
        input_width,
        add_potential,
    )

    if w_sparse.n_sram_required < w_dense.n_sram_required:
        return w_sparse, WeightCompressType.SPARSE
    else:
        return w_dense, WeightCompressType.DENSE


@dataclass
class WeightInfo:
    index: int  # 对应 base weight 的索引
    offset: int  # 左移了多少位


def group_shift_weights_optimized(
    raw_weights: list[np.ndarray], prefix: str = ""
) -> tuple[list[WeightInfo], list[np.ndarray]]:
    if not raw_weights:
        return [], []

    # 1. 预转换成 2D 矩阵以利用向量化计算偏移量
    # 注意：假设 raw_weights 中的 ndarray 长度一致
    matrix = np.vstack(raw_weights)
    n_weights, n_axons = matrix.shape
    dtype = matrix.dtype

    # 2. 向量化查找第一个非零索引 (offset)
    # mask 标记非零位置，argmax 返回第一个 True 的索引
    mask = matrix != 0
    has_nonzero = mask.any(axis=1)
    # 如果全为 0，offset 默认为 0
    offsets = np.where(has_nonzero, mask.argmax(axis=1), 0)

    key_to_index = {}
    base_weights = []
    infos = []

    # 预分配一个全零数组模板，减少循环内的 np.zeros 调用
    zero_template = np.zeros(n_axons, dtype=dtype)

    for i in track(
        range(n_weights),
        description=f"{prefix}computing base weights",
        total=len(range(n_weights)),
    ):
        row = matrix[i]
        offset = int(offsets[i])

        # 3. 构造 Base 权重
        if not has_nonzero[i]:
            base = zero_template.copy()
        else:
            # 这里的切片和赋值是高效的
            base = np.zeros(n_axons, dtype=dtype)
            base[: n_axons - offset] = row[offset:]

        # 4. 关键：使用 tobytes() 作为字典的键，比 tuple() 快一个数量级
        key = base.tobytes()

        if key not in key_to_index:
            idx = len(base_weights)
            key_to_index[key] = idx
            base_weights.append(base)
        else:
            idx = key_to_index[key]

        infos.append(WeightInfo(index=idx, offset=offset))

    return infos, base_weights


def reorder_by_base_weight(
    group_items: list[tuple[Neuron, np.ndarray]], weights_info: list[WeightInfo]
) -> tuple[list[Neuron], list[WeightInfo]]:

    paired = list(zip(group_items, weights_info))

    paired_sorted = sorted(paired, key=lambda x: (x[1].index, x[1].offset))

    reordered_neus = [p[0][0] for p in paired_sorted]
    reordered_infos = [p[1] for p in paired_sorted]

    return reordered_neus, reordered_infos
