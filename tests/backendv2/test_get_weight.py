from dataclasses import dataclass

import numpy as np
import pytest
import torch
from paicorelib import AddPotentialMode, DataWidth, WeightCompressType
from torch import nn
from torch.nn import functional as F

from paibox.backendv2.get_weight import (
    INT32_INDEX_LIMIT,
    _adaptive_pool_window_bounds,
    _index_dtype_for_elems,
    _np_index_dtype,
    _torch_index_dtype,
    adaptive_avgpool1d_weight_matrix,
    adaptive_avgpool2d_weight_matrix,
    adaptive_maxpool1d_weight_matrix,
    adaptive_maxpool2d_weight_matrix,
    choose_weight_strategy,
    conv2d_weight_matrix,
    expanded_path_weight_matrix,
    group_shift_weights_optimized,
    unfold_input_indices_2d,
)
from paibox.backendv2.op_node import CoreOpNode, InNode
from paibox.backendv2.weight import Weight
from paibox.paiir.ir.ir_base import InputNode
from paibox.paiir.ir.op_node import StandaloneCompOp


def _active_cols(row: np.ndarray) -> set[int]:
    return set(np.flatnonzero(row).tolist())


def _row_active_counts(matrix: np.ndarray) -> np.ndarray:
    return np.count_nonzero(matrix, axis=1)


def _adaptive_avgpool_from_connectivity(
    matrix: np.ndarray, input_tensor: torch.Tensor
) -> np.ndarray:
    summed = matrix @ input_tensor.numpy().reshape(-1)
    return summed / _row_active_counts(matrix)


def _input_node(name: str, shape: torch.Size | tuple[int, ...]) -> InNode:
    raw_input = InputNode(shape=torch.Size(shape))
    raw_input.name = name
    return InNode(raw_input.name, raw_input, tuple(raw_input.shape))


def _core_node(name: str, comp: nn.Module, shape: tuple[int, ...]) -> CoreOpNode:
    return CoreOpNode(name, _standalone_comp(comp), shape)


def _standalone_comp(comp: nn.Module) -> StandaloneCompOp:
    raw_node = StandaloneCompOp(comp)
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    return raw_node


def _conv2d_output_shape(
    input_shape: tuple[int, int, int],
    kernel_shape: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    out_channels: int,
) -> tuple[int, int, int]:
    _, height, width = input_shape
    kernel_height, kernel_width = kernel_shape
    out_height = (
        height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1
    ) // stride[0] + 1
    out_width = (
        width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1
    ) // stride[1] + 1
    return out_channels, out_height, out_width


@dataclass(frozen=True)
class ExpandedPathCase:
    predecessor_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    comp: nn.Module
    expected_shape: tuple[int, int]
    probe_row: int
    expected_cols: set[int]


EXPANDED_PATH_CASES = (
    ExpandedPathCase(
        predecessor_shape=(1, 2, 7),
        target_shape=(1, 2, 4),
        comp=nn.AdaptiveMaxPool1d(4),
        expected_shape=(8, 14),
        probe_row=1,
        expected_cols={1, 2, 3},
    ),
    ExpandedPathCase(
        predecessor_shape=(1, 1, 7, 7),
        target_shape=(1, 1, 4, 4),
        comp=nn.AdaptiveMaxPool2d((4, 4)),
        expected_shape=(16, 49),
        probe_row=1,
        expected_cols={1, 2, 3, 8, 9, 10},
    ),
    ExpandedPathCase(
        predecessor_shape=(1, 2, 7),
        target_shape=(1, 2, 4),
        comp=nn.AdaptiveAvgPool1d(4),
        expected_shape=(8, 14),
        probe_row=1,
        expected_cols={1, 2, 3},
    ),
    ExpandedPathCase(
        predecessor_shape=(1, 1, 7, 7),
        target_shape=(1, 1, 4, 4),
        comp=nn.AdaptiveAvgPool2d((4, 4)),
        expected_shape=(16, 49),
        probe_row=1,
        expected_cols={1, 2, 3, 8, 9, 10},
    ),
)


def test_adaptive_pool_window_bounds_matches_pytorch_formula():
    assert [_adaptive_pool_window_bounds(i, 7, 4) for i in range(4)] == [
        (0, 2),
        (1, 4),
        (3, 6),
        (5, 7),
    ]
    assert [_adaptive_pool_window_bounds(i, 8, 4) for i in range(4)] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]


def test_index_dtype_helpers_default_to_int32_and_fallback_to_int64():
    assert _np_index_dtype(INT32_INDEX_LIMIT) == np.int32
    assert _np_index_dtype(INT32_INDEX_LIMIT + 1) == np.int64
    assert _torch_index_dtype(INT32_INDEX_LIMIT) is torch.int32
    assert _torch_index_dtype(INT32_INDEX_LIMIT + 1) is torch.int64
    assert _index_dtype_for_elems({"a": [(1, 2)], "b": [(3, 4)]}) == np.int32
    assert _index_dtype_for_elems({"a": [(1, INT32_INDEX_LIMIT + 1)]}) == np.int64


def test_unfold_input_indices_2d_uses_int32_for_normal_input_size():
    patches = unfold_input_indices_2d(
        1, (4, 4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1)
    )

    assert patches.dtype is torch.int32
    assert patches.shape == (9, 16)
    assert int(patches.max().item()) == 16


def test_conv2d_weight_matrix_matches_grouped_pytorch_conv2d():
    input_shape = (4, 5, 6)
    weight = torch.tensor(
        [
            [[[1, 0], [0, -2], [3, 0]], [[0, 1], [0, 0], [-1, 2]]],
            [[[0, 0], [2, 0], [0, 0]], [[-2, 0], [0, 1], [0, 0]]],
            [[[1, 1], [0, 0], [2, 0]], [[0, -1], [0, 0], [1, 0]]],
            [[[0, 0], [0, 0], [0, 0]], [[1, 0], [0, -1], [0, 2]]],
        ],
        dtype=torch.float32,
    )
    stride = (2, 1)
    padding = (1, 0)
    dilation = (1, 1)
    groups = 2
    output_shape = _conv2d_output_shape(
        input_shape, (3, 2), stride, padding, dilation, out_channels=4
    )

    matrix = conv2d_weight_matrix(
        weight, input_shape, output_shape, stride, padding, dilation, groups, sign=-1
    )
    input_tensor = torch.arange(
        1, 1 + int(np.prod(input_shape)), dtype=torch.float32
    ).reshape(1, *input_shape)

    actual = matrix @ input_tensor.numpy().reshape(-1)
    expected = (
        -F.conv2d(
            input_tensor,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        .numpy()
        .reshape(-1)
    )

    np.testing.assert_array_equal(actual, expected)


def test_adaptive_maxpool2d_weight_matrix_irregular_windows():
    matrix = adaptive_maxpool2d_weight_matrix(1, (1, 7, 7), (1, 4, 4), sign=1)

    assert matrix.shape == (16, 49)
    assert _active_cols(matrix[0]) == {0, 1, 7, 8}
    assert _active_cols(matrix[1]) == {1, 2, 3, 8, 9, 10}
    assert _active_cols(matrix[15]) == {40, 41, 47, 48}


def test_adaptive_maxpool1d_weight_matrix_irregular_windows_and_channels():
    matrix = adaptive_maxpool1d_weight_matrix(2, (2, 7), (2, 4), sign=1)

    assert matrix.shape == (8, 14)
    assert _active_cols(matrix[0]) == {0, 1}
    assert _active_cols(matrix[1]) == {1, 2, 3}
    assert _active_cols(matrix[4]) == {7, 8}
    assert _active_cols(matrix[5]) == {8, 9, 10}


def test_adaptive_maxpool2d_weight_matrix_respects_sign():
    matrix = adaptive_maxpool2d_weight_matrix(1, (1, 4, 4), (1, 2, 2), sign=-1)

    assert set(matrix[0].tolist()) == {-1, 0}
    assert _active_cols(matrix[0]) == {0, 1, 4, 5}


def test_adaptive_maxpool2d_weight_matrix_regular_windows_and_channel_isolation():
    matrix = adaptive_maxpool2d_weight_matrix(2, (2, 8, 8), (2, 4, 4), sign=1)

    assert matrix.shape == (32, 128)
    assert _active_cols(matrix[0]) == {0, 1, 8, 9}
    assert _active_cols(matrix[16]) == {64, 65, 72, 73}
    assert matrix[:16, 64:].sum() == 0
    assert matrix[16:, :64].sum() == 0


def test_adaptive_avgpool2d_weight_matrix_irregular_windows():
    matrix = adaptive_avgpool2d_weight_matrix(1, (1, 7, 7), (1, 4, 4), sign=1)

    assert matrix.shape == (16, 49)
    assert _active_cols(matrix[0]) == {0, 1, 7, 8}
    assert _active_cols(matrix[1]) == {1, 2, 3, 8, 9, 10}
    assert _active_cols(matrix[15]) == {40, 41, 47, 48}


def test_adaptive_avgpool1d_weight_matrix_irregular_windows_and_channels():
    matrix = adaptive_avgpool1d_weight_matrix(2, (2, 7), (2, 4), sign=1)

    assert matrix.shape == (8, 14)
    assert _active_cols(matrix[0]) == {0, 1}
    assert _active_cols(matrix[1]) == {1, 2, 3}
    assert _active_cols(matrix[4]) == {7, 8}
    assert _active_cols(matrix[5]) == {8, 9, 10}


def test_adaptive_avgpool2d_weight_matrix_respects_sign():
    matrix = adaptive_avgpool2d_weight_matrix(1, (1, 4, 4), (1, 2, 2), sign=-1)

    assert set(matrix[0].tolist()) == {-1, 0}
    assert _active_cols(matrix[0]) == {0, 1, 4, 5}


def test_adaptive_avgpool1d_weight_matrix_upsampling_shape():
    matrix = adaptive_avgpool1d_weight_matrix(1, (1, 3), (1, 5), sign=1)

    assert matrix.shape == (5, 3)
    assert [_active_cols(row) for row in matrix] == [{0}, {0, 1}, {1}, {1, 2}, {2}]


def test_adaptive_avgpool1d_matrix_matches_functional_after_window_division():
    input_tensor = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    matrix = adaptive_avgpool1d_weight_matrix(2, (2, 5), (2, 3), sign=1)

    averaged = _adaptive_avgpool_from_connectivity(matrix, input_tensor)
    expected = F.adaptive_avg_pool1d(input_tensor, 3).numpy().reshape(-1)

    np.testing.assert_allclose(averaged, expected)


def test_adaptive_avgpool2d_matrix_matches_pytorch_after_window_division():
    input_tensor = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    matrix = adaptive_avgpool2d_weight_matrix(1, (1, 7, 7), (1, 4, 4), sign=1)

    averaged = _adaptive_avgpool_from_connectivity(matrix, input_tensor)
    expected = F.adaptive_avg_pool2d(input_tensor, (4, 4)).numpy().reshape(-1)

    np.testing.assert_allclose(averaged, expected)


@pytest.mark.parametrize(
    "case", EXPANDED_PATH_CASES, ids=lambda case: type(case.comp).__name__
)
def test_expanded_path_weight_matrix_handles_adaptive_pool(case: ExpandedPathCase):
    predecessor = _input_node("input", case.predecessor_shape)
    target = _core_node("pool", case.comp, case.target_shape)
    matrix = expanded_path_weight_matrix(
        predecessor,
        target,
        target.raw_node.comp,  # type: ignore
        None,
        1,
    )

    assert matrix.shape == case.expected_shape
    assert _active_cols(matrix[case.probe_row]) == case.expected_cols


def _expected_shifted_base(row: np.ndarray) -> tuple[int, np.ndarray]:
    nonzero = np.flatnonzero(row)
    offset = int(nonzero[0]) if nonzero.size else 0
    base = np.zeros_like(row)
    base[: row.size - offset] = row[offset:]
    return offset, base


@pytest.mark.parametrize(
    "raw_weights, expected_base_count",
    [
        (
            [
                np.array([0, 0, 3, 0, 5, 0], dtype=np.int16),
                np.array([0, 3, 0, 5, 0, 0], dtype=np.int16),
            ],
            1,
        ),
        (
            [
                np.zeros(6, dtype=np.int16),
                np.zeros(6, dtype=np.int16),
                np.array([0, 0, 1, 0, 0, 0], dtype=np.int16),
            ],
            2,
        ),
        (
            [
                np.array([0, 1, 2, 0], dtype=np.int8),
                np.array([0, 0, 1, 2], dtype=np.int8),
                np.array([1, 2, 0, 0], dtype=np.int8),
            ],
            1,
        ),
        (
            [
                np.array([0, -3, 0, 5, 0], dtype=np.int16),
                np.array([0, 0, -3, 0, 5], dtype=np.int16),
                np.array([0, 0, 7, 0, 5], dtype=np.int16),
            ],
            2,
        ),
    ],
)
def test_group_shift_weights_preserves_shifted_base_semantics(
    raw_weights: list[np.ndarray], expected_base_count: int
):
    expected = [_expected_shifted_base(row) for row in raw_weights]

    infos, base_weights = group_shift_weights_optimized(raw_weights)

    assert [info.offset for info in infos] == [offset for offset, _ in expected]
    assert len(base_weights) == expected_base_count
    for info, (_offset, expected_base) in zip(infos, expected):
        np.testing.assert_array_equal(base_weights[info.index], expected_base)


def test_group_shift_weights_random_int8_matches_reference():
    rng = np.random.default_rng(12345)
    raw_weights = [rng.integers(-4, 5, size=32, dtype=np.int8) for _ in range(16)]
    raw_weights[3] = np.roll(raw_weights[0], 2)
    raw_weights[3][:2] = 0

    infos, base_weights = group_shift_weights_optimized(raw_weights)

    assert all(base.dtype == np.int8 for base in base_weights)
    for row, info in zip(raw_weights, infos):
        expected_offset, expected_base = _expected_shifted_base(row)
        assert info.offset == expected_offset
        np.testing.assert_array_equal(base_weights[info.index], expected_base)


def test_choose_weight_strategy_sparse_keeps_original_row():
    weight = np.zeros(192, dtype=np.int16)
    weight[[17, 18, 31]] = [3, 4, 5]

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_8BIT, DataWidth.WIDTH_8BIT, AddPotentialMode.NORMAL
    )

    assert selected.compress
    np.testing.assert_array_equal(selected.raw_weights, weight)
    assert compress == WeightCompressType.SPARSE


@pytest.mark.parametrize(
    ("input_width", "last_dense_candidate_index", "expected_compress"),
    [
        (DataWidth.WIDTH_1BIT, 127, WeightCompressType.SPARSE),
        (DataWidth.WIDTH_2BIT, 63, WeightCompressType.SPARSE),
        (DataWidth.WIDTH_4BIT, 31, WeightCompressType.SPARSE),
        (DataWidth.WIDTH_8BIT, 15, WeightCompressType.DENSE),
    ],
    ids=["input-1bit", "input-2bit", "input-4bit", "input-8bit"],
)
def test_choose_weight_strategy_dense_candidate_depends_on_sram_savings(
    input_width,
    last_dense_candidate_index,
    expected_compress,
):
    weight = np.zeros(last_dense_candidate_index + 1, dtype=np.int16)
    weight[[0, last_dense_candidate_index]] = [1, 2]

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_8BIT, input_width, AddPotentialMode.NORMAL
    )

    assert compress == expected_compress
    assert selected.compress is (expected_compress == WeightCompressType.SPARSE)


def test_choose_weight_strategy_dense_tie_remains_dense():
    weight = np.array([1, 2, 3, 4, 5], dtype=np.int16)

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_8BIT, DataWidth.WIDTH_8BIT, AddPotentialMode.NORMAL
    )

    assert not selected.compress
    np.testing.assert_array_equal(selected.raw_weights, [1, 2, 3, 4, 5])
    assert compress == WeightCompressType.DENSE


def test_choose_weight_strategy_single_tap_uses_sparse_when_sram_smaller():
    weight = np.zeros(129, dtype=np.int16)
    weight[128] = 1

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_1BIT, DataWidth.WIDTH_8BIT, AddPotentialMode.NORMAL
    )

    assert selected.compress
    np.testing.assert_array_equal(selected.raw_weights, weight)
    assert compress == WeightCompressType.SPARSE
    assert selected.to_package(weight_skews=(0,)).size == 2


@pytest.mark.parametrize(
    ("compress_type", "weight_skews", "raises"),
    [
        (WeightCompressType.SPARSE, None, True),
        (WeightCompressType.SPARSE, (0,), False),
        (WeightCompressType.DENSE, None, False),
    ],
    ids=["sparse-missing-skew", "sparse-with-skew", "dense-no-skew"],
)
def test_weight_package_skew_contract(compress_type, weight_skews, raises):
    weight = Weight(
        [0, 1, 0] if compress_type == WeightCompressType.SPARSE else [1, 2, 0],
        compress_type=compress_type,
        weight_width=DataWidth.WIDTH_1BIT,
        input_width=DataWidth.WIDTH_1BIT,
    )

    if raises:
        with pytest.raises(ValueError, match="weight_skews"):
            weight.to_package(weight_skews=weight_skews)
        return

    assert weight.to_package(weight_skews=weight_skews).size > 0


def test_choose_weight_strategy_uint8_high_index_uses_sparse_csc_original_row():
    weight = np.zeros(192, dtype=np.int16)
    weight[[144, 146]] = [1, 2]

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_8BIT, DataWidth.WIDTH_8BIT, AddPotentialMode.NORMAL
    )

    assert selected.compress
    np.testing.assert_array_equal(selected.raw_weights, weight)
    assert compress == WeightCompressType.SPARSE


def test_choose_weight_strategy_uint8_long_span_row_still_uses_sparse_if_smaller():
    weight = np.zeros(192, dtype=np.int16)
    weight[[7, 136]] = [1, 2]

    selected, compress = choose_weight_strategy(
        weight, DataWidth.WIDTH_8BIT, DataWidth.WIDTH_8BIT, AddPotentialMode.NORMAL
    )

    assert selected.compress
    np.testing.assert_array_equal(selected.raw_weights, weight)
    assert compress == WeightCompressType.SPARSE
