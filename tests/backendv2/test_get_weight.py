from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from paibox.backendv2.get_weight import (
    _adaptive_pool_window_bounds,
    adaptive_maxpool1d_weight_matrix,
    adaptive_maxpool2d_weight_matrix,
    expanded_path_weight_matrix,
)
from paibox.backendv2.op_node import CoreOpNode, InNode
from paibox.paiir.ir.ir_base import InputNode
from paibox.paiir.ir.op_node import StandaloneCompOp


def _active_cols(row: np.ndarray) -> set[int]:
    return set(np.flatnonzero(row).tolist())


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


@pytest.mark.parametrize(
    "case", EXPANDED_PATH_CASES, ids=lambda case: type(case.comp).__name__
)
def test_expanded_path_weight_matrix_handles_adaptive_maxpool(case: ExpandedPathCase):
    predecessor = _input_node("input", case.predecessor_shape)
    target = _core_node("pool", case.comp, case.target_shape)
    matrix = expanded_path_weight_matrix(
        predecessor,
        target,
        target.raw_node.comp,
        None,
        1,  # type: ignore
    )

    assert matrix.shape == case.expected_shape
    assert _active_cols(matrix[case.probe_row]) == case.expected_cols
