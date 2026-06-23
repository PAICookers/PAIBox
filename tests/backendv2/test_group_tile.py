import pytest
import torch
from paicorelib import DataWidth

from paibox.backendv2.group_tile import (
    MAX_COMPUTE_FANIN,
    compute_fanin_limit,
    try_tile_conv,
)
from paibox.backendv2.op_node import CoreOpNode, InNode
from paibox.paiir.ir.ir_base import InputNode
from paibox.paiir.ir.op_node import StandaloneCompOp
from paibox.paiir.nn.pool import SumPool2d


def _input_node(name: str, shape: tuple[int, ...]) -> InNode:
    raw_input = InputNode(shape=torch.Size(shape))
    raw_input.name = name
    return InNode(raw_input.name, raw_input, tuple(raw_input.shape))


def _sumpool_node(name: str, shape: tuple[int, ...]) -> CoreOpNode:
    raw_node = StandaloneCompOp(SumPool2d(2))
    raw_node.name = name
    raw_node.core_params.input_width = DataWidth.WIDTH_1BIT
    raw_node.core_params.output_width = DataWidth.WIDTH_4BIT
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    return CoreOpNode(raw_node.name, raw_node, shape)


@pytest.mark.parametrize(
    ("max_fanin", "expected_fanins", "expected_output_ys"),
    [
        (MAX_COMPUTE_FANIN, [8 * 64 * 128, 8 * 64 * 128], [0, 32]),
        (
            compute_fanin_limit(1),
            [8 * 62 * 128, 8 * 62 * 128, 8 * 4 * 128],
            [0, 31, 62],
        ),
    ],
    ids=["full-fanin", "margin-one"],
)
def test_sumpool128_tiling_fanin_limit(
    max_fanin: int, expected_fanins: list[int], expected_output_ys: list[int]
):
    in_node = _input_node("input", (1, 8, 128, 128))
    out_node = _sumpool_node("sumpool", (1, 8, 64, 64))
    in_node.output_bit_num_ = 1
    out_node.input_bit_num_ = 1
    out_node.output_bit_num_ = 4
    in_node.successors.append(out_node)
    out_node.predecessors.append(in_node)

    _, groups = try_tile_conv(in_node, out_node, max_fanin)

    fanins = [
        len(group.input_list) * group.raw_elems[0].input_bit_num for group in groups
    ]
    output_ys = [(group.raw_elems[0].index.idx // 64) % 64 for group in groups]

    assert fanins == expected_fanins
    assert all(fanin <= max_fanin for fanin in fanins)
    assert output_ys == expected_output_ys


def test_compute_fanin_limit_defaults_to_full_hardware_capacity():
    assert MAX_COMPUTE_FANIN == 8 * 64 * 128
    assert compute_fanin_limit() == MAX_COMPUTE_FANIN
    assert compute_fanin_limit(1) == MAX_COMPUTE_FANIN - 1


@pytest.mark.parametrize("margin", [-1, 8 * 64 * 128])
def test_compute_fanin_limit_rejects_invalid_margin(margin):
    with pytest.raises(ValueError, match="fanin margin"):
        compute_fanin_limit(margin)
