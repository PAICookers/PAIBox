import torch

from paibox.backendv2.op_node import InNode, RemapNode, get_elem
from paibox.paiir.ir import (
    InputNode,
    LayoutStage,
    ShapeStage,
    TensorLayout,
    TransformOp,
)


def _assert_reorder_map(source: InNode, reorder: RemapNode, expected: dict[int, int]):
    reorder_map = reorder.get_reorder_info()

    for src_idx, dst_idx in expected.items():
        src_elem = get_elem(source, src_idx)
        assert reorder_map[src_elem].index.idx == dst_idx


def test_reorder_node_supports_shape_and_layout_transform_chain():
    raw_input = InputNode(shape=torch.Size((1, 2, 3)))
    source = InNode(raw_input.name, raw_input, raw_input.shape)

    transform = TransformOp(
        (LayoutStage((0, 2, 1)), ShapeStage(lambda _shape: torch.Size((1, 6))))
    )
    transform.input_layouts = (TensorLayout(shape=raw_input.shape, dims=(0, 2, 1)),)
    transform.output_layouts = (TensorLayout(shape=torch.Size((1, 6)), dims=(0, 1)),)

    reorder = RemapNode("ReorderNode_0", transform, transform.output_layouts[0].shape)
    reorder.predecessors = [source]

    _assert_reorder_map(source, reorder, {0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5})


def test_reorder_node_supports_shape_only_transform():
    raw_input = InputNode(shape=torch.Size((1, 2, 3)))
    source = InNode(raw_input.name, raw_input, raw_input.shape)

    transform = TransformOp((ShapeStage(lambda _shape: torch.Size((1, 3, 2))),))
    transform.input_layouts = (TensorLayout(shape=raw_input.shape, dims=(0, 1, 2)),)
    transform.output_layouts = (
        TensorLayout(shape=torch.Size((1, 3, 2)), dims=(0, 1, 2)),
    )

    reorder = RemapNode(
        "ReorderNode_shape_only", transform, transform.output_layouts[0].shape
    )
    reorder.predecessors = [source]

    _assert_reorder_map(source, reorder, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})


def test_reorder_node_supports_explicit_transform_op_permutation():
    raw_input = InputNode(shape=torch.Size((1, 2, 3)))
    source = InNode(raw_input.name, raw_input, raw_input.shape)

    raw_transform = TransformOp((LayoutStage((0, 2, 1)),))
    raw_transform.input_layouts = (TensorLayout(shape=raw_input.shape, dims=(0, 1, 2)),)
    raw_transform.output_layouts = (
        TensorLayout(shape=torch.Size((1, 3, 2)), dims=(0, 2, 1)),
    )

    reorder = RemapNode(
        "ReorderNode_1", raw_transform, raw_transform.output_layouts[0].shape
    )
    reorder.predecessors = [source]

    _assert_reorder_map(source, reorder, {0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5})
