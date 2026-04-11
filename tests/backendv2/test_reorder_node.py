import numpy as np
import paicorelib
import torch

for name, value in {
    "LUT_ACTIVATION_DTYPE": np.int8,
    "LUT_POTENTIAL_DTYPE": np.int8,
    "LUTActivationType": np.ndarray,
    "LUTPotentialType": np.ndarray,
}.items():
    if not hasattr(paicorelib, name):
        setattr(paicorelib, name, value)

from paibox.backendv2.op_node import InNode, ReorderNode, get_elem  # noqa: E402
from paibox.paiir.ir import InputNode, ReshapeOp, TensorLayout  # noqa: E402


def test_reorder_node_respects_input_dims_permutation():
    raw_input = InputNode(shape=torch.Size((1, 2, 3)))
    source = InNode(raw_input.name, raw_input, raw_input.shape)

    raw_reshape = ReshapeOp(shape_fn=lambda _shape: torch.Size((1, 6)))
    raw_reshape.input_layouts = (TensorLayout(shape=raw_input.shape, dims=(0, 2, 1)),)
    raw_reshape.output_layouts = (TensorLayout(shape=torch.Size((1, 6)), dims=(0, 1)),)

    reorder = ReorderNode(
        "ReorderNode_0", raw_reshape, raw_reshape.output_layouts[0].shape
    )
    reorder.predecessors = [source]

    reorder_map = reorder.get_reorder_info()

    expected = {0: 0, 1: 2, 2: 4, 3: 1, 4: 3, 5: 5}
    for src_idx, dst_idx in expected.items():
        src_elem = get_elem(source, src_idx)
        assert reorder_map[src_elem].index.idx == dst_idx
