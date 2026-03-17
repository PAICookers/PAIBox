import numpy as np
import paicorelib
import torch
from torch import nn

for name, value in {
    "LUT_ACTIVATION_DTYPE": np.int8,
    "LUT_POTENTIAL_DTYPE": np.int8,
    "LUTActivationType": np.ndarray,
    "LUTPotentialType": np.ndarray,
}.items():
    if not hasattr(paicorelib, name):
        setattr(paicorelib, name, value)

from paibox.backendv2.neuron import InputElem, Neuron
from paibox.backendv2.op_node import CoreOpNode, CustomIndex, InNode
from paibox.backendv2.routing import get_raw_weights
from paibox.paiir import (
    ANNNodeV25,
    AccumulateOp,
    LutReLU,
    StandaloneActOp,
    StandaloneCompOp,
)


def _configure_raw_node(raw_node):
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    return raw_node


def _input_elems(node: InNode) -> list[InputElem]:
    return [InputElem(node, CustomIndex(i)) for i in range(node.shape.numel())]


def _neurons(node: CoreOpNode) -> list[Neuron]:
    return [Neuron(node, CustomIndex(i)) for i in range(node.shape.numel())]


def test_get_raw_weights_linear_with_unconnected_inputs():
    source = InNode("src", (1, 4))
    unrelated = InNode("other", (1, 2))

    linear = _configure_raw_node(StandaloneCompOp(nn.Linear(4, 3, bias=False)))
    with torch.no_grad():
        linear.comp.weight.copy_(
            torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                dtype=torch.float32,
            )
        )

    linear.output_shape = (1, 3)
    target = CoreOpNode("linear", linear, linear.output_shape)
    target.predecessors = [source]

    weights = get_raw_weights(_neurons(target), _input_elems(source) + _input_elems(unrelated))

    expected = np.array(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 0, 0],
            [9, 10, 11, 12, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(weights, expected)


def test_get_raw_weights_conv2d():
    source = InNode("img", (1, 1, 3, 3))

    conv = _configure_raw_node(StandaloneCompOp(nn.Conv2d(1, 1, 2, bias=False)))
    with torch.no_grad():
        conv.comp.weight.copy_(torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32))

    conv.output_shape = (1, 1, 2, 2)
    target = CoreOpNode("conv", conv, conv.output_shape)
    target.predecessors = [source]

    weights = get_raw_weights(_neurons(target), _input_elems(source))

    expected = np.array(
        [
            [1, 2, 0, 3, 4, 0, 0, 0, 0],
            [0, 1, 2, 0, 3, 4, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 3, 4, 0],
            [0, 0, 0, 0, 1, 2, 0, 3, 4],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(weights, expected)


def test_get_raw_weights_pool2d():
    source = InNode("pool_src", (1, 1, 3, 3))

    pool = _configure_raw_node(StandaloneCompOp(nn.AvgPool2d(2, stride=1)))
    pool.output_shape = (1, 1, 2, 2)
    target = CoreOpNode("pool", pool, pool.output_shape)
    target.predecessors = [source]

    weights = get_raw_weights(_neurons(target), _input_elems(source))

    expected = np.array(
        [
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(weights, expected)


def test_get_raw_weights_standalone_activation_identity():
    source = InNode("act_src", (1, 4))

    act = _configure_raw_node(StandaloneActOp(ANNNodeV25(lut=LutReLU())))
    act.output_shape = (1, 4)
    target = CoreOpNode("act", act, act.output_shape)
    target.predecessors = [source]

    weights = get_raw_weights(_neurons(target), _input_elems(source))

    np.testing.assert_array_equal(weights, np.eye(4, dtype=np.int32))


def test_get_raw_weights_accumulate_respects_negative_branch_sign():
    source_a = InNode("a", (1, 2))
    source_b = InNode("b", (1, 2))

    acc = _configure_raw_node(
        AccumulateOp(
            comps=[nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)],
            act=ANNNodeV25(lut=LutReLU()),
            op_signs=(1, -1),
        )
    )
    with torch.no_grad():
        acc.comps[0].weight.copy_(torch.tensor([[1, 2], [3, 4]], dtype=torch.float32))
        acc.comps[1].weight.copy_(torch.tensor([[5, 6], [7, 8]], dtype=torch.float32))

    acc.output_shape = (1, 2)
    target = CoreOpNode("acc", acc, acc.output_shape)
    target.predecessors = [source_a, source_b]

    input_elems = _input_elems(source_a) + _input_elems(source_b)
    weights = get_raw_weights(_neurons(target), input_elems)

    expected = np.array([[1, 2, -5, -6], [3, 4, -7, -8]], dtype=np.int32)
    np.testing.assert_array_equal(weights, expected)
