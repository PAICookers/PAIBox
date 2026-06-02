from collections.abc import Callable
from typing import TypeVar

import torch
from paicorelib import LeakMultiInputMode
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir import ANNNodeV25, IFNodeV25, LIFNodeV25, LutSigmoid, compile_to_paiir
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.op_node import SequentialOp, StandaloneActOp, TransformOp
from paibox.paiir.nn import SumPool2d
from tests.paiir.conftest import find_transform_nodes

_T = TypeVar("_T")


def _single_node(
    graph: PAIIRGraph, typ: type[_T], predicate: Callable[[_T], bool] | None = None
) -> _T:
    matches = [node for node in graph.nodes.values() if isinstance(node, typ)]
    if predicate is not None:
        matches = [node for node in matches if predicate(node)]
    assert len(matches) == 1
    return matches[0]


class LinearTransformReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 2, 4)
        return self.relu(x)


class LinearTransformReLUTransformBack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 2, 4)
        x = self.relu(x)
        return x.reshape(x.shape[0], 8)


class MaxPoolTransformIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.MaxPool2d(2, 2)
        self.if2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        return self.if2(x)


class AvgPoolTransformANN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], 1, 1, x.shape[2], x.shape[3])
        return self.sigmoid(x)


class AvgPoolTransformIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(2, 2)
        self.if2 = sj.IFNode(v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], 1, 1, x.shape[2], x.shape[3])
        return self.if2(x)


class AvgPoolTransformLIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = sj.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(2, 2)
        self.lif2 = sj.LIFNode(tau=4.0, v_threshold=1.0)

    def forward(self, x):
        x = self.if1(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], 1, 1, x.shape[2], x.shape[3])
        return self.lif2(x)


def test_compile_linear_transform_relu_keeps_visible_post_transform() -> None:
    graph = compile_to_paiir(LinearTransformReLU(), torch.randn(1, 8))

    seq = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, nn.Linear)
    )
    transform = _single_node(graph, TransformOp)
    assert seq.output_layouts[0].shape == torch.Size((1, 8))
    assert transform.input_layouts == seq.output_layouts
    assert transform.output_layouts[0].shape == torch.Size((1, 2, 4))


def test_compile_fixed_point_removes_identity_post_transform_after_commutation() -> (
    None
):
    graph = compile_to_paiir(LinearTransformReLUTransformBack(), torch.randn(1, 8))

    seq = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, nn.Linear)
    )
    assert seq.output_layouts[0].shape == torch.Size((1, 8))
    assert not find_transform_nodes(graph)


def test_compile_maxpool_transform_if_fuses_and_keeps_post_transform() -> None:
    graph = compile_to_paiir(MaxPoolTransformIF(), torch.randn(1, 1, 4, 4))

    seq = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, nn.MaxPool2d)
    )
    transform = _single_node(graph, TransformOp)
    assert isinstance(seq.act, IFNodeV25)
    assert transform.input_layouts == seq.output_layouts


def test_compile_avgpool_transform_ann_uses_shared_core_policy() -> None:
    graph = compile_to_paiir(AvgPoolTransformANN(), torch.randn(1, 1, 4, 4))

    avg_seq = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, nn.AvgPool2d)
    )
    transform = _single_node(graph, TransformOp)
    assert isinstance(avg_seq.act, ANNNodeV25)
    assert isinstance(avg_seq.act.lut, LutSigmoid)
    assert transform.input_layouts == avg_seq.output_layouts


def test_compile_avgpool_transform_if_uses_split_core_policy() -> None:
    graph = compile_to_paiir(AvgPoolTransformIF(), torch.randn(1, 1, 4, 4))

    core1 = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, SumPool2d)
    )
    act = _single_node(
        graph, StandaloneActOp, lambda node: isinstance(node.act, IFNodeV25)
    )
    transform = _single_node(graph, TransformOp)
    assert isinstance(core1.act, ANNNodeV25)
    assert act.output_layouts == core1.output_layouts
    assert transform.input_layouts == act.output_layouts


def test_compile_avgpool_transform_lif_uses_shared_core_policy() -> None:
    graph = compile_to_paiir(AvgPoolTransformLIF(), torch.randn(1, 1, 4, 4))

    avg_seq = _single_node(
        graph, SequentialOp, lambda node: isinstance(node.comp, nn.AvgPool2d)
    )
    transform = _single_node(graph, TransformOp)
    assert isinstance(avg_seq.act, LIFNodeV25)
    assert avg_seq.avgpool_deploy_metadata is not None
    assert avg_seq.act.leak_multi_input == LeakMultiInputMode.ENABLE
    assert transform.input_layouts == avg_seq.output_layouts
