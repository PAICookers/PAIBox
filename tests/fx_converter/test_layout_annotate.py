import operator

import torch
from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.node import Target

from paibox.fx_converter.data_semantic_type import DataSemanticType
from paibox.fx_converter.layout_annotate import (
    DIMS_ANNOTATION_KEY,
    SEMANTIC_TYPE_KEY,
    LayoutAnnotator,
    get_node_semantic_type,
)
from paibox.fx_converter.trace import trace_spikingjelly_model


def print_dims_annotation(gm: fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if DIMS_ANNOTATION_KEY in node.meta:
            print(
                node.name,
                node.meta[DIMS_ANNOTATION_KEY].input_dims,
                node.meta[DIMS_ANNOTATION_KEY].output_dims,
            )


def first_node_meta(gm: fx.GraphModule, op: str, target: Target | None, key: str):
    node = gm.graph.find_nodes(op=op, target=target)
    for n in node:
        if n.op == op and n.target == target:
            if key in n.meta:
                return n.meta[key]

    raise ValueError(f"Node not found: op={op}, target={target}, key={key}")


class TestLayoutAnnotator:
    def test_transpose_transpose(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.transpose(2, 3)
                return x.transpose(1, 2)

        m = M()
        gm = fx.symbolic_trace(m)
        LayoutAnnotator().annotate(gm, torch.randn(1, 2, 3, 4))
        print_dims_annotation(gm)

    def test_permute_permute(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.permute(2, 3, 0, 1)
                return x.permute(1, 2, 3, 0)

        m = M()
        gm = fx.symbolic_trace(m)
        LayoutAnnotator().annotate(gm, torch.randn(1, 2, 3, 4))
        print_dims_annotation(gm)

    def test_transpose_permute(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.transpose(0, 2)
                return x.permute(1, 2, 3, 0)

        m = M()
        gm = fx.symbolic_trace(m)
        LayoutAnnotator().annotate(gm, torch.randn(1, 2, 3, 4))
        print_dims_annotation(gm)

    def test_with_reshape(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.transpose(0, 2)
                x = x.reshape(3, -1)
                return x.permute(1, 0)

        m = M()
        gm = fx.symbolic_trace(m)
        LayoutAnnotator().annotate(gm, torch.randn(1, 2, 3, 4))
        print_dims_annotation(gm)

    def test2(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(3, 3, 3)
                self.relu2 = nn.ReLU()

            def forward(self, x):
                x = self.relu1(self.conv1(x))
                x = x.transpose(2, 3)
                x = self.relu2(self.conv2(x))
                return x.transpose(1, 3)

        m = M()
        gm = fx.symbolic_trace(m)
        gm.graph.print_tabular()
        LayoutAnnotator().annotate(gm, torch.randn(1, 3, 32, 32))
        print_dims_annotation(gm)


class TestDataSemanticTypeAnnotate:
    def test_pipeline_sets_semantic_on_common_nodes(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3)
                self.relu = nn.ReLU()
                self.lif = neuron.LIFNode()

            def forward(self, x: torch.Tensor):
                x = self.conv(x)
                x = x.transpose(2, 3)
                x = self.relu(x)
                x = x.permute(0, 2, 3, 1)
                x = self.lif(x)
                return x

        gm = trace_spikingjelly_model(M())
        LayoutAnnotator().annotate(gm, torch.randn(1, 3, 16, 16))

        assert (
            first_node_meta(gm, op="call_module", target="conv", key=SEMANTIC_TYPE_KEY)
            == DataSemanticType.POTENTIAL
        )
        assert (
            first_node_meta(
                gm, op="call_method", target="transpose", key=SEMANTIC_TYPE_KEY
            )
            == DataSemanticType.POTENTIAL
        )
        assert (
            first_node_meta(gm, op="call_module", target="relu", key=SEMANTIC_TYPE_KEY)
            == DataSemanticType.ACTIVATION
        )
        assert (
            first_node_meta(
                gm, op="call_method", target="permute", key=SEMANTIC_TYPE_KEY
            )
            == DataSemanticType.ACTIVATION
        )
        assert (
            first_node_meta(gm, op="call_module", target="lif", key=SEMANTIC_TYPE_KEY)
            == DataSemanticType.SPIKE
        )

        out = next(n for n in gm.graph.nodes if n.op == "output")
        assert get_node_semantic_type(out) == DataSemanticType.SPIKE

    def test_multi_input_add_merge_max(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, 3)
                self.conv2 = nn.Conv2d(3, 8, 3)
                self.relu = nn.ReLU()
                self.lif = neuron.LIFNode()

            def forward(self, x: torch.Tensor):
                a = self.lif(self.conv1(x))
                b = self.relu(self.conv2(x))
                return a + b

        gm = trace_spikingjelly_model(M())
        LayoutAnnotator().annotate(gm, torch.randn(1, 3, 16, 16))

        add = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target in (operator.add, torch.add)
        )
        assert get_node_semantic_type(add) == DataSemanticType.ACTIVATION

    def test_unknown_module_propagates_input_semantic(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3)
                self.bn = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        gm = trace_spikingjelly_model(M())
        LayoutAnnotator().annotate(gm, torch.randn(1, 3, 16, 16))

        assert (
            first_node_meta(gm, op="call_module", target="bn", key=SEMANTIC_TYPE_KEY)
            == DataSemanticType.POTENTIAL
        )
