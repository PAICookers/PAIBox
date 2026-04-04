"""Tests for lowering-layer converter behavior."""

import warnings

import pytest
import torch
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.ir.core_neuron import ANNNodeV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.lut_activation import LutCustom
from paibox.paiir.ir.op_node import (
    SequentialOp,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.lowering.converter import register_neuron, torch_to_paiir
from tests.paiir.conftest import (
    MultiSpike4,
    UnsupportedSinModel,
    convert_and_fuse,
    find_nodes,
    make_img_3ch_8x8,
    make_multispike4_lut,
    make_vec_8d,
)


def _find_single_act(graph, act_type):
    act_nodes = [
        node
        for node in graph.nodes.values()
        if isinstance(node, StandaloneActOp) and isinstance(node.act, act_type)
    ]
    assert len(act_nodes) == 1
    return act_nodes[0].act


class UnsupportedCountIncludePadAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x)


class SupportedNoPaddingCountIncludePadAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=0,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x)


class TestStrictMode:
    """Test strict mode for unsupported operators."""

    def test_strict_mode_raises(self):
        """strict=True should raise UnsupportedOpError for unsupported ops."""
        model = UnsupportedSinModel()
        with pytest.raises(UnsupportedOpError, match="Unsupported"):
            torch_to_paiir(model, make_img_3ch_8x8(), strict=True)

    def test_non_strict_mode_warns(self):
        """strict=False (default) should warn but bypass unsupported ops."""
        model = UnsupportedSinModel()
        with pytest.warns(UnsupportedOpWarning, match="unsupported"):
            graph = torch_to_paiir(model, make_img_3ch_8x8(), strict=False)
        assert len(graph.nodes) > 0

    def test_strict_mode_rejects_count_include_pad_false_with_padding(self):
        model = UnsupportedCountIncludePadAvgPool2d()
        with pytest.raises(UnsupportedOpError, match="count_include_pad=False"):
            torch_to_paiir(model, make_img_3ch_8x8(), strict=True)

    def test_non_strict_mode_warns_for_count_include_pad_false_with_padding(self):
        model = UnsupportedCountIncludePadAvgPool2d()
        with pytest.warns(UnsupportedOpWarning, match="count_include_pad=False"):
            graph = torch_to_paiir(model, make_img_3ch_8x8(), strict=False)
        assert len(graph.nodes) > 0

    def test_padding_free_count_include_pad_false_is_allowed(self):
        model = SupportedNoPaddingCountIncludePadAvgPool2d()
        graph = torch_to_paiir(model, make_img_3ch_8x8(), strict=True)

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AvgPool2d)
        ]
        assert len(pool_nodes) == 1


class TestRegisterNeuron:
    """Test register_neuron() API for custom neuron registration."""

    def test_register_custom_neuron(self):
        """Register MultiSpike4 -> compile a model -> verify graph structure."""
        register_neuron(
            MultiSpike4, converter=lambda _: ANNNodeV25(make_multispike4_lut())
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = MultiSpike4()

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model()
        fused = convert_and_fuse(model, make_vec_8d())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 1
        assert seq_nodes[0].act.lut is not None
        assert isinstance(seq_nodes[0].act.lut, LutCustom)

        test_inputs = torch.tensor([-100, -2, -1, 0, 1, 2, 3, 4, 5, 20, 100])
        ref_outputs = torch.round(torch.clamp(test_inputs, 0, 4))
        lut_outputs = seq_nodes[0].act(test_inputs)
        assert torch.equal(lut_outputs, ref_outputs), (
            f"LUT mismatch: expected {ref_outputs.tolist()}, got {lut_outputs.tolist()}"
        )


class TestSplitLowering:
    def test_torch_split_branches_lower_to_split_ops(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_left = nn.Conv2d(2, 4, 1, bias=False)
                self.conv_right = nn.Conv2d(3, 5, 1, bias=False)

            def forward(self, x):
                left, right = torch.split(x, [2, 3], dim=1)
                return self.conv_left(left), self.conv_right(right)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 5, 4, 4))

        input_name = graph.input_nodes()[0].name
        split_nodes = [node for node in graph.nodes.values() if isinstance(node, SplitOp)]
        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]

        assert len(split_nodes) == 1
        split = split_nodes[0]
        assert graph.predecessors(split.name) == [input_name]
        assert split.sections == (2, 3)
        assert split.dim == 1
        assert split.output_shapes == (
            torch.Size((1, 2, 4, 4)),
            torch.Size((1, 3, 4, 4)),
        )
        assert len(comp_nodes) == 2
        assert {tuple(graph.predecessors(node.name)) for node in comp_nodes} == {
            (split.name,),
        }
        assert split.successor_output_index == {
            (node.name, 0): 0 if node.comp.in_channels == 2 else 1
            for node in comp_nodes
        }

    def test_method_split_only_materializes_consumed_branch(self):
        class Model(nn.Module):
            def forward(self, x):
                parts = x.split(2, dim=1)
                return parts[0]

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 4, 3, 3))

        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        output_name = graph.output_nodes()[0].name

        assert len(split_nodes) == 1
        assert split_nodes[0].sections == 2
        assert split_nodes[0].output_shapes == (
            torch.Size((1, 2, 3, 3)),
            torch.Size((1, 2, 3, 3)),
        )
        assert split_nodes[0].successor_output_index == {(output_name, 0): 0}
        assert graph.predecessors(output_name) == [split_nodes[0].name]

    def test_method_split_with_sections_list_is_canonicalized(self):
        class Model(nn.Module):
            def forward(self, x):
                parts = x.split([1, 3], dim=1)
                return parts[1]

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 4, 3, 3))

        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        output_name = graph.output_nodes()[0].name

        assert len(split_nodes) == 1
        assert split_nodes[0].sections == (1, 3)
        assert split_nodes[0].output_shapes == (
            torch.Size((1, 1, 3, 3)),
            torch.Size((1, 3, 3, 3)),
        )
        assert split_nodes[0].successor_output_index == {(output_name, 0): 1}

    def test_split_direct_outputs_materialize_single_split_node(self):
        class Model(nn.Module):
            def forward(self, x):
                left, right = torch.split(x, [2, 3], dim=1)
                return left, right

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 5, 4, 4))

        split_nodes = [node for node in graph.nodes.values() if isinstance(node, SplitOp)]
        output_nodes = graph.output_nodes()

        assert len(split_nodes) == 1
        assert len(output_nodes) == 2
        split = split_nodes[0]
        assert split.output_shapes == (
            torch.Size((1, 2, 4, 4)),
            torch.Size((1, 3, 4, 4)),
        )
        assert all(graph.predecessors(node.name) == [split.name] for node in output_nodes)
        assert split.successor_output_index == {
            (output_nodes[0].name, 0): 0,
            (output_nodes[1].name, 0): 1,
        }

    def test_split_after_pending_permute_is_rejected(self):
        class Model(nn.Module):
            def forward(self, x):
                y = x.permute(0, 2, 3, 1)
                left, _right = torch.split(y, [2, 3], dim=3)
                return left

        with pytest.raises(
            UnsupportedOpError, match="pending transpose/permute layout"
        ):
            torch_to_paiir(Model().eval(), torch.randn(1, 5, 3, 3), strict=True)

    def test_chunk_remains_unsupported(self):
        class Model(nn.Module):
            def forward(self, x):
                left, right = torch.chunk(x, 2, dim=1)
                return left + right

        with pytest.raises(UnsupportedOpError, match="chunk"):
            torch_to_paiir(Model().eval(), torch.randn(1, 4, 3, 3), strict=True)

    def test_tensor_split_remains_unsupported(self):
        class Model(nn.Module):
            def forward(self, x):
                left, right = torch.tensor_split(x, 2, dim=1)
                return left + right

        with pytest.raises(UnsupportedOpError, match="tensor_split"):
            torch_to_paiir(Model().eval(), torch.randn(1, 4, 3, 3), strict=True)

    def test_negative_split_getitem_is_rejected(self):
        class Model(nn.Module):
            def forward(self, x):
                parts = torch.split(x, 2, dim=1)
                return parts[-1]

        with pytest.raises(
            UnsupportedOpError, match="non-negative integer getitem consumers"
        ):
            torch_to_paiir(Model().eval(), torch.randn(1, 4, 3, 3), strict=True)


class TestSpikingJellyNeuronAttributeForwarding:
    """Lowering should preserve training-related neuron attributes."""

    def test_activation_based_ifnode_preserves_surrogate_and_detach_reset(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = sj.IFNode(1.25)

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, IFNodeV25)
        assert lowered.surrogate_function is model.act.surrogate_function
        assert lowered.detach_reset is model.act.detach_reset

    def test_activation_based_lifnode_preserves_surrogate_and_detach_reset(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = sj.LIFNode(4.0, False, 1.5)

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, LIFNodeV25)
        assert lowered.surrogate_function is model.act.surrogate_function
        assert lowered.detach_reset is model.act.detach_reset


class TestLegacyClockDrivenCompatibility:
    """Compatibility tests for legacy SpikingJelly clock_driven neurons."""

    def test_clock_driven_ifnode_warns_and_lowers(self):
        legacy_neuron = pytest.importorskip("spikingjelly.clock_driven.neuron")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = legacy_neuron.IFNode()

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        with pytest.warns(
            DeprecationWarning,
            match="activation_based\\.neuron\\.IFNode",
        ):
            graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, IFNodeV25)
        assert lowered.surrogate_function is model.act.surrogate_function
        assert lowered.detach_reset is model.act.detach_reset

    def test_clock_driven_lifnode_warns_and_lowers(self):
        legacy_neuron = pytest.importorskip("spikingjelly.clock_driven.neuron")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = legacy_neuron.LIFNode(2.0, 1.0)

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        with pytest.warns(
            DeprecationWarning,
            match="activation_based\\.neuron\\.LIFNode",
        ):
            graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, LIFNodeV25)
        assert lowered.surrogate_function is model.act.surrogate_function
        assert lowered.detach_reset is model.act.detach_reset

    def test_activation_based_lifnode_does_not_warn(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = sj.LIFNode()

            def forward(self, x):
                return self.act(self.linear(x))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = torch_to_paiir(Model(), make_vec_8d())

        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
        ]
        lowered = _find_single_act(graph, LIFNodeV25)
        assert isinstance(lowered, LIFNodeV25)
