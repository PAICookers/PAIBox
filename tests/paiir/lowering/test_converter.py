"""Tests for lowering-layer converter behavior."""

import pytest
import torch
from torch import nn

from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.ir.core_neuron import ANNNodeV25
from paibox.paiir.ir.lut_activation import LutCustom
from paibox.paiir.ir.op_node import SequentialOp, StandaloneCompOp
from paibox.paiir.lowering.converter import register_neuron, torch_to_paiir
from tests.paiir.conftest import (
    UnsupportedSinModel,
    convert_and_fuse,
    find_nodes,
    make_img_3ch_8x8,
    make_multispike4_lut,
    make_vec_8d,
)


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
        from tests.paiir.conftest import MultiSpike4

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
        assert torch.equal(
            lut_outputs, ref_outputs
        ), f"LUT mismatch: expected {ref_outputs.tolist()}, got {lut_outputs.tolist()}"
