import pytest
import torch
from paicorelib import DataSign, DataWidth
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir import CompileConfig, compile_to_paiir
from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.op_node import AccumulateOp, ConcatOp, SequentialOp

from .conftest import (
    ANNClassifier,
    SimpleCNN,
    SNNResidualAdd,
    SNNTwoLayer,
    UnsupportedSoftmax,
    find_nodes,
    make_img_3ch_8x8,
    make_img_3ch_32x32,
    offline_nodes,
)


class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(3, 4, 1)
        self.conv3 = nn.Conv2d(8, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv3(torch.cat([self.conv1(x), self.conv2(x)], dim=1)))


class TestCompileBasic:
    """Basic compilation and graph structure."""

    def test_simple_cnn(self):
        graph = compile_to_paiir(SimpleCNN(), make_img_3ch_32x32())

        assert "InputNode_0" in graph.nodes
        assert "OutputNode_0" in graph.nodes
        assert len(offline_nodes(graph)) >= 1

    def test_snn_two_layer(self):
        graph = compile_to_paiir(SNNTwoLayer(), make_img_3ch_8x8())
        assert len(find_nodes(graph, SequentialOp)) == 2

    def test_ann_classifier(self):
        graph = compile_to_paiir(ANNClassifier(), make_img_3ch_8x8())
        assert len(find_nodes(graph, SequentialOp)) >= 2

    def test_residual_add(self):
        graph = compile_to_paiir(SNNResidualAdd(), make_img_3ch_8x8())

        accum_nodes = find_nodes(graph, AccumulateOp)
        assert len(accum_nodes) == 1
        assert accum_nodes[0].signs == (1, 1)

    def test_concat(self):
        graph = compile_to_paiir(ConcatModel(), make_img_3ch_8x8())

        concat_nodes = find_nodes(graph, ConcatOp)
        assert len(concat_nodes) == 1
        assert concat_nodes[0].dim == 1


class TestDataFormat:
    """Verify data/weight sign & width are correctly inferred after compilation."""

    def test_snn_output_unsigned_1bit(self):
        """SNN (IF/LIF default): output UNSIGNED 1BIT, input propagated."""
        graph = compile_to_paiir(SNNTwoLayer(), make_img_3ch_8x8())

        for node in offline_nodes(graph):
            cp = node.core_params
            # SNN default: spike output is unsigned 1-bit
            assert cp.output_sign == DataSign.UNSIGNED
            assert cp.output_width == DataWidth.WIDTH_1BIT
            # Input format must be filled
            assert cp.input_sign == DataSign.UNSIGNED
            assert cp.input_width == DataWidth.WIDTH_1BIT

    def test_ann_output_8bit(self):
        """ANN (ReLU/Sigmoid): output 8BIT, sign matches activation."""
        graph = compile_to_paiir(ANNClassifier(), make_img_3ch_8x8())

        for node in offline_nodes(graph):
            cp = node.core_params
            assert cp.output_width == DataWidth.WIDTH_8BIT
            assert cp.output_sign in (DataSign.UNSIGNED, DataSign.SIGNED)

    def test_quantized_int8_weights(self):
        """Quantized int8 weights -> SIGNED WIDTH_8BIT."""
        model = SNNTwoLayer()
        with torch.no_grad():
            model.conv1.weight.copy_(
                torch.randint(-128, 128, model.conv1.weight.shape, dtype=torch.int8)
            )
            model.conv2.weight.copy_(
                torch.randint(-128, 128, model.conv2.weight.shape, dtype=torch.int8)
            )

        graph = compile_to_paiir(model, make_img_3ch_8x8())

        for node in offline_nodes(graph):
            assert node.core_params.weight_sign == DataSign.SIGNED
            assert node.core_params.weight_width == DataWidth.WIDTH_8BIT

    def test_binary_weights(self):
        """Binary {0, 1} weights -> UNSIGNED WIDTH_1BIT."""

        class BinarySNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4, bias=False)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.linear(x))

        model = BinarySNN()
        with torch.no_grad():
            model.linear.weight.copy_(
                torch.randint(0, 2, model.linear.weight.shape, dtype=torch.int8)
            )

        graph = compile_to_paiir(model, torch.randn(1, 8))

        nodes = offline_nodes(graph)
        assert len(nodes) == 1
        assert nodes[0].core_params.weight_sign == DataSign.UNSIGNED
        assert nodes[0].core_params.weight_width == DataWidth.WIDTH_1BIT

    def test_ternary_weights(self):
        """Ternary {-1, 0, 1} weights -> SIGNED WIDTH_2BIT."""

        class TernarySNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4, bias=False)
                self.ifn = sj.IFNode()

            def forward(self, x):
                return self.ifn(self.linear(x))

        model = TernarySNN()
        with torch.no_grad():
            model.linear.weight.copy_(
                torch.randint(-1, 2, model.linear.weight.shape, dtype=torch.int8)
            )

        graph = compile_to_paiir(model, torch.randn(1, 8))

        nodes = offline_nodes(graph)
        assert len(nodes) == 1
        assert nodes[0].core_params.weight_sign == DataSign.SIGNED
        assert nodes[0].core_params.weight_width == DataWidth.WIDTH_2BIT

    def test_input_formats_override(self):
        """Explicit input_formats overrides default inference (kwarg and config)."""
        sample_input = make_img_3ch_32x32()
        fmt = {"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)}

        graph_kw = compile_to_paiir(SimpleCNN(), sample_input, input_formats=fmt)
        first_kw = offline_nodes(graph_kw)[0]
        assert first_kw.core_params.input_sign == DataSign.UNSIGNED
        assert first_kw.core_params.input_width == DataWidth.WIDTH_1BIT

    def test_input_formats_override2(self):
        sample_input = make_img_3ch_32x32()
        fmt = {"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)}

        config = CompileConfig(input_formats=fmt)
        graph_cfg = compile_to_paiir(SimpleCNN(), sample_input, compile_config=config)
        first_cfg = offline_nodes(graph_cfg)[0]
        assert first_cfg.core_params.input_sign == DataSign.UNSIGNED
        assert first_cfg.core_params.input_width == DataWidth.WIDTH_1BIT


class TestTickParams:
    """Verify timing parameters after compilation."""

    def test_tick_start_assigned(self):
        """All OfflineCoreOps get tick_start > 0."""
        graph = compile_to_paiir(SNNTwoLayer(), make_img_3ch_8x8())

        for node in offline_nodes(graph):
            assert node.core_params.tick_start is not None
            assert node.core_params.tick_start >= 1

    @pytest.mark.parametrize(
        "tick_duration, auto_reset, expected_duration, expected_initial",
        [(None, None, 0, 1), (100, True, 100, 1), (100, False, 100, 1)],
        ids=["default", "reset_enabled", "reset_disabled"],
    )
    def test_tick_duration_and_auto_reset(
        self, tick_duration, auto_reset, expected_duration, expected_initial
    ):
        """ANN mode: tick_initial is always 1 (clear membrane every step)."""
        kwargs = {}
        if tick_duration is not None:
            kwargs["tick_duration"] = tick_duration
        if auto_reset is not None:
            kwargs["auto_reset"] = auto_reset

        sample_input = make_img_3ch_32x32()
        graph = compile_to_paiir(SimpleCNN(), sample_input, **kwargs)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == expected_duration
            assert node.core_params.tick_initial == expected_initial


class TestCompileConfig:
    """CompileConfig and parameter precedence."""

    def test_config_applies(self):
        config = CompileConfig(tick_duration=50, auto_reset=False)
        sample_input = make_img_3ch_32x32()
        graph = compile_to_paiir(SimpleCNN(), sample_input, compile_config=config)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 50
            # ANN mode: tick_initial is always 1
            assert node.core_params.tick_initial == 1

    def test_explicit_kwarg_overrides_config(self):
        config = CompileConfig(tick_duration=50, auto_reset=False)
        sample_input = make_img_3ch_32x32()
        graph = compile_to_paiir(
            SimpleCNN(),
            sample_input,
            tick_duration=100,
            auto_reset=True,
            compile_config=config,
        )

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 100
            # ANN mode: tick_initial is always 1
            assert node.core_params.tick_initial == 1


class TestStrictMode:
    """strict parameter controls unsupported op handling."""

    def test_strict_raises(self):
        with pytest.raises(UnsupportedOpError):
            compile_to_paiir(UnsupportedSoftmax(), torch.randn(1, 10), strict=True)

    def test_non_strict_warns(self):
        with pytest.warns(UnsupportedOpWarning):
            compile_to_paiir(UnsupportedSoftmax(), torch.randn(1, 10), strict=False)
