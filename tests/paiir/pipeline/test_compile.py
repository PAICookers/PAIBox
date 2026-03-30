import importlib
import warnings

import pytest
import torch
import torch.nn.functional as F
from paicorelib import DataSign, DataWidth
from spikingjelly.activation_based import neuron as sj
from torch import nn

import paibox.paiir.pipeline.avgpool.fusion as avgpool_fusion
import paibox.paiir.pipeline.compile as compile_mod
from paibox.paiir import CompileConfig, LIFNodeV25, compile_to_paiir, torch_to_paiir
from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    ReshapeOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.lowering.converter import _analyze_graph, _LoweringContext
from paibox.paiir.nn import SumPool1d, SumPool2d
from paibox.paiir.pipeline.avgpool import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
    calibrate_avgpool_threshold,
)
from paibox.paiir.pipeline.avgpool.metadata import AvgPoolDeployMetadata
from paibox.paiir.pipeline.passes import GraphCleanupWarning
from tests.paiir.conftest import (
    ANNClassifier,
    SimpleCNN,
    SNNResidualAdd,
    SNNTwoLayer,
    SNNWithAvgPool1dIF,
    SNNWithAvgPool1dLIF,
    SNNWithAvgPoolLIF,
    UnsupportedSoftmax,
    find_nodes,
    make_img_3ch_8x8,
    make_img_3ch_32x32,
    make_vec_64d,
    offline_nodes,
)
from tests.paiir.tracing import trace_for_lowering


class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(3, 4, 1)
        self.conv3 = nn.Conv2d(8, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv3(torch.cat([self.conv1(x), self.conv2(x)], dim=1)))


class PoolAfterReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = x.reshape(x.shape)
        return self.pool(x)


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


class TestPackageExports:
    def test_public_packages_reexport_compile_symbols(self):
        paiir_mod = importlib.import_module("paibox.paiir")
        pipeline_mod = importlib.import_module("paibox.paiir.pipeline")
        avgpool_mod = importlib.import_module("paibox.paiir.pipeline.avgpool")

        assert paiir_mod.compile_to_paiir is compile_mod.compile_to_paiir
        assert paiir_mod.torch_to_paiir is torch_to_paiir
        assert pipeline_mod.CompileConfig is CompileConfig
        assert pipeline_mod.compile_to_paiir is compile_mod.compile_to_paiir
        assert avgpool_mod.calibrate_avgpool_threshold is not None
        assert not hasattr(paiir_mod, "OfflineCoreOp")
        assert not hasattr(pipeline_mod, "DataFormat")
        assert not hasattr(avgpool_mod, "AvgPoolDeployMetadata")


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

    def test_transpose_then_flatten_before_linear_compiles(self):
        class TransposeFlattenLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                return self.linear(x.transpose(1, 2).flatten(1))

        graph = compile_to_paiir(TransposeFlattenLinear(), torch.randn(1, 2, 3))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 1
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[0].name]

    def test_permute_then_reshape_before_linear_compiles(self):
        class PermuteReshapeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(24, 5, bias=False)

            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(x.size(0), -1)
                return self.linear(x)

        graph = compile_to_paiir(PermuteReshapeLinear(), torch.randn(1, 2, 3, 4))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 1
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[0].name]

    def test_maxpool_after_reshape_compiles(self):
        graph = compile_to_paiir(PoolAfterReshape(), make_img_3ch_8x8())

        reshape_nodes = find_nodes(graph, ReshapeOp)
        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.MaxPool2d)
        ]

        assert len(reshape_nodes) == 1
        assert len(pool_nodes) == 1
        assert graph.predecessors(pool_nodes[0].name) == [reshape_nodes[0].name]
        assert pool_nodes[0].core_params.input_sign is not None
        assert pool_nodes[0].core_params.input_width is not None

    def test_function_unsqueeze_before_linear_compiles(self):
        class FunctionUnsqueezeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                x = torch.unsqueeze(x, 1)
                x = x.flatten(1)
                return self.linear(x)

        graph = compile_to_paiir(FunctionUnsqueezeLinear(), torch.randn(1, 2, 3))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 2
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(reshape_nodes[1].name) == [reshape_nodes[0].name]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[1].name]

    def test_tuple_repeat_all_ones_before_linear_compiles(self):
        class TupleRepeatLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                x = x.repeat((1, 1, 1))
                x = x.flatten(1)
                return self.linear(x)

        graph = compile_to_paiir(TupleRepeatLinear(), torch.randn(1, 2, 3))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 2
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(reshape_nodes[1].name) == [reshape_nodes[0].name]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[1].name]

    def test_method_squeeze_before_linear_compiles(self):
        class MethodSqueezeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                x = x.squeeze(1)
                x = x.flatten(1)
                return self.linear(x)

        graph = compile_to_paiir(MethodSqueezeLinear(), torch.randn(1, 1, 2, 3))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 2
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(reshape_nodes[1].name) == [reshape_nodes[0].name]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[1].name]

    def test_function_squeeze_before_linear_compiles(self):
        class FunctionSqueezeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x):
                x = torch.squeeze(x, 1)
                x = x.flatten(1)
                return self.linear(x)

        graph = compile_to_paiir(FunctionSqueezeLinear(), torch.randn(1, 1, 2, 3))

        reshape_nodes = find_nodes(graph, ReshapeOp)
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 2
        assert len(linear_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(reshape_nodes[1].name) == [reshape_nodes[0].name]
        assert graph.predecessors(linear_nodes[0].name) == [reshape_nodes[1].name]


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
        """ANN mode: cores with activation functions get tick_initial=1."""
        kwargs = {}
        if tick_duration is not None:
            kwargs["tick_duration"] = tick_duration
        if auto_reset is not None:
            kwargs["auto_reset"] = auto_reset

        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(ANNClassifier(), sample_input, **kwargs)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == expected_duration
            # Nodes with activation (membrane potential accumulates) need tick_initial=1
            if isinstance(node, (SequentialOp, AccumulateOp, StandaloneActOp)):
                assert node.core_params.tick_initial == expected_initial
            # StandaloneCompOp has no activation, snn_mode=SNN, tick_initial depends on auto_reset
            elif isinstance(node, StandaloneCompOp):
                if auto_reset and expected_duration > 0:
                    assert node.core_params.tick_initial == expected_duration
                else:
                    assert node.core_params.tick_initial == 0


class TestCompileConfig:
    """CompileConfig and parameter precedence."""

    def test_config_applies(self):
        config = CompileConfig(tick_duration=50, auto_reset=False)
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(ANNClassifier(), sample_input, compile_config=config)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 50
            if isinstance(node, (SequentialOp, AccumulateOp, StandaloneActOp)):
                assert node.core_params.tick_initial == 1
            elif isinstance(node, StandaloneCompOp):
                assert node.core_params.tick_initial == 0

    def test_explicit_kwarg_overrides_config(self):

        config = CompileConfig(tick_duration=50, auto_reset=False)
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(
            ANNClassifier(),
            sample_input,
            tick_duration=100,
            auto_reset=True,
            compile_config=config,
        )

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 100
            # Only nodes with neurons get tick_initial=1
            if not isinstance(node, StandaloneCompOp):
                assert node.core_params.tick_initial == 1


class TestStrictMode:
    """strict parameter controls unsupported op handling."""

    def test_strict_raises(self):
        with pytest.raises(UnsupportedOpError):
            compile_to_paiir(UnsupportedSoftmax(), torch.randn(1, 10), strict=True)

    def test_non_strict_warns(self):
        with pytest.warns(UnsupportedOpWarning):
            compile_to_paiir(UnsupportedSoftmax(), torch.randn(1, 10), strict=False)

    def test_strict_raises_for_count_include_pad_false_with_padding(self):
        with pytest.raises(UnsupportedOpError, match="count_include_pad=False"):
            compile_to_paiir(
                UnsupportedCountIncludePadAvgPool2d(),
                make_img_3ch_8x8(),
                strict=True,
            )

    def test_padding_free_count_include_pad_false_still_compiles(self):
        graph = compile_to_paiir(
            SupportedNoPaddingCountIncludePadAvgPool2d(),
            make_img_3ch_8x8(),
            strict=True,
        )

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AvgPool2d)
        ]
        assert len(pool_nodes) == 1


class FunctionalQuantizedConv(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias = torch.randn(4)
        scale = torch.tensor(0.125)
        self.register_buffer("weight_int8", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        w = self.weight_int8.to(x.dtype) * self.weight_scale
        return F.conv2d(x, w, self.bias, stride=1, padding=1, dilation=1, groups=1)


class FunctionalQuantizedConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3), dtype=torch.int8)
        bias = torch.randn(4)
        scale = torch.tensor(0.125)
        self.register_buffer("weight_int8", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        w = self.weight_int8.to(x.dtype) * self.weight_scale
        return F.conv1d(x, w, self.bias, stride=1, padding=1, dilation=1, groups=1)


class FunctionalQuantizedConvWithShapeReshape(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias = torch.randn(4)
        scale = torch.tensor(0.125)
        self.register_buffer("weight_int8", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        expanded = x.unsqueeze(0).repeat(1, 1, 1, 1, 1)
        flat = expanded.flatten(0, 1)
        w = self.weight_int8.to(flat.dtype) * self.weight_scale
        y = F.conv2d(flat, w, self.bias, stride=1, padding=1, dilation=1, groups=1)
        y = y.reshape(
            expanded.shape[0], expanded.shape[1], -1, y.shape[-2], y.shape[-1]
        )
        return self.relu(y)


class TestFunctionalConv:
    def test_quantized_functional_conv2d_supported_in_strict_mode(self):
        model = FunctionalQuantizedConv()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(model, torch.randn(1, 3, 8, 8), strict=True)

        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]
        assert len(comp_nodes) == 1
        assert isinstance(comp_nodes[0].comp, nn.Conv2d)
        assert torch.equal(comp_nodes[0].comp.raw_weight, model.weight_int8)
        assert graph.predecessors(comp_nodes[0].name) == ["InputNode_0"]
        assert not any("conv2d" in str(w.message) for w in caught)

    def test_quantized_functional_conv1d_supported_in_strict_mode(self):
        model = FunctionalQuantizedConv1d()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(model, torch.randn(1, 3, 16), strict=True)

        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv1d)
        ]
        assert len(comp_nodes) == 1
        assert isinstance(comp_nodes[0].comp, nn.Conv1d)
        assert torch.equal(comp_nodes[0].comp.raw_weight, model.weight_int8)
        assert graph.predecessors(comp_nodes[0].name) == ["InputNode_0"]
        assert not any("conv1d" in str(w.message) for w in caught)

    def test_shape_only_reshape_args_do_not_become_data_predecessors(self):
        graph = torch_to_paiir(
            FunctionalQuantizedConvWithShapeReshape(),
            torch.randn(1, 3, 8, 8),
            strict=False,
        )
        graph.summary()

        reshape_nodes = [
            node for node in graph.nodes.values() if node.name == "ReshapeOp_0"
        ]
        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]
        assert len(reshape_nodes) == 1
        assert len(comp_nodes) == 1
        assert graph.predecessors("ReshapeOp_0") == ["InputNode_0"]
        assert graph.predecessors("ReshapeOp_1") == ["ReshapeOp_0"]
        assert graph.predecessors("ReshapeOp_2") == ["ReshapeOp_1"]
        assert graph.predecessors(comp_nodes[0].name) == ["ReshapeOp_2"]

    def test_unsqueeze_repeat_all_ones_compile_path_succeeds(self):
        graph = compile_to_paiir(
            FunctionalQuantizedConvWithShapeReshape(),
            torch.randn(1, 3, 8, 8),
            strict=True,
        )

        reshape_nodes = find_nodes(graph, ReshapeOp)
        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]

        assert reshape_nodes
        assert comp_nodes
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]

    def test_view_as_reference_path_does_not_become_data_predecessor(self):
        class ViewAsReferenceFromFlatten(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(12, 4, bias=False)

            def forward(self, x):
                ref = x.flatten(1)
                y = x.view_as(ref)
                return self.linear(y)

        model = ViewAsReferenceFromFlatten()

        with pytest.warns(GraphCleanupWarning, match="disconnected"):
            graph = compile_to_paiir(model, torch.randn(1, 3, 2, 2))

        reshape_nodes = [
            node for node in graph.nodes.values() if isinstance(node, ReshapeOp)
        ]
        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Linear)
        ]

        assert len(reshape_nodes) == 1
        assert len(comp_nodes) == 1
        assert graph.predecessors(reshape_nodes[0].name) == ["InputNode_0"]
        assert graph.predecessors(comp_nodes[0].name) == [reshape_nodes[0].name]

    def test_analysis_prebuilds_functional_conv_and_marks_shape_aux_nodes(self):
        sample = torch.randn(1, 3, 8, 8)
        gm = trace_for_lowering(FunctionalQuantizedConvWithShapeReshape(), sample)
        ctx = _LoweringContext()

        _analyze_graph(gm, ctx)

        conv_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "__name__", "") == "conv2d"
        ]
        assert len(conv_nodes) == 1
        assert conv_nodes[0] in ctx.prebuilt_ir_nodes

        shape_getattrs = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is getattr
            and len(node.args) >= 2
            and node.args[1] == "shape"
        ]
        assert shape_getattrs
        assert all(node in ctx.aux_bypass_nodes for node in shape_getattrs)
        assert ctx.shape_analysis is not None
        reshape_nodes = [
            node
            for node in gm.graph.nodes
            if ctx.shape_analysis.sink_for(node) is not None
        ]
        assert reshape_nodes

    def test_analysis_prebuilds_functional_conv1d_node(self):
        sample = torch.randn(1, 3, 16)
        gm = trace_for_lowering(FunctionalQuantizedConv1d(), sample)
        ctx = _LoweringContext()

        _analyze_graph(gm, ctx)

        conv_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "__name__", "") == "conv1d"
        ]
        assert len(conv_nodes) == 1
        assert conv_nodes[0] in ctx.prebuilt_ir_nodes


# Parametric test values: kernel_size for AvgPool1d
AVGPOOL1D_KERNEL_SIZES = [1, 2, 3, 4, 5, 8, 9, 16]


class TestAvgPool1dCompilation:
    """Parametric compilation tests for AvgPool1d with various kernel sizes.

    Tests the split-core deployment pattern: Linear-IF -> AvgPool1d-IF.
    """

    @pytest.mark.parametrize("kernel_size", AVGPOOL1D_KERNEL_SIZES)
    def test_avgpool1d_split_core_fusion(self, kernel_size):
        """Verify AvgPool1d-IF is fused into split-core deployment."""
        model = SNNWithAvgPool1dIF(kernel_size)
        graph = compile_to_paiir(model, make_vec_64d())

        # Split-core: Core 1 (SequentialOp) + Core 2 (StandaloneActOp)
        seq_nodes = find_nodes(graph, SequentialOp)
        act_nodes = find_nodes(graph, StandaloneActOp)

        # Should have: Linear-IF (SequentialOp) + AvgPool-core1 (SequentialOp) + IF-core2 (StandaloneActOp)
        assert len(seq_nodes) == 2
        assert len(act_nodes) == 1

    @pytest.mark.parametrize("kernel_size", AVGPOOL1D_KERNEL_SIZES)
    def test_avgpool1d_leak_tau(self, kernel_size):
        """Verify leak_tau=0 for split-core SumPool (no division-by-shift needed).

        For split-core deployment:
        - SumPool outputs sum values directly (no division)
        - LUT thresholds are scaled to sum domain
        - No leak parameters needed since there's no division-by-shift
        """
        model = SNNWithAvgPool1dIF(kernel_size)
        graph = compile_to_paiir(model, make_vec_64d())

        seq_nodes = find_nodes(graph, SequentialOp)
        # Find the SumPool core (second SequentialOp)
        sumpool_core = seq_nodes[-1]

        # Split-core uses SumPool, no leak parameters needed
        assert isinstance(sumpool_core.comp, SumPool1d)
        assert sumpool_core.neuron_params.leak_tau == 0

    @pytest.mark.parametrize("kernel_size", AVGPOOL1D_KERNEL_SIZES)
    def test_avgpool1d_data_format(self, kernel_size):
        """Verify data format propagation through split-core AvgPool."""
        model = SNNWithAvgPool1dIF(kernel_size)
        graph = compile_to_paiir(model, make_vec_64d())

        # All cores should have data format filled
        for node in offline_nodes(graph):
            cp = node.core_params
            assert cp.input_sign is not None
            assert cp.input_width is not None
            assert cp.output_sign is not None
            assert cp.output_width is not None

    @pytest.mark.parametrize("kernel_size", AVGPOOL1D_KERNEL_SIZES)
    def test_avgpool1d_split_core_lut_compensation(self, kernel_size):
        """Verify split-core deployment uses SumPool with scaled LUT thresholds."""
        model = SNNWithAvgPool1dIF(kernel_size)
        graph = compile_to_paiir(model, make_vec_64d())

        seq_nodes = find_nodes(graph, SequentialOp)
        avgpool_core = seq_nodes[-1]

        # Verify split-core deployment: comp is SumPool1d, not AvgPool1d
        assert isinstance(avgpool_core.comp, SumPool1d)

        # Verify LUT thresholds are scaled: threshold' = threshold * window_size
        # Identity LUT has thresholds [0, 1, 2, ...], after compensation [0, k, 2k, ...]
        lut_data = avgpool_core.lut_data
        assert lut_data is not None
        assert lut_data.thresholds[1].item() == kernel_size
        assert lut_data.thresholds[2].item() == kernel_size * 2


class TestAvgPoolLIFSplitCore:
    def test_default_remains_shared_core(self):
        model = SNNWithAvgPoolLIF(kernel_size=3, tau=4.0)
        graph = compile_to_paiir(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(graph, SequentialOp)
        act_nodes = find_nodes(graph, StandaloneActOp)

        avgpool_seq = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)]
        assert len(avgpool_seq) == 1
        assert len(act_nodes) == 0

    def test_split_flag_still_keeps_shared_core_when_scores_tie(self):
        model = SNNWithAvgPoolLIF(kernel_size=3, tau=4.0)
        graph = compile_to_paiir(
            model, make_img_3ch_8x8(), enable_split_avgpool_lif=True
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        act_nodes = find_nodes(graph, StandaloneActOp)

        avgpool_seq = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)]
        assert len(avgpool_seq) == 1
        assert len(act_nodes) == 0

    def test_split_core_selected_when_scoring_prefers_it(self):
        class AvgPoolLIFNoDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.pool = nn.AvgPool2d(2)
                self.lif2 = sj.LIFNode(
                    tau=5.0, decay_input=False, v_threshold=1.0, v_reset=0.0
                )

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif2(self.pool(x))

        graph = compile_to_paiir(
            AvgPoolLIFNoDecay(),
            make_img_3ch_8x8(),
            enable_split_avgpool_lif=True,
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        act_nodes = find_nodes(graph, StandaloneActOp)

        sumpool_core = [n for n in seq_nodes if isinstance(n.comp, SumPool2d)]
        assert len(sumpool_core) == 1
        assert len(act_nodes) == 1

        core1 = sumpool_core[0]
        core2 = act_nodes[0]
        assert core1.act.lut is not None
        assert core1.act.lut.thresholds[1].item() == 1
        assert core1.act.lut.lut_values[1].item() == 1
        assert core1.act.lut.lut_values[4].item() == 4

        # window_size = 4, so all voltage-domain parameters scale by 4
        assert core2.act.thres_pos == 4
        assert core2.act.thres_neg < 0

    def test_split_core_lif_scales_reset_and_init(self):
        class AvgPoolLIFWithOffsets(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.pool = nn.AvgPool2d(2)
                self.lif2 = sj.LIFNode(
                    tau=4.0,
                    decay_input=False,
                    v_threshold=0.25,
                    v_reset=-0.25,
                )

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif2(self.pool(x))

        graph = compile_to_paiir(
            AvgPoolLIFWithOffsets(),
            make_img_3ch_8x8(),
            enable_split_avgpool_lif=True,
        )

        act_nodes = find_nodes(graph, StandaloneActOp)
        assert len(act_nodes) == 1
        core2 = act_nodes[0]
        assert core2.act.thres_pos == 1
        assert core2.act.reset_v == -1
        assert core2.act.init_v == -1

    def test_split_core_lif_falls_back_for_large_window(self):
        model = SNNWithAvgPool1dLIF(kernel_size=256, tau=4.0)
        graph = compile_to_paiir(
            model,
            torch.randn(1, 1, 512),
            enable_split_avgpool_lif=True,
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        act_nodes = find_nodes(graph, StandaloneActOp)

        avgpool_seq = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool1d)]
        assert len(avgpool_seq) == 1
        assert len(act_nodes) == 0

    def test_split_core_lif_kwarg_overrides_config(self):
        model = SNNWithAvgPoolLIF(kernel_size=3, tau=4.0)
        config = CompileConfig(enable_split_avgpool_lif=True)
        graph = compile_to_paiir(
            model,
            make_img_3ch_8x8(),
            compile_config=config,
            enable_split_avgpool_lif=False,
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        assert len([n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)]) == 1


class TestAvgPoolCalibration:
    """Integration tests for AvgPool+LIF threshold calibration.

    Note: Calibration only applies to shared-core AvgPool+LIF deployment.
    AvgPool+IF uses split-core pattern with LUT scaling (lossless).
    AvgPool+ReLU uses LUT scaling (lossless).

    These tests verify the calibration pass infrastructure by checking:
    1. The pass can be enabled/disabled via kwarg and config
    2. CalibrationResult data is accessible
    3. The pass integrates correctly with the compilation pipeline
    """

    def test_calibrate_off_by_default(self):
        """Default compile: calibration is off."""
        # AvgPool+IF uses split-core, no calibration needed
        model = SNNWithAvgPool1dIF(kernel_size=3)
        graph = compile_to_paiir(model, make_vec_64d())
        # Should compile without error
        assert len(graph.nodes) > 0

    def test_calibrate_via_kwarg(self):
        """compile(..., enable_avgpool_calibration=True) enables calibration."""
        calls = 0

        def fake_calibrate_avgpool_thresholds(graph):
            nonlocal calls
            calls += 1
            return {}

        # Build model with SJ LIFNode (FX-compatible) + AvgPool1d
        class AvgPoolLIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(3)
                self.lif = sj.LIFNode(tau=9.0, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        model = AvgPoolLIF()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            compile_mod,
            "calibrate_avgpool_thresholds",
            fake_calibrate_avgpool_thresholds,
        )
        graph = compile_to_paiir(model, make_vec_64d(), enable_avgpool_calibration=True)
        monkeypatch.undo()

        # Verify calibration ran
        assert calls == 1
        seq_nodes = find_nodes(graph, SequentialOp)
        for node in seq_nodes:
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d):
                assert isinstance(node.avgpool_deploy_metadata, AvgPoolDeployMetadata)
                return

        pytest.fail("No AvgPool+LIF SequentialOp node found")

    def test_calibration_flag_is_passed_into_candidate_selection(self, monkeypatch):
        calls: list[bool] = []

        def fake_select_avgpool_lif_candidate(
            act,
            pred_out_width,
            window_size,
            allow_split_lif=False,
            try_calibration=False,
            avg_divisor=None,
        ):
            calls.append(try_calibration)
            return AvgPoolLIFCandidateScore(
                AvgPoolDeployScheme.SHARED_CORE, True, 0.0, 0.0, 0.0, 0.0
            )

        class AvgPoolLIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(3)
                self.lif = sj.LIFNode(tau=9.0, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        monkeypatch.setattr(
            avgpool_fusion,
            "select_avgpool_lif_candidate",
            fake_select_avgpool_lif_candidate,
        )
        monkeypatch.setattr(
            compile_mod, "calibrate_avgpool_thresholds", lambda graph: {}
        )

        graph = compile_to_paiir(
            AvgPoolLIF(), make_vec_64d(), enable_avgpool_calibration=True
        )

        assert calls == [True]
        seq_nodes = find_nodes(graph, SequentialOp)
        avgpool_node = next(
            node
            for node in seq_nodes
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d)
        )
        assert isinstance(avgpool_node.avgpool_deploy_metadata, AvgPoolDeployMetadata)
        assert avgpool_node.avgpool_deploy_metadata.uses_calibration is True

    def test_calibrate_via_config(self, monkeypatch):
        """CompileConfig(enable_avgpool_calibration=True) enables calibration."""
        calls = 0

        def fake_calibrate_avgpool_thresholds(graph):
            nonlocal calls
            calls += 1
            return {}

        class AvgPoolLIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(3)
                self.lif = sj.LIFNode(tau=9.0, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        model = AvgPoolLIF()
        config = CompileConfig(enable_avgpool_calibration=True)
        monkeypatch.setattr(
            compile_mod,
            "calibrate_avgpool_thresholds",
            fake_calibrate_avgpool_thresholds,
        )
        graph = compile_to_paiir(model, make_vec_64d(), compile_config=config)

        assert calls == 1
        seq_nodes = find_nodes(graph, SequentialOp)
        for node in seq_nodes:
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d):
                assert isinstance(node.avgpool_deploy_metadata, AvgPoolDeployMetadata)
                return

        pytest.fail("No AvgPool+LIF SequentialOp node found")

    def test_calibrated_threshold_value(self):
        """After calibration, threshold is set to calibrated value."""

        # For tau=4.0 (power of 2), baseline should be optimal: 1 * 4 = 4
        class AvgPoolLIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(4)
                self.lif = sj.LIFNode(tau=4.0, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        model = AvgPoolLIF()
        graph = compile_to_paiir(model, make_vec_64d(), enable_avgpool_calibration=True)

        seq_nodes = find_nodes(graph, SequentialOp)
        for node in seq_nodes:
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d):
                # For power-of-2 tau, calibration should keep baseline: 1 * 4 = 4
                assert node.neuron_params.thres_pos == 4
                return

        pytest.fail("No AvgPool+LIF SequentialOp node found")

    def test_shared_core_node_marked_uncalibrated_skips_calibration_pass(
        self, monkeypatch
    ):
        def fake_select_avgpool_lif_candidate(
            act,
            pred_out_width,
            window_size,
            allow_split_lif=False,
            try_calibration=False,
            avg_divisor=None,
        ):
            return AvgPoolLIFCandidateScore(
                AvgPoolDeployScheme.SHARED_CORE, False, 0.0, 0.0, 0.0, 0.0
            )

        class AvgPoolLIFNoDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(9)
                self.lif = sj.LIFNode(tau=5.0, decay_input=False, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        monkeypatch.setattr(
            avgpool_fusion,
            "select_avgpool_lif_candidate",
            fake_select_avgpool_lif_candidate,
        )

        graph = compile_to_paiir(
            AvgPoolLIFNoDecay(), make_vec_64d(), enable_avgpool_calibration=True
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        avgpool_node = next(
            node
            for node in seq_nodes
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d)
        )
        assert avgpool_node.avgpool_deploy_metadata is not None
        assert avgpool_node.avgpool_deploy_metadata.uses_calibration is False
        assert avgpool_node.neuron_params.thres_pos == 2

    def test_shared_core_node_marked_calibrated_writes_back_searched_threshold(
        self, monkeypatch
    ):
        def fake_select_avgpool_lif_candidate(
            act,
            pred_out_width,
            window_size,
            allow_split_lif=False,
            try_calibration=False,
            avg_divisor=None,
        ):
            return AvgPoolLIFCandidateScore(
                AvgPoolDeployScheme.SHARED_CORE, True, 0.0, 0.0, 0.0, 0.0
            )

        class AvgPoolLIFNoDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(9)
                self.lif = sj.LIFNode(tau=5.0, decay_input=False, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        monkeypatch.setattr(
            avgpool_fusion,
            "select_avgpool_lif_candidate",
            fake_select_avgpool_lif_candidate,
        )

        graph = compile_to_paiir(
            AvgPoolLIFNoDecay(), make_vec_64d(), enable_avgpool_calibration=True
        )

        seq_nodes = find_nodes(graph, SequentialOp)
        avgpool_node = next(
            node
            for node in seq_nodes
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d)
        )
        assert avgpool_node.avgpool_deploy_metadata is not None
        assert avgpool_node.avgpool_deploy_metadata.uses_calibration is True
        assert avgpool_node.neuron_params.thres_pos == 1

    def test_calibration_search_range(self):
        """Calibration searches in correct range around baseline."""
        # Test the calibration function directly
        act = LIFNodeV25(tau=9.0, v_threshold=1.0)
        window_size = 4
        baseline = 4  # 1 * 4

        result = calibrate_avgpool_threshold(
            act, window_size, baseline, n_steps=100, seed=42, search_ratio=0.5
        )

        # Verify result structure
        assert result.baseline_thres == baseline
        assert result.best_thres <= baseline  # Search range is [baseline*0.5, baseline]
        assert result.best_thres >= baseline * 0.5
        assert result.alpha > 0
        assert result.n_candidates > 0

    def test_kwarg_overrides_config(self, monkeypatch):
        """Explicit kwarg overrides CompileConfig.enable_avgpool_calibration."""
        calls = 0

        def fake_calibrate_avgpool_thresholds(graph):
            nonlocal calls
            calls += 1
            return {}

        class AvgPoolLIF(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.avgpool = nn.AvgPool1d(4)
                self.lif = sj.LIFNode(tau=4.0, v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif(self.avgpool(x))

        model = AvgPoolLIF()

        # Config says True, kwarg says False -> kwarg wins (no calibration)
        config = CompileConfig(enable_avgpool_calibration=True)
        monkeypatch.setattr(
            compile_mod,
            "calibrate_avgpool_thresholds",
            fake_calibrate_avgpool_thresholds,
        )
        graph = compile_to_paiir(
            model,
            make_vec_64d(),
            compile_config=config,
            enable_avgpool_calibration=False,
        )

        assert calls == 0
        seq_nodes = find_nodes(graph, SequentialOp)
        for node in seq_nodes:
            if hasattr(node, "comp") and isinstance(node.comp, nn.AvgPool1d):
                return

        pytest.fail("No AvgPool+LIF SequentialOp node found")
