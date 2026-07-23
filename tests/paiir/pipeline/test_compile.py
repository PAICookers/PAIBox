import warnings
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from paicorelib import (
    RM,
    DataSign,
    DataWidth,
    ThresholdNegMode,
    ThresholdPosMode,
)
from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import neuron as sj
from torch import Tensor, nn

import paibox.paiir.pipeline.avgpool.fusion as avgpool_fusion
import paibox.paiir.pipeline.compile as compile_mod
from paibox.paiir import (
    CompileConfig,
    CoreNeuronV25,
    LIFNodeV25,
    compile_to_paiir,
    register_module,
    torch_to_paiir,
)
from paibox.paiir.exceptions import (
    GraphValidationError,
    UnsupportedOpError,
    UnsupportedOpWarning,
)
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    LayoutStage,
    PadOp,
    SequentialOp,
    ShapeStage,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.ir.signal_domain import SignalDomain
from paibox.paiir.lowering.converter import _analyze_graph, _LoweringContext
from paibox.paiir.nn import SumPool1d, SumPool2d
from paibox.paiir.pipeline.avgpool import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
    calibrate_avgpool_threshold,
)
from paibox.paiir.pipeline.avgpool.metadata import AvgPoolDeployMetadata
from paibox.paiir.pipeline.passes import GraphCleanupWarning, validate_graph
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
    find_transform_nodes,
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
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        left = self.relu1(self.conv1(x))
        right = self.relu2(self.conv2(x))
        return self.relu3(self.conv3(torch.cat([left, right], dim=1)))


class PotentialConcatIntoWeightedConsumer(nn.Module):
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


class AvgPool2dWrapper(nn.Module):
    def __init__(self, *, padding: int, count_include_pad: bool) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1, padding, count_include_pad=count_include_pad)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class Pool2dWrapper(nn.Module):
    def __init__(self, pool: nn.Module) -> None:
        super().__init__()
        self.pool = pool

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class SpikingJellyLayerCompileSmoke(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = sj_layer.MaxPool2d(2, step_mode="m")
        self.flatten = sj_layer.Flatten(step_mode="m")
        self.linear = sj_layer.Linear(3 * 4 * 4, 2, step_mode="m")

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.flatten(self.pool(x)))


class VotingLayerCompileSmoke(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vote = sj_layer.VotingLayer(2, step_mode="m")

    def forward(self, x: Tensor) -> Tensor:
        return self.vote(x)


class AdaptiveAvgPoolCompileSmoke(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class AdaptiveAvgPoolRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.pool(x))


class AdaptiveMaxPoolCompileSmoke(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((4, 4))

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class TransformThenLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        transform_fn: Callable[[Tensor], Tensor],
        *,
        out_features: int = 4,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.transform_fn = transform_fn

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.transform_fn(x))


def _make_linear_after_transform_model(
    in_features: int,
    transform_fn: Callable[[Tensor], Tensor],
    *,
    out_features: int = 4,
) -> nn.Module:
    return TransformThenLinear(
        in_features,
        transform_fn,
        out_features=out_features,
    )


def _assert_single_transform_before_linear(
    graph, expected_stage_types: tuple[type[object], ...]
) -> None:
    transform_nodes = find_transform_nodes(graph)
    linear_nodes = [
        node
        for node in find_nodes(graph, StandaloneCompOp)
        if isinstance(node.comp, nn.Linear)
    ]

    assert len(transform_nodes) == 1
    assert len(linear_nodes) == 1
    assert graph.predecessors(transform_nodes[0].name) == ["InputNode_0"]
    assert graph.predecessors(linear_nodes[0].name) == [transform_nodes[0].name]
    assert (
        tuple(type(stage) for stage in transform_nodes[0].stages)
        == expected_stage_types
    )


def _find_single_conv_comp(graph, conv_type: type[nn.Conv1d] | type[nn.Conv2d]):
    comp_nodes = [
        node
        for node in graph.nodes.values()
        if isinstance(node, StandaloneCompOp) and isinstance(node.comp, conv_type)
    ]
    assert len(comp_nodes) == 1
    comp = comp_nodes[0].comp
    assert isinstance(comp, conv_type)
    return comp_nodes[0], comp


TRANSFORM_BEFORE_LINEAR_CASES = (
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: x.transpose(1, 2).flatten(1)
        ),
        torch.randn(1, 2, 3),
        (LayoutStage, ShapeStage),
        id="transpose_then_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(6, lambda x: torch.flatten(x, 1)),
        torch.randn(1, 2, 3),
        (ShapeStage,),
        id="function_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: torch.flatten(x.transpose(1, 2), 1)
        ),
        torch.randn(1, 2, 3),
        (LayoutStage, ShapeStage),
        id="transpose_then_function_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            24, lambda x: x.permute(0, 2, 3, 1).reshape(x.size(0), -1)
        ),
        torch.randn(1, 2, 3, 4),
        (LayoutStage, ShapeStage),
        id="permute_then_reshape",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            24, lambda x: torch.reshape(x, (x.size(0), -1))
        ),
        torch.randn(1, 2, 3, 4),
        (ShapeStage,),
        id="function_reshape",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            24, lambda x: torch.reshape(x.permute(0, 2, 3, 1), (x.size(0), -1))
        ),
        torch.randn(1, 2, 3, 4),
        (LayoutStage, ShapeStage),
        id="permute_then_function_reshape",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: torch.unsqueeze(x, 1).flatten(1)
        ),
        torch.randn(1, 2, 3),
        (ShapeStage, ShapeStage),
        id="unsqueeze_then_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: x.repeat((1, 1, 1)).flatten(1)
        ),
        torch.randn(1, 2, 3),
        (ShapeStage,),
        id="repeat_all_ones_then_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: x.squeeze(1).flatten(1)
        ),
        torch.randn(1, 1, 2, 3),
        (ShapeStage, ShapeStage),
        id="method_squeeze_then_flatten",
    ),
    pytest.param(
        lambda: _make_linear_after_transform_model(
            6, lambda x: torch.squeeze(x, 1).flatten(1)
        ),
        torch.randn(1, 1, 2, 3),
        (ShapeStage, ShapeStage),
        id="function_squeeze_then_flatten",
    ),
)


class PotentialIntoWeightedConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.relu(x)


class PotentialIntoStandalonePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))


class TestUnsupported32BitConsumers:
    def test_compile_rejects_weighted_consumer_of_potential_domain(self):
        with pytest.raises(
            GraphValidationError, match="SequentialOp.*WIDTH_32BIT|WIDTH_32BIT"
        ):
            compile_to_paiir(PotentialIntoWeightedConsumer().eval(), torch.randn(1, 4))

    def test_compile_rejects_standalone_compute_consumer_of_potential_domain(self):
        with pytest.raises(
            GraphValidationError,
            match="Standalone MaxPool .*VALUE-domain predecessor|WIDTH_32BIT",
        ):
            compile_to_paiir(PotentialIntoStandalonePool().eval(), make_img_3ch_8x8())

    def test_compile_rejects_potential_concat_into_weighted_consumer(self):
        with pytest.raises(
            GraphValidationError, match="SequentialOp.*WIDTH_32BIT|WIDTH_32BIT"
        ):
            compile_to_paiir(
                PotentialConcatIntoWeightedConsumer().eval(), make_img_3ch_8x8()
            )


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

    def test_spikingjelly_layer_compile_smoke(self):
        graph = compile_to_paiir(SpikingJellyLayerCompileSmoke(), make_img_3ch_8x8())
        # SJ layer wrappers lower via inheritance; comp is an nn.X subclass instance.
        comp_types = [type(node.comp) for node in find_nodes(graph, StandaloneCompOp)]
        assert any(issubclass(comp_type, nn.MaxPool2d) for comp_type in comp_types)
        assert any(issubclass(comp_type, nn.Linear) for comp_type in comp_types)

    def test_VotingLayer_compile_smoke(self):
        graph = compile_to_paiir(VotingLayerCompileSmoke(), torch.randn(1, 8))

        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.AvgPool1d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.kernel_size == (2,)
        assert pool_nodes[0].comp.stride == (2,)
        assert graph.predecessors(pool_nodes[0].name) == ["InputNode_0"]
        assert pool_nodes[0].core_params.input_sign is not None
        assert pool_nodes[0].core_params.input_width is not None

    @pytest.mark.parametrize(
        ("model_factory", "sample_input", "expected_stage_types"),
        TRANSFORM_BEFORE_LINEAR_CASES,
    )
    def test_transform_before_linear_compiles(
        self,
        model_factory: Callable[[], nn.Module],
        sample_input: torch.Tensor,
        expected_stage_types: tuple[type[object], ...],
    ):
        graph = compile_to_paiir(model_factory(), sample_input)
        _assert_single_transform_before_linear(graph, expected_stage_types)

    def test_shape_then_layout_then_flatten_before_linear_compiles(self):
        class ReshapePermuteFlattenLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(24, 5, bias=False)

            def forward(self, x):
                x = x.reshape(x.size(0), 2, 2, 6)
                x = x.permute(0, 3, 1, 2)
                x = x.flatten(1)
                return self.linear(x)

        graph = compile_to_paiir(ReshapePermuteFlattenLinear(), torch.randn(1, 2, 3, 4))

        _assert_single_transform_before_linear(
            graph, (ShapeStage, LayoutStage, ShapeStage)
        )

    def test_maxpool_after_reshape_compiles(self):
        graph = compile_to_paiir(PoolAfterReshape(), make_img_3ch_8x8())

        reshape_nodes = find_transform_nodes(graph)
        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.MaxPool2d)
        ]

        assert len(reshape_nodes) == 0
        assert len(pool_nodes) == 1
        assert graph.predecessors(pool_nodes[0].name) == ["InputNode_0"]
        assert pool_nodes[0].core_params.input_sign is not None
        assert pool_nodes[0].core_params.input_width is not None

    def test_adaptive_avgpool2d_standalone_compiles(self):
        graph = compile_to_paiir(AdaptiveAvgPoolCompileSmoke(), torch.randn(1, 3, 7, 7))

        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.AdaptiveAvgPool2d)
        ]
        assert len(pool_nodes) == 1
        assert graph.predecessors(pool_nodes[0].name) == ["InputNode_0"]
        assert pool_nodes[0].core_params.input_sign is not None
        assert pool_nodes[0].core_params.input_width is not None

    def test_adaptive_maxpool2d_standalone_compiles_as_value_maxpool(self):
        graph = compile_to_paiir(AdaptiveMaxPoolCompileSmoke(), torch.randn(1, 3, 7, 7))

        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.AdaptiveMaxPool2d)
        ]
        assert len(pool_nodes) == 1
        assert graph.predecessors(pool_nodes[0].name) == ["InputNode_0"]
        assert pool_nodes[0].signal_semantics.output_domain is SignalDomain.VALUE
        assert pool_nodes[0].core_params.input_sign is not None
        assert pool_nodes[0].core_params.input_width is not None

    def test_adaptive_avgpool2d_relu_does_not_use_fixed_avgpool_fusion(self):
        graph = compile_to_paiir(AdaptiveAvgPoolRelu(), torch.randn(1, 3, 7, 7))

        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.AdaptiveAvgPool2d)
        ]
        act_nodes = find_nodes(graph, StandaloneActOp)
        seq_nodes = [
            node
            for node in find_nodes(graph, SequentialOp)
            if isinstance(node.comp, nn.AdaptiveAvgPool2d)
        ]
        assert len(pool_nodes) == 1
        assert len(act_nodes) == 1
        assert seq_nodes == []
        assert graph.predecessors(act_nodes[0].name) == [pool_nodes[0].name]


class TestPadCompilation:
    def test_symmetric_pad_before_conv2d_is_folded_into_conv_padding(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=(1, 1), bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(F.pad(x, (1, 1, 2, 2))))

        graph = compile_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        assert find_nodes(graph, PadOp) == []
        seq_nodes = [
            node
            for node in find_nodes(graph, SequentialOp)
            if isinstance(node.comp, nn.Conv2d)
        ]
        assert len(seq_nodes) == 1
        assert seq_nodes[0].comp.padding == (3, 2)
        assert graph.predecessors(seq_nodes[0].name) == ["InputNode_0"]

    def test_symmetric_pad_before_conv1d_is_folded_into_conv_padding(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 4, 3, padding=1, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(F.pad(x, (2, 2))))

        graph = compile_to_paiir(Model().eval(), torch.randn(1, 3, 16), strict=True)

        assert find_nodes(graph, PadOp) == []
        seq_nodes = [
            node
            for node in find_nodes(graph, SequentialOp)
            if isinstance(node.comp, nn.Conv1d)
        ]
        assert len(seq_nodes) == 1
        assert seq_nodes[0].comp.padding == (3,)
        assert graph.predecessors(seq_nodes[0].name) == ["InputNode_0"]

    def test_asymmetric_pad_before_conv_is_preserved_and_compiles(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=0, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(F.pad(x, (1, 2, 0, 1))))

        graph = compile_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        seq_nodes = [
            node
            for node in find_nodes(graph, SequentialOp)
            if isinstance(node.comp, nn.Conv2d)
        ]
        assert len(pad_nodes) == 1
        assert len(seq_nodes) == 1
        assert seq_nodes[0].comp.padding == (0, 0)
        assert graph.predecessors(seq_nodes[0].name) == [pad_nodes[0].name]
        assert pad_nodes[0].signal_semantics.known_code_range is not None
        lo, hi = pad_nodes[0].signal_semantics.known_code_range
        assert lo <= 0 <= hi

    def test_pad_before_maxpool_is_preserved_and_compiles(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2, 2)

            def forward(self, x):
                return self.pool(F.pad(x, (1, 2, 0, 1)))

        graph = compile_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        pool_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.MaxPool2d)
        ]
        assert len(pad_nodes) == 1
        assert len(pool_nodes) == 1
        assert graph.predecessors(pool_nodes[0].name) == [pad_nodes[0].name]

    def test_direct_output_pad_is_preserved_and_compiles(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (0, 1, 2, 0))

        graph = compile_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        assert len(pad_nodes) == 1
        assert graph.predecessors(graph.output_nodes()[0].name) == [pad_nodes[0].name]
        assert pad_nodes[0].output_layouts[0].shape == torch.Size((1, 3, 10, 9))

    def test_multi_consumer_pad_is_preserved_and_compiles(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 2, 3, padding=0, bias=False)
                self.conv2 = nn.Conv2d(3, 2, 3, padding=0, bias=False)
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()

            def forward(self, x):
                y = F.pad(x, (1, 1, 1, 1))
                return torch.cat(
                    [self.relu1(self.conv1(y)), self.relu2(self.conv2(y))],
                    dim=1,
                )

        graph = compile_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        conv_nodes = [
            node
            for node in find_nodes(graph, SequentialOp)
            if isinstance(node.comp, nn.Conv2d)
        ]
        assert len(pad_nodes) == 1
        assert len(conv_nodes) == 2
        assert sorted(graph.successors(pad_nodes[0].name)) == sorted(
            node.name for node in conv_nodes
        )


class TestDataFormat:
    """Verify data/weight sign & width are correctly inferred after compilation."""

    def test_snn_output_unsigned_1bit(self):
        """SNN (IF/LIF default): fixed signed-8 input, spike outputs remain 1BIT."""
        graph = compile_to_paiir(SNNTwoLayer(), make_img_3ch_8x8())

        nodes = offline_nodes(graph)
        topo = graph.topo_sort()
        first, second = sorted(nodes, key=lambda node: topo.index(node.name))

        assert first.core_params.input_sign == DataSign.SIGNED
        assert first.core_params.input_width == DataWidth.WIDTH_8BIT
        assert first.core_params.output_sign == DataSign.UNSIGNED
        assert first.core_params.output_width == DataWidth.WIDTH_1BIT

        assert second.core_params.input_sign == DataSign.UNSIGNED
        assert second.core_params.input_width == DataWidth.WIDTH_1BIT
        assert second.core_params.output_sign == DataSign.UNSIGNED
        assert second.core_params.output_width == DataWidth.WIDTH_1BIT

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
        "kwargs, expected_duration, expected_initial",
        [
            ({}, 0, 1),
            ({"auto_reset": False}, 1, 0),
            ({"timesteps": 7}, 0, 7),
            ({"timesteps": 7, "auto_reset": False}, 7, 0),
        ],
        ids=[
            "default_auto_reset",
            "manual_reset_default_timesteps",
            "multi_step_auto_reset",
            "multi_step_manual_reset",
        ],
    )
    def test_timesteps_and_auto_reset_mapping(
        self, kwargs, expected_duration, expected_initial
    ):
        """Public timesteps/auto_reset map to internal core tick parameters."""
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(ANNClassifier(), sample_input, **kwargs)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == expected_duration
            assert node.core_params.tick_initial == expected_initial

    @pytest.mark.parametrize(
        ("auto_reset", "expected_duration"),
        [(True, 0), (False, 3)],
    )
    def test_standalone_potential_resets_every_tick(
        self, auto_reset, expected_duration
    ):
        linear = nn.Linear(3, 2)
        graph = compile_to_paiir(
            linear.eval(),
            torch.zeros(1, 3),
            input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
            timesteps=3,
            auto_reset=auto_reset,
        )
        node = next(
            node for node in graph.nodes.values() if isinstance(node, StandaloneCompOp)
        )

        assert node.core_params.tick_duration == expected_duration
        assert node.core_params.tick_initial == (3 if auto_reset else 0)
        assert node.neu_params.reset_mode == RM.MODE_NORMAL
        assert node.neu_params.reset_v == 0
        assert node.neu_params.thres_neg_mode == ThresholdNegMode.FIRE
        assert node.neu_params.thres_pos_mode == ThresholdPosMode.FIRE
        assert node.neu_params.thres_neg == 0
        assert node.neu_params.thres_pos == 0

        neuron = CoreNeuronV25(
            reset_mode=node.neu_params.reset_mode,
            reset_v=node.neu_params.reset_v,
            thres_neg_mode=node.neu_params.thres_neg_mode,
            thres_pos_mode=node.neu_params.thres_pos_mode,
            thres_neg=node.neu_params.thres_neg,
            thres_pos=node.neu_params.thres_pos,
        ).eval()
        neuron(torch.tensor([[5.0, -5.0]]))
        assert torch.equal(neuron.v, torch.zeros(1, 2))
        neuron(torch.tensor([[-2.0, 2.0]]))
        assert torch.equal(neuron.v, torch.zeros(1, 2))

    @pytest.mark.parametrize("timesteps", [0, -1], ids=["zero", "negative"])
    def test_invalid_timesteps_raises(self, timesteps):
        """compile_to_paiir rejects non-positive public timesteps."""
        with pytest.raises(ValueError, match="timesteps.*positive"):
            compile_to_paiir(ANNClassifier(), make_img_3ch_8x8(), timesteps=timesteps)


class TestCompileConfig:
    """CompileConfig and parameter precedence."""

    def test_config_applies(self):
        config = CompileConfig(timesteps=50, auto_reset=False)
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(ANNClassifier(), sample_input, compile_config=config)

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 50
            assert node.core_params.tick_initial == 0

    def test_explicit_kwarg_overrides_config(self):

        config = CompileConfig(timesteps=50, auto_reset=False)
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(
            ANNClassifier(),
            sample_input,
            timesteps=100,
            auto_reset=True,
            compile_config=config,
        )

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 0
            assert node.core_params.tick_initial == 100

    def test_explicit_default_timing_kwarg_overrides_config(self):
        config = CompileConfig(timesteps=50, auto_reset=False)
        sample_input = make_img_3ch_8x8()
        graph = compile_to_paiir(
            ANNClassifier(),
            sample_input,
            timesteps=1,
            auto_reset=True,
            compile_config=config,
        )

        for node in offline_nodes(graph):
            assert node.core_params.tick_duration == 0
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
                AvgPool2dWrapper(padding=1, count_include_pad=False),
                make_img_3ch_8x8(),
                strict=True,
            )

    def test_padding_free_count_include_pad_false_still_compiles(self):
        graph = compile_to_paiir(
            AvgPool2dWrapper(padding=0, count_include_pad=False),
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

    @pytest.mark.parametrize(
        "pool_factory",
        [
            pytest.param(
                lambda: nn.AvgPool2d(3, stride=2, ceil_mode=True),
                id="avgpool2d",
            ),
            pytest.param(
                lambda: nn.MaxPool2d(3, stride=2, ceil_mode=True),
                id="maxpool2d",
            ),
        ],
    )
    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "non_strict"])
    def test_ceil_mode_pooling_is_hard_error(
        self, pool_factory: Callable[[], nn.Module], strict: bool
    ):
        with pytest.raises(UnsupportedOpError, match="ceil_mode=True"):
            compile_to_paiir(
                Pool2dWrapper(pool_factory()),
                make_img_3ch_8x8(),
                strict=strict,
            )


class FunctionalDirectConv(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(4, 3, 3, 3)
        bias = torch.randn(4)
        self.register_buffer("weight_buf", weight)
        self.register_buffer("bias_buf", bias)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight_buf,  # type: ignore
            self.bias_buf,  # type: ignore
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )


class FunctionalDirectConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(4, 3, 3)
        bias = torch.randn(4)
        self.register_buffer("weight_buf", weight)
        self.register_buffer("bias_buf", bias)

    def forward(self, x):
        return F.conv1d(
            x,
            self.weight_buf,  # type: ignore
            self.bias_buf,  # type: ignore
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )


class FunctionalQuantizedConvExpression(nn.Module):
    weight_int8_buf: Tensor
    weight_scale_buf: Tensor
    bias_buf: Tensor

    def __init__(self):
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias = torch.randn(4)
        scale = torch.tensor(0.125)
        self.register_buffer("weight_int8_buf", weight)
        self.register_buffer("weight_scale_buf", scale)
        self.register_buffer("bias_buf", bias)
        self.weight_int8_buf = weight
        self.weight_scale_buf = scale
        self.bias_buf = bias

    def forward(self, x):
        w = self.weight_int8_buf.to(x.dtype) * self.weight_scale_buf
        return F.conv2d(x, w, self.bias_buf, stride=1, padding=1, dilation=1, groups=1)


class FunctionalMixedDynamicToConvExpression(nn.Module):
    weight_int8_buf: Tensor
    bias_buf: Tensor

    def __init__(self):
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias = torch.randn(4)
        self.register_buffer("weight_int8_buf", weight)
        self.register_buffer("bias_buf", bias)
        self.weight_int8_buf = weight
        self.bias_buf = bias

    def forward(self, x):
        w = self.weight_int8_buf.to(x.dtype, copy=False)
        return F.conv2d(x, w, self.bias_buf, stride=1, padding=1, dilation=1, groups=1)


class FunctionalDirectConvWithShapeReshape(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(4, 3, 3, 3)
        bias = torch.randn(4)
        self.register_buffer("weight_buf", weight)
        self.register_buffer("bias_buf", bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        expanded = x.unsqueeze(0).repeat(1, 1, 1, 1, 1)
        flat = expanded.flatten(0, 1)
        y = F.conv2d(
            flat,
            self.weight_buf,  # type: ignore
            self.bias_buf,  # type: ignore
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )
        y = y.reshape(
            expanded.shape[0], expanded.shape[1], -1, y.shape[-2], y.shape[-1]
        )
        return self.relu(y)


class ExplicitQuantizedConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        weight_int8 = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias_int32 = torch.randint(-16, 16, (4,), dtype=torch.int32)
        weight_scale = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
        self.register_buffer("weight_int8_buf", weight_int8)
        self.register_buffer("bias_int32_buf", bias_int32)
        self.register_buffer("weight_scale_buf", weight_scale)
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x):
        weight = self.weight_int8_buf.to(x.dtype) * self.weight_scale_buf.view(
            -1, 1, 1, 1
        )  # type: ignore
        bias = self.bias_int32_buf.to(x.dtype)
        return F.conv2d(
            x,
            weight,
            bias,  # type: ignore
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _canonicalize_explicit_quantized_conv(module: nn.Module) -> nn.Module:
    assert isinstance(module, ExplicitQuantizedConvModule)
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=True,
    )
    with torch.no_grad():
        conv.weight.copy_(
            module.weight_int8_buf.to(conv.weight.dtype)
            * module.weight_scale_buf.view(-1, 1, 1, 1)
        )  # type: ignore
        assert conv.bias is not None
        conv.bias.copy_(module.bias_int32_buf.to(conv.bias.dtype))  # type: ignore
        conv.weight.requires_grad_(False)
        conv.bias.requires_grad_(False)
    return conv


class TestFunctionalConv:
    def test_direct_functional_conv2d_supported_in_strict_mode(self):
        model = FunctionalDirectConv()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(model, torch.randn(1, 3, 8, 8), strict=True)

        comp_node, comp = _find_single_conv_comp(graph, nn.Conv2d)
        assert comp.weight.detach().equal(model.weight_buf)  # type: ignore
        assert comp.bias is not None
        assert comp.bias.detach().equal(model.bias_buf)  # type: ignore
        assert graph.predecessors(comp_node.name) == ["InputNode_0"]
        assert not any("conv2d" in str(w.message) for w in caught)

    def test_direct_functional_conv1d_supported_in_strict_mode(self):
        model = FunctionalDirectConv1d()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = compile_to_paiir(model, torch.randn(1, 3, 16), strict=True)

        comp_node, comp = _find_single_conv_comp(graph, nn.Conv1d)
        assert comp.weight.detach().equal(model.weight_buf)  # type: ignore
        assert comp.bias is not None
        assert comp.bias.detach().equal(model.bias_buf)  # type: ignore
        assert graph.predecessors(comp_node.name) == ["InputNode_0"]
        assert not any("conv1d" in str(w.message) for w in caught)

    def test_functional_conv_quantized_weight_expression_is_unsupported(self):
        model = FunctionalQuantizedConvExpression()

        with pytest.raises(UnsupportedOpError):
            compile_to_paiir(model, torch.randn(1, 3, 8, 8), strict=True)

    def test_functional_conv_mixed_dynamic_to_is_unsupported(self):
        model = FunctionalMixedDynamicToConvExpression()

        with pytest.raises(UnsupportedOpError):
            compile_to_paiir(model, torch.randn(1, 3, 8, 8), strict=True)

    def test_registered_custom_conv_module_compiles_via_explicit_canonical_mapping(
        self,
    ):
        register_module(
            ExplicitQuantizedConvModule, _canonicalize_explicit_quantized_conv
        )

        model = ExplicitQuantizedConvModule().eval()
        graph = compile_to_paiir(model, torch.randn(1, 3, 8, 8), strict=True)

        _, comp = _find_single_conv_comp(graph, nn.Conv2d)
        expected_weight = model.weight_int8_buf.to(
            comp.weight.dtype
        ) * model.weight_scale_buf.view(
            -1, 1, 1, 1
        )  # type: ignore
        assert comp.weight.detach().equal(expected_weight)
        assert comp.bias is not None
        assert comp.bias.detach().equal(
            model.bias_int32_buf.to(comp.bias.dtype)  # type: ignore
        )

    def test_shape_only_reshape_args_do_not_become_data_predecessors(self):
        graph = torch_to_paiir(
            FunctionalDirectConvWithShapeReshape(),
            torch.randn(1, 3, 8, 8),
            strict=False,
        )
        graph.summary()

        transform_nodes = find_transform_nodes(graph)
        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]
        assert len(transform_nodes) == 4
        assert len(comp_nodes) == 1
        pre_comp_transforms = transform_nodes[:3]
        assert graph.predecessors(pre_comp_transforms[0].name) == ["InputNode_0"]
        assert graph.predecessors(pre_comp_transforms[1].name) == [
            pre_comp_transforms[0].name
        ]
        assert graph.predecessors(pre_comp_transforms[2].name) == [
            pre_comp_transforms[1].name
        ]
        assert graph.predecessors(comp_nodes[0].name) == [pre_comp_transforms[2].name]

    def test_unsqueeze_repeat_all_ones_compile_path_succeeds(self):
        graph = compile_to_paiir(
            FunctionalDirectConvWithShapeReshape(),
            torch.randn(1, 3, 8, 8),
            strict=True,
        )

        transform_nodes = find_transform_nodes(graph)
        seq_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, SequentialOp) and isinstance(node.comp, nn.Conv2d)
        ]

        assert len(transform_nodes) == 1
        assert len(seq_nodes) == 1
        assert graph.predecessors(seq_nodes[0].name) == [graph.input_nodes()[0].name]
        assert graph.predecessors(transform_nodes[0].name) == [seq_nodes[0].name]

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

        reshape_nodes = find_transform_nodes(graph)
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
        gm = trace_for_lowering(FunctionalDirectConvWithShapeReshape(), sample)
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
        gm = trace_for_lowering(FunctionalDirectConv1d(), sample)
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


class TestSplitCompilation:
    def test_torch_to_paiir_split_concat_reshape_summary_smoke(self):
        class Model(nn.Module):
            def forward(self, x):
                left, right = torch.split(x, [2, 3], dim=1)
                y = torch.cat([right, left], dim=1)
                return y.reshape(y.shape[0], -1)

        sample = torch.randn(1, 5, 4, 4)
        graph = torch_to_paiir(Model().eval(), sample)
        validate_graph(graph)

        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        concat_nodes = [
            node for node in graph.nodes.values() if isinstance(node, ConcatOp)
        ]
        reshape_nodes = find_transform_nodes(graph)

        assert len(split_nodes) == 1
        assert len(concat_nodes) == 1
        assert len(reshape_nodes) == 1

        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            graph.summary(verbose=True)
        summary = buf.getvalue()
        assert "out[0] shape=(1, 2, 4, 4), dims=(0, 1, 2, 3)" in summary
        assert "out[1] shape=(1, 3, 4, 4), dims=(0, 1, 2, 3)" in summary
        assert "src_port=0 ->" in summary
        assert "src_port=1 ->" in summary

    def test_compile_to_paiir_rejects_frontend_only_split_op(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, 1, bias=False)

            def forward(self, x):
                left, _right = torch.split(x, [2, 3], dim=1)
                return self.conv(left)

        with pytest.raises(GraphValidationError, match="frontend-only IR"):
            compile_to_paiir(Model().eval(), torch.randn(1, 5, 4, 4))


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
        assert sumpool_core.neu_params.leak_tau == 0

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
        assert avgpool_core.act.lut is not None
        logical_lut = avgpool_core.act.lut.logical_lut_data
        assert logical_lut.thresholds[1].item() == kernel_size
        assert logical_lut.thresholds[2].item() == kernel_size * 2


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
                assert node.neu_params.thres_pos == 4
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
        assert avgpool_node.neu_params.thres_pos == 2

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
        assert avgpool_node.neu_params.thres_pos == 1

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
