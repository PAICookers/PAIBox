"""Tests for lowering-layer converter behavior."""

import warnings
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from paicorelib import OnlineCoreUpdateType, OnlineCoreWorkMode
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import neuron as sj
from torch import Tensor, nn

from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.ir.calc_params import (
    OnlineCoreSemanticMode,
    OnlineGradientRole,
    OnlineUpdateDirection,
)
from paibox.paiir.ir.core_neuron import (
    ANNNodeV25,
    IFNodeV25,
    LIFNodeV25,
)
from paibox.paiir.ir.lut_activation import LutCustom, LutReLU
from paibox.paiir.ir.op_node import (
    OnlineCoreOp,
    PadOp,
    SequentialOp,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.lowering.converter import (
    build_default_module_map,
    mark_online,
    register_module,
    register_neuron,
    torch_to_paiir,
)
from tests.paiir.conftest import (
    MultiSpike4,
    UnsupportedSinModel,
    convert_and_fuse,
    find_nodes,
    make_img_3ch_8x8,
    make_multispike4_lut,
    make_vec_8d,
)
from tests.paiir.tracing import trace_with_paiir_tracer


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


class MaxPoolReturnIndicesValuesOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        values, _indices = self.pool(x)
        return values


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

    @pytest.mark.parametrize(
        ("pool_factory", "sample_factory"),
        [
            pytest.param(
                lambda: nn.AvgPool1d(3, stride=2, ceil_mode=True),
                lambda: torch.randn(1, 3, 8),
                id="avgpool1d",
            ),
            pytest.param(
                lambda: nn.AvgPool2d(3, stride=2, ceil_mode=True),
                make_img_3ch_8x8,
                id="avgpool2d",
            ),
            pytest.param(
                lambda: nn.MaxPool1d(3, stride=2, ceil_mode=True),
                lambda: torch.randn(1, 3, 8),
                id="maxpool1d",
            ),
            pytest.param(
                lambda: nn.MaxPool2d(3, stride=2, ceil_mode=True),
                make_img_3ch_8x8,
                id="maxpool2d",
            ),
        ],
    )
    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "non_strict"])
    def test_ceil_mode_pooling_is_hard_error(
        self,
        pool_factory: Callable[[], nn.Module],
        sample_factory: Callable[[], Tensor],
        strict: bool,
    ):
        model = nn.Sequential(pool_factory()).eval()

        with pytest.raises(UnsupportedOpError, match="ceil_mode=True"):
            torch_to_paiir(model, sample_factory(), strict=strict)

    def test_strict_mode_rejects_maxpool_return_indices(self):
        model = MaxPoolReturnIndicesValuesOnly()
        with pytest.raises(UnsupportedOpError, match="return_indices=True"):
            torch_to_paiir(model, make_img_3ch_8x8(), strict=True)

    def test_non_strict_mode_warns_for_maxpool_return_indices(self):
        model = MaxPoolReturnIndicesValuesOnly()
        with pytest.warns(UnsupportedOpWarning, match="return_indices=True"):
            graph = torch_to_paiir(model, make_img_3ch_8x8(), strict=False)
        assert len(graph.nodes) > 0


class TestPadLowering:
    def test_functional_constant_zero_pad_lowers_to_pad_op(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 2, 3, 4), mode="constant", value=0)

        graph = torch_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        assert len(pad_nodes) == 1
        assert pad_nodes[0].padding == (1, 2, 3, 4)
        assert graph.predecessors(pad_nodes[0].name) == ["InputNode_0"]
        assert pad_nodes[0].output_layouts[0].shape == torch.Size((1, 3, 15, 11))

    def test_functional_constant_none_value_lowers_to_pad_op(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (2, 2), mode="constant", value=None)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 3, 8), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        assert len(pad_nodes) == 1
        assert pad_nodes[0].padding == (2, 2)

    @pytest.mark.parametrize(
        ("module", "sample", "expected_padding", "expected_shape"),
        [
            (
                nn.ZeroPad1d((1, 2)),
                torch.randn(1, 3, 8),
                (1, 2),
                torch.Size((1, 3, 11)),
            ),
            (
                nn.ZeroPad2d((1, 2, 3, 4)),
                make_img_3ch_8x8(),
                (1, 2, 3, 4),
                torch.Size((1, 3, 15, 11)),
            ),
            (
                nn.ConstantPad1d((2, 1), 0),
                torch.randn(1, 3, 8),
                (2, 1),
                torch.Size((1, 3, 11)),
            ),
            (
                nn.ConstantPad2d((1, 1, 2, 2), 0.0),
                make_img_3ch_8x8(),
                (1, 1, 2, 2),
                torch.Size((1, 3, 12, 10)),
            ),
        ],
        ids=["zero1d", "zero2d", "constant1d-zero", "constant2d-zero"],
    )
    def test_zero_module_pads_lower_to_pad_op(
        self,
        module: nn.Module,
        sample: Tensor,
        expected_padding: tuple[int, ...],
        expected_shape: torch.Size,
    ):
        graph = torch_to_paiir(nn.Sequential(module).eval(), sample, strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        assert len(pad_nodes) == 1
        assert pad_nodes[0].padding == expected_padding
        assert pad_nodes[0].output_layouts[0].shape == expected_shape

    def test_nonzero_constant_pad_is_unsupported(self):
        with pytest.raises(UnsupportedOpError, match="nonzero"):
            torch_to_paiir(
                nn.Sequential(nn.ConstantPad2d(1, 3)).eval(),
                make_img_3ch_8x8(),
                strict=True,
            )

    @pytest.mark.parametrize(
        "module",
        [
            nn.ReflectionPad1d(1),
            nn.ReflectionPad2d(1),
            nn.ReplicationPad1d(1),
            nn.ReplicationPad2d(1),
            nn.CircularPad1d(1),
            nn.CircularPad2d(1),
        ],
        ids=lambda module: type(module).__name__,
    )
    def test_non_constant_pad_modules_are_unsupported(self, module: nn.Module):
        sample = (
            torch.randn(1, 3, 8)
            if "1d" in type(module).__name__.lower()
            else make_img_3ch_8x8()
        )
        with pytest.raises(UnsupportedOpError, match=type(module).__name__):
            torch_to_paiir(nn.Sequential(module).eval(), sample, strict=True)

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_functional_non_constant_pad_modes_are_unsupported(self, mode: str):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1, 1, 1), mode=mode)

        with pytest.raises(UnsupportedOpError, match=f"mode='{mode}'"):
            torch_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

    def test_functional_nonzero_constant_pad_is_unsupported(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1), mode="constant", value=1)

        with pytest.raises(UnsupportedOpError, match="nonzero"):
            torch_to_paiir(Model().eval(), torch.randn(1, 3, 8), strict=True)


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
        assert torch.equal(
            lut_outputs, ref_outputs
        ), f"LUT mismatch: expected {ref_outputs.tolist()}, got {lut_outputs.tolist()}"

    def test_exact_registration_overrides_builtin_core_neuron_fallback(self):
        register_neuron(
            IFNodeV25, converter=lambda _: ANNNodeV25(make_multispike4_lut())
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = IFNodeV25()

            def forward(self, x):
                return self.act(self.linear(x))

        graph = torch_to_paiir(Model().eval(), make_vec_8d())

        lowered = _find_single_act(graph, ANNNodeV25)
        assert isinstance(lowered.lut, LutCustom)

    def test_register_custom_lut_activation_as_neuron_compat_layer(self):
        lut = make_multispike4_lut()
        register_neuron(MultiSpike4, converter=lambda _: lut)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = MultiSpike4()

            def forward(self, x):
                return self.act(self.linear(x))

        graph = torch_to_paiir(Model().eval(), make_vec_8d())

        lowered = _find_single_act(graph, ANNNodeV25)
        assert lowered.lut is not None
        assert isinstance(lowered.lut, LutCustom)
        assert lowered.lut is not lut
        assert torch.equal(lowered.lut.thresholds, lut.thresholds)
        assert torch.equal(lowered.lut.lut_values, lut.lut_values)
        assert lowered.lut.thresholds is not lut.thresholds
        assert lowered.lut.lut_values is not lut.lut_values

    def test_register_custom_core_neuron_converter_result_is_cloned(self):
        neuron = IFNodeV25(v_threshold=2.0)
        neuron(torch.full((1, 4), 1.0))
        register_neuron(MultiSpike4, converter=lambda _: neuron)

        graph = torch_to_paiir(MultiSpike4().eval(), make_vec_8d())

        lowered = _find_single_act(graph, IFNodeV25)
        assert lowered is not neuron
        assert lowered.thres_pos == neuron.thres_pos
        assert lowered.v == lowered.init_v

    def test_register_spikingjelly_per_channel_ifnode_lowers_to_deploy_only_act(self):
        class _PerChannelThresholdMixin:
            per_channel_threshold: Tensor

            def _threshold_view(self) -> Tensor:
                shape = [1] * self.v.ndim
                shape[1] = self.per_channel_threshold.numel()
                return self.per_channel_threshold.view(shape)

            def neuronal_fire(self):
                return self.surrogate_function(self.v - self._threshold_view())

            def neuronal_reset(self, spike):
                spike_d = spike.detach() if self.detach_reset else spike
                if self.v_reset is None:
                    self.v = self.v - spike_d * self._threshold_view()
                else:
                    self.v = spike_d * self.v_reset + (1.0 - spike_d) * self.v

        class SpikingJellyPerChannelIFNode(_PerChannelThresholdMixin, sj.IFNode):
            per_channel_threshold: Tensor

            def __init__(self, thresholds: Tensor) -> None:
                super().__init__(v_threshold=1.0, v_reset=0.0)
                self.register_buffer("per_channel_threshold", thresholds)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = SpikingJellyPerChannelIFNode(
                    torch.tensor([1.0, 2.0, 3.0, 4.0])
                )

            def forward(self, x):
                return self.act(self.linear(x))

        register_neuron(
            SpikingJellyPerChannelIFNode,
            lambda m: IFNodeV25(
                m.per_channel_threshold, m.v_reset, m.surrogate_function, m.detach_reset
            ),
        )

        fused = convert_and_fuse(Model(), make_vec_8d())
        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 1
        assert isinstance(seq_nodes[0].act, IFNodeV25)
        assert isinstance(seq_nodes[0].act.thres_pos, torch.Tensor)
        assert torch.equal(
            seq_nodes[0].act.thres_pos, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        assert isinstance(seq_nodes[0].neuron_params.thres_pos, torch.Tensor)


class TestOnlineMarking:
    def test_mark_online_linear_expands_to_minimal_training_graph(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x):
                return self.linear(x)

        graph = torch_to_paiir(mark_online(Model()), make_vec_8d())
        graph.lint()

        online_nodes = [
            graph.nodes[name]
            for name in graph.topo_sort()
            if isinstance(graph.nodes[name], OnlineCoreOp)
        ]
        assert len(online_nodes) == 4
        assert [node.core_params.semantic_mode.value for node in online_nodes] == [
            "forward",
            "loss",
            "gradient",
            "update",
        ]
        assert isinstance(online_nodes[0].comp, nn.Linear)
        assert online_nodes[1].comp is None
        assert online_nodes[2].core_params.gradient_role.value == "output"
        assert len(graph.output_nodes()) == 2

    def test_mark_online_subtree_tags_supported_descendants_only(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.linear(x))

        graph = torch_to_paiir(mark_online(Model()), make_vec_8d())

        assert len([n for n in graph.nodes.values() if isinstance(n, OnlineCoreOp)]) == 4
        assert len(
            [n for n in graph.nodes.values() if isinstance(n, StandaloneActOp)]
        ) == 1

    def test_two_layer_online_chain_expands_loss_gradient_update_stages(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 6)
                self.linear2 = nn.Linear(6, 4)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        graph = torch_to_paiir(mark_online(Model()), make_vec_8d())
        graph.lint()
        online_nodes = [
            graph.nodes[name]
            for name in graph.topo_sort()
            if isinstance(graph.nodes[name], OnlineCoreOp)
        ]

        assert [node.core_params.semantic_mode.value for node in online_nodes] == [
            "forward",
            "forward",
            "loss",
            "gradient",
            "gradient",
            "update",
            "update",
        ]
        gradient_nodes = [
            node
            for node in online_nodes
            if node.core_params.semantic_mode.value == "gradient"
        ]
        assert [node.core_params.gradient_role.value for node in gradient_nodes] == [
            "output",
            "hidden",
        ]

    def test_online_training_expansion_rejects_branched_online_path(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(8, 4)
                self.right = nn.Linear(8, 4)

            def forward(self, x):
                return self.left(x) + self.right(x)

        with pytest.raises(NotImplementedError, match="single serial online path"):
            torch_to_paiir(mark_online(Model()), make_vec_8d())

    def test_mark_online_rejects_empty_subtree(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(x)

        with pytest.raises(ValueError, match="nn.Linear only"):
            mark_online(Model())

    def test_mark_online_non_recursive_rejects_unsupported_module(self):
        with pytest.raises(ValueError, match="direct marking of nn.Linear only"):
            mark_online(nn.ReLU(), recursive=False)

    def test_mark_online_rejects_non_field_override_name(self):
        with pytest.raises(TypeError, match="unknown online-core parameter"):
            mark_online(
                nn.Linear(8, 4),
                recursive=False,
                validate_tick_params=True,
            )

    def test_mark_online_rejects_linear_subclass_until_explicitly_supported(self):
        class LinearSubclass(nn.Linear):
            pass

        with pytest.raises(ValueError, match="direct marking of nn.Linear only"):
            mark_online(LinearSubclass(8, 4), recursive=False)

    def test_mark_online_rejects_non_forward_semantic_override(self):
        with pytest.raises(ValueError, match="semantic_mode='forward' only"):
            mark_online(
                nn.Linear(8, 4),
                recursive=False,
                semantic_mode=OnlineCoreSemanticMode.UPDATE,
            )

    def test_mark_online_rejects_gradient_role_override(self):
        with pytest.raises(ValueError, match="does not accept gradient_role"):
            mark_online(
                nn.Linear(8, 4),
                recursive=False,
                gradient_role=OnlineGradientRole.OUTPUT,
            )

    def test_mark_online_rejects_prebound_work_mode(self):
        with pytest.raises(ValueError, match="does not accept prebound work_mode"):
            mark_online(
                nn.Linear(8, 4),
                recursive=False,
                work_mode=OnlineCoreWorkMode.BACKWARD_WEIGHT_UPDATE,
            )

    def test_mark_online_rejects_update_stage_output_width_override(self):
        with pytest.raises(
            ValueError, match="does not accept update-stage output_width"
        ):
            mark_online(
                nn.Linear(8, 4),
                recursive=False,
                output_width=OnlineCoreUpdateType.KAHAN_WEIGHT,
            )

    def test_online_training_expansion_preserves_update_direction_hint(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x):
                return self.linear(x)

        graph = torch_to_paiir(
            mark_online(Model(), update_direction=OnlineUpdateDirection.BACKWARD),
            make_vec_8d(),
        )
        update_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, OnlineCoreOp)
            and node.core_params.semantic_mode is OnlineCoreSemanticMode.UPDATE
        ]

        assert len(update_nodes) == 1
        assert (
            update_nodes[0].core_params.update_direction
            is OnlineUpdateDirection.BACKWARD
        )

    def test_online_training_expansion_rejects_multi_output_graph(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x):
                y = self.linear(x)
                return y, y

        with pytest.raises(NotImplementedError, match="single-output graphs only"):
            torch_to_paiir(mark_online(Model()), make_vec_8d())


class ExplicitQuantConv(nn.Module):
    """Custom deploy module that requires register_module(...) to canonicalize it."""

    def __init__(self) -> None:
        super().__init__()
        weight = torch.randint(-8, 8, (4, 3, 3, 3), dtype=torch.int8)
        bias = torch.randint(-16, 16, (4,), dtype=torch.int32)
        scale = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
        self.register_buffer("weight_int8", weight)
        self.register_buffer("bias_int32", bias)
        self.register_buffer("weight_scale", scale)
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight_int8.to(torch.float32) * self.weight_scale.view(
            -1, 1, 1, 1
        )  # type: ignore
        bias = self.bias_int32.to(torch.float32)
        return F.conv2d(
            x,
            weight,
            bias,  # type: ignore
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _to_canonical_conv2d(m: nn.Module) -> nn.Module:
    """Materialize a canonical Conv2d from ExplicitQuantConv's deploy buffers."""
    assert isinstance(m, ExplicitQuantConv)
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=m.stride,
        padding=m.padding,
        dilation=m.dilation,
        groups=m.groups,
        bias=True,
    )
    with torch.no_grad():
        conv.weight.copy_(
            m.weight_int8.to(conv.weight.dtype) * m.weight_scale.view(-1, 1, 1, 1)  # type: ignore
        )
        assert conv.bias is not None
        conv.bias.copy_(m.bias_int32.to(conv.bias.dtype))  # type: ignore
    return conv


class TestRegisterCanonicalModule:
    def test_register_custom_module_to_canonical_conv(self):
        register_module(ExplicitQuantConv, _to_canonical_conv2d)

        model = ExplicitQuantConv().eval()
        graph = torch_to_paiir(model, make_img_3ch_8x8(), strict=True)

        comp_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]
        assert len(comp_nodes) == 1
        comp = comp_nodes[0].comp
        assert isinstance(comp, nn.Conv2d)
        expected_weight = model.weight_int8.to(
            comp.weight.dtype
        ) * model.weight_scale.view(
            -1, 1, 1, 1
        )  # type: ignore
        assert comp.weight.detach().equal(expected_weight)
        assert comp.bias is not None
        assert comp.bias.detach().equal(model.bias_int32.to(comp.bias.dtype))  # type: ignore
        assert graph.predecessors(comp_nodes[0].name) == ["InputNode_0"]

    def test_register_canonical_module_accepts_neuron_target(self):
        register_module(MultiSpike4, lambda _: ANNNodeV25(make_multispike4_lut()))
        graph = torch_to_paiir(MultiSpike4().eval(), make_vec_8d(), strict=True)

        lowered = _find_single_act(graph, ANNNodeV25)
        assert isinstance(lowered.lut, LutCustom)

    def test_register_canonical_module_accepts_activation_target(self):
        register_module(MultiSpike4, lambda _: nn.ReLU())
        graph = torch_to_paiir(MultiSpike4().eval(), make_vec_8d(), strict=True)

        lowered = _find_single_act(graph, ANNNodeV25)
        assert isinstance(lowered.lut, LutReLU)

    def test_register_canonical_module_rejects_duplicate_registration(self):
        register_module(ExplicitQuantConv, _to_canonical_conv2d)
        with pytest.raises(ValueError, match="already registered"):
            register_module(ExplicitQuantConv, _to_canonical_conv2d)

    def test_register_canonical_module_rejects_ceil_mode_pool(self):
        class CustomPool(nn.Module):
            def forward(self, x):
                return x

        register_module(CustomPool, lambda _: nn.AvgPool2d(3, 2, ceil_mode=True))

        with pytest.raises(UnsupportedOpError, match="ceil_mode=True"):
            torch_to_paiir(CustomPool().eval(), make_img_3ch_8x8(), strict=False)

    def test_register_canonical_module_preempts_unsupported_sj_layer_guard(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.voting = layer.VotingLayer(2)

            def forward(self, x):
                return self.voting(x)

        register_module(
            layer.VotingLayer,
            lambda m: nn.AvgPool1d(int(m.voting_size), int(m.voting_size)),
        )

        graph = torch_to_paiir(Model().eval(), make_vec_8d(), strict=True)

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AvgPool1d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.kernel_size == (2,)
        assert pool_nodes[0].comp.stride == (2,)

    def test_register_canonical_module_can_return_zero_pad(self):
        class CustomPad(nn.Module):
            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad = CustomPad()

            def forward(self, x):
                return self.pad(x)

        register_module(CustomPad, lambda _: nn.ZeroPad2d((1, 2, 3, 4)))

        graph = torch_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=True)

        pad_nodes = find_nodes(graph, PadOp)
        assert len(pad_nodes) == 1
        assert pad_nodes[0].padding == (1, 2, 3, 4)


class TestCoreNeuronV25Lowering:
    def test_ifnodev25_lowers_without_registration_and_does_not_alias_state(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = IFNodeV25(v_threshold=2.0)

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, IFNodeV25)
        assert lowered is not model.act
        assert lowered.thres_pos == model.act.thres_pos
        assert lowered.reset_mode == model.act.reset_mode
        assert lowered.v == lowered.init_v

        model.act(torch.full((1, 4), 1.0))
        assert isinstance(model.act.v, torch.Tensor)
        assert model.act.v.abs().sum().item() > 0
        assert lowered.v == lowered.init_v

    def test_lifnodev25_root_module_lowers_without_registration(self):
        model = LIFNodeV25().eval()
        graph = torch_to_paiir(model, torch.ones(1, 4))

        lowered = _find_single_act(graph, LIFNodeV25)
        assert lowered is not model
        assert lowered.tau == model.tau
        assert lowered.thres_pos == model.thres_pos

    def test_annnodev25_lowers_without_registration_and_clones_lut(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = ANNNodeV25(make_multispike4_lut())

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, ANNNodeV25)
        assert lowered is not model.act
        assert lowered.lut is not None
        assert model.act.lut is not None
        assert lowered.lut is not model.act.lut
        assert torch.equal(lowered.lut.thresholds, model.act.lut.thresholds)
        assert torch.equal(lowered.lut.lut_values, model.act.lut.lut_values)
        assert lowered.lut.thresholds is not model.act.lut.thresholds
        assert lowered.lut.lut_values is not model.act.lut.lut_values


class TestLutActivationLowering:
    def test_direct_lut_module_lowers_without_registration_and_clones_lut(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
                self.act = LutReLU(min_val=-16, max_val=16, output_sign=0)

            def forward(self, x):
                return self.act(self.linear(x))

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        lowered = _find_single_act(graph, ANNNodeV25)
        assert lowered.lut is not None
        assert isinstance(lowered.lut, LutReLU)
        assert lowered.lut is not model.act
        assert torch.equal(lowered.lut.thresholds, model.act.thresholds)
        assert torch.equal(lowered.lut.lut_values, model.act.lut_values)
        assert lowered.lut.thresholds is not model.act.thresholds
        assert lowered.lut.lut_values is not model.act.lut_values

    def test_root_lutcustom_module_lowers_without_registration(self):
        thresholds = torch.arange(256, dtype=torch.float32)
        values = torch.arange(256, dtype=torch.float32)
        model = LutCustom(thresholds, values).eval()

        graph = torch_to_paiir(model, torch.ones(1, 4))

        lowered = _find_single_act(graph, ANNNodeV25)
        assert lowered.lut is not None
        assert isinstance(lowered.lut, LutCustom)
        assert lowered.lut is not model
        assert torch.equal(lowered.lut.thresholds, model.thresholds)
        assert torch.equal(lowered.lut.lut_values, model.lut_values)
        assert lowered.lut.thresholds is not model.thresholds
        assert lowered.lut.lut_values is not model.lut_values


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
        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
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
        assert len(comp_nodes) == 2
        assert {tuple(graph.predecessors(node.name)) for node in comp_nodes} == {
            (split.name,),
        }
        outgoing = sorted(graph.outgoing_edges(split.name), key=lambda edge: edge.dst)
        assert {(edge.dst, edge.dst_port, edge.src_port) for edge in outgoing} == {
            (node.name, 0, 0 if node.comp.in_channels == 2 else 1)
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
        outgoing = graph.outgoing_edges(split_nodes[0].name)
        assert [(edge.dst, edge.dst_port, edge.src_port) for edge in outgoing] == [
            (output_name, 0, 0)
        ]
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
        outgoing = graph.outgoing_edges(split_nodes[0].name)
        assert [(edge.dst, edge.dst_port, edge.src_port) for edge in outgoing] == [
            (output_name, 0, 1)
        ]

    def test_split_direct_outputs_materialize_single_split_node(self):
        class Model(nn.Module):
            def forward(self, x):
                left, right = torch.split(x, [2, 3], dim=1)
                return left, right

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 5, 4, 4))

        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        output_nodes = graph.output_nodes()

        assert len(split_nodes) == 1
        assert len(output_nodes) == 2
        split = split_nodes[0]
        assert all(
            graph.predecessors(node.name) == [split.name] for node in output_nodes
        )
        outgoing = sorted(graph.outgoing_edges(split.name), key=lambda edge: edge.dst)
        assert [(edge.dst, edge.dst_port, edge.src_port) for edge in outgoing] == [
            (output_nodes[0].name, 0, 0),
            (output_nodes[1].name, 0, 1),
        ]

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
        assert type(lowered.surrogate_function) is type(model.act.surrogate_function)
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
        assert type(lowered.surrogate_function) is type(model.act.surrogate_function)
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
        assert type(lowered.surrogate_function) is type(model.act.surrogate_function)
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
        assert type(lowered.surrogate_function) is type(model.act.surrogate_function)
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


class TestSpikingJellyLayerCanonicalization:
    def test_adaptive_pool_modules_use_default_module_map(self):
        module_map = build_default_module_map()

        assert nn.AdaptiveMaxPool1d in module_map
        assert nn.AdaptiveMaxPool2d in module_map
        assert nn.AdaptiveAvgPool1d in module_map
        assert nn.AdaptiveAvgPool2d in module_map

    def test_conv2d_layer_lowers_to_canonical_conv2d(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = layer.Conv2d(3, 4, 3, padding=1, bias=True, step_mode="m")

            def forward(self, x):
                return self.conv(x)

        model = Model().eval()
        with torch.no_grad():
            model.conv.weight.fill_(2.0)
            model.conv.bias.fill_(3.0)  # type: ignore

        graph = torch_to_paiir(model, make_img_3ch_8x8())

        conv_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Conv2d)
        ]
        assert len(conv_nodes) == 1
        comp = conv_nodes[0].comp
        assert isinstance(comp, nn.Conv2d)
        assert torch.equal(comp.weight, model.conv.weight)
        assert torch.equal(comp.bias, model.conv.bias)  # type: ignore

    def test_linear_layer_lowers_to_canonical_linear(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = layer.Linear(8, 4, bias=True, step_mode="m")

            def forward(self, x):
                return self.linear(x)

        model = Model().eval()
        graph = torch_to_paiir(model, make_vec_8d())

        linear_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.Linear)
        ]
        assert len(linear_nodes) == 1
        assert isinstance(linear_nodes[0].comp, nn.Linear)

    def test_pool_and_flatten_layers_lower_to_canonical_nodes(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = layer.MaxPool2d(2, step_mode="m")
                self.avgpool = layer.AvgPool2d(2, step_mode="m")
                self.flatten = layer.Flatten(start_dim=1, step_mode="m")

            def forward(self, x):
                return self.flatten(self.avgpool(self.maxpool(x)))

        graph = torch_to_paiir(Model().eval(), make_img_3ch_8x8())

        assert (
            len(
                [
                    node
                    for node in graph.nodes.values()
                    if isinstance(node, StandaloneCompOp)
                    and isinstance(node.comp, nn.MaxPool2d)
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    node
                    for node in graph.nodes.values()
                    if isinstance(node, StandaloneCompOp)
                    and isinstance(node.comp, nn.AvgPool2d)
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    node
                    for node in graph.nodes.values()
                    if type(node).__name__ == "TransformOp"
                ]
            )
            == 1
        )

    def test_voting_layer_lowers_to_canonical_avgpool1d(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vote = layer.VotingLayer(2, step_mode="m")

            def forward(self, x):
                return self.vote(x)

        graph = torch_to_paiir(Model().eval(), make_vec_8d())

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AvgPool1d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.kernel_size == (2,)
        assert pool_nodes[0].comp.stride == (2,)

    def test_voting_layer_traces_as_leaf_without_exposed_squeeze_unsqueeze(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vote = layer.VotingLayer(2, step_mode="m")

            def forward(self, x):
                return self.vote(x)

        gm = trace_with_paiir_tracer(Model().eval())

        vote_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_module" and node.target == "vote"
        ]
        assert len(vote_nodes) == 1
        assert not [
            node
            for node in gm.graph.nodes
            if (
                node.op == "call_function"
                and node.target in (torch.squeeze, torch.unsqueeze)
            )
            or (node.op == "call_method" and node.target in {"squeeze", "unsqueeze"})
        ]

    def test_dropout_layer_is_erased_in_eval_deploy_path(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = layer.Dropout(p=0.5, step_mode="m")
                self.linear = nn.Linear(8, 4)

            def forward(self, x):
                return self.linear(self.dropout(x))

        graph = torch_to_paiir(Model().eval(), make_vec_8d())

        assert (
            len(
                [
                    node
                    for node in graph.nodes.values()
                    if isinstance(node, StandaloneCompOp)
                    and isinstance(node.comp, nn.Linear)
                ]
            )
            == 1
        )

    def test_regular_adaptive_maxpool2d_lowers_to_standalone_comp(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveMaxPool2d((4, 4))

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), make_img_3ch_8x8())

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AdaptiveMaxPool2d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.output_size == (4, 4)

    def test_irregular_adaptive_maxpool2d_lowers_to_standalone_comp(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveMaxPool2d((4, 4))

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 3, 7, 7))

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AdaptiveMaxPool2d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.output_size == (4, 4)

    def test_irregular_adaptive_maxpool1d_lowers_to_standalone_comp(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveMaxPool1d(4)

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 2, 7))

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AdaptiveMaxPool1d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.output_size == 4

    def test_adaptive_avgpool2d_lowers_to_standalone_comp(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((4, 4))

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 3, 7, 7))

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AdaptiveAvgPool2d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.output_size == (4, 4)

    def test_adaptive_avgpool1d_lowers_to_standalone_comp(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool1d(4)

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), torch.randn(1, 2, 7))

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.AdaptiveAvgPool1d)
        ]
        assert len(pool_nodes) == 1
        assert pool_nodes[0].comp.output_size == 4

    @pytest.mark.parametrize(
        ("module", "sample", "expected_type"),
        [
            (
                layer.AdaptiveAvgPool1d(4, step_mode="m"),
                torch.randn(1, 2, 7),
                nn.AdaptiveAvgPool1d,
            ),
            (
                layer.AdaptiveAvgPool2d((4, 4), step_mode="m"),
                torch.randn(1, 3, 7, 7),
                nn.AdaptiveAvgPool2d,
            ),
        ],
        ids=["layer-adaptive-avgpool1d", "layer-adaptive-avgpool2d"],
    )
    def test_sj_adaptive_avgpool_layers_lower_to_standalone_comp(
        self, module: nn.Module, sample: torch.Tensor, expected_type: type[nn.Module]
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = module

            def forward(self, x):
                return self.pool(x)

        graph = torch_to_paiir(Model().eval(), sample)

        pool_nodes = [
            node
            for node in graph.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, expected_type)
        ]
        assert len(pool_nodes) == 1

    @pytest.mark.parametrize(
        ("module", "sample"),
        [
            (layer.BatchNorm2d(3), make_img_3ch_8x8()),
            (layer.ConvTranspose2d(3, 4, 3), make_img_3ch_8x8()),
            (layer.Conv3d(1, 1, 3), torch.randn(1, 1, 5, 5, 5)),
            (layer.NeuNorm(3, 8, 8), make_img_3ch_8x8()),
        ],
    )
    def test_unsupported_layer_modules_are_rejected_by_lowering(self, module, sample):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

        with pytest.raises(UnsupportedOpError, match="spikingjelly"):
            torch_to_paiir(Model().eval(), sample)

    def test_non_strict_unsupported_layer_warns_and_bypasses_during_lowering(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = layer.ConvTranspose2d(3, 3, 3, padding=1)

            def forward(self, x):
                return self.module(x)

        with pytest.warns(UnsupportedOpWarning, match="spikingjelly"):
            graph = torch_to_paiir(Model().eval(), make_img_3ch_8x8(), strict=False)

        assert "InputNode_0" in graph.nodes
        assert "OutputNode_0" in graph.nodes
