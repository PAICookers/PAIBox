"""Tests for converter and passes using shared helpers from conftest."""

import pytest
import torch
from paicorelib import SNNMode
from torch import nn

from paibox.paiir.calc_params import OfflineCoreParams
from paibox.paiir.converter import register_neuron, torch_to_paiir
from paibox.paiir.core_neuron import ANNNodeV25, LIFNodeV25
from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.lut_activation import LutCustom, LutReLU, LutSigmoid, LutTanh
from paibox.paiir.op_node import (
    AccumulateOp,
    AddOp,
    ConcatOp,
    OfflineCoreOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.passes import (
    assign_tick_params,
    fuse_to_offline_cores,
    propagate_data_format,
)

from .conftest import (
    ANNClassifier,
    ANNConvBNReLU,
    ANNResidualSubtract,
    AvgPool2IF,
    MultiInputMerge,
    SNNDepthwiseSeparable,
    SNNFlattenTransition,
    SNNResidualAdd,
    SNNWithAvgPoolIF,
    SNNWithMaxPool,
    SPPFBlock,
    StandaloneConv,
    UnsupportedSinModel,
    convert_and_fuse,
    find_first,
    find_node_names,
    find_nodes,
    make_img_1ch_4x4,
    make_img_3ch_8x8,
    make_img_3ch_9x9,
    make_img_16ch_8x8,
    make_multispike4_lut,
    make_vec_8d,
)


class TestSNNConversion:
    """Test SNN network conversion through the full pipeline."""

    def test_two_layer_snn(self):
        """Conv-LIF -> Conv-IF: two SequentialOps after fusion."""
        from .conftest import SNNTwoLayer

        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        comp_types = {type(n.comp) for n in seq_nodes}
        assert comp_types == {nn.Conv2d}
        act_types = sorted(type(n.act).__name__ for n in seq_nodes)
        assert act_types == ["IFNodeV25", "LIFNodeV25"]

        assert find_nodes(fused, StandaloneCompOp) == []
        assert find_nodes(fused, StandaloneActOp) == []

    def test_residual_add(self):
        """Two conv branches + add + LIF: AccumulateOp after fusion."""
        model = SNNResidualAdd()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert len(accum_nodes[0].comps) == 2
        assert isinstance(accum_nodes[0].act, LIFNodeV25)
        assert accum_nodes[0].signs == (1, 1)

        assert find_nodes(fused, AddOp) == []
        assert find_nodes(fused, StandaloneCompOp) == []

    def test_maxpool_chain(self):
        """Conv-LIF -> MaxPool-LIF: pooling mode inferred."""
        model = SNNWithMaxPool()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        pool_ops = [n for n in seq_nodes if isinstance(n.comp, nn.MaxPool2d)]
        assert len(pool_ops) == 1

        from paicorelib import PoolingMode

        assert pool_ops[0].core_params.pooling_mode == PoolingMode.MAX

    def test_depthwise_separable(self):
        """DWConv-LIF -> PWConv-LIF: grouped conv handled."""
        model = SNNDepthwiseSeparable()
        fused = convert_and_fuse(model, make_img_16ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        conv_ops = [n for n in seq_nodes if isinstance(n.comp, nn.Conv2d)]
        groups = sorted([n.comp.groups for n in conv_ops])
        assert groups == [1, 16]

    def test_flatten_transition(self):
        """Conv-IF -> flatten -> Linear-IF: flatten bypassed."""
        model = SNNFlattenTransition()
        fused = convert_and_fuse(model, make_img_1ch_4x4())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        comp_types = sorted(type(n.comp).__name__ for n in seq_nodes)
        assert comp_types == ["Conv2d", "Linear"]


class TestANNConversion:
    """Test ANN network conversion through the full pipeline."""

    def test_classifier_head(self):
        """Conv-ReLU -> AvgPool -> flatten -> Linear-Sigmoid."""
        model = ANNClassifier()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) >= 2

        lut_types = {type(n.act.lut) for n in seq_nodes if n.act.lut is not None}
        assert LutReLU in lut_types
        assert LutSigmoid in lut_types

    def test_conv_bn_relu(self):
        """Conv-BN-ReLU chain: BN bypassed, Conv fuses with ReLU."""
        model = ANNConvBNReLU()
        model.eval()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        assert all(isinstance(n.act.lut, LutReLU) for n in seq_nodes)

        for node in fused.nodes.values():
            if hasattr(node, "comp"):
                assert not isinstance(node.comp, (nn.BatchNorm1d, nn.BatchNorm2d))

    def test_subtract_branch(self):
        """Two linear branches with subtraction -> tanh."""
        model = ANNResidualSubtract()
        fused = convert_and_fuse(model, make_vec_8d())

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert accum_nodes[0].signs == (1, -1)
        assert isinstance(accum_nodes[0].act.lut, LutTanh)


class TestComplexPatterns:
    """Test complex/real-world network patterns."""

    def test_multi_input(self):
        """Two separate inputs merged by add + LIF."""
        model = MultiInputMerge()
        unfused = torch_to_paiir(model, make_img_3ch_8x8(), make_img_3ch_8x8())
        fused = fuse_to_offline_cores(unfused)

        assert len(fused.input_nodes()) == 2

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert len(accum_nodes[0].comps) == 2

    def test_sppf_block(self):
        """SPPF: cascaded MaxPool with shared module, cat merge."""
        model = SPPFBlock()
        unfused = torch_to_paiir(model, make_img_16ch_8x8())

        comp_nodes = find_nodes(unfused, StandaloneCompOp)
        pool_nodes = [n for n in comp_nodes if isinstance(n.comp, nn.MaxPool2d)]
        assert len(pool_nodes) >= 3

    def test_standalone_conv(self):
        """Conv with no activation stays StandaloneCompOp."""
        model = StandaloneConv()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        comp_nodes = find_nodes(fused, StandaloneCompOp)
        assert len(comp_nodes) == 1
        assert isinstance(comp_nodes[0].comp, nn.Conv2d)

    def test_concat_op(self):
        """torch.cat creates ConcatOp with correct dim and port ordering."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 1)
                self.conv2 = nn.Conv2d(3, 4, 1)
                self.conv3 = nn.Conv2d(8, 2, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                c = torch.cat([a, b], dim=1)
                return self.relu(self.conv3(c))

        model = M()
        unfused = torch_to_paiir(model, make_img_3ch_8x8())

        # Check ConcatOp exists
        concat_nodes = find_nodes(unfused, ConcatOp)
        assert len(concat_nodes) == 1
        assert concat_nodes[0].dim == 1

        # Check port ordering: conv1 -> port 0, conv2 -> port 1
        concat_name = find_node_names(unfused, ConcatOp)[0]
        preds = unfused.predecessors(concat_name)
        assert len(preds) == 2

        fused = fuse_to_offline_cores(unfused)
        propagate_data_format(fused)
        assign_tick_params(fused)

        # Verify data format propagation through ConcatOp
        assert all(
            node.core_params.input_sign is not None
            for node in fused.nodes.values()
            if isinstance(node, OfflineCoreOp)
        )


class TestShapeAndDims:
    """Test shape and dims propagation."""

    def test_shape_propagation(self):
        """Shapes propagate correctly through a two-layer SNN."""
        from .conftest import SNNTwoLayer

        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        for node in seq_nodes:
            assert node.output_shape != ()
            assert all(s != () for s in node.input_shapes)

    def test_no_sample_input(self):
        """Converter works without sample input, shapes default to ()."""
        from .conftest import SNNTwoLayer

        model = SNNTwoLayer()
        unfused = torch_to_paiir(model)
        fused = fuse_to_offline_cores(unfused)

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        for node in seq_nodes:
            assert node.output_shape == ()

    def test_dims_identity(self):
        """Identity dims for standard conv/linear (no transpose)."""
        from .conftest import SNNTwoLayer

        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        for node in seq_nodes:
            assert node.output_dims == (0, 1, 2, 3)

    def test_flatten_dims(self):
        """flatten resets dims to identity."""
        model = SNNFlattenTransition()
        fused = convert_and_fuse(model, make_img_1ch_4x4())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        for node in seq_nodes:
            assert node.output_dims in [(0, 1, 2, 3), (0, 1)]


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


class TestAssignTickParams:
    """Test assign_tick_params pass."""

    @pytest.fixture
    def fused_snn(self):
        """Fixture: fused SNNTwoLayer graph."""
        from .conftest import SNNTwoLayer

        return convert_and_fuse(SNNTwoLayer(), make_img_3ch_8x8())

    def test_tick_start_from_depth(self, fused_snn):
        """tick_start is assigned based on DAG depth."""
        assign_tick_params(fused_snn)

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        assert len(seq_nodes) == 2

        for node in seq_nodes:
            assert node.core_params.tick_start is not None
            assert node.core_params.tick_start >= 1

        tick_starts = sorted([n.core_params.tick_start for n in seq_nodes])
        assert tick_starts[0] < tick_starts[1]

    def test_tick_start_explicit_override(self, fused_snn):
        """User-set tick_start is preserved by the pass."""
        first_op = find_first(fused_snn, SequentialOp)
        first_op.core_params.tick_start = 42

        assign_tick_params(fused_snn)
        assert first_op.core_params.tick_start == 42

    @pytest.mark.parametrize(
        "tick_duration, expected",
        [(0, 0), (100, 100)],
        ids=["default_always_working", "global_override"],
    )
    def test_tick_duration(self, fused_snn, tick_duration, expected):
        """tick_duration: default (0) or graph-level override applied to all nodes."""
        assign_tick_params(fused_snn, tick_duration=tick_duration)

        for node in fused_snn.nodes.values():
            if isinstance(node, SequentialOp):
                assert node.core_params.tick_duration == expected

    def test_tick_duration_per_node_override_via_core_params(self, fused_snn):
        """Per-node tick_duration set on core_params takes priority."""
        first_op = find_first(fused_snn, SequentialOp)
        first_op.core_params.tick_duration = 50

        assign_tick_params(fused_snn, tick_duration=100)
        assert first_op.core_params.tick_duration == 50

    @pytest.mark.parametrize(
        "tick_duration, auto_reset, expected_initial",
        [(100, True, 100), (0, True, 0), (100, False, 0)],
        ids=["reset_with_duration", "reset_always_working", "no_reset"],
    )
    def test_auto_reset(self, fused_snn, tick_duration, auto_reset, expected_initial):
        """auto_reset controls tick_initial derivation from tick_duration."""
        assign_tick_params(
            fused_snn, tick_duration=tick_duration, auto_reset=auto_reset
        )

        for node in fused_snn.nodes.values():
            if isinstance(node, SequentialOp):
                assert node.core_params.tick_initial == expected_initial

    def test_tick_override_tick_start(self, fused_snn):
        """overrides dict can set tick_start for a specific node."""
        seq_names = find_node_names(fused_snn, SequentialOp)
        assert len(seq_names) == 2

        assign_tick_params(fused_snn, overrides={seq_names[0]: {"tick_start": 10}})

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        by_name = {n.name: n for n in seq_nodes}
        assert by_name[seq_names[0]].core_params.tick_start == 10
        assert by_name[seq_names[1]].core_params.tick_start is not None
        assert by_name[seq_names[1]].core_params.tick_start != 10

    def test_tick_override_duration_and_auto_reset(self, fused_snn):
        """overrides dict can set tick_duration and auto_reset per-node."""
        seq_names = find_node_names(fused_snn, SequentialOp)

        assign_tick_params(
            fused_snn,
            tick_duration=100,
            auto_reset=True,
            overrides={seq_names[0]: {"tick_duration": 200, "auto_reset": False}},
        )

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        by_name = {n.name: n for n in seq_nodes}

        cp0 = by_name[seq_names[0]].core_params
        assert cp0.tick_duration == 200
        assert cp0.tick_initial == 0

        cp1 = by_name[seq_names[1]].core_params
        assert cp1.tick_duration == 100
        assert cp1.tick_initial == 100

    def test_residual_tick_start(self):
        """Residual (AccumulateOp) gets correct tick_start."""
        fused = convert_and_fuse(SNNResidualAdd(), make_img_3ch_8x8())
        assign_tick_params(fused)

        accum_ops = find_nodes(fused, AccumulateOp)
        assert len(accum_ops) == 1
        assert accum_ops[0].core_params.tick_start == 1

    @pytest.mark.parametrize(
        "kwargs, error_match",
        [
            ({"tick_duration": -1}, "tick_duration.*must be non-negative"),
            ({"tick_start": -5}, "tick_start.*non-negative"),
        ],
        ids=["negative_graph_duration", "negative_override_start"],
    )
    def test_negative_param_raises(self, fused_snn, kwargs, error_match):
        """Negative tick parameters raise ValueError immediately."""
        if "tick_duration" in kwargs:
            with pytest.raises(ValueError, match=error_match):
                assign_tick_params(fused_snn, tick_duration=kwargs["tick_duration"])
        else:
            seq_name = find_node_names(fused_snn, SequentialOp)[0]
            with pytest.raises(ValueError, match=error_match):
                assign_tick_params(fused_snn, overrides={seq_name: kwargs})

    def test_override_negative_tick_duration_raises(self, fused_snn):
        """Negative tick_duration in overrides raises immediately."""
        seq_name = find_node_names(fused_snn, SequentialOp)[0]
        with pytest.raises(ValueError, match="tick_duration.*non-negative"):
            assign_tick_params(fused_snn, overrides={seq_name: {"tick_duration": -10}})

    def test_override_unknown_node_raises(self):
        """Override key for non-existent node raises KeyError."""
        from .conftest import SNNTwoLayer

        fused = convert_and_fuse(SNNTwoLayer(), make_img_3ch_8x8())
        with pytest.raises(KeyError, match="does not match any node"):
            assign_tick_params(fused, overrides={"nonexistent_node": {"tick_start": 1}})

    def test_validate_tick_params_unassigned(self):
        """validate_tick_params raises if tick_start is still None."""
        cp = OfflineCoreParams()
        assert cp.tick_start is None
        with pytest.raises(ValueError, match="tick_start is None"):
            cp.validate_tick_params()

    @pytest.mark.parametrize(
        "params, error_match",
        [
            ({"tick_start": -1}, "tick_start"),
            ({"tick_start": 1, "tick_duration": -1}, "tick_duration"),
            ({"tick_start": 1, "tick_initial": -1}, "tick_initial"),
        ],
        ids=["negative_start", "negative_duration", "negative_initial"],
    )
    def test_validate_tick_params_out_of_range(self, params, error_match):
        """validate_tick_params raises on out-of-range values."""
        cp = OfflineCoreParams(**params)
        with pytest.raises(ValueError, match=error_match):
            cp.validate_tick_params()


class TestRegisterNeuron:
    """Test register_neuron() API for custom neuron registration."""

    def test_register_custom_neuron(self):
        """Register MultiSpike4 -> compile a model -> verify graph structure."""
        from .conftest import MultiSpike4

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


class TestSplitCoreAvgPoolIF:
    """Test split-core deployment for AvgPool + IFNodeV25 pattern."""

    def test_split_core_structure(self):
        """AvgPool + IF splits into SequentialOp(AvgPool, ANN) + StandaloneActOp(IF)."""
        model = SNNWithAvgPoolIF()
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        # First layer: Conv + IF -> SequentialOp
        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2  # Conv+IF and AvgPool+ANN

        avgpool_cores = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)]
        assert len(avgpool_cores) == 1

        # Split-core: standalone IF
        standalone_acts = find_nodes(fused, StandaloneActOp)
        assert len(standalone_acts) == 1

    def test_core1_is_ann_with_identity_lut(self):
        """Core 1 (AvgPool core) operates in ANN mode with identity LUT."""
        model = SNNWithAvgPoolIF()
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        seq_nodes = find_nodes(fused, SequentialOp)
        avgpool_core = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)][0]

        assert avgpool_core.core_params.snn_mode == SNNMode.ANN
        assert avgpool_core.act.lut is not None
        assert isinstance(avgpool_core.act.lut, LutCustom)

        # Identity LUT: output == input in the non-negative range.
        # Predecessor is IFNodeV25 (unsigned spikes), so AvgPool sum >= 0.
        test_vals = torch.arange(0, 128)
        outputs, _ = avgpool_core.act.lut.lookup(test_vals.float())
        assert torch.equal(outputs, test_vals)

    def test_core2_has_compensated_threshold(self):
        """Core 2 (IF core) has threshold compensated for 3x3 pool shift error."""
        model = SNNWithAvgPoolIF()
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        standalone_acts = find_nodes(fused, StandaloneActOp)
        assert len(standalone_acts) == 1
        if_core = standalone_acts[0]

        # window_size = 9 (3x3 pool), factor = 9/8 = 1.125
        # Original threshold = 1.0, reset_v = 0.0
        # theta' = 0 + (1 - 0) * 1.125 = 1.125
        assert if_core.act.thres_pos == pytest.approx(1.125)
        assert if_core.act.lut is None

    def test_edge_connectivity(self):
        """Core 1 -> Core 2 edge exists in the fused graph."""
        model = SNNWithAvgPoolIF()
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        seq_nodes = find_nodes(fused, SequentialOp)
        avgpool_core = [n for n in seq_nodes if isinstance(n.comp, nn.AvgPool2d)][0]

        standalone_acts = find_nodes(fused, StandaloneActOp)
        if_core = standalone_acts[0]

        # Check that avgpool_core -> if_core edge exists
        succs = fused.successors(avgpool_core.name)
        assert if_core.name in succs

    def test_window_size_power_of_2_no_threshold_compensation(self):
        """When window_size is power of 2, IF threshold is unchanged."""
        model = AvgPool2IF()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        standalone_acts = find_nodes(fused, StandaloneActOp)
        assert len(standalone_acts) == 1
        assert standalone_acts[0].act.thres_pos == pytest.approx(1.0)
