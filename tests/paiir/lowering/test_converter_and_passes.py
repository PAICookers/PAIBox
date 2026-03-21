"""Tests for converter and passes using shared helpers from conftest."""

import pytest
import torch
from paicorelib import SNNMode
from torch import nn

from paibox.paiir.exceptions import UnsupportedOpError, UnsupportedOpWarning
from paibox.paiir.ir.calc_params import OfflineCoreParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, LIFNodeV25
from paibox.paiir.ir.lut_activation import LutCustom, LutReLU, LutSigmoid, LutTanh
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    AddOp,
    ConcatOp,
    OfflineCoreOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from paibox.paiir.lowering.converter import register_neuron, torch_to_paiir
from paibox.paiir.nn import SumPool2d
from paibox.paiir.pipeline.avgpool import AvgPoolDeployMetadata
from paibox.paiir.pipeline.passes import (
    assign_tick_params,
    fuse_to_offline_cores,
    propagate_data_format,
)
from tests.paiir.conftest import (
    ANNClassifier,
    ANNConvBNReLU,
    ANNResidualSubtract,
    MultiInputMerge,
    SNNDepthwiseSeparable,
    SNNFlattenTransition,
    SNNResidualAdd,
    SNNWithAvgPoolIF,
    SNNWithMaxPool,
    SPPFBlock,
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
        from tests.paiir.conftest import SNNTwoLayer

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
        model = nn.Conv2d(3, 8, 3)
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
        from tests.paiir.conftest import SNNTwoLayer

        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        for node in seq_nodes:
            assert node.output_shape != ()
            assert all(s != () for s in node.input_shapes)

    def test_no_sample_input(self):
        """Converter works without sample input, shapes default to ()."""
        from tests.paiir.conftest import SNNTwoLayer

        model = SNNTwoLayer()
        unfused = torch_to_paiir(model)
        fused = fuse_to_offline_cores(unfused)

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        for node in seq_nodes:
            assert node.output_shape == ()

    def test_dims_identity(self):
        """Identity dims for standard conv/linear (no transpose)."""
        from tests.paiir.conftest import SNNTwoLayer

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
        from tests.paiir.conftest import SNNTwoLayer

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
        from tests.paiir.conftest import SNNTwoLayer

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
        assert torch.equal(lut_outputs, ref_outputs), (
            f"LUT mismatch: expected {ref_outputs.tolist()}, got {lut_outputs.tolist()}"
        )


class TestSplitCoreAvgPoolIF:
    """Test split-core deployment for AvgPool + IFNodeV25 pattern."""

    def test_split_core_structure(self):
        """AvgPool + IF splits into SequentialOp(SumPool, ANN) + StandaloneActOp(IF)."""
        model = SNNWithAvgPoolIF(3)
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        # First layer: Conv + IF -> SequentialOp
        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2  # Conv+IF and SumPool+ANN

        # Split-core uses SumPool, not AvgPool
        sumpool_cores = [n for n in seq_nodes if isinstance(n.comp, SumPool2d)]
        assert len(sumpool_cores) == 1

        # Split-core: standalone IF
        standalone_acts = find_nodes(fused, StandaloneActOp)
        assert len(standalone_acts) == 1

    def test_core1_is_ann_with_identity_lut(self):
        """Core 1 (SumPool core) operates in ANN mode with scaled LUT.

        Split-core uses SumPool (sum domain), so LUT thresholds are scaled
        by window_size during fusion.
        """
        model = SNNWithAvgPoolIF(3)
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        seq_nodes = find_nodes(fused, SequentialOp)
        sumpool_core = [n for n in seq_nodes if isinstance(n.comp, SumPool2d)][0]

        assert sumpool_core.core_params.snn_mode == SNNMode.ANN
        assert sumpool_core.act.lut is not None
        assert sumpool_core.act.leak_tau == 0

        # LUT thresholds are scaled to sum domain (window_size = 9 for 3x3 pool)
        # Identity LUT: thresholds [0, 1, 2, ...] scaled to [0, 9, 18, ...]
        assert sumpool_core.act.lut.thresholds[1] == 9

        # Deployed LUT is the same (already in sum domain)
        lut_data = sumpool_core.lut_data
        assert lut_data is not None
        assert lut_data.thresholds[1] == 9

    def test_core2_no_compensation_needed(self):
        """Core 2 (IF core) uses original threshold, no compensation.

        Core 1's LUT handles the domain conversion, so Core 2 receives
        correct values and needs no adjustment.
        """
        model = SNNWithAvgPoolIF(3)
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        standalone_acts = find_nodes(fused, StandaloneActOp)
        assert len(standalone_acts) == 1
        if_core = standalone_acts[0]

        # Original threshold unchanged
        assert if_core.act.thres_pos == 1.0
        assert if_core.act.lut is None

    def test_edge_connectivity(self):
        """Core 1 -> Core 2 edge exists in the fused graph."""
        model = SNNWithAvgPoolIF(3)
        fused = convert_and_fuse(model, make_img_3ch_9x9())

        seq_nodes = find_nodes(fused, SequentialOp)
        sumpool_core = [n for n in seq_nodes if isinstance(n.comp, SumPool2d)][0]

        standalone_acts = find_nodes(fused, StandaloneActOp)
        if_core = standalone_acts[0]

        # Check that sumpool_core -> if_core edge exists
        succs = fused.successors(sumpool_core.name)
        assert if_core.name in succs

    def test_window_size_power_of_2_lut_compensation(self):
        """LUT thresholds are scaled by window_size for split-core.

        For window_size=4 (2x2 pool), LUT threshold is scaled from 1 to 4.
        Core 2 IF needs no compensation.
        """
        model = SNNWithAvgPoolIF(2)
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        sumpool_core = [n for n in seq_nodes if isinstance(n.comp, SumPool2d)][0]

        # LUT thresholds are scaled to sum domain (window_size = 4 for 2x2 pool)
        assert sumpool_core.act.lut is not None
        assert sumpool_core.act.lut.thresholds[1] == 4

        # Deployed LUT is the same (already in sum domain)
        lut_data = sumpool_core.lut_data
        assert lut_data is not None
        assert lut_data.thresholds[1] == 4

        # Core 2 has no compensation
        standalone_acts = find_nodes(fused, StandaloneActOp)
        if_core = standalone_acts[0]
        assert if_core.act.thres_pos == 1.0


class TestAvgPoolDeploymentWriteback:
    def test_shared_core_lif_records_avgpool_deploy_metadata(self):
        import spikingjelly.activation_based.neuron as sj

        class AvgPoolLIFNoDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.pool = nn.AvgPool2d(3)
                self.lif2 = sj.LIFNode(
                    tau=4.0,
                    decay_input=False,
                    v_threshold=1.0,
                    v_reset=0.0,
                )

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif2(self.pool(x))

        unfused = torch_to_paiir(AvgPoolLIFNoDecay(), torch.randn(1, 3, 9, 9))
        fused = fuse_to_offline_cores(unfused, enable_split_avgpool_lif=False)

        avgpool_lif_nodes = [
            node
            for node in find_nodes(fused, SequentialOp)
            if isinstance(node.comp, nn.AvgPool2d) and node.act.has_lif_dynamics
        ]
        assert len(avgpool_lif_nodes) == 1

        shared_lif = avgpool_lif_nodes[0]
        assert isinstance(shared_lif.avgpool_deploy_metadata, AvgPoolDeployMetadata)
        assert shared_lif.avgpool_deploy_metadata.source_decay_input is False
        assert shared_lif.avgpool_deploy_metadata.uses_calibration is False

    def test_split_core_lif_is_scaled_during_fusion(self):
        import spikingjelly.activation_based.neuron as sj

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

        unfused = torch_to_paiir(AvgPoolLIFNoDecay(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(unfused, enable_split_avgpool_lif=True)

        act_nodes = find_nodes(fused, StandaloneActOp)
        assert len(act_nodes) == 1
        split_lif = act_nodes[0]
        split_core1 = [
            n for n in find_nodes(fused, SequentialOp) if isinstance(n.comp, SumPool2d)
        ]
        assert len(split_core1) == 1
        assert split_core1[0].avgpool_deploy_metadata is None

        # Fusion rewrites topology and immediately moves Core 2 into the
        # split-core sum domain (window_size=4).
        assert split_lif.act.thres_pos == 4
        assert split_lif.act.reset_v == 0
        assert split_lif.act.init_v == 0
