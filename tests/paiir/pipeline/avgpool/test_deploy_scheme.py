"""Tests for AvgPool-specific fusion/deploy writeback behavior."""

import torch
from paicorelib import SNNMode
from spikingjelly.activation_based import neuron as sj
from torch import nn

from paibox.paiir.ir.op_node import SequentialOp, StandaloneActOp
from paibox.paiir.lowering.converter import torch_to_paiir
from paibox.paiir.nn import SumPool2d
from paibox.paiir.pipeline.avgpool import AvgPoolDeployMetadata
from paibox.paiir.pipeline.passes import fuse_to_offline_cores, specialize_general_adds
from tests.paiir.conftest import (
    SNNWithAvgPoolIF,
    convert_and_fuse,
    find_nodes,
    make_img_3ch_8x8,
    make_img_3ch_9x9,
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
        fused = fuse_to_offline_cores(specialize_general_adds(unfused), enable_split_avgpool_lif=False)

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
        fused = fuse_to_offline_cores(specialize_general_adds(unfused), enable_split_avgpool_lif=True)

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

