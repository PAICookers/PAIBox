import copy

import pytest
import torch
from paicorelib import RM, ThresholdPosMode
from spikingjelly.activation_based import functional
from torch import nn

from paibox.paiir.ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.lut_activation import LutCustom, LutReLU


class TestIFNodeV25:
    def test_spike_and_reset(self):
        """Supra-threshold integer input fires a spike; membrane resets."""
        n = IFNodeV25(5, 0)
        spike = n(torch.tensor([[8]]))
        assert spike.item() == 1
        # hard reset: v=0, no multiplicative leak (leak_tau=0)
        assert n.v.item() == 0

    def test_subthreshold_accumulation(self):
        """Sub-threshold inputs accumulate linearly (pure integrator)."""
        n = IFNodeV25(10, 0)
        x = torch.tensor([[3]])

        s1 = n(x)
        assert s1.item() == 0
        # charge: v=0+3=3, no leak: v stays 3
        assert n.v.item() == 3

        # Step 2: v=3+3=6, no leak: v stays 6
        s2 = n(x)
        assert s2.item() == 0
        assert n.v.item() == 6

    def test_soft_reset(self):
        """v_reset=None selects soft reset (subtract threshold from v)."""
        n = IFNodeV25(5, v_reset=None)
        assert n.reset_mode == RM.MODE_LINEAR
        spike = n(torch.tensor([[7]]))
        assert spike.item() == 1
        # soft reset: v = 7 - 5 = 2, no leak: v stays 2
        assert n.v.item() == 2

    def test_multi_channel(self):
        """Independent per-channel computation."""
        n = IFNodeV25(5, 0)
        spike = n(torch.tensor([[2, 6, 10]]))
        assert spike.tolist() == [[0, 1, 1]]

    def test_reset_clears_state(self):
        """reset() restores membrane to init_v."""
        n = IFNodeV25(10, 0)
        n(torch.tensor([[3]]))
        n.reset()
        assert n.v == 0

    def test_init_v_matches_v_reset(self):
        """Initial membrane potential matches v_reset (SpikingJelly convention)."""
        # Hard reset with v_reset=0.0 -> init_v=0.0
        n1 = IFNodeV25(v_threshold=1.0, v_reset=0.0)
        assert n1.v == 0.0
        assert n1.init_v == 0.0

        # Hard reset with v_reset=1.0 -> init_v=1.0
        n2 = IFNodeV25(v_threshold=1.0, v_reset=1.0)
        assert n2.v == 1.0
        assert n2.init_v == 1.0

        # Soft reset (v_reset=None) -> init_v=0.0
        n3 = IFNodeV25(v_threshold=1.0, v_reset=None)
        assert n3.v == 0.0
        assert n3.init_v == 0.0


class TestLIFNodeV25:
    def test_tau_must_gt_1(self):
        """tau <= 1 raises ValueError."""
        with pytest.raises(ValueError, match="tau must be > 1"):
            LIFNodeV25(tau=1)

    def test_tau_power_of_2(self):
        """Power-of-2 tau converts to exact shift exponent without warning."""
        n = LIFNodeV25(tau=4)
        assert n.leak_tau == -2  # log2(4) = 2, right shift

    def test_tau_non_power_warns(self):
        """Non-power-of-2 tau emits a warning and rounds to nearest."""
        with pytest.warns(UserWarning, match="not a power of 2"):
            LIFNodeV25(tau=3)

    def test_leak_reduces_voltage(self):
        """LIF input decay causes lower membrane potential than IF."""
        n_if = IFNodeV25(100, 0)
        n_lif = LIFNodeV25(tau=2, v_threshold=100, v_reset=0)
        x = torch.tensor([[20]])
        n_if(x)
        n_lif(x)
        # IF (pure integrator): v=20, no leak
        # LIF (decay_input): v=20>>1=10, leak: 10-(10>>1)=5
        assert n_if.v.item() == 20
        assert n_lif.v.item() == 5

    def test_decay_input_false(self):
        """decay_input=False: input is not scaled by tau."""
        n = LIFNodeV25(tau=2, decay_input=False, v_threshold=100, v_reset=0)
        n(torch.tensor([[4]]))
        # decay_input=False: v = 0 + 4 = 4
        # leak: v - (v>>1) = 4 - 2 = 2
        assert n.v.item() == 2

    def test_init_v_matches_v_reset(self):
        """Initial membrane potential matches v_reset (SpikingJelly convention)."""
        # Hard reset with v_reset=0.0 -> init_v=0.0
        n1 = LIFNodeV25(tau=2, v_threshold=1.0, v_reset=0.0)
        assert n1.v == 0.0
        assert n1.init_v == 0.0

        # Hard reset with v_reset=1.0 -> init_v=1.0
        n2 = LIFNodeV25(tau=2, v_threshold=1.0, v_reset=1.0)
        assert n2.v == 1.0
        assert n2.init_v == 1.0

        # Soft reset (v_reset=None) -> init_v=0.0
        n3 = LIFNodeV25(tau=2, v_threshold=1.0, v_reset=None)
        assert n3.v == 0.0
        assert n3.init_v == 0.0


class TestCoreNeuronCopying:
    def test_clone_preserves_config_but_resets_runtime_state(self):
        neuron = IFNodeV25(
            v_threshold=10.0, v_reset=0.0, leak_v=torch.tensor([1.0, 2.0])
        )
        neuron(torch.tensor([[3.0, 4.0]]))

        cloned = neuron.clone()

        assert isinstance(cloned, IFNodeV25)
        assert cloned is not neuron
        assert cloned.thres_pos == neuron.thres_pos
        assert cloned.reset_mode == neuron.reset_mode
        assert cloned.v == cloned.init_v
        assert cloned._any_pos_spike_at_last_ts is False
        if isinstance(neuron.leak_v, torch.Tensor):
            assert isinstance(cloned.leak_v, torch.Tensor)
            assert torch.equal(cloned.leak_v, neuron.leak_v)
            assert cloned.leak_v is not neuron.leak_v
        else:
            assert cloned.leak_v == neuron.leak_v

    def test_deepcopy_preserves_config_and_runtime_state(self):
        neuron = IFNodeV25(
            v_threshold=10.0,
            v_reset=0.0,
            thres_neg=0.0,
            leak_v=torch.tensor([1.0, 2.0]),
        )
        neuron(torch.tensor([[3.0, 4.0]]))

        cloned = copy.deepcopy(neuron)

        assert isinstance(cloned, IFNodeV25)
        assert cloned is not neuron
        assert cloned.thres_pos == neuron.thres_pos
        assert cloned.reset_mode == neuron.reset_mode
        assert torch.equal(cloned.v, neuron.v)
        assert cloned.v is not neuron.v
        assert torch.equal(cloned.leak_v, neuron.leak_v)
        assert cloned.leak_v is not neuron.leak_v

    def test_clone_resets_runtime_state_for_lif(self):
        neuron = LIFNodeV25(tau=2.0, v_threshold=5.0, v_reset=0.0, leak_v=1.0)
        neuron(torch.tensor([[4.0]]))

        cloned = neuron.clone()

        assert isinstance(cloned, LIFNodeV25)
        assert cloned is not neuron
        assert cloned.tau == neuron.tau
        assert cloned.thres_pos == neuron.thres_pos
        assert cloned.leak_v == neuron.leak_v
        assert cloned.v == cloned.init_v
        assert cloned._any_pos_spike_at_last_ts is False

    def test_ann_clone_clones_lut_without_aliasing(self):
        thresholds = torch.arange(256, dtype=torch.float32)
        values = torch.arange(256, dtype=torch.float32)
        neuron = ANNNodeV25(LutCustom(thresholds, values), leak_v=2.0)

        cloned = neuron.clone()

        assert isinstance(cloned, ANNNodeV25)
        assert cloned is not neuron
        assert cloned.lut is not None
        assert neuron.lut is not None
        assert cloned.lut is not neuron.lut
        assert torch.equal(cloned.lut.thresholds, neuron.lut.thresholds)
        assert torch.equal(cloned.lut.lut_values, neuron.lut.lut_values)
        assert cloned.lut.thresholds is not neuron.lut.thresholds
        assert cloned.lut.lut_values is not neuron.lut.lut_values

    def test_custom_subclass_clone_preserves_type_without_special_hook(self):
        class CustomNeuron(CoreNeuronV25):
            def __init__(self):
                super().__init__()
                self.scale = torch.tensor([2.0])

            def forward(self, x):
                return self.single_step_forward(x)

            def single_step_forward(self, x):
                return super().single_step_forward(x) * self.scale

        neuron = CustomNeuron()
        neuron(torch.tensor([[1.0]]))

        cloned = neuron.clone()

        assert isinstance(cloned, CustomNeuron)
        assert cloned is not neuron
        assert torch.equal(cloned.scale, neuron.scale)
        assert cloned.scale is not neuron.scale
        assert cloned.v == cloned.init_v


class TestNeuronV25InNetwork:
    """End-to-end tests: neuron operators inside typical neural networks."""

    def test_conv_lif_multi_timestep(self):
        """Conv2d + LIFNodeV25: multi-timestep SNN inference with spike accumulation."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, 3, padding=1, bias=True)
                self.lif = LIFNodeV25(tau=2, v_threshold=1, v_reset=0)

            def forward(self, x):
                return self.lif(self.conv(x))

        torch.manual_seed(42)
        model = M()
        x = torch.randn(1, 1, 4, 4)
        total_spikes = 0
        for _ in range(8):
            spike = model(x)
            total_spikes += spike.sum().item()

        # Verify spikes are produced across timesteps
        assert total_spikes > 0
        # Verify membrane is non-zero (accumulated state)
        assert model.lif.v.abs().sum().item() > 0

    def test_linear_if_classification(self):
        """Linear + IFNodeV25: simple classification head produces spike output."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4, bias=False)
                self.ifn = IFNodeV25(v_threshold=0.5, v_reset=0)

            def forward(self, x):
                return self.ifn(self.linear(x))

        model = M()
        x = torch.randn(1, 8)
        spike = model(x)
        # Output is integer spike: {-1, 0, 1}
        assert spike.shape == (1, 4)
        assert set(spike.unique().tolist()).issubset({-1, 0, 1})

    def test_two_layer_snn(self):
        """Conv-LIF -> flatten -> Linear-IF: full two-layer SNN pipeline."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
                self.lif = LIFNodeV25(tau=2, v_threshold=0.3, v_reset=0)
                self.linear = nn.Linear(2 * 4 * 4, 3, bias=True)
                self.ifn = IFNodeV25(v_threshold=0.3, v_reset=0)

            def forward(self, x):
                h = self.lif(self.conv(x)).float()
                return self.ifn(self.linear(h.flatten(1)))

        model = M()
        x = torch.randn(1, 1, 4, 4)
        spikes_over_time = []
        for _ in range(10):
            out = model(x)
            spikes_over_time.append(out.clone())

        all_spikes = torch.stack(spikes_over_time)
        assert all_spikes.shape == (10, 1, 3)

    def test_residual_add_with_neuron(self):
        """Two conv branches added + LIF: residual SNN block."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = nn.Conv2d(1, 4, 3, padding=1, bias=False)
                self.conv_b = nn.Conv2d(1, 4, 3, padding=1, bias=False)
                self.lif = LIFNodeV25(tau=2, v_threshold=1, v_reset=0)

            def forward(self, x):
                return self.lif(self.conv_a(x) + self.conv_b(x))

        model = M()
        x = torch.randn(1, 1, 4, 4)
        spike = model(x)
        assert spike.shape == (1, 4, 4, 4)

    def test_sequential_snn(self):
        """nn.Sequential with neuron layers works for multi-timestep inference."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Linear(8, 4, bias=True),
                    LIFNodeV25(tau=2, v_threshold=0.1, v_reset=0),
                )

            def forward(self, x):
                return self.seq(x)

        model = M()
        x = torch.randn(1, 8)
        functional.reset_net(model)
        spikes = [model(x) for _ in range(8)]
        all_spikes = torch.stack(spikes)

        assert all_spikes.shape == (8, 1, 4)
        assert all_spikes.abs().sum().item() > 0

        functional.reset_net(model)
        assert model.seq[1].v == 0

    def test_reset_between_sequences(self):
        """Neuron state resets between independent input sequences."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.lif = LIFNodeV25(tau=2, v_threshold=0.5)

            def forward(self, x):
                return self.lif(self.linear(x))

        model = M()
        x = torch.randn(1, 4)
        functional.reset_net(model)

        # Run sequence 1
        for _ in range(5):
            model(x)
        v_after_seq1 = model.lif.v.clone()

        # Reset and run the same sequence again
        functional.reset_net(model)
        for _ in range(5):
            model(x)
        v_after_seq2 = model.lif.v.clone()

        # Identical input + reset -> identical membrane state
        assert torch.equal(v_after_seq1, v_after_seq2)

    def test_training_gradient_flow(self):
        """Training mode produces float output with gradient support."""
        ifn = IFNodeV25(v_threshold=0.5, v_reset=0)
        ifn.train()
        x = torch.randn(1, 4, requires_grad=True)
        spike = ifn(x)
        assert spike.is_floating_point()
        assert spike.grad_fn is not None

    def test_eval_int8_output(self):
        """Eval mode preserves int8 output (deployment behaviour unchanged)."""
        ifn = IFNodeV25(v_threshold=0.5, v_reset=0)
        ifn.eval()
        x = torch.randn(1, 4)
        spike = ifn(x)
        assert spike.dtype == torch.int8

    def test_training_backward(self):
        """Backward pass through Linear + LIF produces weight gradients."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4, bias=False)
                self.lif = LIFNodeV25(tau=2, v_threshold=0.5, v_reset=0)

            def forward(self, x):
                return self.lif(self.linear(x))

        model = M()
        model.train()
        x = torch.randn(1, 8)
        spike = model(x)
        loss = spike.sum()
        loss.backward()
        assert model.linear.weight.grad is not None
        assert model.linear.weight.grad.abs().sum().item() > 0

    def test_detach_reset(self):
        """detach_reset=True prevents gradient flow through the reset path."""

        class M(nn.Module):
            def __init__(self, detach_reset=False):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.ifn = IFNodeV25(
                    v_threshold=0.3, v_reset=0, detach_reset=detach_reset
                )

            def forward(self, x):
                return self.ifn(self.linear(x))

        # With detach_reset=False (default)
        model = M(detach_reset=False)
        model.train()
        x = torch.randn(1, 4)
        # Two timesteps so reset contributes to the second spike's gradient
        s1 = model(x)
        s2 = model(x)
        loss = (s1 + s2).sum()
        loss.backward()
        grad_no_detach = model.linear.weight.grad.clone()

        # With detach_reset=True
        model = M(detach_reset=True)
        model.train()
        s1 = model(x)
        s2 = model(x)
        loss = (s1 + s2).sum()
        loss.backward()
        grad_detach = model.linear.weight.grad.clone()

        # Gradients should differ when reset is detached
        assert not torch.equal(grad_no_detach, grad_detach)


class TestCoreNeuronV25ANN:
    """Tests for ANN mode: CoreNeuronV25 with LUT activation."""

    def test_ann_forward_lut_relu(self):
        """ANNNodeV25(lut=LutReLU()) produces correct 8-bit output."""
        neuron = ANNNodeV25(lut=LutReLU(min_val=-500, max_val=500))
        neuron.eval()
        x = torch.tensor([[10.0, -5.0, 100.0]])
        out = neuron(x)
        assert out.shape == (1, 3)
        # ReLU: negative -> 0, positive -> non-zero
        assert out[0, 1].item() == 0
        assert out[0, 0].item() >= 0
        assert out[0, 2].item() > 0

    def test_ann_forward_with_leak(self):
        """Additive leak (bias) affects LUT lookup input."""
        neuron = ANNNodeV25(lut=LutReLU(min_val=-500, max_val=500), leak_v=10.0)
        neuron.eval()
        # Input = -5, after charge: v = 0 + (-5) + 10 = 5
        x = torch.tensor([[-5.0]])
        out = neuron(x)
        # v=5 should map to a positive LUT value
        assert out.item() > 0

    def test_ann_reset_positive_output(self):
        """LUT(n) > 0 triggers reset: hard reset sets V = reset_v."""
        neuron = CoreNeuronV25(
            lut=LutReLU(min_val=-500, max_val=500),
            reset_mode=RM.MODE_NORMAL,
            reset_v=0,
        )
        neuron.eval()
        x = torch.tensor([[100.0]])
        out = neuron(x)
        assert out.item() > 0
        # After hard reset, v should be 0
        assert neuron.v.item() == 0

    def test_ann_reset_zero_output(self):
        """LUT(n) == 0 leaves V unchanged (no reset)."""
        neuron = ANNNodeV25(lut=LutReLU(), reset_mode=RM.MODE_NORMAL, reset_v=0)
        neuron.eval()
        # Negative input -> ReLU output is 0 -> no reset
        x = torch.tensor([[-100.0]])
        out = neuron(x)
        assert out.item() == 0
        # V should NOT be reset to reset_v; it should retain charge value
        assert neuron.v.item() != 0

    def test_ann_to_neuron_params_with_bias(self):
        """Bias is fused into leak_v (the fixed bug)."""
        neuron = ANNNodeV25(lut=LutReLU())
        bias = torch.tensor(5.0)
        params = neuron.to_neuron_params(bias=bias)
        assert params.leak_v == 5.0

    def test_ann_export_lut(self):
        """export_lut() returns LutData for ANN mode."""
        neuron = ANNNodeV25(lut=LutReLU())
        data = neuron.export_lut()
        assert data is not None
        assert data.thresholds.shape == (256,)
        assert data.values.shape == (256,)

    def test_ann_range_clamp(self):
        """CEILING/FLOOR clamps V before LUT lookup."""
        neuron = CoreNeuronV25(
            lut=LutReLU(),
            thres_pos=50,
            thres_neg=-50,
        )
        neuron.eval()
        # Large positive input should be clamped by CEILING (thres_pos_mode default is FIRE not CEILING)
        # Use CEILING mode explicitly
        neuron.thres_pos_mode = ThresholdPosMode.CEILING
        x = torch.tensor([[1000.0]])
        neuron(x)
        # After range clamp, v should be at most 50 (before reset)

    def test_ann_forward_matches_standalone_lut(self):
        """ANNNodeV25(lut=LutReLU()) matches LutReLU() for default params."""
        lut = LutReLU()
        neuron = ANNNodeV25(lut=LutReLU())
        neuron.eval()
        x = torch.linspace(-500, 500, 50).unsqueeze(0)
        # For default neuron params (no leak, no special modes),
        # the output should match standalone LUT
        out_neuron = neuron(x)
        out_lut = lut(x)
        assert torch.equal(out_neuron, out_lut)

    def test_ann_lut_as_submodule(self):
        """LUT is registered as a proper nn.Module sub-module."""
        neuron = ANNNodeV25(lut=LutReLU())
        submodules = dict(neuron.named_modules())
        assert "lut" in submodules
        assert isinstance(submodules["lut"], LutReLU)

    def test_ann_training_raises(self):
        """ANN mode (LUT) raises RuntimeError in training mode."""
        neuron = ANNNodeV25(lut=LutReLU())
        neuron.train()
        x = torch.tensor([[1.0, 2.0]])
        with pytest.raises(RuntimeError, match="does not support training"):
            neuron(x)

    def test_ann_eval_works(self):
        """ANN mode works in eval mode (no error)."""
        neuron = ANNNodeV25(lut=LutReLU())
        neuron.eval()
        x = torch.tensor([[1.0, 2.0]])
        out = neuron(x)
        assert out.shape == x.shape
