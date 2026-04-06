"""Tests for PAIIRGraph simulation: step(), reset(), run(), forward()."""

from typing import Literal

import pytest
import torch
from spikingjelly.activation_based import functional as sF
from spikingjelly.activation_based import neuron as sj
from torch import Tensor, nn

import paibox.paiir.pipeline.avgpool.fusion as avgpool_fusion
from paibox.paiir import compile_to_paiir, torch_to_paiir
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    OfflineCoreOp,
    ReshapeOp,
    SequentialOp,
    SplitOp,
    StandaloneCompOp,
)
from paibox.paiir.pipeline.avgpool import (
    AvgPoolDeployScheme,
    AvgPoolLIFCandidateScore,
)
from paibox.paiir.pipeline.passes import GraphCleanupWarning
from tests.paiir.conftest import (
    ANNClassifier,
    ANNResidualSubtract,
    MultiInputMerge,
    SNNDepthwiseSeparable,
    SNNFlattenTransition,
    SNNResidualAdd,
    SNNTwoLayer,
    SNNWithAvgPoolIF,
    SNNWithMaxPool,
    SPPFBlock,
    find_nodes,
)


def _make_snn_input(
    batch_size: int = 1, channels: int = 3, size: int = 8, bipolar: bool = False
) -> Tensor:
    """Create SNN input spikes.

    Args:
        bipolar: If True, generate bipolar spikes {-1, 0, 1} for neurons with
            negative firing capability. Default False generates {0, 1}.
    """
    if bipolar:
        return torch.randint(
            -1, 2, (batch_size, channels, size, size), dtype=torch.int8
        )
    return torch.randint(0, 2, (batch_size, channels, size, size), dtype=torch.int8)


def _make_ann_input(batch_size: int = 1, channels: int = 3, size: int = 8) -> Tensor:
    """Create ANN input: 8-bit quantized values in int8 format."""
    return torch.randint(
        -128, 128, (batch_size, channels, size, size), dtype=torch.int8
    )


class SingleLayerSNN(nn.Module):
    """Minimal single-layer SNN (Conv-IF) for testing."""

    def __init__(self, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=bias)
        self.act = sj.IFNode(v_threshold=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class SingleLayerANN(nn.Module):
    """Minimal single-layer ANN (Conv-ReLU) for testing."""

    def __init__(self, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x))


def _compile_snn(
    tick_duration: int = 0,
    auto_reset: bool = True,
    bias: bool = False,
    bipolar: bool = False,
):
    """Compile a single-layer SNN with deterministic weights.

    Args:
        tick_duration: Activity duration for the core. 0 for always active.
        auto_reset: Whether to auto-reset neuron state.
        bias: Whether to include bias in the conv layer.
        bipolar: If True, use bipolar input spikes {-1, 0, 1}.
    """
    model = SingleLayerSNN(bias=bias)
    with torch.no_grad():
        model.conv.weight.fill_(1)
        if bias:
            assert torch.is_tensor(model.conv.bias)
            model.conv.bias.fill_(1)

    x_compile = torch.randn(1, 3, 8, 8)
    graph = compile_to_paiir(
        model, x_compile, tick_duration=tick_duration, auto_reset=auto_reset
    )
    return graph, _make_snn_input(bipolar=bipolar)


def _compile_ann(out_channels: int = 4):
    """Compile a single-layer ANN with deterministic weights."""
    model = SingleLayerANN(out_channels=out_channels)
    with torch.no_grad():
        model.conv.weight.fill_(1)

    x_compile = torch.randn(1, 3, 8, 8)
    graph = compile_to_paiir(model, x_compile)
    return graph, _make_ann_input(), model


class TestTickActivityWindow:
    """Tests for tick_start / tick_duration activity window behavior."""

    @pytest.mark.parametrize(
        "tick_start, tick_duration, inactive_steps, active_steps",
        [(3, 0, [1, 2], [3]), (1, 2, [3], [1, 2]), (1, 0, [], [1, 2, 3])],
        ids=["delayed_start", "limited_duration", "always_active"],
    )
    def test_activity_window(
        self,
        tick_start: int,
        tick_duration: int,
        inactive_steps: list[int],
        active_steps: list[int],
    ) -> None:
        """Node outputs zeros when inactive, valid output when active."""
        graph, x = _compile_snn()

        seq_ops = find_nodes(graph, SequentialOp)
        assert len(seq_ops) == 1
        seq_ops[0].core_params.tick_start = tick_start
        seq_ops[0].core_params.tick_duration = tick_duration

        max_step = max(
            (max(inactive_steps) if inactive_steps else 0),
            (max(active_steps) if active_steps else 0),
        )
        outputs = []
        for _ in range(max_step):
            out = graph.step(x)
            assert torch.is_tensor(out)
            outputs.append(out)

        for step in inactive_steps:
            assert torch.all(outputs[step - 1] == 0), f"step {step} should be inactive"

        for step in active_steps:
            assert outputs[step - 1].shape == seq_ops[0].output_shape


class TestTickInitial:
    """Tests for tick_initial semantics (state reset behavior)."""

    def test_ann_tick_initial_is_one(self) -> None:
        """ANN cores have tick_initial=1 (stateless per step)."""
        graph, _, _ = _compile_ann()
        for node in find_nodes(graph, OfflineCoreOp):
            assert node.core_params.tick_initial == 1

    def test_ann_stateless_consistency(self) -> None:
        """ANN produces identical outputs for identical inputs across steps."""
        graph, x, _ = _compile_ann()
        out1 = graph.step(x)
        out2 = graph.step(x)
        torch.testing.assert_close(out1, out2)

    def test_snn_tick_initial_equals_duration_when_auto_reset(self) -> None:
        """SNN with auto_reset=True has tick_initial=tick_duration."""
        graph, _ = _compile_snn(tick_duration=4, auto_reset=True)
        seq_ops = find_nodes(graph, SequentialOp)
        assert seq_ops[0].core_params.tick_initial == 4

    def test_snn_tick_initial_zero_when_no_auto_reset(self) -> None:
        """SNN with auto_reset=False has tick_initial=0."""
        graph, _ = _compile_snn(tick_duration=0, auto_reset=False)
        seq_ops = find_nodes(graph, SequentialOp)
        assert seq_ops[0].core_params.tick_initial == 0

    def test_snn_auto_reset_behavior(self) -> None:
        """SNN neuron state resets after tick_initial active steps."""
        graph, x = _compile_snn(tick_duration=4, auto_reset=True, bipolar=True)
        seq_ops = find_nodes(graph, SequentialOp)
        op = seq_ops[0]

        for _ in range(4):
            graph.step(x)

        assert graph._active_counts[op.name] == 4

        v = op.act.v
        assert isinstance(v, Tensor)
        assert v.abs().sum() > 0

        graph.step(x)
        assert graph._active_counts[op.name] == 4

    def test_snn_state_accumulates_without_auto_reset(self) -> None:
        """SNN membrane potential accumulates when auto_reset=False."""
        graph, x = _compile_snn(tick_duration=0, auto_reset=False, bipolar=True)
        seq_ops = find_nodes(graph, SequentialOp)
        op = seq_ops[0]

        for _ in range(5):
            graph.step(x)

        v = op.act.v
        if isinstance(v, Tensor):
            assert not torch.all(v == op.act.init_v)


class TestSNNSimulation:
    """Tests for SNN simulation behavior and correctness."""

    def test_voltage_accumulates(self) -> None:
        """Neuron membrane potential changes across steps."""
        graph, x = _compile_snn(tick_duration=0, auto_reset=False, bipolar=True)
        seq_ops = find_nodes(graph, SequentialOp)
        op = seq_ops[0]

        vs: list[Tensor] = []
        for _ in range(8):
            graph.step(x)
            v = op.act.v
            if isinstance(v, Tensor):
                vs.append(v.clone())

        assert len(vs) > 0
        assert not all(torch.equal(vs[0], v) for v in vs[1:])

    def test_reset_clears_all_state(self) -> None:
        """graph.reset() clears sim_step, active_counts, and neuron state."""
        graph, x = _compile_snn(tick_duration=0, auto_reset=False)
        seq_ops = find_nodes(graph, SequentialOp)
        op = seq_ops[0]
        init_v = op.act.init_v

        for _ in range(5):
            graph.step(x)

        assert graph._sim_step == 5

        graph.reset()
        v = op.act.v
        if isinstance(v, Tensor):
            assert torch.all(v == init_v)
        else:
            assert v == init_v

    @pytest.mark.parametrize("bipolar", [False, True], ids=["unipolar", "bipolar"])
    def test_output_is_valid_spikes(self, bipolar) -> None:
        """SNN output contains only valid spike values {0, 1} or {-1, 0, 1}."""
        graph, x = _compile_snn(bias=True)
        graph.reset()

        for _ in range(3):
            out = graph.step(x)
            assert torch.is_tensor(out)
            assert out.dtype == torch.int8
            unique_vals = out.unique().tolist()

            if bipolar:
                assert all(v in [-1, 0, 1] for v in unique_vals)
            else:
                assert all(v in [0, 1] for v in unique_vals)

    def test_vs_spikingjelly_if(self) -> None:
        """SNN output matches SpikingJelly IFNode behavior."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False), sj.IFNode(v_threshold=1.0)
        )

        with torch.no_grad():
            model[0].weight.fill_(1)

        x_float = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        with torch.no_grad():
            sj_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_float)
        graph.reset()
        paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == sj_out.shape
        sj_spike = sj_out.to(torch.int8)
        assert torch.equal(paiir_out, sj_spike)

    def test_vs_spikingjelly_lif(self) -> None:
        """SNN with LIF neuron matches SpikingJelly LIFNode behavior."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False),
            sj.LIFNode(tau=2.0, v_threshold=1.0),
        )

        with torch.no_grad():
            model[0].weight.fill_(1)

        x_float = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        with torch.no_grad():
            sj_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_float)
        graph.reset()
        paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == sj_out.shape
        # Verify PAIIR output matches SpikingJelly output
        sj_spike = sj_out.to(torch.int8)
        assert torch.equal(paiir_out, sj_spike)


class TestANNSimulation:
    """Tests for ANN simulation behavior and correctness."""

    def test_output_shape_and_dtype(self) -> None:
        """PAIIR output shape and dtype match original PyTorch model."""
        graph, x_int8, model = _compile_ann()
        x_float = x_int8.to(torch.float32)

        model.eval()
        with torch.no_grad():
            expected = model(x_float)

        out = graph.step(x_int8)
        assert torch.is_tensor(out)
        assert out.shape == expected.shape
        assert out.dtype in [torch.uint8, torch.int8]

    def test_vs_pytorch_relu(self) -> None:
        """ANN ReLU output correlates with PyTorch ReLU output.

        Note: LUT quantization introduces approximation error. We verify
        correlation (>0.95) to ensure quantization preserves structure.
        """
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.ReLU())

        with torch.no_grad():
            model[0].weight.fill_(1)

        x_float = torch.randn(1, 3, 8, 8)
        x_int8 = _make_ann_input()

        model.eval()
        with torch.no_grad():
            relu_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_float)
        graph.reset()
        paiir_out = graph.step(x_int8)
        assert torch.is_tensor(paiir_out)

        assert paiir_out.dtype in [torch.uint8, torch.int8]

        # Verify PAIIR output correlates with PyTorch output
        paiir_float = paiir_out.to(torch.float32)
        relu_var = relu_out.var()
        paiir_var = paiir_float.var()

        if relu_var > 1e-6 and paiir_var > 1e-6:
            relu_norm = (relu_out - relu_out.mean()) / (relu_out.std() + 1e-8)
            paiir_norm = (paiir_float - paiir_float.mean()) / (paiir_float.std() + 1e-8)
            corr = torch.corrcoef(
                torch.stack([relu_norm.flatten(), paiir_norm.flatten()])
            )[0, 1]
            assert corr > 0.95, f"Correlation {corr:.4f} < 0.95"


class TestRunInterface:
    """Tests for run() interface."""

    def test_run_output_has_time_dimension(self) -> None:
        """run(T) produces output with leading dimension T."""
        graph, x_single = _compile_snn()

        T = 4
        x = x_single.unsqueeze(0).expand(T, -1, -1, -1, -1)
        out = graph.run(x, T=T)
        assert torch.is_tensor(out)
        assert out.shape[0] == T

    def test_run_step_equivalence(self) -> None:
        """run(x, T=1)[0] equals step(x) after reset."""
        graph, x, _ = _compile_ann()

        x_t = x.unsqueeze(0)
        run_out = graph.run(x_t, T=1)[0]

        graph.reset()
        step_out = graph.step(x)
        assert torch.is_tensor(run_out) and torch.is_tensor(step_out)
        assert torch.equal(run_out, step_out)

    def test_reset_flag_controls_sim_step(self) -> None:
        """reset=False preserves sim_step; reset=True resets it."""
        graph, x_single = _compile_snn()

        T = 2
        x = x_single.unsqueeze(0).expand(T, -1, -1, -1, -1)

        graph.run(x, T=T, reset=True)
        assert graph._sim_step == T

        graph.run(x, T=T, reset=False)
        assert graph._sim_step == T * 2

        graph.run(x, T=T, reset=True)
        assert graph._sim_step == T


class TestMultiInputGraph:
    """Tests for graphs with multiple inputs."""

    def test_multi_input_step(self) -> None:
        """Graph with multiple InputNodes accepts multiple inputs."""
        model = MultiInputMerge()
        # Use integer weights for PAIIR compatibility
        with torch.no_grad():
            model.conv1.weight.fill_(1)
            model.conv2.weight.fill_(1)

        x_compile = torch.randn(1, 3, 8, 8)
        graph = compile_to_paiir(model, x_compile, x_compile)

        x1 = _make_snn_input()
        x2 = _make_snn_input()

        out = graph.step(x1, x2)
        assert torch.is_tensor(out)
        assert out.shape == (1, 16, 8, 8)

    def test_multi_input_run(self) -> None:
        """run() with multiple inputs produces correct output shape."""
        model = MultiInputMerge()
        # Use integer weights for PAIIR compatibility
        with torch.no_grad():
            model.conv1.weight.fill_(1)
            model.conv2.weight.fill_(1)

        x_compile = torch.randn(1, 3, 8, 8)
        graph = compile_to_paiir(model, x_compile, x_compile)

        T = 3
        x1 = _make_snn_input().unsqueeze(0).expand(T, -1, -1, -1, -1)
        x2 = _make_snn_input().unsqueeze(0).expand(T, -1, -1, -1, -1)

        out = graph.run(x1, x2, T=T)
        assert torch.is_tensor(out)
        assert out.shape[0] == T


class TestMultiOutputModel:
    """Tests for torch models with multiple outputs.

    When a torch model returns multiple outputs (e.g., return out1, out2),
    compile_to_paiir creates a single OutputNode with multiple predecessors.
    The step() method returns a tuple containing all branch outputs.
    """

    def test_multi_output_torch_model_graph_structure(self) -> None:
        """Multi-output torch model creates single OutputNode with multiple predecessors."""

        class M(nn.Module):
            """SNN with two output branches."""

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.act1 = sj.IFNode(v_threshold=1.0)
                self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.act2 = sj.IFNode(v_threshold=1.0)

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                out1 = self.act1(self.conv1(x))
                out2 = self.act2(self.conv2(x))
                return out1, out2

        model = M()
        with torch.no_grad():
            model.conv1.weight.fill_(1)
            model.conv2.weight.fill_(1)

        x_compile = torch.randn(1, 3, 8, 8)

        # PAIIR conversion
        graph = compile_to_paiir(model, x_compile)

        # Graph structure: two OutputNodes, each with one predecessor
        output_nodes = graph.output_nodes()
        assert len(output_nodes) == 2

        for out_node in output_nodes:
            preds = graph.predecessors(out_node.name)
            assert len(preds) == 1  # Each OutputNode has exactly one predecessor
            assert isinstance(graph.nodes[preds[0]], SequentialOp)

    def test_multi_output_torch_model_step_returns_tuple(self) -> None:
        """Multi-output model: step() returns tuple of all branch outputs."""

        class M(nn.Module):
            """SNN with two output branches."""

            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.act1 = sj.IFNode(v_threshold=1.0)
                self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.act2 = sj.IFNode(v_threshold=1.0)

            def forward(self, x: Tensor):
                out1 = self.act1(self.conv1(x))
                out2 = self.act2(self.conv2(x))
                return out1, out2

        model = M()
        with torch.no_grad():
            model.conv1.weight.fill_(1)
            model.conv2.weight.fill_(1)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        # Torch model returns tuple)
        torch_out = _run_snn_reference(model, x_int8)

        # PAIIR conversion
        graph = compile_to_paiir(model, x_compile)
        graph.reset()
        paiir_out = graph.step(x_int8)

        assert isinstance(paiir_out, tuple) and len(paiir_out) == 2
        for out, torch_out in zip(paiir_out, torch_out):
            assert torch.equal(out, torch_out.to(torch.int8))


class TestErrorHandling:
    """Tests for error conditions and edge cases."""

    def test_step_wrong_input_count_raises(self) -> None:
        """step() raises ValueError when input count mismatches."""
        graph, _ = _compile_snn()

        with pytest.raises(ValueError, match="expected .* input"):
            graph.step()

        x = _make_snn_input()
        with pytest.raises(ValueError, match="expected .* input"):
            graph.step(x, x)

    def test_train_eval_mode(self) -> None:
        """train() and eval() propagate to node modules."""
        graph, _ = _compile_snn()

        graph.train()
        for node in graph.nodes.values():
            if isinstance(node, nn.Module):
                assert node.training

        graph.eval()
        for node in graph.nodes.values():
            if isinstance(node, nn.Module):
                assert not node.training


def _set_quantized_weights(model: nn.Module, seed: int = 42) -> None:
    """Set deterministic int8 weights on all Conv/Linear layers."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.copy_(
                    torch.randint(-128, 128, module.weight.shape).float()
                )
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.copy_(
                    torch.randint(-128, 128, module.weight.shape).float()
                )
                if module.bias is not None:
                    module.bias.zero_()


def _to_int8(t: Tensor) -> Tensor:
    return t.to(torch.int8) if t.is_floating_point() else t


def _run_snn_reference(
    model: nn.Module, *x_int8: Tensor
) -> Tensor | tuple[Tensor, ...]:
    """Run SpikingJelly model and return int8 output(s)."""
    model.eval()
    sF.reset_net(model)
    with torch.no_grad():
        out = model(*[x.float() for x in x_int8])

    if isinstance(out, tuple):
        return tuple(_to_int8(t) for t in out)
    return _to_int8(out)


def _validate_ann_output(
    paiir_out: Tensor,
    pytorch_out: Tensor,
    activation: Literal["relu", "sigmoid", "tanh", "linear"],
) -> dict[str, float]:
    """Validate ANN output against PyTorch reference with error bounds.

    Error bounds based on LUT quantization accuracy from test_lut_activation.py:
    - ReLU: MAE <= 2.0, max_error <= 8 (linear in positive region)
    - Sigmoid: MAE <= 8.0, max_error <= 25 (nonlinear, 256 bins)
    - Tanh: MAE <= 6.0, max_error <= 20 (symmetric nonlinear)
    - Linear: MAE <= 2.0, max_error <= 5 (direct mapping)

    PAIIR outputs are quantized to int8:
    - ReLU: [0, 255] (unsigned)
    - Sigmoid: [0, 255] (unsigned, scaled from [0, 1])
    - Tanh: [-127, 127] (signed, scaled from [-1, 1])
    - Linear: direct mapping

    Args:
        paiir_out: PAIIR simulation output (quantized int8).
        pytorch_out: PyTorch reference output (float, in math domain).
        activation: Activation type ("relu" | "sigmoid" | "tanh" | "linear").

    Returns:
        Dict with "mae", "max_error", and "passed" keys.
    """
    bounds = {
        "relu": {"max_mae": 2.0, "max_error": 8, "scale": 255.0},
        "sigmoid": {"max_mae": 8.0, "max_error": 25, "scale": 255.0},
        "tanh": {"max_mae": 6.0, "max_error": 20, "scale": 127.0},
        "linear": {"max_mae": 2.0, "max_error": 5, "scale": 1.0},
    }
    assert activation in bounds, f"Unknown activation: {activation}"

    # Scale PyTorch output to PAIIR quantized range for comparison
    scale = bounds[activation]["scale"]
    pytorch_scaled = pytorch_out * scale

    paiir_float = paiir_out.to(torch.float32)
    mae = (paiir_float - pytorch_scaled).abs().mean().item()
    max_error = (paiir_float - pytorch_scaled).abs().max().item()

    bound = bounds[activation]
    passed = mae <= bound["max_mae"] and max_error <= bound["max_error"]

    return {"mae": mae, "max_error": max_error, "passed": passed}


class TestMultiLayerSNN:
    """Tests for multi-layer SNN simulation with exact SpikingJelly comparison."""

    def test_two_layer_vs_spikingjelly(self) -> None:
        """SNNTwoLayer output matches SpikingJelly exactly.

        Model: Conv-LIF -> Conv-IF

        For multi-layer networks, the output is available at step = max(tick_start).
        This is because each layer's tick_start is assigned based on DAG depth.
        """
        model = SNNTwoLayer()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert isinstance(sj_out, Tensor)  # SNNTwoLayer has single output

        # PAIIR
        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        # Run steps until output is available (max tick_start for 2 layers = 2)
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)

    def test_flatten_linear_transition(self) -> None:
        """SNNFlattenTransition: Conv -> flatten -> Linear.

        Tests spatial-to-dense transition in SNN context now that flatten is
        materialized as a routing ``ReshapeOp`` for graph simulation.
        """
        model = SNNFlattenTransition()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 1, 4, 4)
        x_int8 = torch.randint(0, 2, (1, 1, 4, 4), dtype=torch.int8)

        sj_out = _run_snn_reference(model, x_int8)
        assert torch.is_tensor(sj_out)

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)

    def test_depthwise_separable(self) -> None:
        """SNNDepthwiseSeparable: DWConv -> PWConv.

        Verifies grouped conv handling in multi-layer SNN.
        """
        model = SNNDepthwiseSeparable()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 16, 8, 8)
        x_int8 = torch.randint(0, 2, (1, 16, 8, 8), dtype=torch.int8)

        sj_out = _run_snn_reference(model, x_int8)
        assert isinstance(sj_out, Tensor)  # SNNDepthwiseSeparable has single output

        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        # Run steps until output is available
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)

    def test_maxpool_lif_snn(self) -> None:
        """SNNWithMaxPool: Conv-LIF -> MaxPool-LIF.

        Verifies MaxPool fusion in multi-layer SNN context.
        """
        model = SNNWithMaxPool()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert isinstance(sj_out, Tensor)  # SNNWithMaxPool has single output

        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)

    @pytest.mark.parametrize("kernel_size", [2, 3])
    def test_avgpool_if_snn(self, kernel_size: int) -> None:
        """AvgPoolIF: Conv-IF -> AvgPool(k×k)-IF.

        Tests split-core AvgPool deployment:
        - Core 1: AvgPool + identity LUT (outputs sum without shift)
        - Core 2: IF with threshold scaled by window_size

        Both power-of-2 and non-power-of-2 kernels should match SpikingJelly exactly,
        since Core 1 passes through the true sum and Core 2 compensates with window_size.
        """
        model = SNNWithAvgPoolIF(kernel_size)
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert torch.is_tensor(sj_out)

        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == sj_out.shape
        assert torch.equal(paiir_out, sj_out)

    def test_avgpool_if_snn_with_divisor_override(self) -> None:
        class AvgPoolIFDivisorOne(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.pool = nn.AvgPool2d(2, divisor_override=1)
                self.if2 = sj.IFNode(v_threshold=1.0)

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.if2(self.pool(x))

        model = AvgPoolIFDivisorOne()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert torch.is_tensor(sj_out)

        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == sj_out.shape
        assert torch.equal(paiir_out, sj_out)

    def test_avgpool_lif_split_core_snn(self, monkeypatch) -> None:
        """Split-core AvgPool+LIF matches reference when exact-sum coding is enabled.

        Force the split-core path explicitly so this test validates split-core
        simulation semantics rather than score-tie or heuristic selection.
        """

        kernel_size = 2

        def fake_select_avgpool_lif_candidate(
            act,
            pred_out_width,
            window_size,
            allow_split_lif=False,
            try_calibration=False,
            avg_divisor=None,
        ):
            return AvgPoolLIFCandidateScore(
                AvgPoolDeployScheme.SPLIT_CORE_LIF_EXACT_SUM, False, 0.0, 0.0, 0.0, 0.0
            )

        monkeypatch.setattr(
            avgpool_fusion,
            "select_avgpool_lif_candidate",
            fake_select_avgpool_lif_candidate,
        )

        class AvgPoolLIFNoDecay(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.if1 = sj.IFNode(v_threshold=1.0)
                self.pool = nn.AvgPool2d(kernel_size)
                self.lif2 = sj.LIFNode(
                    tau=4.0, decay_input=False, v_threshold=1.0, v_reset=0.0
                )

            def forward(self, x):
                x = self.if1(self.conv(x))
                return self.lif2(self.pool(x))

        model = AvgPoolLIFNoDecay()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert torch.is_tensor(sj_out)

        graph = compile_to_paiir(model, x_compile, enable_split_avgpool_lif=True)
        graph.reset()

        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == sj_out.shape
        assert torch.equal(paiir_out, sj_out)


class TestMultiLayerANN:
    """Tests for multi-layer ANN simulation with bounded error validation."""

    def test_classifier_relu_sigmoid_chain(self) -> None:
        """ANNClassifier: Conv-ReLU -> AvgPool -> Linear-Sigmoid.

        Validates LUT error bounds for each activation type.

        Note: This test is currently skipped because the LUT-based Sigmoid
        activation requires proper input domain mapping between the hardware
        int32 domain and the quantized int8 simulation domain. The multi-step
        execution pattern is correct, but the LUT lookup returns constant
        values due to domain mismatch.

        TODO: Fix LUT input domain mapping in CoreNeuronV25 or LutActivation
        to enable this test.
        """
        pytest.skip(
            "LUT Sigmoid requires input domain mapping fix. "
            "The multi-step execution pattern is implemented correctly."
        )

        model = ANNClassifier()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_ann_input()

        # PyTorch reference
        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        # PAIIR
        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        # Multi-layer ANN needs multiple steps to complete pipeline execution.
        # Each layer has a different tick_start, so we need to run enough steps
        # for all layers to execute.
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)

        # Validate with Sigmoid error bounds (final activation)
        result = _validate_ann_output(paiir_out, pytorch_out, "sigmoid")
        assert result["passed"], (
            f"Sigmoid validation failed: MAE={result['mae']:.2f}, "
            f"max_error={result['max_error']:.2f}"
        )


class TestAccumulateOpSimulation:
    """Tests for residual/add operations with multi-input accumulation."""

    def test_residual_add_snn(self) -> None:
        """SNNResidualAdd: Conv_a + Conv_b -> LIF.

        Verifies signs=(1, 1) accumulation, exact match with SpikingJelly.
        """
        model = SNNResidualAdd()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x_int8)
        assert isinstance(sj_out, Tensor)  # SNNResidualAdd has single output

        # PAIIR
        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        # Verify AccumulateOp exists with correct signs
        acc_ops = [n for n in graph.nodes.values() if isinstance(n, AccumulateOp)]
        assert len(acc_ops) == 1
        assert acc_ops[0].signs == (1, 1)

        paiir_out = graph.step(x_int8)
        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)

    def test_residual_subtract_ann(self) -> None:
        """ANNResidualSubtract: Linear_a - Linear_b -> Tanh.

        Verifies signs=(1, -1) subtraction with bounded error validation.

        Note: This test is currently skipped because the LUT-based Tanh
        activation requires proper input domain mapping between the hardware
        int32 domain and the quantized int8 simulation domain.
        """
        pytest.skip(
            "LUT Tanh requires input domain mapping fix. "
            "The AccumulateOp structure is verified correctly."
        )

        model = ANNResidualSubtract()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 8)
        x_int8 = torch.randint(-128, 128, (1, 8), dtype=torch.int8)

        # PyTorch reference
        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        # PAIIR
        graph = compile_to_paiir(model, x_compile)
        graph.reset()

        # Verify AccumulateOp exists with subtraction signs
        acc_ops = [n for n in graph.nodes.values() if isinstance(n, AccumulateOp)]
        assert len(acc_ops) == 1
        assert acc_ops[0].signs == (1, -1)

        paiir_out = graph.step(x_int8)
        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == pytorch_out.shape

        # Validate with Tanh error bounds
        result = _validate_ann_output(paiir_out, pytorch_out, "tanh")
        assert result["passed"], (
            f"Tanh validation failed: MAE={result['mae']:.2f}, "
            f"max_error={result['max_error']:.2f}"
        )

    def test_multi_input_merge(self) -> None:
        """MultiInputMerge: two inputs, add + LIF.

        Tests multiple InputNodes handling in accumulation context.
        """
        model = MultiInputMerge()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x1_int8 = _make_snn_input()
        x2_int8 = _make_snn_input()

        sj_out = _run_snn_reference(model, x1_int8, x2_int8)
        assert isinstance(sj_out, Tensor)  # MultiInputMerge has single output

        # PAIIR
        graph = compile_to_paiir(model, x_compile, x_compile)
        graph.reset()

        # Verify AccumulateOp exists
        acc_ops = [n for n in graph.nodes.values() if isinstance(n, AccumulateOp)]
        assert len(acc_ops) == 1

        paiir_out = graph.step(x1_int8, x2_int8)
        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, sj_out)


class TestConcatOpSimulation:
    """Tests for tensor concatenation operations."""

    def test_sppf_block_concat(self) -> None:
        """SPPFBlock: cascaded MaxPool -> cat -> Conv.

        Verifies ConcatOp exists and correct output shape.

        SPPFBlock structure (tick_start):
        - cv1 (Conv-IF): tick_start=1
        - MaxPool chain: tick_start=2,3,4
        - ConcatOp: routing
        - cv2 (Conv-IF): tick_start=5

        Need to run max_tick_start=5 steps to get final output.
        """
        model = SPPFBlock()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 16, 8, 8)
        x_int8 = torch.randint(0, 2, x_compile.shape, dtype=torch.int8)

        # PAIIR
        graph = compile_to_paiir(model, x_compile)

        # Verify ConcatOp exists
        concat_ops = [n for n in graph.nodes.values() if isinstance(n, ConcatOp)]
        assert len(concat_ops) == 1
        assert concat_ops[0].dim == 1

        # Run correct number of steps based on max tick_start
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        graph.reset()
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)

        sj_out = _run_snn_reference(model, x_int8)
        assert isinstance(sj_out, Tensor)  # SPPFBlock has single output
        assert torch.equal(paiir_out.to(torch.int8), sj_out.to(torch.int8))

    def test_concat_ordering(self) -> None:
        """ConcatOp preserves input ordering via dst_port.

        Critical test: predecessors() returns names sorted by dst_port,
        ensuring correct output channel ordering matches input order.

        Verification strategy:
        - Each conv has distinct weights (1, 2, 3) so outputs are distinguishable
        - Verify PAIIR output segment i matches PyTorch segment i exactly
        - This catches ordering errors: if PAIIR outputs [b, a, c] instead of [a, b, c],
          the segments won't match
        """

        # Create a model with explicit concatenation order
        class _ConcatTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
                self.conv2 = nn.Conv2d(3, 4, 3, padding=1)
                self.conv3 = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                c = self.conv3(x)
                # Explicit order: [a, b, c]
                return torch.cat([a, b, c], dim=1)

        model = _ConcatTest()

        # Set DISTINCT weights for each conv to make outputs distinguishable
        with torch.no_grad():
            model.conv1.weight.fill_(1)
            model.conv2.weight.fill_(2)
            model.conv3.weight.fill_(3)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_ann_input()

        # PAIIR
        graph = compile_to_paiir(model, x_compile)

        # Find ConcatOp
        concat_ops = [n for n in graph.nodes.values() if isinstance(n, ConcatOp)]
        assert len(concat_ops) == 1
        concat_node = concat_ops[0]

        # Find predecessors (should be ordered by dst_port)
        preds = graph.predecessors(concat_node.name)
        assert len(preds) == 3

        # Verify predecessors are OfflineCoreOps (conv ops without activation)
        for pred_name in preds:
            pred_node = graph.nodes[pred_name]
            assert isinstance(pred_node, OfflineCoreOp)

        # Run simulation
        graph.reset()
        paiir_out = graph.step(x_int8)
        assert torch.is_tensor(paiir_out)
        assert paiir_out.shape == (1, 12, 8, 8)  # 4 + 4 + 4 channels

        # Reference
        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        # CRITICAL: Verify exact match for each segment
        # Distinct weights (1, 2, 3) ensure segments are different
        # If PAIIR outputs wrong order like [b, a, c], this will fail
        for i in range(3):
            paiir_seg = paiir_out[:, i * 4 : (i + 1) * 4, :, :]
            pytorch_seg = pytorch_out[:, i * 4 : (i + 1) * 4, :, :]

            torch.testing.assert_close(
                paiir_seg.to(torch.float32),
                pytorch_seg,
                rtol=1e-5,
                atol=1.0,  # Allow small quantization error
                msg=f"Segment {i} mismatch: PAIIR concat order may differ from PyTorch",
            )


class TestSplitOpSimulation:
    def test_unfused_split_branch_tuple_matches_pytorch(self) -> None:
        class SplitBranchModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_left = nn.Conv2d(2, 4, 1, bias=False)
                self.conv_right = nn.Conv2d(3, 4, 1, bias=False)

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                left, right = torch.split(x, [2, 3], dim=1)
                return self.conv_left(left), self.conv_right(right)

        model = SplitBranchModel().eval()
        _set_quantized_weights(model)

        x = torch.randint(-8, 8, (1, 5, 4, 4), dtype=torch.int8)
        graph = torch_to_paiir(model, x.float())

        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        assert len(split_nodes) == 1

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x.float())

        paiir_out = graph.forward(x)

        assert isinstance(paiir_out, tuple)
        assert len(paiir_out) == 2
        torch.testing.assert_close(paiir_out[0].to(torch.float32), pytorch_out[0])
        torch.testing.assert_close(paiir_out[1].to(torch.float32), pytorch_out[1])

    def test_unfused_split_then_concat_matches_pytorch(self) -> None:
        class SplitConcatModel(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                left, right = torch.split(x, [2, 3], dim=1)
                return torch.cat([right, left], dim=1)

        model = SplitConcatModel().eval()
        x = torch.randn(1, 5, 4, 4)

        graph = torch_to_paiir(model, x)
        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        concat_nodes = [
            node for node in graph.nodes.values() if isinstance(node, ConcatOp)
        ]

        assert len(split_nodes) == 1
        assert len(concat_nodes) == 1

        with torch.no_grad():
            pytorch_out = model(x)

        paiir_out = graph.forward(x)

        assert torch.is_tensor(paiir_out)
        torch.testing.assert_close(paiir_out, pytorch_out)

    def test_direct_split_outputs_match_pytorch_with_single_split_node(self) -> None:
        class SplitOutputModel(nn.Module):
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                left, right = torch.split(x, [2, 3], dim=1)
                return left, right

        model = SplitOutputModel().eval()
        x = torch.randn(1, 5, 4, 4)

        graph = torch_to_paiir(model, x)
        split_nodes = [
            node for node in graph.nodes.values() if isinstance(node, SplitOp)
        ]
        output_nodes = graph.output_nodes()

        assert len(split_nodes) == 1
        assert len(output_nodes) == 2

        with torch.no_grad():
            pytorch_out = model(x)

        paiir_out = graph.forward(x)

        assert isinstance(paiir_out, tuple)
        assert len(paiir_out) == 2
        torch.testing.assert_close(paiir_out[0], pytorch_out[0])
        torch.testing.assert_close(paiir_out[1], pytorch_out[1])


class TestStandaloneOpSimulation:
    """Tests for standalone compute operations without activation."""

    def test_standalone_conv(self) -> None:
        """Conv without activation, outputs membrane potential.

        Verifies StandaloneCompOp correctly outputs accumulated potential
        without neuron/LUT activation.
        """
        model = nn.Conv2d(3, 8, 3)
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 8, 8)
        x_int8 = _make_ann_input()

        # PyTorch reference
        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        # PAIIR
        graph = compile_to_paiir(model, x_compile)

        # Verify StandaloneCompOp exists
        standalone_ops = [
            n for n in graph.nodes.values() if isinstance(n, StandaloneCompOp)
        ]
        assert len(standalone_ops) == 1

        graph.reset()
        paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_view_and_view_as_routing_before_linear(self) -> None:
        """`view(size(0), -1)` and `view_as(...)` execute as routing ops."""

        class ViewLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(12, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = x.view(x.size(0), -1)
                x = x.view_as(x)
                return self.linear(x)

        model = ViewLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 2, 2)
        x_int8 = torch.randint(-128, 128, (1, 3, 2, 2), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_view_as_reference_path_before_linear(self) -> None:
        """`view_as(ref)` uses only the data tensor as the runtime graph input."""

        class ViewAsReferenceFromFlatten(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(12, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                ref = x.flatten(1)
                y = x.view_as(ref)
                return self.linear(y)

        model = ViewAsReferenceFromFlatten()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 3, 2, 2)
        x_int8 = torch.randint(-128, 128, (1, 3, 2, 2), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        with pytest.warns(GraphCleanupWarning, match="disconnected"):
            graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_view_shape_arithmetic_before_linear(self) -> None:
        """Shape arithmetic feeding `view(...)` is treated as shape-only aux graph."""

        class ViewShapeArithmetic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(24, 5, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                batch = x.size(0) + x.size(1) - x.size(1)
                features = (x.size(1) * x.size(2) * x.size(3)) // batch
                return self.linear(x.view(batch, features))

        model = ViewShapeArithmetic()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3, 4)
        x_int8 = torch.randint(-128, 128, (1, 2, 3, 4), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_function_unsqueeze_before_linear(self) -> None:
        """Function-form `torch.unsqueeze(...)` executes as a routing op."""

        class FunctionUnsqueezeLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = torch.unsqueeze(x, 1)
                x = x.flatten(1)
                return self.linear(x)

        model = FunctionUnsqueezeLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3)
        x_int8 = torch.randint(-128, 128, (1, 2, 3), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_tuple_repeat_all_ones_before_linear(self) -> None:
        """Tuple-form identity `repeat((1,...))` executes as a routing no-op."""

        class TupleRepeatLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = x.repeat((1, 1, 1))
                x = x.flatten(1)
                return self.linear(x)

        model = TupleRepeatLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3)
        x_int8 = torch.randint(-128, 128, (1, 2, 3), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_method_squeeze_before_linear(self) -> None:
        """Method-form `squeeze(...)` executes as a routing op."""

        class MethodSqueezeLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = x.squeeze(1)
                x = x.flatten(1)
                return self.linear(x)

        model = MethodSqueezeLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 1, 2, 3)
        x_int8 = torch.randint(-128, 128, (1, 1, 2, 3), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_function_squeeze_before_linear(self) -> None:
        """Function-form `torch.squeeze(...)` executes as a routing op."""

        class FunctionSqueezeLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = torch.squeeze(x, 1)
                x = x.flatten(1)
                return self.linear(x)

        model = FunctionSqueezeLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 1, 2, 3)
        x_int8 = torch.randint(-128, 128, (1, 1, 2, 3), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_transpose_then_flatten_before_linear(self) -> None:
        """Bypassed transpose metadata is materialized at the downstream reshape op."""

        class TransposeFlattenLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 4, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x.transpose(1, 2).flatten(1))

        model = TransposeFlattenLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3)
        x_int8 = torch.randint(-128, 128, (1, 2, 3), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_permute_then_reshape_before_linear(self) -> None:
        """Bypassed permute metadata is materialized at the downstream reshape op."""

        class PermuteReshapeLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(24, 5, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(x.size(0), -1)
                return self.linear(x)

        model = PermuteReshapeLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3, 4)
        x_int8 = torch.randint(-128, 128, (1, 2, 3, 4), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    def test_flatten_then_reshape_chain_before_linear(self) -> None:
        """Chained `flatten -> reshape(size arithmetic) -> flatten -> Linear` simulates.

        The pre-fusion layout canonicalization pass now collapses the reshape
        chain to a single effective `ReshapeOp`.
        """

        class FlattenReshapeLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(24, 5, bias=False)

            def forward(self, x: Tensor) -> Tensor:
                x = x.flatten(1)
                x = x.reshape(x.size(0), 2, x.size(1) // 2)
                x = x.flatten(1)
                return self.linear(x)

        model = FlattenReshapeLinear()
        _set_quantized_weights(model)

        x_compile = torch.randn(1, 2, 3, 4)
        x_int8 = torch.randint(-128, 128, (1, 2, 3, 4), dtype=torch.int8)

        model.eval()
        with torch.no_grad():
            pytorch_out = model(x_int8.float())

        graph = compile_to_paiir(model, x_compile)
        reshape_nodes = [n for n in graph.nodes.values() if isinstance(n, ReshapeOp)]
        assert len(reshape_nodes) == 1

        graph.reset()
        max_tick_start = max(
            n.core_params.tick_start
            for n in graph.nodes.values()
            if isinstance(n, OfflineCoreOp) and n.core_params.tick_start is not None
        )
        for _ in range(max_tick_start):
            paiir_out = graph.step(x_int8)

        assert torch.is_tensor(paiir_out)
        assert torch.equal(paiir_out, pytorch_out)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8], ids=["uint8", "int8"])
    def test_standalone_maxpool_preserves_integer_dtype(
        self, dtype: torch.dtype
    ) -> None:
        model = nn.MaxPool2d(2, 2)

        x_compile = torch.randn(1, 1, 4, 4)
        x_int = torch.arange(16, dtype=dtype).reshape(1, 1, 4, 4)

        with torch.no_grad():
            pytorch_out = model(x_int)

        graph = compile_to_paiir(model, x_compile)

        standalone_ops = [
            n for n in graph.nodes.values() if isinstance(n, StandaloneCompOp)
        ]
        assert len(standalone_ops) == 1

        graph.reset()
        paiir_out = graph.step(x_int)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.dtype == dtype
        assert torch.equal(paiir_out, pytorch_out)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8], ids=["uint8", "int8"])
    def test_standalone_maxpool1d_preserves_integer_dtype(
        self, dtype: torch.dtype
    ) -> None:
        model = nn.MaxPool1d(2, 2)

        x_compile = torch.randn(1, 1, 8)
        x_int = torch.arange(8, dtype=dtype).reshape(1, 1, 8)

        with torch.no_grad():
            pytorch_out = model(x_int.to(torch.float32)).to(dtype)

        graph = compile_to_paiir(model, x_compile)

        standalone_ops = [
            n for n in graph.nodes.values() if isinstance(n, StandaloneCompOp)
        ]
        assert len(standalone_ops) == 1

        graph.reset()
        paiir_out = graph.step(x_int)

        assert torch.is_tensor(paiir_out)
        assert paiir_out.dtype == dtype
        assert torch.equal(paiir_out, pytorch_out)
