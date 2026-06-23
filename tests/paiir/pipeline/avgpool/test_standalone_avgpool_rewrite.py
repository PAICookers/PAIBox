import pytest
import torch
from paicorelib import DataSign, DataWidth, ThresholdNegMode
from spikingjelly.activation_based import layer, neuron
from torch import nn

from paibox.paiir import ANNNodeV25, IFNodeV25, LutLinear
from paibox.paiir.exceptions import GraphValidationError, OutputApproxWarning
from paibox.paiir.ir.op_node import SequentialOp, StandaloneCompOp
from paibox.paiir.ir.signal_domain import SignalDomain
from paibox.paiir.lowering.converter import register_neuron
from paibox.paiir.nn import SumPool1d, SumPool2d
from paibox.paiir.pipeline import compile_to_paiir
from paibox.paiir.pipeline.data_format import infer_output_format


class DirectStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.pool(x)


class SpikeStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = neuron.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.pool(self.if1(self.conv(x)))


class PotentialStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.pool(self.conv(x))


class SpikeStandaloneAvgPoolDivisorOverride(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = neuron.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(3, 3, divisor_override=1)

    def forward(self, x):
        return self.pool(self.if1(self.conv(x)))


class UnsignedAnnStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class MixedModeStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_snn = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = neuron.IFNode(v_threshold=1.0)
        self.conv_ann = nn.Conv2d(1, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        a = self.if1(self.conv_snn(x))
        b = self.relu(self.conv_ann(x))
        return self.pool(torch.cat((a, b), dim=1))


class ResidualAnnStandaloneAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(1, 1, 1, bias=False)
        self.conv_b = nn.Conv2d(1, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        y = self.conv_a(x) + self.conv_b(x)
        return self.pool(self.relu(y))


class SpikeVotingLayer10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.if1 = neuron.IFNode(v_threshold=1.0)
        self.vote = layer.VotingLayer(10, step_mode="s")

    def forward(self, x):
        return self.vote(self.if1(x))


class SpikeAvgPoolThenLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.if1 = neuron.IFNode(v_threshold=1.0)
        self.pool = nn.AvgPool2d(3, 3)
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        y = self.pool(self.if1(self.conv(x)))
        return self.linear(y.flatten(1))


def _find_rewritten_avgpool_seq(graph) -> SequentialOp:
    return next(
        node
        for node in graph.nodes.values()
        if isinstance(node, SequentialOp) and isinstance(node.comp, SumPool2d)
    )


def _find_rewritten_sumpool1d_seq(graph) -> SequentialOp:
    return next(
        node
        for node in graph.nodes.values()
        if isinstance(node, SequentialOp) and isinstance(node.comp, SumPool1d)
    )


def _find_standalone_avgpool(graph) -> StandaloneCompOp:
    return next(
        node
        for node in graph.nodes.values()
        if isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.AvgPool2d)
    )


def test_direct_input_avgpool_stays_standalone_when_source_mode_is_unknown() -> None:
    graph = compile_to_paiir(
        DirectStandaloneAvgPool(),
        torch.zeros(1, 1, 6, 6),
        input_formats={"InputNode_0": (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
    )

    pool = _find_standalone_avgpool(graph)
    assert isinstance(pool.comp, nn.AvgPool2d)


def test_potential_predecessor_avgpool_is_rejected_by_32bit_contract() -> None:
    model = PotentialStandaloneAvgPool().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    with pytest.raises(
        GraphValidationError, match="StandaloneCompOp.*WIDTH_32BIT|WIDTH_32BIT"
    ):
        compile_to_paiir(model, torch.zeros(1, 1, 6, 6))


def test_binary_majority_specializes_spike_predecessor_avgpool2d() -> None:
    model = SpikeStandaloneAvgPool().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    with pytest.warns(OutputApproxWarning, match="majority spikes"):
        graph = compile_to_paiir(model, torch.zeros(1, 1, 6, 6))
    seq = _find_rewritten_avgpool_seq(graph)

    assert isinstance(seq.comp, SumPool2d)
    assert isinstance(seq.act, IFNodeV25)
    assert seq.act.thres_pos == 1
    assert seq.act.leak_v == -4
    assert seq.act.thres_neg_mode == ThresholdNegMode.FLOOR
    assert seq.act.thres_neg == 0
    assert infer_output_format(seq.act) == (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)
    assert not any(
        isinstance(node, StandaloneCompOp) and isinstance(node.comp, nn.AvgPool2d)
        for node in graph.nodes.values()
    )


def test_binary_majority_forward_matches_manual_majority_rule() -> None:
    model = SpikeStandaloneAvgPool().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1)
    with pytest.warns(OutputApproxWarning, match="majority spikes"):
        graph = compile_to_paiir(model, torch.zeros(1, 1, 6, 6))
    seq = _find_rewritten_avgpool_seq(graph)

    x = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    seq.act.reset()
    actual = seq(x).to(torch.int64)
    expected = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.int64)

    assert torch.equal(actual, expected)


def test_binary_majority_falls_back_to_standalone_when_divisor_differs() -> None:
    model = SpikeStandaloneAvgPoolDivisorOverride().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1)

    graph = compile_to_paiir(model, torch.zeros(1, 1, 6, 6))

    pool = _find_standalone_avgpool(graph)
    assert isinstance(pool.comp, nn.AvgPool2d)


def test_unsigned_ann_standalone_avgpool_rewrites_to_exact_ann_path() -> None:
    graph = compile_to_paiir(UnsignedAnnStandaloneAvgPool(), torch.zeros(1, 1, 6, 6))
    seq = _find_rewritten_avgpool_seq(graph)

    assert isinstance(seq.comp, SumPool2d)
    assert isinstance(seq.act, ANNNodeV25)
    assert infer_output_format(seq.act) == (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)

    x = torch.tensor(
        [
            [
                [
                    [0, 1, 2, 10, 20, 30],
                    [3, 4, 5, 40, 50, 60],
                    [6, 7, 8, 70, 80, 90],
                    [100, 110, 120, 130, 140, 150],
                    [160, 170, 180, 190, 200, 210],
                    [220, 230, 240, 250, 255, 255],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    seq.act.reset()
    actual = seq(x).to(torch.int64)
    expected = torch.round(nn.AvgPool2d(3, 3)(x.to(torch.float32))).to(torch.int64)

    assert torch.equal(actual, expected)


def test_signed_ann_standalone_avgpool_rewrites_to_exact_ann_path() -> None:
    class SignedAnnIdentity(nn.Module):
        def forward(self, x):
            return x.to(torch.float32)

    register_neuron(
        SignedAnnIdentity,
        converter=lambda _: ANNNodeV25(
            LutLinear(min_val=-128, max_val=127, output_signed=True)
        ),
    )

    class SignedAnnStandaloneAvgPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            self.signed_ann = SignedAnnIdentity()
            self.pool = nn.AvgPool2d(3, 3)

        def forward(self, x):
            return self.pool(self.signed_ann(self.conv(x)))

    graph = compile_to_paiir(SignedAnnStandaloneAvgPool(), torch.zeros(1, 1, 6, 6))
    seq = _find_rewritten_avgpool_seq(graph)

    assert isinstance(seq.comp, SumPool2d)
    assert isinstance(seq.act, ANNNodeV25)
    assert infer_output_format(seq.act) == (DataSign.SIGNED, DataWidth.WIDTH_8BIT)

    x = torch.tensor(
        [
            [
                [
                    [-128, -120, -64, -10, 0, 10],
                    [-127, -100, -32, -8, 8, 32],
                    [-90, -45, -1, 1, 45, 90],
                    [-70, -35, -5, 5, 35, 70],
                    [-16, -8, -4, 4, 8, 16],
                    [0, 20, 40, 60, 100, 127],
                ]
            ]
        ],
        dtype=torch.int32,
    )
    seq.act.reset()
    actual = seq(x).to(torch.int64)
    expected = torch.round(nn.AvgPool2d(3, 3)(x.to(torch.float32))).to(torch.int64)

    assert torch.equal(actual, expected)


def test_accumulate_ann_predecessor_avgpool_rewrites_to_exact_ann_path() -> None:
    model = ResidualAnnStandaloneAvgPool().eval()
    with torch.no_grad():
        model.conv_a.weight.fill_(1)
        model.conv_b.weight.fill_(1)

    graph = compile_to_paiir(model, torch.zeros(1, 1, 6, 6))
    seq = _find_rewritten_avgpool_seq(graph)

    assert isinstance(seq.comp, SumPool2d)
    assert isinstance(seq.act, ANNNodeV25)
    assert infer_output_format(seq.act) == (DataSign.UNSIGNED, DataWidth.WIDTH_8BIT)


def test_mixed_mode_standalone_avgpool_stays_standalone() -> None:
    model = MixedModeStandaloneAvgPool().eval()
    with torch.no_grad():
        model.conv_snn.weight.fill_(1)
        model.conv_ann.weight.fill_(1)

    graph = compile_to_paiir(model, torch.zeros(1, 1, 6, 6))

    pool = _find_standalone_avgpool(graph)
    assert isinstance(pool.comp, nn.AvgPool2d)


def test_output_voting_layer_default_keeps_majority_fallback() -> None:
    model = SpikeVotingLayer10().eval()

    with pytest.warns(OutputApproxWarning, match="SumPool1d \\+ IFNodeV25"):
        graph = compile_to_paiir(model, torch.zeros(1, 110))

    seq = _find_rewritten_sumpool1d_seq(graph)
    assert isinstance(seq.comp, SumPool1d)
    assert isinstance(seq.act, IFNodeV25)
    assert infer_output_format(seq.act) == (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)


def test_output_voting_layer_sum_approx_exports_counts() -> None:
    model = SpikeVotingLayer10().eval()

    with pytest.warns(OutputApproxWarning, match="SumPool1d \\+ identity LUT"):
        graph = compile_to_paiir(
            model, torch.zeros(1, 110), output_approx="sum_approx_if_avgpool"
        )

    seq = _find_rewritten_sumpool1d_seq(graph)
    assert isinstance(seq.comp, SumPool1d)
    assert isinstance(seq.act, ANNNodeV25)
    assert seq.signal_semantics.output_domain is SignalDomain.VALUE
    assert seq.signal_semantics.known_code_range == (0, 10)
    assert seq.core_params.output_sign == DataSign.UNSIGNED
    assert seq.core_params.output_width == DataWidth.WIDTH_4BIT
    assert graph.output_nodes()[0].shape == torch.Size((1, 11))


def test_output_voting_layer_sum_counts_preserve_time_accumulated_argmax() -> None:
    model = SpikeVotingLayer10().eval()
    with pytest.warns(OutputApproxWarning, match="identity LUT"):
        graph = compile_to_paiir(
            model, torch.zeros(1, 110), output_approx="sum_approx_if_avgpool"
        )
    seq = _find_rewritten_sumpool1d_seq(graph)

    spike_frames = torch.zeros(4, 1, 110, dtype=torch.int32)
    spike_frames[0, 0, 0:10] = 1
    spike_frames[1, 0, 10:20] = 1
    spike_frames[2, 0, 10:20] = 1
    spike_frames[3, 0, 20:30] = 1

    avgpool = nn.AvgPool1d(10, 10)
    count_sum = torch.zeros(1, 11, dtype=torch.int64)
    avg_sum = torch.zeros(1, 11, dtype=torch.float32)
    for frame in spike_frames:
        seq.act.reset()
        count_sum += seq(frame).to(torch.int64)
        avg_sum += avgpool(frame.to(torch.float32))

    assert torch.equal(count_sum.argmax(dim=1), avg_sum.argmax(dim=1))


def test_output_sum_approx_does_not_change_non_output_avgpool_policy() -> None:
    model = SpikeAvgPoolThenLinear().eval()
    with torch.no_grad():
        model.conv.weight.fill_(1)
        model.linear.weight.fill_(1)

    graph = compile_to_paiir(
        model,
        torch.zeros(1, 1, 6, 6),
        enable_delayed_avgpool_division=False,
        output_approx="sum_approx_if_avgpool",
    )

    seq = _find_rewritten_avgpool_seq(graph)
    assert isinstance(seq.comp, SumPool2d)
    assert isinstance(seq.act, IFNodeV25)
