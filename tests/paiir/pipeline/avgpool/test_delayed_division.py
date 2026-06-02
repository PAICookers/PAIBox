import pytest
import torch
from torch import nn

from paibox.paiir import ANNNodeV25, IFNodeV25, LIFNodeV25, compile_to_paiir
from paibox.paiir.ir.lut_activation import LutCustom
from paibox.paiir.ir.op_node import SequentialOp, StandaloneCompOp
from paibox.paiir.lowering.converter import register_neuron
from paibox.paiir.nn import SumPool2d


def _make_identity_u8_lut() -> LutCustom:
    thresholds = torch.arange(256, dtype=torch.int32)
    values = torch.arange(256, dtype=torch.uint8)
    return LutCustom(thresholds, values, output_sign=0, is_float=False)


def _make_clamp_u4_lut() -> LutCustom:
    thresholds = torch.arange(256, dtype=torch.int32)
    values = torch.arange(256, dtype=torch.int32).clamp_(0, 15).to(torch.uint8)
    return LutCustom(thresholds, values, output_sign=0, is_float=False)


class ClampUint4(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.floor(x), 0, 15)


class IdentityUint8(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.floor(x), 0, 255)


@pytest.fixture(autouse=True)
def register_delayed_division_neurons(restore_default_module_map) -> None:
    register_neuron(ClampUint4, converter=lambda m: ANNNodeV25(_make_clamp_u4_lut()))
    register_neuron(
        IdentityUint8, converter=lambda m: ANNNodeV25(_make_identity_u8_lut())
    )


class LowRangeAvgPoolConvLUT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.quant = ClampUint4()
        self.pool = nn.AvgPool2d(3, 3)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=True)
        self.out = IdentityUint8()

        with torch.no_grad():
            self.conv1.weight.fill_(1)
            self.conv2.weight.fill_(1)
            self.conv2.bias.fill_(2)

    def forward(self, x):
        x = self.quant(self.conv1(x))
        x = self.pool(x)
        x = self.out(self.conv2(x))
        return x


class FullRangeAvgPoolConvLUT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.quant = IdentityUint8()
        self.pool = nn.AvgPool2d(3, 3)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
        self.out = IdentityUint8()

        with torch.no_grad():
            self.conv1.weight.fill_(1)
            self.conv2.weight.fill_(1)

    def forward(self, x):
        x = self.quant(self.conv1(x))
        x = self.pool(x)
        x = self.out(self.conv2(x))
        return x


class LowRangeAvgPoolMaxPoolLinearLUT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.quant = ClampUint4()
        self.pool = nn.AvgPool2d(3, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4, 4, bias=True)
        self.out = IdentityUint8()

        with torch.no_grad():
            self.conv.weight.fill_(1)
            self.linear.weight.copy_(torch.eye(4))
            self.linear.bias.fill_(3)

    def forward(self, x):
        x = self.quant(self.conv(x))
        x = self.pool(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.out(self.linear(x))
        return x


class LowRangeAvgPoolAvgPoolConvIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.quant = ClampUint4()
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=True)
        self.out_if = IFNodeV25(v_threshold=4.0, leak_v=1.0)

        with torch.no_grad():
            self.conv1.weight.fill_(1)
            self.conv2.weight.fill_(1)
            self.conv2.bias.fill_(1)

    def forward(self, x):
        x = self.quant(self.conv1(x))
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.conv2(x)
        return self.out_if(x)


class LowRangeAvgPoolAvgPoolConvLIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
        self.quant = ClampUint4()
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=True)
        self.out_lif = LIFNodeV25(tau=2.0, v_threshold=3.0)

        with torch.no_grad():
            self.conv1.weight.fill_(1)
            self.conv2.weight.fill_(1)
            self.conv2.bias.fill_(1)

    def forward(self, x):
        x = self.quant(self.conv1(x))
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.conv2(x)
        return self.out_lif(x)


def _reset_stateful_modules(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset", None)
        if callable(reset):
            reset()


def _compile_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
    timesteps: int = 1,
    auto_reset: bool = True,
    **kwargs,
):
    return compile_to_paiir(
        model.eval(),
        sample_input.to(torch.float32),
        timesteps=timesteps,
        auto_reset=auto_reset,
        **kwargs,
    )


def _run_and_compare(model: nn.Module, sample_input: torch.Tensor, **kwargs) -> None:
    # These tests compare operator values, so use the default continuously active
    # auto-reset policy for compiled cores.
    graph = _compile_graph(model, sample_input)
    _reset_stateful_modules(model)
    graph.reset()

    with torch.no_grad():
        expected = model(sample_input.to(torch.float32)).to(torch.int64)
        horizon = max(
            node.core_params.tick_start
            for node in graph.nodes.values()
            if hasattr(node, "core_params") and node.core_params.tick_start is not None
        )
        repeated = sample_input.to(torch.float32).repeat(
            horizon, *([1] * sample_input.ndim)
        )
        actual = graph.run(repeated, T=horizon)[-1].to(torch.int64)

    assert torch.equal(actual, expected)


def _find_sumpool_nodes(graph) -> list[SequentialOp]:
    return [
        node
        for node in graph.nodes.values()
        if isinstance(node, SequentialOp) and isinstance(node.comp, SumPool2d)
    ]


def _find_standalone_pool_nodes(
    graph, pool_type: type[nn.Module]
) -> list[StandaloneCompOp]:
    return [
        node
        for node in graph.nodes.values()
        if isinstance(node, StandaloneCompOp) and isinstance(node.comp, pool_type)
    ]


def test_default_delayed_division_rewrites_low_range_avgpool_chain() -> None:
    sample = torch.tensor(
        [
            [
                [
                    [0, 1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5, 6],
                    [2, 3, 4, 5, 6, 7],
                    [3, 4, 5, 6, 7, 8],
                    [4, 5, 6, 7, 8, 9],
                    [5, 6, 7, 8, 9, 10],
                ]
            ]
        ],
        dtype=torch.int32,
    )

    graph = _compile_graph(LowRangeAvgPoolConvLUT(), sample)
    sumpools = _find_sumpool_nodes(graph)

    assert len(sumpools) == 1
    assert sumpools[0].act.lut is not None
    assert sumpools[0].act.lut.thresholds[1].item() == 1
    _run_and_compare(LowRangeAvgPoolConvLUT(), sample)


def test_disabling_delayed_division_keeps_low_range_avgpool_standalone() -> None:
    sample = torch.arange(36, dtype=torch.int32).reshape(1, 1, 6, 6)

    graph = _compile_graph(
        LowRangeAvgPoolConvLUT(), sample, enable_delayed_avgpool_division=False
    )

    assert not _find_sumpool_nodes(graph)
    assert len(_find_standalone_pool_nodes(graph, nn.AvgPool2d)) == 1
    _run_and_compare(
        LowRangeAvgPoolConvLUT(), sample, enable_delayed_avgpool_division=False
    )


def test_full_range_source_falls_back_to_existing_exact_avg_rewrite() -> None:
    sample = torch.arange(36, dtype=torch.int32).reshape(1, 1, 6, 6)

    graph = _compile_graph(FullRangeAvgPoolConvLUT(), sample)
    sumpools = _find_sumpool_nodes(graph)

    assert len(sumpools) == 1
    assert sumpools[0].act.lut is not None
    assert sumpools[0].act.lut.thresholds[1].item() == 5
    _run_and_compare(FullRangeAvgPoolConvLUT(), sample)


def test_delayed_division_does_not_cross_standalone_maxpool() -> None:
    sample = torch.tensor(
        [
            [
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15],
                    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15],
                    [7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15],
                    [8, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15],
                    [9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15],
                    [10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15],
                    [11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15],
                ]
            ]
        ],
        dtype=torch.int32,
    )

    graph = _compile_graph(LowRangeAvgPoolMaxPoolLinearLUT(), sample)

    # Delayed division still stops before standalone MaxPool, so both pooling
    # ops remain standalone in the compiled graph.
    assert not _find_sumpool_nodes(graph)
    assert len(_find_standalone_pool_nodes(graph, nn.AvgPool2d)) == 1
    assert len(_find_standalone_pool_nodes(graph, nn.MaxPool2d)) == 1
    _run_and_compare(LowRangeAvgPoolMaxPoolLinearLUT(), sample)


def test_delayed_division_supports_avgpool_chain_into_if_with_leak_v() -> None:
    sample = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ]
            ]
        ],
        dtype=torch.int32,
    )

    graph = _compile_graph(LowRangeAvgPoolAvgPoolConvIF(), sample)
    assert len(_find_sumpool_nodes(graph)) == 2
    _run_and_compare(LowRangeAvgPoolAvgPoolConvIF(), sample)


def test_delayed_division_supports_avgpool_chain_into_lif() -> None:
    sample = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ]
            ]
        ],
        dtype=torch.int32,
    )

    graph = _compile_graph(LowRangeAvgPoolAvgPoolConvLIF(), sample)
    assert len(_find_sumpool_nodes(graph)) == 2
    _run_and_compare(LowRangeAvgPoolAvgPoolConvLIF(), sample)
