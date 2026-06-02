import torch
from paicorelib import (
    AddPotentialMode,
    DataSign,
    DataWidth,
    LeakMultiInputMode,
    OutputType,
)
from torch import nn

from paibox.backendv2.mapper import Mapper
from paibox.paiir import compile_to_paiir
from paibox.paiir.ir import (
    AccumulateOp,
    PotentialPassthroughNodeV25,
    QuantizedConvAddReLU2dOp,
    QuantizedSequentialOp,
    SequentialOp,
    StandaloneCompOp,
)
from paibox.paiir.ir.lut_activation import LutReLU
from paibox.paiir.pipeline.quantized_materialize import materialize_quantized_ops
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode, TensorLayout
from paibox.paiir.ir.core_neuron import ANNNodeV25
from tests.paiir.conftest import find_nodes


def _layout(shape):
    return TensorLayout(torch.Size(shape), tuple(range(len(shape))))


class ManualResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        from simples_quantize.quantize_tools.ops import ManualConvAddReLU2d

        conv = nn.Conv2d(3, 3, 1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1]], [[-2]], [[3]]],
                        [[[-4]], [[5]], [[-6]]],
                        [[[7]], [[-8]], [[9]]],
                    ],
                    dtype=conv.weight.dtype,
                )
            )
        self.res = ManualConvAddReLU2d(
            original_conv2=conv,
            y_in_scale=0.25,
            y_in_zp=0,
            w_scale=0.5,
            w_zp=0,
            conv2_out_scale=0.125,
            out_scale=0.25,
            out_zp=0,
            x_scale=0.125,
            x_zp=0,
            activation_symmetric=True,
        )

    def forward(self, y, x):
        return self.res(y, x)


def test_quantized_sequential_materializes_to_sequential_op():
    graph = PAIIRGraph("quant_seq")
    inp = InputNode(torch.Size((1, 3, 4, 4)))
    conv = nn.Conv2d(3, 2, 1)
    qop = QuantizedSequentialOp(conv, ANNNodeV25(LutReLU()))
    qop.input_layouts = (_layout((1, 3, 4, 4)),)
    qop.output_layouts = (_layout((1, 2, 4, 4)),)
    out = OutputNode(torch.Size((1, 2, 4, 4)))

    graph.add_node(inp)
    graph.add_node(qop)
    graph.add_node(out)
    graph.add_edge(inp.name, qop.name)
    graph.add_edge(qop.name, out.name)

    lowered = materialize_quantized_ops(graph)

    assert not find_nodes(lowered, QuantizedSequentialOp)
    seq_nodes = find_nodes(lowered, SequentialOp)
    assert len(seq_nodes) == 1
    assert isinstance(seq_nodes[0].comp, nn.Conv2d)


def test_quantized_conv_add_relu_materializes_to_backend_ready_fragment():
    graph = PAIIRGraph("quant_residual")
    y = InputNode(torch.Size((1, 3, 4, 4)))
    x = InputNode(torch.Size((1, 2, 4, 4)))
    conv = nn.Conv2d(3, 2, 1)
    bias = torch.tensor([5, -7], dtype=conv.bias.dtype)
    with torch.no_grad():
        conv.bias.copy_(bias)
    qop = QuantizedConvAddReLU2dOp(
        conv=conv,
        act=ANNNodeV25(LutReLU()),
        shortcut_m=7,
        shortcut_n=-3,
    )
    qop.input_layouts = (_layout((1, 3, 4, 4)), _layout((1, 2, 4, 4)))
    qop.output_layouts = (_layout((1, 2, 4, 4)),)
    out = OutputNode(torch.Size((1, 2, 4, 4)))

    graph.add_node(y)
    graph.add_node(x)
    graph.add_node(qop)
    graph.add_node(out)
    graph.add_edge(y.name, qop.name, dst_port=0)
    graph.add_edge(x.name, qop.name, dst_port=1)
    graph.add_edge(qop.name, out.name)

    lowered = materialize_quantized_ops(graph)

    assert not find_nodes(lowered, QuantizedConvAddReLU2dOp)
    conv_cores = [
        node for node in find_nodes(lowered, StandaloneCompOp) if isinstance(node.comp, nn.Conv2d)
    ]
    shortcut_cores = [
        node
        for node in find_nodes(lowered, SequentialOp)
        if isinstance(node.comp, nn.Conv2d)
        and node.comp.groups == 2
        and node.comp.in_channels == 2
        and node.comp.out_channels == 2
    ]

    assert len(conv_cores) == 1
    assert torch.equal(conv_cores[0].neuron_params.leak_v, bias)
    assert conv_cores[0].neuron_params.output_type == OutputType.POTENTIAL
    assert len(shortcut_cores) == 1
    shortcut = shortcut_cores[0]
    assert shortcut.comp.kernel_size == (1, 1)
    assert shortcut.comp.bias is None
    assert torch.equal(
        shortcut.comp.weight.detach(),
        torch.full_like(shortcut.comp.weight.detach(), 7),
    )
    assert isinstance(shortcut.act, PotentialPassthroughNodeV25)
    assert shortcut.core_params.weight_sign == DataSign.UNSIGNED
    assert shortcut.core_params.weight_width == DataWidth.WIDTH_8BIT
    assert shortcut.neuron_params.leak_multi_input == LeakMultiInputMode.ENABLE
    assert shortcut.neuron_params.leak_tau == -3
    assert shortcut.neuron_params.output_type == OutputType.POTENTIAL


def test_quantized_conv_add_relu_allows_uint8_shortcut_gain_255():
    graph = PAIIRGraph("quant_residual_gain")
    y = InputNode(torch.Size((1, 2, 4, 4)))
    x = InputNode(torch.Size((1, 2, 4, 4)))
    qop = QuantizedConvAddReLU2dOp(
        conv=nn.Conv2d(2, 2, 1),
        act=ANNNodeV25(LutReLU()),
        shortcut_m=255,
        shortcut_n=0,
    )
    qop.input_layouts = (_layout((1, 2, 4, 4)), _layout((1, 2, 4, 4)))
    qop.output_layouts = (_layout((1, 2, 4, 4)),)

    graph.add_node(y)
    graph.add_node(x)
    graph.add_node(qop)
    graph.add_edge(y.name, qop.name, dst_port=0)
    graph.add_edge(x.name, qop.name, dst_port=1)

    lowered = materialize_quantized_ops(graph)
    shortcut = [
        node
        for node in find_nodes(lowered, SequentialOp)
        if isinstance(node.comp, nn.Conv2d)
        and node.comp.groups == node.comp.in_channels
        and isinstance(node.act, PotentialPassthroughNodeV25)
    ][0]

    assert shortcut.core_params.weight_sign == DataSign.UNSIGNED
    assert shortcut.core_params.weight_width == DataWidth.WIDTH_8BIT
    weights = shortcut.weights
    assert weights is not None
    assert weights[0].dtype == torch.uint8
    assert torch.equal(weights[0], torch.full_like(weights[0], 255))


def test_manual_quantized_conv_add_relu_compile_path():
    from simples_quantize.quantize_tools.paiir import register_manual_quantized_paiir

    register_manual_quantized_paiir()
    graph = compile_to_paiir(
        ManualResidualModel(),
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 3, 4, 4),
    )

    assert not find_nodes(graph, QuantizedConvAddReLU2dOp)
    acc_nodes = find_nodes(graph, AccumulateOp)
    assert len(acc_nodes) == 1
    assert acc_nodes[0].core_params.add_potential == AddPotentialMode.DIRECT_ADD
    assert acc_nodes[0].comps == [None, None]

    shortcut_cores = [
        node
        for node in find_nodes(graph, SequentialOp)
        if isinstance(node.comp, nn.Conv2d)
        and node.comp.groups == node.comp.in_channels
        and node.comp.out_channels == node.comp.in_channels
        and node.comp.kernel_size == (1, 1)
        and isinstance(node.act, PotentialPassthroughNodeV25)
    ]
    assert len(shortcut_cores) == 1
    assert shortcut_cores[0].core_params.weight_sign == DataSign.UNSIGNED
    assert shortcut_cores[0].core_params.weight_width == DataWidth.WIDTH_8BIT


def test_manual_quantized_conv_add_relu_backendv2_smoke(tmp_path):
    from simples_quantize.quantize_tools.paiir import register_manual_quantized_paiir

    register_manual_quantized_paiir()
    graph = compile_to_paiir(
        ManualResidualModel(),
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 3, 4, 4),
    )

    mapper = Mapper()
    mapper.compile(graph, output_path=tmp_path, target_platform="x86", debug=False)

    assert mapper.routing_groups
