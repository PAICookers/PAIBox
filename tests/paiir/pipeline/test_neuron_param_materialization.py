import pytest
import torch
from paicorelib import LeakAddMode, LeakMultiMode
from torch import Tensor, nn

from paibox.paiir import (
    ANNNodeV25,
    CoreNeuronV25,
    IFNodeV25,
    LeakyBeta0NodeV25,
    LutReLU,
    compile_to_paiir,
)
from paibox.paiir.ir.calc_params import NeuronParams
from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    OfflineCoreOp,
    SequentialOp,
    StandaloneCompOp,
    TensorLayout,
)
from paibox.paiir.pipeline.neuron_param_materialization import (
    materialize_neuron_params,
)


class _ParamOp(OfflineCoreOp):
    def __init__(self, params: NeuronParams, bias: Tensor | None = None) -> None:
        super().__init__()
        self.params_source = params
        self.bias_source = bias

    def _src_params(self) -> tuple[NeuronParams, Tensor | None]:
        return self.params_source, self.bias_source


def _graph_with(*nodes: OfflineCoreOp) -> PAIIRGraph:
    graph = PAIIRGraph("materialization")
    for node in nodes:
        graph.add_node(node)
    return graph


def _set_output_shape(node: OfflineCoreOp, shape: tuple[int, ...]) -> None:
    node.output_layouts = (TensorLayout(torch.Size(shape), tuple(range(len(shape)))),)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(7, 7, id="python-scalar"),
        pytest.param(torch.tensor(7), 7, id="zero-dimensional-tensor"),
        pytest.param(
            torch.tensor([1, 2]),
            torch.tensor([1] * 6 + [2] * 6),
            id="per-channel",
        ),
        pytest.param(
            torch.arange(12),
            torch.arange(12),
            id="already-flat",
        ),
        pytest.param(
            torch.tensor([[[1]], [[2]]]),
            torch.tensor([1] * 6 + [2] * 6),
            id="channel-broadcast-shape",
        ),
        pytest.param(
            torch.tensor([[[[1]], [[2]]]]),
            torch.tensor([1] * 6 + [2] * 6),
            id="singleton-batch",
        ),
        pytest.param(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([1, 2, 3, 4, 5, 6] * 2),
            id="generic-broadcast",
        ),
    ],
)
def test_materializes_supported_tensor_shapes(value, expected):
    node = _ParamOp(NeuronParams(thres_pos=value))
    _set_output_shape(node, (1, 2, 2, 3))

    materialize_neuron_params(_graph_with(node))

    actual = node.neu_params.thres_pos
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        assert actual.shape == (12,)
    else:
        assert not torch.is_tensor(actual)
        assert actual == expected


def test_materialization_is_atomic_when_a_later_node_is_invalid():
    valid = _ParamOp(NeuronParams(thres_pos=torch.tensor([1, 2])))
    invalid = _ParamOp(NeuronParams(thres_pos=torch.ones(4, 4)))
    for node in (valid, invalid):
        _set_output_shape(node, (1, 2, 2, 3))

    with pytest.raises(ValueError, match="cannot broadcast"):
        materialize_neuron_params(_graph_with(valid, invalid))

    for node in (valid, invalid):
        with pytest.raises(RuntimeError, match="not materialized"):
            _ = node.neu_params


def test_rejects_batch_varying_parameter_tensor():
    node = _ParamOp(NeuronParams(thres_pos=torch.ones(2, 2, 2, 3)))
    _set_output_shape(node, (1, 2, 2, 3))

    with pytest.raises(ValueError, match="batch-varying"):
        materialize_neuron_params(_graph_with(node))


@pytest.mark.parametrize("value", [1.0, torch.tensor([0.0, -1.0])])
def test_rejects_non_integer_leak_tau(value):
    node = _ParamOp(NeuronParams(leak_tau=value))
    _set_output_shape(node, (1, 2))

    with pytest.raises(TypeError, match="leak_tau must"):
        materialize_neuron_params(_graph_with(node))


def test_rank_one_output_materializes_one_logical_neuron():
    node = _ParamOp(NeuronParams(thres_pos=torch.tensor([7])))
    _set_output_shape(node, (1,))

    materialize_neuron_params(_graph_with(node))

    assert torch.equal(node.neu_params.thres_pos, torch.tensor([7]))


def test_rejects_elementwise_inverted_thresholds_after_broadcast():
    node = _ParamOp(
        NeuronParams(
            thres_pos=torch.tensor([1, 3]),
            thres_neg=torch.tensor([2, 2]),
        )
    )
    _set_output_shape(node, (1, 2, 2))

    with pytest.raises(ValueError, match="thres_pos >= thres_neg elementwise"):
        materialize_neuron_params(_graph_with(node))

    with pytest.raises(RuntimeError, match="not materialized"):
        _ = node.neu_params


def test_merges_output_channel_bias_after_leak_materialization():
    params = NeuronParams(leak_v=torch.tensor([1.0, 2.0]))
    node = _ParamOp(params, torch.tensor([10.0, 20.0]))
    _set_output_shape(node, (1, 2, 2, 3))

    materialize_neuron_params(_graph_with(node))
    params.leak_v = torch.tensor([100.0, 200.0])

    assert torch.equal(node.neu_params.leak_v, torch.tensor([11.0] * 6 + [22.0] * 6))


def test_accumulate_keeps_signed_bias_separate_until_materialization():
    left = nn.Linear(3, 2)
    right = nn.Linear(3, 2)
    with torch.no_grad():
        left.bias.copy_(torch.tensor([4.0, 6.0]))
        right.bias.copy_(torch.tensor([1.0, 2.0]))
    node = AccumulateOp(
        (left, right),
        IFNodeV25(leak_v=torch.tensor([10.0, 20.0])),
        op_signs=(1, -1),
    )
    _set_output_shape(node, (1, 2))

    raw_params, raw_bias = node.src_params()
    assert torch.equal(raw_params.leak_v, torch.tensor([10.0, 20.0]))
    assert torch.equal(raw_bias, torch.tensor([3.0, 4.0]))

    materialize_neuron_params(_graph_with(node))

    assert torch.equal(node.neu_params.leak_v, torch.tensor([13.0, 24.0]))


def test_standalone_compute_materializes_output_channel_bias():
    linear = nn.Linear(3, 2)
    with torch.no_grad():
        linear.bias.copy_(torch.tensor([4.0, 6.0]))
    node = StandaloneCompOp(linear)
    _set_output_shape(node, (1, 2))

    materialize_neuron_params(_graph_with(node))

    assert torch.equal(node.neu_params.leak_v, torch.tensor([4.0, 6.0]))


def test_rejects_bias_with_backward_additive_leak():
    node = _ParamOp(
        NeuronParams(leak_add_mode=LeakAddMode.BACKWARD), torch.tensor([1.0, 2.0])
    )
    _set_output_shape(node, (1, 2))

    with pytest.raises(ValueError, match="leak_add_mode is BACKWARD"):
        materialize_neuron_params(_graph_with(node))


def test_compile_materializes_after_fusion_and_merges_linear_bias():
    linear = nn.Linear(3, 2)
    with torch.no_grad():
        linear.bias.copy_(torch.tensor([1.0, 2.0]))
    graph = compile_to_paiir(
        nn.Sequential(linear, IFNodeV25(leak_v=torch.tensor([10.0, 20.0]))),
        torch.zeros(1, 3),
    )
    offline_nodes = [
        node for node in graph.nodes.values() if isinstance(node, OfflineCoreOp)
    ]

    assert offline_nodes
    sequential = next(node for node in offline_nodes if isinstance(node, SequentialOp))
    assert torch.equal(sequential.neu_params.leak_v, torch.tensor([11.0, 22.0]))


def test_compile_materializes_ann_vector_parameters():
    graph = compile_to_paiir(
        nn.Sequential(
            nn.Linear(3, 2, bias=False),
            ANNNodeV25(
                LutReLU(),
                reset_v=torch.tensor([0.0, 1.0]),
                thres_neg=torch.tensor([-2.0, -3.0]),
                thres_pos=torch.tensor([2.0, 3.0]),
                leak_tau_shift=torch.tensor([0, -1]),
                leak_v=torch.tensor([4.0, 5.0]),
                init_v=torch.tensor([0.0, 1.0]),
            ),
        ),
        torch.zeros(1, 3),
    )
    sequential = next(
        node for node in graph.nodes.values() if isinstance(node, SequentialOp)
    )

    assert torch.equal(sequential.neu_params.reset_v, torch.tensor([0.0, 1.0]))
    assert torch.equal(sequential.neu_params.leak_tau, torch.tensor([0, -1]))
    assert torch.equal(sequential.neu_params.leak_v, torch.tensor([4.0, 5.0]))


def test_compile_accepts_leaky_beta_zero_node():
    graph = compile_to_paiir(
        nn.Sequential(nn.Linear(3, 2), LeakyBeta0NodeV25()),
        torch.zeros(1, 3),
    )

    sequential = next(
        node for node in graph.nodes.values() if isinstance(node, SequentialOp)
    )
    assert sequential.act.has_lif_dynamics
    assert sequential.neu_params.leak_tau == 0


def test_compile_materializes_conv_channel_parameters_in_flat_order():
    conv = nn.Conv2d(1, 2, 1)
    with torch.no_grad():
        conv.bias.copy_(torch.tensor([10.0, 20.0]))
    graph = compile_to_paiir(
        nn.Sequential(
            conv,
            CoreNeuronV25(
                reset_v=torch.tensor([0.0, 1.0]),
                thres_neg=torch.tensor([-10.0, -11.0]),
                thres_pos=torch.tensor([5.0, 6.0]),
                leak_multi_mode=LeakMultiMode.ENABLE,
                leak_tau_shift=torch.tensor([0, -1]),
                leak_v=torch.tensor([1.0, 2.0]),
                init_v=torch.tensor([0.0, 1.0]),
            ),
        ),
        torch.zeros(1, 1, 2, 2),
    )
    sequential = next(
        node for node in graph.nodes.values() if isinstance(node, SequentialOp)
    )
    params = sequential.neu_params

    assert torch.equal(params.thres_pos, torch.tensor([5.0] * 4 + [6.0] * 4))
    assert torch.equal(params.leak_tau, torch.tensor([0] * 4 + [-1] * 4))
    assert torch.equal(params.leak_v, torch.tensor([11.0] * 4 + [22.0] * 4))
