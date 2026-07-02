import pytest
import torch
from paicorelib import RM
from torch import nn

from paibox.backendv2.op_node import CoreOpNode
from paibox.paiir import compile_to_paiir, torch_to_paiir
from paibox.paiir.exceptions import UnsupportedOpError
from paibox.paiir.ir.core_neuron import CoreNeuronV25, IFNodeV25
from paibox.paiir.ir.op_node import OfflineCoreOp, StandaloneActOp, StandaloneCompOp
from tests.paiir.conftest import find_nodes, make_vec_8d

snn = pytest.importorskip("snntorch")


def _leaky(**kwargs):
    params = {
        "beta": 1.0,
        "threshold": 1.0,
        "init_hidden": True,
        "output": False,
        "reset_delay": False,
        "surrogate_disable": True,
    }
    params.update(kwargs)
    return snn.Leaky(**params)


def _lowered_leaky_act(module: nn.Module) -> CoreNeuronV25:
    graph = torch_to_paiir(nn.Sequential(module), make_vec_8d(), strict=True)
    act_nodes = find_nodes(graph, StandaloneActOp)
    assert len(act_nodes) == 1
    act = act_nodes[0].act
    assert isinstance(act, CoreNeuronV25)
    return act


@pytest.mark.parametrize(
    ("reset_mechanism", "reset_delay", "expected_type", "reset_mode"),
    [
        ("subtract", False, IFNodeV25, RM.MODE_LINEAR),
        ("zero", False, IFNodeV25, RM.MODE_NORMAL),
        ("zero", True, IFNodeV25, RM.MODE_NORMAL),
        ("none", False, CoreNeuronV25, RM.MODE_NONRESET),
        ("none", True, CoreNeuronV25, RM.MODE_NONRESET),
    ],
)
def test_supported_reset_mechanisms_lower_to_expected_neuron(
    reset_mechanism, reset_delay, expected_type, reset_mode
):
    act = _lowered_leaky_act(
        _leaky(reset_mechanism=reset_mechanism, reset_delay=reset_delay, threshold=2.0)
    )

    assert type(act) is expected_type
    assert act.reset_mode == reset_mode
    assert act.reset_v == 0
    assert act.thres_pos == 2.0
    assert act.leak_tau == 0


@pytest.mark.parametrize(
    ("reset_mechanism", "reset_delay"),
    [
        ("subtract", False),
        ("zero", False),
        ("zero", True),
        ("none", False),
        ("none", True),
    ],
)
def test_supported_subset_matches_snntorch_spikes(reset_mechanism, reset_delay):
    leaky = _leaky(
        reset_mechanism=reset_mechanism, reset_delay=reset_delay, threshold=1.0
    )
    act = _lowered_leaky_act(
        _leaky(reset_mechanism=reset_mechanism, reset_delay=reset_delay, threshold=1.0)
    )
    act.eval()
    inputs = torch.tensor(
        [[0.4, 0.8, 1.2, -0.3], [0.7, 0.15, 0.1, 1.5], [0.6, 0.6, 0.6, 0.2]]
    )

    snntorch_spikes = [leaky(x.unsqueeze(0)).to(torch.int8) for x in inputs]
    paiir_spikes = [act(x.unsqueeze(0)).to(torch.int8) for x in inputs]

    assert torch.equal(torch.stack(snntorch_spikes), torch.stack(paiir_spikes))


def test_two_linear_leaky_blocks_lower_all_supported_frontend_ops():
    model = nn.Sequential(
        nn.Linear(8, 4, bias=False),
        _leaky(reset_mechanism="zero", threshold=2.0),
        nn.Linear(4, 2, bias=False),
        _leaky(reset_mechanism="none", reset_delay=True, threshold=3.0),
    )

    graph = torch_to_paiir(model, make_vec_8d(), strict=True)
    comp_nodes = find_nodes(graph, StandaloneCompOp)
    act_nodes = find_nodes(graph, StandaloneActOp)

    assert [type(node.comp) for node in comp_nodes] == [nn.Linear, nn.Linear]
    assert [type(node.act) for node in act_nodes] == [IFNodeV25, CoreNeuronV25]
    assert [node.act.reset_mode for node in act_nodes] == [
        RM.MODE_NORMAL,
        RM.MODE_NONRESET,
    ]
    assert [node.act.thres_pos for node in act_nodes] == [2.0, 3.0]


def test_state_quant_none_is_supported():
    act = _lowered_leaky_act(_leaky(state_quant=None))

    assert act.reset_mode == RM.MODE_LINEAR


def test_init_hidden_is_not_part_of_lowering_contract():
    act = _lowered_leaky_act(_leaky(init_hidden=False))

    assert act.reset_mode == RM.MODE_LINEAR


def test_leaky_subclass_is_not_recognized_as_supported_source_op():
    class CustomLeaky(snn.Leaky):
        pass

    with pytest.raises(UnsupportedOpError) as excinfo:
        torch_to_paiir(
            nn.Sequential(
                CustomLeaky(
                    beta=1.0,
                    init_hidden=True,
                    output=False,
                    reset_delay=False,
                    surrogate_disable=True,
                )
            ),
            make_vec_8d(),
            strict=True,
        )

    message = str(excinfo.value)
    assert "frontend=snntorch" in message
    assert "field=module" in message
    assert "only exact snntorch.Leaky is supported" in message


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            snn.Synaptic(
                alpha=0.5,
                beta=0.5,
                init_hidden=True,
                output=False,
                reset_delay=False,
                surrogate_disable=True,
            ),
            id="synaptic",
        ),
        pytest.param(
            snn.RLeaky(
                beta=1.0,
                all_to_all=False,
                init_hidden=True,
                output=False,
                reset_delay=False,
                surrogate_disable=True,
            ),
            id="rleaky",
        ),
        pytest.param(
            snn.Alpha(
                alpha=0.8,
                beta=0.5,
                init_hidden=True,
                output=False,
                surrogate_disable=True,
            ),
            id="alpha",
        ),
        pytest.param(
            snn.Lapicque(
                R=1,
                C=2,
                time_step=1,
                init_hidden=True,
                output=False,
                reset_mechanism="none",
                surrogate_disable=True,
            ),
            id="lapicque",
        ),
    ],
)
def test_unsupported_snntorch_neurons_are_frontend_hard_errors(module):
    with pytest.raises(UnsupportedOpError) as excinfo:
        torch_to_paiir(nn.Sequential(module), make_vec_8d(), strict=False)

    message = str(excinfo.value)
    assert "frontend=snntorch" in message
    assert "field=module" in message
    assert "only exact snntorch.Leaky is supported" in message


@pytest.mark.parametrize(
    ("module", "field", "value", "reason"),
    [
        pytest.param(
            _leaky(reset_delay=True),
            "reset_delay",
            "True",
            "not equivalent for subtract reset",
            id="reset_delay",
        ),
        pytest.param(_leaky(beta=0.5), "beta", "0.5", "beta != 1", id="beta_value"),
        pytest.param(
            _leaky(beta=1.0, learn_beta=True),
            "beta",
            "Parameter",
            "learnable or non-scalar",
            id="learn_beta",
        ),
        pytest.param(
            _leaky(threshold=torch.ones(2)),
            "threshold",
            "Tensor",
            "learnable or non-scalar",
            id="threshold_tensor",
        ),
        pytest.param(
            _leaky(threshold=1.0, learn_threshold=True),
            "threshold",
            "Parameter",
            "learnable or non-scalar",
            id="learn_threshold",
        ),
        pytest.param(
            _leaky(output=True),
            "output",
            "True",
            "explicit membrane output",
            id="output",
        ),
        pytest.param(
            _leaky(inhibition=True),
            "inhibition",
            "True",
            "not equivalent",
            id="inhibition",
        ),
        pytest.param(
            _leaky(state_quant=lambda x: x),
            "state_quant",
            "function",
            "not represented",
            id="state_quant",
        ),
        pytest.param(
            _leaky(graded_spikes_factor=2.0),
            "graded_spikes_factor",
            "2.0",
            "not represented",
            id="graded_spikes_factor",
        ),
        pytest.param(
            _leaky(graded_spikes_factor=1.0, learn_graded_spikes_factor=True),
            "graded_spikes_factor",
            "Parameter",
            "learnable or non-scalar",
            id="learn_graded_spikes_factor",
        ),
    ],
)
def test_unsupported_leaky_configs_report_field_value_and_reason(
    module, field, value, reason
):
    with pytest.raises(UnsupportedOpError) as excinfo:
        torch_to_paiir(nn.Sequential(module), strict=True)

    message = str(excinfo.value)
    assert "stage=source" in message
    assert "frontend=snntorch" in message
    assert "op=Leaky" in message
    assert f"field={field}" in message
    assert value in message
    assert reason in message


def test_linear_leaky_compile_preserves_neuron_attrs_for_backendv2():
    model = nn.Sequential(
        nn.Linear(8, 4, bias=False),
        _leaky(reset_mechanism="subtract", threshold=2.0),
    )

    graph = compile_to_paiir(model, make_vec_8d(), strict=True)
    cores = find_nodes(graph, OfflineCoreOp)
    sequentials = [
        node
        for node in cores
        if hasattr(node, "act") and isinstance(node.act, CoreNeuronV25)
    ]
    assert len(sequentials) == 1

    attrs = CoreOpNode(
        "snntorch_leaky", sequentials[0], torch.Size((1, 4))
    ).attrs_part2()
    assert RM(attrs.reset_mode) == RM.MODE_LINEAR
    assert attrs.threshold_pos == 2
    assert attrs.leak_tau == 0
