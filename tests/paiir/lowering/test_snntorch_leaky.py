import numpy as np
import pytest
import torch
from paicorelib import (
    RM,
    DataSign,
    DataWidth,
    LeakMultiComparisonOrder,
    LeakMultiMode,
)
from torch import nn

from paibox.backendv2 import Mapper
from paibox.exceptions import AutoOptimizationWarning
from paibox.paiir import compile_to_paiir, torch_to_paiir
from paibox.paiir.exceptions import UnsupportedOpError
from paibox.paiir.ir.core_neuron import CoreNeuronV25, IFNodeV25, LeakyBeta0NodeV25
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
    assert act.thres_pos == 3
    assert act.leak_tau == 0


@pytest.mark.parametrize(
    ("reset_mechanism", "reset_delay"),
    [
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
    inputs = torch.tensor([[0, 1, 2, -1], [1, 0, 0, 3], [2, 2, 0, 0]])

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
    assert [node.act.thres_pos for node in act_nodes] == [3, 4]


def test_subtract_reset_prioritizes_strict_spike_boundary():
    leaky = _leaky(reset_mechanism="subtract", threshold=1.0)
    with pytest.warns(AutoOptimizationWarning, match="maximum extra reset 1"):
        act = _lowered_leaky_act(_leaky(reset_mechanism="subtract", threshold=1.0))

    source_spike = leaky(torch.tensor([[1.0, 2.0]])).to(torch.int8)
    target_spike = act(torch.tensor([[1.0, 2.0]])).to(torch.int8)

    assert torch.equal(source_spike, target_spike)
    assert act.thres_pos == 2
    assert torch.equal(leaky.mem, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(act.v, torch.tensor([[1.0, 0.0]]))


def test_vector_beta_lowers_to_per_neuron_mode_and_shift():
    beta = torch.tensor([0.0, 0.5, 0.75, 1.0] * 2)
    threshold = torch.arange(1.0, 9.0)

    with pytest.warns(AutoOptimizationWarning, match="subtract reset"):
        act = _lowered_leaky_act(_leaky(beta=beta, threshold=threshold))

    assert type(act) is CoreNeuronV25
    assert torch.equal(act.leak_multi_mode, torch.tensor([1, 1, 1, 0, 1, 1, 1, 0]))
    assert torch.equal(act.leak_tau, torch.tensor([0, -1, -2, 0] * 2))
    assert act.leak_multi_sequence == LeakMultiComparisonOrder.AFTER_COMPARE
    assert torch.equal(act.thres_pos, torch.arange(2, 10))
    assert act.has_mixed_dynamics


@pytest.mark.parametrize("beta", [0.5, 0.75])
def test_exact_hardware_beta_matches_snntorch_spikes(beta):
    source = _leaky(beta=beta, reset_mechanism="none", threshold=2.0)
    act = _lowered_leaky_act(
        _leaky(beta=beta, reset_mechanism="none", threshold=2.0)
    ).eval()
    inputs = [torch.full((1, 8), 2.0) for _ in range(3)]

    source_spikes = [source(x).to(torch.int8) for x in inputs]
    paicore_spikes = [act(x).to(torch.int8) for x in inputs]

    assert act.leak_multi_sequence == LeakMultiComparisonOrder.AFTER_COMPARE
    assert torch.equal(torch.stack(paicore_spikes), torch.stack(source_spikes))


def test_all_beta_zero_uses_beta_zero_node():
    act = _lowered_leaky_act(_leaky(beta=torch.zeros(8), reset_mechanism="zero"))

    assert type(act) is LeakyBeta0NodeV25
    assert act.has_lif_dynamics


def test_all_beta_zero_nonreset_uses_equivalent_beta_zero_node():
    act = _lowered_leaky_act(_leaky(beta=torch.zeros(8), reset_mechanism="none"))

    assert type(act) is LeakyBeta0NodeV25
    assert act.has_lif_dynamics


def test_non_grid_beta_uses_nearest_positive_hardware_beta():
    with pytest.warns(AutoOptimizationWarning, match="max absolute error"):
        act = _lowered_leaky_act(
            _leaky(beta=torch.full((8,), 0.1), reset_mechanism="zero")
        )

    assert torch.equal(act.leak_multi_mode, torch.zeros(8, dtype=torch.int64))
    assert torch.equal(act.leak_tau, torch.full((8,), -3, dtype=torch.int64))


def test_learning_metadata_does_not_affect_current_parameter_values():
    module = _leaky(
        beta=torch.full((8,), 0.5),
        threshold=torch.arange(1.0, 9.0),
        learn_beta=True,
        learn_threshold=True,
        reset_mechanism="zero",
    )

    act = _lowered_leaky_act(module)

    assert torch.equal(act.leak_tau, torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(act.thres_pos, torch.arange(2, 10))


@pytest.mark.parametrize("field", ["beta", "threshold"])
def test_conv_output_rejects_ambiguous_1d_leaky_parameters(field):
    kwargs = {field: torch.tensor([0.5, 0.75])}
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, bias=False),
        _leaky(reset_mechanism="zero", **kwargs),
    )

    with pytest.raises(UnsupportedOpError, match="rank-2 Linear/per-feature"):
        torch_to_paiir(model, torch.zeros(1, 1, 4, 4), strict=True)


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
        pytest.param(
            _leaky(beta=torch.ones(2, 2)),
            "beta",
            "shape=(2, 2)",
            "scalar or 1D",
            id="beta_rank",
        ),
        pytest.param(
            _leaky(beta=-0.1),
            "beta",
            "-0.1",
            "values must be in [0, 1]",
            id="beta_below_range",
        ),
        pytest.param(
            _leaky(beta=1.1),
            "beta",
            "1.1",
            "values must be in [0, 1]",
            id="beta_above_range",
        ),
        pytest.param(
            _leaky(threshold=torch.ones(2, 2)),
            "threshold",
            "shape=(2, 2)",
            "scalar or 1D",
            id="threshold_rank",
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


def test_linear_leaky_compiles_and_exports_with_vector_neuron_attrs(tmp_path):
    beta = torch.tensor([0.0, 0.5, 0.75, 1.0])
    threshold = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model = nn.Sequential(
        nn.Linear(8, 4, bias=False),
        _leaky(reset_mechanism="subtract", threshold=threshold, beta=beta),
    )

    with pytest.warns(AutoOptimizationWarning, match="subtract reset"):
        graph = compile_to_paiir(
            model,
            make_vec_8d(),
            input_formats={"InputNode_0": (DataSign.SIGNED, DataWidth.WIDTH_8BIT)},
            strict=True,
        )
    cores = find_nodes(graph, OfflineCoreOp)
    sequentials = [
        node
        for node in cores
        if hasattr(node, "act") and isinstance(node.act, CoreNeuronV25)
    ]
    assert len(sequentials) == 1

    mapper = Mapper()
    mapper.compile(graph, tmp_path, target_platform="x86", debug=False)
    placements = sorted(
        (
            placement
            for core in mapper.coreplacements
            for placement in core.neus
            if len(placement.raw_neus) == 1
            and placement.raw_neus[0].target.name == sequentials[0].name
        ),
        key=lambda placement: placement.raw_neus[0].index.idx,
    )
    attrs = [placement.neu_attrs_part2 for placement in placements]
    assert len(attrs) == 4
    assert [RM(item.reset_mode) for item in attrs] == [RM.MODE_LINEAR] * 4
    assert [item.threshold_pos for item in attrs] == [2, 3, 4, 5]
    assert [item.leak_multi_mode for item in attrs] == [
        LeakMultiMode.ENABLE,
        LeakMultiMode.ENABLE,
        LeakMultiMode.ENABLE,
        LeakMultiMode.DISABLE,
    ]
    assert [item.leak_tau for item in attrs] == [0, -1, -2, 0]

    assert np.load(tmp_path / "cfg_frames.npy").size > 0
