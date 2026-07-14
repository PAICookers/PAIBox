import pytest
import torch

from paibox.backendv2.op_node import CoreOpNode
from paibox.paiir import IFNodeV25
from paibox.paiir.ir.calc_params import NeuronParams
from paibox.paiir.ir.op_node import StandaloneActOp


def _core_node(params: NeuronParams) -> CoreOpNode:
    raw_node = StandaloneActOp(IFNodeV25())
    raw_node.neu_params = params
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    return CoreOpNode("act", raw_node, (1, 2, 1, 2))


def test_attrs_part2_requires_materialized_params():
    raw_node = StandaloneActOp(IFNodeV25())
    raw_node.core_params.tick_start = 1
    raw_node.core_params.tick_duration = 0
    raw_node.core_params.tick_initial = 1
    node = CoreOpNode("act", raw_node, (1, 1))

    with pytest.raises(RuntimeError, match="not materialized"):
        node.attrs_part2()


def test_attrs_part2_resolves_all_flat_neuron_params():
    node = _core_node(
        NeuronParams(
            reset_v=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            thres_neg=torch.tensor([-1.0, -2.0, -3.0, -4.0]),
            thres_pos=torch.tensor([5.0, 6.0, 7.0, 8.0]),
            leak_tau=torch.tensor([0, -1, -2, -3]),
            leak_v=torch.tensor([9.0, 10.0, 11.0, 12.0]),
            init_v=torch.tensor([13.0, 14.0, 15.0, 16.0]),
        )
    )

    attrs = node.attrs_part2(2)

    assert attrs.reset_v == 3
    assert attrs.threshold_neg == -3
    assert attrs.threshold_pos == 7
    assert attrs.leak_tau == -2
    assert attrs.leak_v == 11
    assert attrs.vjt_initial == 15


def test_attrs_part2_retains_scalar_rounding():
    node = _core_node(
        NeuronParams(
            reset_v=1.6,
            thres_neg=-2.6,
            thres_pos=3.6,
            leak_tau=-2,
            leak_v=4.6,
            init_v=5.6,
        )
    )

    attrs = node.attrs_part2(3)

    assert attrs.reset_v == 2
    assert attrs.threshold_neg == -3
    assert attrs.threshold_pos == 4
    assert attrs.leak_tau == -2
    assert attrs.leak_v == 5
    assert attrs.vjt_initial == 6
