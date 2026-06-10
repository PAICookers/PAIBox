import operator

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_fx

from paibox.quantize_tools import (
    ManualConvAddReLU2d,
    ManualConv2d,
    ManualConvReLU2d,
    ManualLinear,
    ManualLinearReLU,
    build_bked_backend_config,
    build_bked_qconfig_mapping,
    build_manual_prepare_custom_config,
    convert_prepared_fx_to_manual,
)
from paibox.quantize_tools.observed_ops import (
    ObservedManualConv2d,
    ObservedManualConvAddReLU2d,
    ObservedManualConvReLU2d,
    ObservedManualLinear,
    ObservedManualLinearReLU,
)


def _prepared(model: nn.Module, sample: torch.Tensor):
    backend_config = build_bked_backend_config()
    fused = fuse_fx(model.eval(), backend_config=backend_config)
    prepared = prepare_fx(
        fused,
        build_bked_qconfig_mapping(symmetric=True),
        (sample,),
        prepare_custom_config=build_manual_prepare_custom_config(),
        backend_config=backend_config,
    )
    prepared(sample)
    return prepared, backend_config


def _assert_clean_graph(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        target = getattr(node.target, "__name__", str(node.target))
        assert "quantize_per_tensor" not in target
        assert "dequantize" not in target
        if node.op == "call_module":
            assert not hasattr(gm.get_submodule(
                str(node.target)), "calculate_qparams")
        if node.op == "get_attr":
            assert "scale" not in str(node.target)
            assert "zero_point" not in str(node.target)


def _single_module(gm: torch.fx.GraphModule, module_type: type[nn.Module]):
    matches = [m for m in gm.modules() if isinstance(m, module_type)]
    assert len(matches) == 1
    return matches[0]


def test_conv2d_custom_convert_to_manual():
    model = nn.Sequential(nn.Conv2d(3, 4, 1))
    prepared, backend_config = _prepared(model, torch.randn(1, 3, 4, 4))
    _single_module(prepared, ObservedManualConv2d)

    converted = convert_prepared_fx_to_manual(
        prepared, backend_config=backend_config)
    _single_module(converted, ManualConv2d)
    _assert_clean_graph(converted)


def test_convrelu_custom_convert_to_manual():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
    prepared, backend_config = _prepared(model, torch.randn(1, 3, 4, 4))
    _single_module(prepared, ObservedManualConvReLU2d)

    converted = convert_prepared_fx_to_manual(
        prepared, backend_config=backend_config)
    _single_module(converted, ManualConvReLU2d)
    _assert_clean_graph(converted)


def test_linear_custom_convert_to_manual():
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 5))
    prepared, backend_config = _prepared(model, torch.randn(1, 3, 4, 4))
    _single_module(prepared, ObservedManualLinear)

    converted = convert_prepared_fx_to_manual(
        prepared, backend_config=backend_config)
    _single_module(converted, ManualLinear)
    _assert_clean_graph(converted)


def test_linearrelu_custom_convert_to_manual():
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 5), nn.ReLU())
    prepared, backend_config = _prepared(model, torch.randn(1, 3, 4, 4))
    _single_module(prepared, ObservedManualLinearReLU)

    converted = convert_prepared_fx_to_manual(
        prepared, backend_config=backend_config)
    _single_module(converted, ManualLinearReLU)
    _assert_clean_graph(converted)


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, y, x):
        return F.relu(operator.add(self.conv(y), x))


def test_convaddrelu_custom_convert_to_manual():
    backend_config = build_bked_backend_config()
    sample_y = torch.randn(1, 3, 4, 4)
    sample_x = torch.randn(1, 3, 4, 4)
    fused = fuse_fx(ResidualModel().eval(), backend_config=backend_config)
    assert any(isinstance(m, nni.ConvAddReLU2d) for m in fused.modules())
    prepared = prepare_fx(
        fused,
        build_bked_qconfig_mapping(symmetric=True),
        (sample_y, sample_x),
        prepare_custom_config=build_manual_prepare_custom_config(),
        backend_config=backend_config,
    )
    prepared(sample_y, sample_x)
    _single_module(prepared, ObservedManualConvAddReLU2d)

    converted = convert_prepared_fx_to_manual(
        prepared, backend_config=backend_config)
    _single_module(converted, ManualConvAddReLU2d)
    _assert_clean_graph(converted)
