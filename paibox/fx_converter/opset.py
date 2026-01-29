import itertools
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron
from torch import fx, nn

SUPPORTED_ACT_OPS = [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softsign]
SUPPORTED_CONV_OPS = [nn.Conv1d, nn.Conv2d, nn.Linear]
SUPPORTED_POOL_OPS = [nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d]
SUPPORTED_NEU_OPS = [neuron.LIFNode, neuron.IFNode]
ADD_OR_SUB_OPS = [operator.add, operator.sub, torch.add, torch.sub]
IGNORED_MODULES = [nn.Dropout, nn.Identity]


def make_comp_act_patterns():
    """Make comp-act patterns"""
    return itertools.product(
        SUPPORTED_CONV_OPS + SUPPORTED_POOL_OPS, SUPPORTED_ACT_OPS + SUPPORTED_NEU_OPS
    )


def make_conv_relu_patterns():
    """Make conv-relu patterns"""
    return itertools.product(SUPPORTED_CONV_OPS, SUPPORTED_ACT_OPS)


def make_conv_neuron_patterns():
    """Make conv-neuron patterns"""
    return itertools.product(SUPPORTED_CONV_OPS, SUPPORTED_NEU_OPS)


def make_pool_relu_patterns():
    """Make pool-relu patterns"""
    return itertools.product(SUPPORTED_POOL_OPS, SUPPORTED_ACT_OPS)


def make_pool_neuron_patterns():
    """Make pool-neuron patterns"""
    return itertools.product(SUPPORTED_POOL_OPS, SUPPORTED_NEU_OPS)


def make_add_act_patterns():
    """Make add-act patterns"""
    return itertools.product(ADD_OR_SUB_OPS, SUPPORTED_ACT_OPS + SUPPORTED_NEU_OPS)


def is_module_supported_neuron(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_NEU_OPS))


def is_module_supported_act(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_ACT_OPS))


def is_module_supported_neu_act(m: nn.Module) -> bool:
    return is_module_supported_neuron(m) or is_module_supported_act(m)


def is_module_supported_comp(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_CONV_OPS + SUPPORTED_POOL_OPS))


def is_module_supported_pool(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_POOL_OPS))


def is_module_supported_maxpool(m: nn.Module) -> bool:
    maxpool = (nn.MaxPool1d, nn.MaxPool2d)
    assert all(op in SUPPORTED_POOL_OPS for op in maxpool)
    return isinstance(m, maxpool)


def is_module_supported(m: nn.Module) -> bool:
    return (
        is_module_supported_neuron(m)
        or is_module_supported_act(m)
        or is_module_supported_comp(m)
    )


def _is_node_supported_module(
    node: fx.Node,
    modules: dict[str, Any],
    fn: Callable[[nn.Module], bool] | None = None,
) -> bool:
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False

    m = modules[node.target]
    return True if fn is None else fn(m)


def is_node_supported_neuron(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    return _is_node_supported_module(node, modules, is_module_supported_neuron)


def is_node_supported_act(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    return _is_node_supported_module(node, modules, is_module_supported_act)


def is_node_supported_neu_act(
    node: fx.Node, modules: dict[str, fx.GraphModule]
) -> bool:
    return _is_node_supported_module(node, modules, is_module_supported_neu_act)


def is_node_supported_comp(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    return _is_node_supported_module(node, modules, is_module_supported_comp)


def is_node_supported_pool(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    return _is_node_supported_module(node, modules, is_module_supported_pool)


_EQUIVALENT_TYPES = [
    {nn.Conv1d, F.conv1d},
    {nn.Conv2d, F.conv2d},
    {nn.Linear, F.linear},
    {nn.MaxPool1d, F.max_pool1d},
    {nn.MaxPool2d, F.max_pool2d},
    {nn.AvgPool1d, F.avg_pool1d},
    {nn.AvgPool2d, F.avg_pool2d},
    {nn.ReLU, F.relu, F.relu_},
    {nn.Sigmoid, F.sigmoid},
    {nn.Tanh, F.tanh},
    {nn.Softsign, F.softsign},
    {torch.add, operator.add, operator.iadd, "add", "add_"},
    {torch.sub, operator.sub, operator.isub, "sub", "sub_"},
    {torch.transpose, "transpose", "transpose_"},
    {torch.permute, "permute", "permute_"},
]


def _create_equivalent_types_dict(equiv_sets: list[set]):
    dct = {}
    for values in equiv_sets:
        for v in values:
            dct[v] = list(values)
    return dct


_EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict(_EQUIVALENT_TYPES)


def get_equivalent_types() -> list[set]:
    return _EQUIVALENT_TYPES


def _get_matching_types(partition_type: Any) -> list[Any]:
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])

    return matching_types


def _valid_type_sequence(partition_types: list[Any]) -> bool:
    partition_types_set = set()
    for partition_type in partition_types:
        matching_types = _get_matching_types(partition_type)
        matching_types_set = set(matching_types)
        if len(partition_types_set & matching_types_set) > 0:
            return False
        partition_types_set |= matching_types_set
    return True
