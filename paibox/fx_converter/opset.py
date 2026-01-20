import operator

import torch
from spikingjelly.activation_based import neuron
from torch import fx, nn

SUPPORTED_ACT_OPS = [nn.ReLU, nn.PReLU]
SUPPORTED_COMP_OPS = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Linear,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.AvgPool1d,
    nn.AvgPool2d,
]

SUPPORTED_NEU_OPS = [neuron.LIFNode, neuron.IFNode]
IMPLICIT_SUM_OPS = [operator.add, operator.sub, torch.add, torch.sub]


def is_module_supported_neuron(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_NEU_OPS))


def is_module_supported_act(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_ACT_OPS))


def is_module_supported_comp(m: nn.Module) -> bool:
    return isinstance(m, tuple(SUPPORTED_COMP_OPS))


def is_module_supported(m: nn.Module) -> bool:
    return (
        is_module_supported_neuron(m)
        or is_module_supported_act(m)
        or is_module_supported_comp(m)
    )


def is_node_supported_neuron(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    m = modules[node.target]
    return is_module_supported_neuron(m)


def is_node_supported_act(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    m = modules[node.target]
    return is_module_supported_act(m)


def is_node_supported_comp(node: fx.Node, modules: dict[str, fx.GraphModule]) -> bool:
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    m = modules[node.target]
    return is_module_supported_comp(m)
