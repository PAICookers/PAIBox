from collections.abc import Callable, Iterable
from typing import Any

from torch import fx


def matches_func_module_pattern(
    pattern: tuple[Callable, type], node: fx.Node, modules: dict[str, Any]
) -> bool:
    """Match a pattern of (function, module) in fx graph."""
    if len(node.args) == 0:
        return False

    if not isinstance(node.args[0], fx.Node):
        return False
    if node.args[0].op != "call_function":
        return False
    if not callable(node.args[0].target):
        return False
    if node.args[0].target is not pattern[0]:
        return False

    if not isinstance(node, fx.Node):
        return False
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    if type(modules[node.target]) is not pattern[1]:
        return False
    return True


def matches_function_pattern(pattern: Iterable[Callable], node: fx.Node) -> bool:
    if len(node.args) == 0:
        return False

    nodes: tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != "call_function":
            return False
        if current_node.target != expected_type:
            return False

    return True
