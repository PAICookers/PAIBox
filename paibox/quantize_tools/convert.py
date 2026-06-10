from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch.ao.quantization.quantize_fx import convert_to_reference_fx

from .custom_convert_config import (
    build_manual_convert_custom_config,
    MANUAL_MODULE_TYPES,
)


def convert_fx_to_manual(
    prepared_model: torch.fx.GraphModule,
    *,
    qconfig_mapping=None,
    backend_config=None,
    inplace: bool = False,
) -> torch.fx.GraphModule:
    model = prepared_model if inplace else copy.deepcopy(prepared_model)
    converted = convert_to_reference_fx(
        model,
        convert_custom_config=build_manual_convert_custom_config(),
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )
    return strip_manual_qdq(converted)


def strip_manual_qdq(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    modules = dict(model.named_modules())
    graph = model.graph

    _strip_all_quantize_dequantize_nodes(graph)

    for node in list(graph.nodes):
        if node.op != "call_module" or not isinstance(node.target, str):
            continue
        module = modules.get(node.target)
        if not isinstance(module, MANUAL_MODULE_TYPES):
            continue

        node.args = tuple(_strip_input_arg(arg, graph) for arg in node.args)

    graph.eliminate_dead_code()
    model.recompile()
    model.delete_all_unused_submodules()
    _assert_no_quant_artifacts(model)
    return model


def _strip_all_quantize_dequantize_nodes(graph: torch.fx.Graph) -> None:
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if not (_is_quantize_node(node) or _is_dequantize_node(node)):
                continue
            if not node.args or not isinstance(node.args[0], torch.fx.Node):
                continue
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            changed = True


def _strip_input_arg(arg, graph: torch.fx.Graph):
    if not isinstance(arg, torch.fx.Node):
        return arg

    current = arg
    if _is_dequantize_node(current) and current.args:
        source = current.args[0]
        if isinstance(source, torch.fx.Node):
            current = source

    if _is_quantize_node(current) and current.args:
        source = current.args[0]
        if isinstance(source, torch.fx.Node):
            return source

    return current


def _is_quantize_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    name = getattr(target, "__name__", str(target))
    return "quantize_per_tensor" in name


def _is_dequantize_node(node: torch.fx.Node) -> bool:
    if node.op == "call_method" and node.target == "dequantize":
        return True
    if node.op == "call_function":
        name = getattr(node.target, "__name__", str(node.target))
        return "dequantize" in name
    return False


def _assert_no_quant_artifacts(model: torch.fx.GraphModule) -> None:
    bad_nodes: list[str] = []
    for node in model.graph.nodes:
        if _is_quantize_node(node) or _is_dequantize_node(node):
            bad_nodes.append(node.name)
            continue
        if node.op == "call_module":
            mod = model.get_submodule(str(node.target))
            if hasattr(mod, "calculate_qparams"):
                bad_nodes.append(node.name)
        if node.op == "get_attr" and _looks_like_qparam_attr(str(node.target)):
            bad_nodes.append(node.name)

    if bad_nodes:
        joined = ", ".join(bad_nodes)
        raise RuntimeError(
            f"manual FX graph still contains quantization artifacts: {joined}")


def _looks_like_qparam_attr(target: str) -> bool:
    parts: Iterable[str] = target.replace(".", "_").split("_")
    return any(part in {"scale", "zero", "zero_point"} for part in parts)
