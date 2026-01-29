import copy
import operator
from collections.abc import Callable
from typing import Any

import torch
from torch import fx, nn
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from paibox import _logging

from ..exceptions import NotSupportedError
from .core_op import AccumCoreOp, CalcCoreOp, SeqCoreOp, SingleCompOp, SingleNeuLUTOp
from .ir_base import PAIIR, OpLoc
from .match_utils import matches_func_module_pattern, matches_function_pattern
from .opset import (
    ADD_OR_SUB_OPS,
    SUPPORTED_CONV_OPS,
    is_node_supported_comp,
    is_node_supported_neu_act,
    make_add_act_patterns,
    make_comp_act_patterns,
)

fuse_log = _logging.get_artifact_logger(__name__, "fuse")


# NOTE: in case of the arguments of fuse() is changing, define it here.
# torch.fx.experimental.optimization.fuse()
def fuse_conv_bn(
    model: nn.Module, inplace: bool = False, no_trace: bool = False
) -> nn.Module:
    """
    Fuses convolution/BN and linear/BN layers for inference purposes.
    Will deepcopy your model by default, but can modify the model inplace as well.
    """
    patterns = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
        (nn.Linear, nn.BatchNorm1d),
    ]
    if not inplace:
        model = copy.deepcopy(model)
    if not no_trace or not isinstance(model, fx.GraphModule):
        fx_model = fx.symbolic_trace(model)
    else:
        fx_model = model
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    # Output of conv/linear is used by other nodes
                    continue
                first_layer = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                if pattern[0] in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
                    fused_layer = fuse_conv_bn_eval(first_layer, bn)
                else:  # nn.Linear
                    fused_layer = fuse_linear_bn_eval(first_layer, bn)
                replace_node_module(node.args[0], modules, fused_layer)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def _parent_name(target: str) -> tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module_with_ir(
    node: fx.Node, modules: dict[str, Any], ir: PAIIR
) -> None:
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)

    new_name = ir.name
    new_target = f"{parent_name}.{new_name}" if parent_name else new_name

    modules[new_target] = ir
    setattr(modules[parent_name], new_name, ir)
    node.target = new_target
    node.name = new_name


def fuse_compute_act(gm: fx.GraphModule) -> fx.GraphModule:
    patterns = make_comp_act_patterns()
    modules = dict(gm.named_modules())

    # Ensure all call_module targets are present in modules (handle flattened modules)
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target not in modules:
            try:
                modules[node.target] = gm.get_submodule(node.target)
            except AttributeError:
                pass
    new_graph = copy.deepcopy(gm.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                node_prev = node.all_input_nodes[0]
                if len(node_prev.users) > 1:
                    continue

                fused = SeqCoreOp.build(node_prev, node, modules, OpLoc.OFFLINE_CORE)

                modules[fused.name] = fused
                with new_graph.inserting_after(node):
                    new_node = new_graph.call_module(fused.name, args=node_prev.args)

                node.replace_all_uses_with(new_node)
                new_graph.erase_node(node)
                new_graph.erase_node(node_prev)

                fuse_log.debug(
                    f"Replaced {node_prev} + {node.name} with new node: {fused.name}"
                )

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def nodes_have_same_type(nodes: list[fx.Node], modules: dict[str, Any]) -> bool:
    if not nodes:
        return True

    assert isinstance(nodes[0].target, str)
    if nodes[0].target not in modules:
        return False

    first_module_type = type(modules[nodes[0].target])
    for node in nodes[1:]:
        if node.target not in modules:
            return False

        assert isinstance(node.target, str)
        if type(modules[node.target]) is not first_module_type:
            return False
    return True


def fuse_implicit_add(gm: fx.GraphModule) -> fx.GraphModule:
    patterns = make_add_act_patterns()
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_func_module_pattern(pattern, node, modules):
                add_node = node.all_input_nodes[0]
                if len(add_node.users) > 1:
                    raise NotSupportedError("'add' node must be used only once")

                add_prev_nodes = add_node.all_input_nodes
                assert len(add_prev_nodes) == 2

                # Ensure previous nodes are supported ops
                valid_ops = True
                for op in add_prev_nodes:
                    if not isinstance(modules[op.target], tuple(SUPPORTED_CONV_OPS)):
                        valid_ops = False
                        break
                if not valid_ops:
                    continue

                # check the previous nodes are of same type
                # TODO support ops with different types: conv + maxpool?
                if not nodes_have_same_type(add_prev_nodes, modules):
                    raise NotSupportedError("operands of 'add' must be same type")

                assert isinstance(node.target, str)
                if node.target in (operator.add, torch.add):
                    fused = AccumCoreOp.build(
                        add_prev_nodes, node, modules, (1, 1), OpLoc.OFFLINE_CORE
                    )
                else:  # sub
                    fused = AccumCoreOp.build(
                        add_prev_nodes, node, modules, (1, -1), OpLoc.OFFLINE_CORE
                    )

                modules[fused.name] = fused

                # Merge arguments
                conv_args = []
                for op in add_prev_nodes:
                    conv_args.extend(op.args)

                with new_graph.inserting_after(node):
                    new_node = new_graph.call_module(fused.name, args=tuple(conv_args))

                node.replace_all_uses_with(new_node)
                new_graph.erase_node(node)
                new_graph.erase_node(add_node)
                for op in add_prev_nodes:
                    new_graph.erase_node(op)

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def fuse_standalone_comp(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if is_node_supported_comp(node, modules):
            if len(node.all_input_nodes) > 1:
                raise NotSupportedError(
                    "standalone computing node must have only one input"
                )

            fused = SingleCompOp.build(node, modules, OpLoc.OFFLINE_CORE)
            modules[fused.name] = fused

            with new_graph.inserting_after(node):
                new_node = new_graph.call_module(fused.name, args=node.args)

            node.replace_all_uses_with(new_node)
            new_graph.erase_node(node)
            fuse_log.debug(
                f"Wrapped standalone computing node {node.name} with {fused.name}"
            )

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def fuse_standalone_neu(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if is_node_supported_neu_act(node, modules):
            if len(node.all_input_nodes) > 1:
                raise NotSupportedError(
                    "standalone neuron or activation node must have only one input"
                )

            fused = SingleNeuLUTOp.build(node, modules, OpLoc.OFFLINE_CORE)
            modules[fused.name] = fused

            with new_graph.inserting_after(node):
                new_node = new_graph.call_module(fused.name, args=node.args)

            node.replace_all_uses_with(new_node)
            new_graph.erase_node(node)
            fuse_log.debug(
                f"Wrapped standalone neuron/activation node {node.name} with {fused.name}"
            )

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def fuse_calc_op(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if node.op == "call_function" and node.target in ADD_OR_SUB_OPS:
            fused = CalcCoreOp.build(node, modules, OpLoc.OFFLINE_CORE)
            modules[fused.name] = fused

            with new_graph.inserting_after(node):
                new_node = new_graph.call_module(fused.name, args=node.args)

            node.replace_all_uses_with(new_node)
            new_graph.erase_node(node)
            fuse_log.debug(f"Fused calc op {node.name} into {fused.name}")

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def remove_shape_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove those nodes for calculating shape"""
    new_graph = copy.deepcopy(gm.graph)
    nodes_to_remove = set()

    for node in new_graph.nodes:
        # check op getitem
        if matches_function_pattern((getattr, operator.getitem), node):
            op_getattr = node.args[0]
            assert isinstance(op_getattr, fx.Node)
            if len(op_getattr.args) == 2 and op_getattr.args[1] == "shape":
                # like x.shape
                nodes_to_remove.add(node)
                nodes_to_remove.add(op_getattr)

        # Check op floordiv/mul & remove
        elif matches_function_pattern((operator.getitem, operator.floordiv), node):
            nodes_to_remove.add(node)

        elif matches_function_pattern((operator.getitem, operator.mul), node):
            nodes_to_remove.add(node)

        elif node.op == "call_method" and node.target in ["flatten", "contiguous"]:
            nodes_to_remove.add(node)

        elif node.op == "call_method" and node.target in ["reshape", "view"]:
            nodes_to_remove.add(node)

    for node in nodes_to_remove:
        node.replace_all_uses_with(node.args[0])
        new_graph.erase_node(node)
        fuse_log.debug(f"Removed node '{node.name}'")

    new_graph.lint()
    return fx.GraphModule(gm, new_graph)


_FUSE_PASSES: list[Callable[[fx.GraphModule], fx.GraphModule]] = []


def apply_passes(gm: fx.GraphModule) -> fx.GraphModule:
    _FUSE_PASSES.clear()
    _FUSE_PASSES.append(remove_shape_nodes)
    _FUSE_PASSES.append(fuse_compute_act)
    _FUSE_PASSES.append(fuse_implicit_add)
    _FUSE_PASSES.append(fuse_standalone_comp)
    _FUSE_PASSES.append(fuse_standalone_neu)
    _FUSE_PASSES.append(fuse_calc_op)

    for i, fuse in enumerate(_FUSE_PASSES):
        fuse_log.debug(f"Applying pass #{i}: {fuse.__name__}")
        gm = fuse(gm)
        gm.graph.print_tabular()
        print("\n")

    return gm
