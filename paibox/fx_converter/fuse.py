import copy
from collections.abc import Callable
from typing import Any

from torch import fx, nn
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from paibox import _logging

from .core_op import AccumCoreOp, CalcCoreOp, SeqCoreOp, SingleConvMaxOp, SingleNeuLUTOp
from .ir_base import PAIIR, OpLoc
from .opset import (
    IMPLICIT_SUM_OPS,
    SUPPORTED_ACT_OPS,
    SUPPORTED_COMP_OPS,
    SUPPORTED_CONV_OPS,
    SUPPORTED_NEU_OPS,
)

fuse_log = _logging.get_artifact_logger(__name__, "fuse")


# NOTE: in case of the arguments of fuse() is changing, define it here.
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

    # Ensure all call_module targets are present in modules
    for node in fx_model.graph.nodes:
        if node.op == "call_module" and node.target not in modules:
            try:
                modules[node.target] = fx_model.get_submodule(node.target)
            except AttributeError:
                pass

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
    patterns = [(c, a) for c in SUPPORTED_COMP_OPS for a in SUPPORTED_ACT_OPS] + [
        (c, n) for c in SUPPORTED_COMP_OPS for n in SUPPORTED_NEU_OPS
    ]
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


def fuse_implicit_add(gm: fx.GraphModule) -> fx.GraphModule:
    patterns = [(op, n) for op in IMPLICIT_SUM_OPS for n in SUPPORTED_NEU_OPS] + [
        (op, a) for op in IMPLICIT_SUM_OPS for a in SUPPORTED_ACT_OPS
    ]
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_func_module_pattern(pattern, node, modules):
                add_node = node.all_input_nodes[0]
                if len(add_node.users) > 1:
                    continue

                add_prev_nodes = add_node.all_input_nodes
                if len(add_prev_nodes) > 2 or len(add_prev_nodes) == 1:
                    # TODO more complicated case: add_1(o1, o2), add_2(add_1, o3)
                    continue

                # TODO check the previous nodes are supported comp ops & of same type: conv & conv, maxpool & maxpool, etc.
                # But max pool & avg pool fusion may not make sense.
                # Ensure previous nodes are supported linear ops (Conv, Linear)
                valid_ops = True
                for op in add_prev_nodes:
                    if not isinstance(modules[op.target], tuple(SUPPORTED_CONV_OPS)):
                        valid_ops = False
                        break
                if not valid_ops:
                    continue

                assert isinstance(node.target, str)
                fused = AccumCoreOp.build(
                    add_prev_nodes, node, modules, OpLoc.OFFLINE_CORE
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


def fuse_standalone_conv_max(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if node.op == "call_module" and node.target in modules:
            module = modules[node.target]
            if isinstance(module, tuple(SUPPORTED_COMP_OPS)):
                # Check if it is already fused by previous steps (it shouldn'be because strict matching)
                # But if some other pass wrapped it, we skip.
                # Since we iterate original modules and graph, and we are mutating new_graph,
                # we should be careful.
                # wait, create_file overwrote everything so I am defining this function now.

                fused = SingleConvMaxOp.build(node, modules, OpLoc.OFFLINE_CORE)
                modules[fused.name] = fused

                with new_graph.inserting_after(node):
                    new_node = new_graph.call_module(fused.name, args=node.args)

                node.replace_all_uses_with(new_node)
                # Do not erase node immediately if it is referenced?
                # replace_all_uses_with replaces usages.
                # Note: node is from new_graph iteration.

                new_graph.erase_node(node)
                fuse_log.debug(f"Wrapped standalone conv {node.name} with {fused.name}")

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def fuse_standalone_neu(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if node.op == "call_module" and node.target in modules:
            module = modules[node.target]
            # Check for supported neurons OR activations (ANN neurons)
            is_neu = isinstance(module, tuple(SUPPORTED_NEU_OPS))
            is_act = isinstance(module, tuple(SUPPORTED_ACT_OPS))

            if is_neu or is_act:
                fused = SingleNeuLUTOp.build(node, modules, OpLoc.OFFLINE_CORE)
                modules[fused.name] = fused

                with new_graph.inserting_after(node):
                    new_node = new_graph.call_module(fused.name, args=node.args)

                node.replace_all_uses_with(new_node)
                new_graph.erase_node(node)
                fuse_log.debug(f"Wrapped standalone neu {node.name} with {fused.name}")

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


def fuse_calc_op(gm: fx.GraphModule) -> fx.GraphModule:
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)

    for node in new_graph.nodes:
        if node.op == "call_function" and node.target in IMPLICIT_SUM_OPS:
            fused = CalcCoreOp.build(node, modules, OpLoc.OFFLINE_CORE)
            modules[fused.name] = fused

            with new_graph.inserting_after(node):
                new_node = new_graph.call_module(fused.name, args=node.args)

            node.replace_all_uses_with(new_node)
            new_graph.erase_node(node)
            fuse_log.debug(f"Fused calc op {node.name} into {fused.name}")

    new_graph.lint()
    return fx.GraphModule(modules, new_graph)


_FUSE_PASSES: list[Callable[[fx.GraphModule], fx.GraphModule]] = []


def apply_fuse_passes(gm: fx.GraphModule) -> fx.GraphModule:
    _FUSE_PASSES.clear()
    _FUSE_PASSES.append(fuse_compute_act)
    _FUSE_PASSES.append(fuse_implicit_add)
    _FUSE_PASSES.append(fuse_standalone_conv_max)
    _FUSE_PASSES.append(fuse_standalone_neu)
    _FUSE_PASSES.append(fuse_calc_op)

    gm = fuse_conv_bn(gm, inplace=True, no_trace=True)  # type: ignore

    for i, fuse in enumerate(_FUSE_PASSES):
        fuse_log.debug(f"Applying fuse pass #{i}: {fuse.__name__}")
        gm = fuse(gm)
        gm.graph.print_tabular()
        print("\n")

    return gm
