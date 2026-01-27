from typing import Any

import torch
from spikingjelly.activation_based import functional as sF
from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import ShapeProp

from paibox.fx_converter.fuse import fuse_conv_bn


class NeuronAsOpTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, neuron.BaseNode):  # Treat neuron as leaf module
            return True

        return super().is_leaf_module(m, module_qualified_name)


class DropoutRemover(torch.fx.Transformer):
    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        if isinstance(self.submodules[target], nn.Dropout):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def trace_spikingjelly_model(m: nn.Module) -> fx.GraphModule:
    m.eval()
    m.requires_grad_(False)
    sF.reset_net(m)

    tracer = NeuronAsOpTracer()
    traced_graph = tracer.trace(m)
    traced = fx.GraphModule(m, traced_graph)
    flatten_module_hierarchy(traced)
    traced.graph.lint()
    return traced


def flatten_module_hierarchy(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Flattens the module hierarchy by lifting all submodules referenced in the graph
    to the top-level GraphModule. This eliminates chain usage like `self.seq.0(x)`
    or `getattr(self.seq, '0')(x)`.
    """
    nodes_to_modify = {}

    for node in gm.graph.nodes:
        if node.op == 'call_module':
            target = node.target
            if isinstance(target, str) and '.' in target:
                nodes_to_modify[node] = target

    for node, old_target in nodes_to_modify.items():
        submod = gm.get_submodule(old_target)

        # New name construction
        new_target_name = old_target.replace('.', '_')

        # Check collision
        if hasattr(gm, new_target_name):
            existing = getattr(gm, new_target_name)
            if existing is not submod:
                count = 1
                while hasattr(gm, f"{new_target_name}_{count}"):
                    count += 1
                new_target_name = f"{new_target_name}_{count}"

        # Add to top level
        gm.add_module(new_target_name, submod)
        # Update node
        node.target = new_target_name

    gm.recompile()
    return gm


def propagate_tensor_shape(gm: fx.GraphModule, *input: torch.Tensor) -> None:
    # TODO dtype?
    shape_prop = ShapeProp(gm)
    shape_prop.propagate(*input)

    for node in gm.graph.nodes:
        print(node.name, node.meta["tensor_meta"].dtype,
              node.meta["tensor_meta"].shape)


def remove_dropout_and_fuse_conv_bn(m: nn.Module) -> fx.GraphModule:
    gm = trace_spikingjelly_model(m)
    m = DropoutRemover(gm).transform()
    m.graph.print_tabular()
    m = fuse_conv_bn(m, inplace=True, no_trace=True)
    return m  # type: ignore


def trace_spikingjelly_model_with_shape(
    m: nn.Module, input: torch.Tensor
) -> fx.GraphModule:
    m.eval()
    sF.reset_net(m)

    tracer = NeuronAsOpTracer()
    traced_graph = tracer.trace(m)
    traced = fx.GraphModule(m, traced_graph)
    traced.graph.lint()

    propagate_tensor_shape(traced, input)

    return traced
