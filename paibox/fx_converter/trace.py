from typing import Any

import torch
from spikingjelly.activation_based import functional as sF
from spikingjelly.activation_based import neuron
from torch import fx, nn
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import ShapeProp

from paibox import _logging
from paibox.fx_converter.fuse import fuse_conv_bn

from ..exceptions import NotSupportedError
from .opset import IGNORED_MODULES

trace_log = _logging.get_artifact_logger(__name__, "trace")


class NeuronAsOpTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, neuron.BaseNode):  # Treat neuron as leaf module
            return True

        return super().is_leaf_module(m, module_qualified_name)


class IgnoredModuleRemover(torch.fx.Transformer):
    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        if isinstance(self.submodules[target], tuple(IGNORED_MODULES)):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def trace_spikingjelly_model(
    m: nn.Module, concrete_args: dict[str, Any] | None = None
) -> fx.GraphModule:
    m.eval()
    m.cpu()
    m.requires_grad_(False)
    sF.reset_net(m)

    traced_graph = NeuronAsOpTracer().trace(m, concrete_args)
    gm = fx.GraphModule(m, traced_graph)
    # Remove ignored modules from the graph
    traced = IgnoredModuleRemover(gm).transform()
    # Eliminate dead code
    return eliminate_dead_code(traced)


def eliminate_dead_code(gm: fx.GraphModule) -> fx.GraphModule:
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def flatten_module_sequential(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Flattens the module sequential by lifting all submodules referenced in the graph
    to the top-level GraphModule. This eliminates chain usage like `self.seq.0(x)`
    or `getattr(self.seq, '0')(x)`.
    """
    nodes_to_modify = {}

    for node in gm.graph.nodes:
        if node.op == "call_module":
            target = node.target
            if isinstance(target, str) and "." in target:
                nodes_to_modify[node] = target

    for node, old_target in nodes_to_modify.items():
        submod = gm.get_submodule(old_target)

        # New name construction
        new_target_name = old_target.replace(".", "_")

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
    for i in input:
        if (batch := i.shape[0]) != 1:
            raise NotSupportedError(
                f"only support batch size 1 for now, but got {batch}"
            )

    shape_prop = ShapeProp(gm)
    shape_prop.propagate(*input)

    for node in gm.graph.nodes:
        if "tensor_meta" in node.meta:
            print(
                node.name,
                node.meta["tensor_meta"].dtype,
                node.meta["tensor_meta"].shape,
            )


def remove_dropout_identity_and_fuse_conv_bn(m: nn.Module) -> fx.GraphModule:
    m = trace_spikingjelly_model(m)
    m = IgnoredModuleRemover(m).transform()
    m.graph.print_tabular()
    m = fuse_conv_bn(m, inplace=True, no_trace=True)
    flatten_module_sequential(m)
    return m  # type: ignore
