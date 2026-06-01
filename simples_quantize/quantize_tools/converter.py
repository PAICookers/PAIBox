
import copy

import torch
import torch.nn as nn
import torch.fx
import torch.ao.nn.intrinsic as nni

from typing import Dict, Tuple, cast

from .graph_analysis import (
    collect_observer_qparams,
    find_next_observer_params,
    find_prev_observer_params,
    get_weight_qparams,
)
from .ops import ManualQuantConvReLU2d, ManualQuantLinear, ManualQuantConv2d, ManualQuantLinearReLU
from .ops import ManualConvAddReLU2d
from .summary import collect_quantized_layer_records, export_quantized_model_summary


class FxGraphConverter:
    """
    Automatic FX Graph converter that replaces standard Conv/Linear with ManualQuant versions.
    Encapsulates traversal, parameter extraction, and node replacement logic.
    """

    def __init__(self, prepared_model: torch.fx.GraphModule, activation_symmetric: bool = False):
        self.original_model = prepared_model
        # Use deepcopy to avoid modifying the original prepared model during conversion
        self.model = copy.deepcopy(prepared_model)
        self.modules = cast(Dict[str, nn.Module],
                            dict(self.model.named_modules()))

        self.activation_symmetric = activation_symmetric

        # State
        self.obs_params: Dict[str, Tuple[float, int]] = {}

        # Strategy Registry: module_type -> handler_function
        self.handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        self.register_handler(nni.ConvReLU2d, self._handle_convrelu2d)
        self.register_handler(nni.ConvAddReLU2d, self._handle_convaddrelu2d)
        self.register_handler(nni.LinearReLU, self._handle_linearrelu)
        self.register_handler(nn.Linear, self._handle_linear)
        self.register_handler(nn.Conv2d, self._handle_conv2d)

    def register_handler(self, module_type, handler_func):
        """Register a handler for a specific module type."""
        self.handlers[module_type] = handler_func

    def _collect_calibration_stats(self):
        """Collect scale/zp from all observer nodes in the graph."""
        self.obs_params = collect_observer_qparams(
            self.model, self.activation_symmetric)

    def _find_next_observer_params(self, node) -> Tuple[float, int]:
        """Find the output scale/zp by looking ahead in the graph."""
        return find_next_observer_params(node, self.obs_params)

    def _find_prev_observer_params(self, node) -> Tuple[float, int]:
        """Find the input scale/zp by backtracking the graph."""
        return find_prev_observer_params(node, self.obs_params)

    def _get_weight_qparams(self, mod) -> Tuple[float, int]:
        """Extract weight quantization parameters from qconfig."""
        return get_weight_qparams(mod)

    def _rewrite_with_manual_module(self, node, mod, manual_cls, label, target_mod=None, show_module_type=False):
        target_module = target_mod if target_mod is not None else mod

        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(target_module)

        print(f"[{node.name}] ({label})")
        if show_module_type:
            print(f"  Type: {type(target_module).__name__}")
        print(f"  Input  : scale={s_in:.6f}, zp={z_in}")
        print(f"  Weight : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output : scale={s_out:.6f}, zp={z_out}")

        new_mod = manual_cls(
            target_module, s_in, z_in, s_w, z_w, s_out, z_out,
            activation_symmetric=self.activation_symmetric,
        )

        new_target = f"mq_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_convrelu2d(self, node, mod):
        """Handler for fused ConvReLU2d nodes."""
        self._rewrite_with_manual_module(
            node,
            mod,
            ManualQuantConvReLU2d,
            "ConvReLU2d",
            target_mod=getattr(mod, '0', mod),
        )

    def _handle_linear(self, node, mod):
        """Handler for purely Linear nodes."""
        self._rewrite_with_manual_module(
            node,
            mod,
            ManualQuantLinear,
            "Linear",
        )

    def _handle_linearrelu(self, node, mod):
        """Handler for LinearReLU fused nodes."""
        self._rewrite_with_manual_module(
            node,
            mod,
            ManualQuantLinearReLU,
            "LinearReLU",
            target_mod=getattr(mod, '0', mod),
        )

    def _handle_convaddrelu2d(self, node, mod):
        """Handler for backend-fused ConvAddReLU2d residual nodes."""
        conv = getattr(mod, "0", mod)
        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(conv)

        shortcut_node = node.args[1] if len(node.args) > 1 else None
        x_scale, x_zp = (
            find_prev_observer_params(shortcut_node, self.obs_params)
            if isinstance(shortcut_node, torch.fx.Node)
            else (1.0, 0)
        )
        conv_out_scale = s_in * s_w

        print(f"[{node.name}] (ConvAddReLU2d)")
        print(f"  Shortcut Input : scale={x_scale:.6f}, zp={x_zp}")
        print(f"  Conv Input     : scale={s_in:.6f}, zp={z_in}")
        print(f"  Conv Weight    : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output         : scale={s_out:.6f}, zp={z_out}")

        new_mod = ManualConvAddReLU2d(
            original_conv2=conv,
            y_in_scale=s_in,
            y_in_zp=z_in,
            w_scale=s_w,
            w_zp=z_w,
            conv2_out_scale=conv_out_scale,
            out_scale=s_out,
            out_zp=z_out,
            x_scale=x_scale,
            x_zp=x_zp,
            activation_symmetric=self.activation_symmetric,
        )

        new_target = f"mq_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_conv2d(self, node, mod):
        """Handler for standalone Conv2d nodes."""
        self._rewrite_with_manual_module(
            node,
            mod,
            ManualQuantConv2d,
            "Conv2d",
            show_module_type=True,
        )

    def _cleanup_graph(self):
        """Remove observers."""
        # Remove Observers (replace with identity/passthrough)
        nodes_to_pass = []
        for node in self.model.graph.nodes:
            if node.op == 'call_module' and node.name in self.obs_params:
                if len(node.args) > 0:
                    replacement = node.args[0]
                    if isinstance(replacement, torch.fx.Node):
                        node.replace_all_uses_with(replacement)
                        nodes_to_pass.append(node)

        for n in nodes_to_pass:
            self.model.graph.erase_node(n)

        self.model.graph.lint()
        self.model.recompile()

    def convert(self):
        """Execute the conversion process."""
        self._collect_calibration_stats()

        # Iterate over a copy of nodes to allow modification
        for node in list(self.model.graph.nodes):
            if node.op == 'call_module':
                if not isinstance(node.target, str):
                    continue
                mod = self.modules.get(node.target)
                if mod is None:
                    continue

                # Dynamic Dispatch
                # Check for exact type match first
                handler = self.handlers.get(type(mod))

                if handler:
                    handler(node, mod)

        self._cleanup_graph()
        return self.model


def convert_fx_to_manual(prepared_model, activation_symmetric=False):
    """
    Legacy wrapper for FxGraphConverter.
    """
    converter = FxGraphConverter(
        prepared_model, activation_symmetric=activation_symmetric)
    return converter.convert()
