
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
import torch.ao.nn.intrinsic as nni
from typing import Tuple

from .utils import get_obs_params
from .ops import ManualQuantConvReLU2d, ManualQuantLinear, ManualQuantConv2d, ManualQuantLinearReLU, ManualIntAddResidual


class FloatSafeAvgPool2d(nn.Module):
    """
    Safe AvgPool2d that casts uint8/int8 to float before pooling to bypass PyTorch restrictions.
    """

    def __init__(self, original_module):
        super().__init__()
        self.kernel_size = original_module.kernel_size
        self.stride = original_module.stride
        self.padding = original_module.padding
        self.ceil_mode = original_module.ceil_mode
        self.count_include_pad = original_module.count_include_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype in (torch.uint8, torch.int8):
            out = F.avg_pool2d(x.float(), self.kernel_size, self.stride,
                               self.padding, self.ceil_mode, self.count_include_pad)
            return torch.round(out).to(dtype)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class FxGraphConverter:
    """
    Automatic FX Graph converter that replaces standard Conv/Linear with ManualQuant versions.
    Encapsulates traversal, parameter extraction, and node replacement logic.
    """

    def __init__(self, prepared_model: torch.fx.GraphModule, use_lut: bool = False, activation_symmetric: bool = False):
        self.original_model = prepared_model
        # Use deepcopy to avoid modifying the original prepared model during conversion
        self.model = copy.deepcopy(prepared_model)
        self.modules = dict(self.model.named_modules())
        self.use_lut = use_lut
        self.activation_symmetric = activation_symmetric

        # State
        self.obs_params = {}

        # Strategy Registry: module_type -> handler_function
        self.handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        self.register_handler(nni.ConvReLU2d, self._handle_convrelu2d)
        self.register_handler(nni.LinearReLU, self._handle_linearrelu)
        self.register_handler(nn.Linear, self._handle_linear)
        self.register_handler(nn.Conv2d, self._handle_conv2d)
        # self.register_handler(nn.AvgPool2d, self._handle_avgpool2d)

    def register_handler(self, module_type, handler_func):
        """Register a handler for a specific module type."""
        self.handlers[module_type] = handler_func

    def _collect_calibration_stats(self):
        """Collect scale/zp from all observer nodes in the graph."""
        print("\n=== Calibration Statistics ===")
        self.obs_params = {}
        for node in self.model.graph.nodes:
            if node.op == 'call_module':
                mod = getattr(self.model, node.target, None)
                if mod and hasattr(mod, 'calculate_qparams'):
                    s, zp = get_obs_params(mod)

                    if self.activation_symmetric and zp != 0:
                        # PyTorch QNNPACK requires quint8 (0~255) for activations.
                        # We intercept its observer stats and forcefully remap into symmetric int8 (-128~127).
                        float_max = (255 - zp) * s
                        float_min = (0 - zp) * s
                        abs_max = max(abs(float_max), abs(float_min))
                        s = abs_max / 127.0 if abs_max > 0 else 1.0
                        zp = 0

                    self.obs_params[node.name] = (s, zp)
                    print(f"Observer [{node.name}]: scale={s:.6f}, zp={zp}")

    def _find_next_observer_params(self, node) -> Tuple[float, int]:
        """Find the output scale/zp by looking ahead in the graph."""
        # Simple heuristic: The first user that is an Observer determines the output params.

        # We might need to skip some pure-functional non-observer nodes like flatten/max_pool2d
        # To do this robustly, we do a BFS
        queue = list(node.users.keys())
        while queue:
            curr = queue.pop(0)
            if curr.name in self.obs_params:
                return self.obs_params[curr.name]
            queue.extend(list(curr.users.keys()))

        return (1.0, 0)

    def _find_prev_observer_params(self, node) -> Tuple[float, int]:
        """Find the input scale/zp by backtracking the graph."""
        if hasattr(node, 'name') and node.name in self.obs_params:
            return self.obs_params[node.name]

        if len(node.args) > 0:
            input_node = node.args[0]
            curr = input_node
            # Trace back through non-observer nodes if necessary
            while isinstance(curr, torch.fx.Node) and curr.op != 'placeholder':
                if curr.name in self.obs_params:
                    return self.obs_params[curr.name]
                if len(curr.args) > 0 and isinstance(curr.args[0], torch.fx.Node):
                    curr = curr.args[0]
                else:
                    break
        return (1.0, 0)

    def _get_weight_qparams(self, mod) -> Tuple[float, int]:
        """Extract weight quantization parameters from qconfig."""
        s_w, z_w = 1.0, 0
        if hasattr(mod, 'qconfig') and mod.qconfig is not None:
            weight_obs = mod.qconfig.weight()
            weight_obs(mod.weight)
            s_w, z_w = get_obs_params(weight_obs)
        return s_w, z_w

    def _handle_convrelu2d(self, node, mod):
        """Handler for fused ConvReLU2d nodes."""
        target_mod = getattr(mod, '0', mod)

        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(target_mod)

        print(f"[{node.name}] (ConvReLU2d)")
        print(f"  Input  : scale={s_in:.6f}, zp={z_in}")
        print(f"  Weight : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output : scale={s_out:.6f}, zp={z_out}")

        new_mod = ManualQuantConvReLU2d(
            target_mod, s_in, z_in, s_w, z_w, s_out, z_out,
            activation_symmetric=self.activation_symmetric
        )

        new_target = f"manual_quant_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_linear(self, node, mod):
        """Handler for purely Linear nodes."""
        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(mod)

        print(f"[{node.name}] (Linear)")
        print(f"  Input  : scale={s_in:.6f}, zp={z_in}")
        print(f"  Weight : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output : scale={s_out:.6f}, zp={z_out}")

        new_mod = ManualQuantLinear(
            mod, s_in, z_in, s_w, z_w, s_out, z_out,
            activation_symmetric=self.activation_symmetric
        )

        new_target = f"manual_quant_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_linearrelu(self, node, mod):
        """Handler for LinearReLU fused nodes."""
        target_mod = getattr(mod, '0', mod)

        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(target_mod)

        print(f"[{node.name}] (LinearReLU)")
        print(f"  Input  : scale={s_in:.6f}, zp={z_in}")
        print(f"  Weight : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output : scale={s_out:.6f}, zp={z_out}")

        new_mod = ManualQuantLinearReLU(
            target_mod, s_in, z_in, s_w, z_w, s_out, z_out,
            activation_symmetric=self.activation_symmetric
        )

        new_target = f"manual_quant_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_avgpool2d(self, node, mod):
        """Handler for AvgPool2d to make it Byte safe."""
        print(f"[{node.name}] (AvgPool2d) safely wrapped.")
        new_mod = FloatSafeAvgPool2d(mod)
        new_target = f"float_safe_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _handle_conv2d(self, node, mod):
        """Handler for standalone Conv2d nodes."""
        s_in, z_in = self._find_prev_observer_params(node)
        s_out, z_out = self._find_next_observer_params(node)
        s_w, z_w = self._get_weight_qparams(mod)

        print(f"[{node.name}] (Conv2d) Type: {type(mod).__name__}")
        print(f"  Input  : scale={s_in:.6f}, zp={z_in}")
        print(f"  Weight : scale={s_w:.6f}, zp={z_w}")
        print(f"  Output : scale={s_out:.6f}, zp={z_out}")

        new_mod = ManualQuantConv2d(
            mod, s_in, z_in, s_w, z_w, s_out, z_out,
            activation_symmetric=self.activation_symmetric
        )
        new_target = f"manual_quant_{node.name}"
        self.model.add_module(new_target, new_mod)
        node.target = new_target

    def _replace_add_relu(self):
        import operator
        import builtins
        import torch.nn.functional as F

        count = 0
        nodes = list(self.model.graph.nodes)

        for node in nodes:
            if node.op == 'call_function' and node.target in (F.relu, torch.relu):
                arg_node = node.args[0]

                # Verify trailing add
                add_node = arg_node
                while getattr(add_node, 'op', '') == 'call_module' and 'activation_post_process' in add_node.name:
                    add_node = add_node.args[0]

                if getattr(add_node, 'op', '') == 'call_function' and add_node.target in (operator.add, ):

                    conv_node = None
                    x_node_obs = None
                    y_node_obs = None

                    for arg in add_node.args:
                        if not isinstance(arg, torch.fx.Node):
                            continue
                        curr = arg
                        while curr.op == 'call_module' and 'activation_post_process' in curr.name:
                            curr = curr.args[0]

                        if curr.op == 'call_module':
                            mod = self.modules.get(curr.target)
                            # Pytorch prepared_model uses nn.Conv2d natively unquantized
                            if isinstance(mod, nn.Conv2d):
                                conv_node = curr
                                y_node_obs = arg
                                x_node_obs = add_node.args[1] if add_node.args[0] == arg else add_node.args[0]
                                break

                    if conv_node:
                        conv_mod = self.modules.get(conv_node.target)
                        s_in, z_in = self._find_prev_observer_params(conv_node)
                        s_out, z_out = self._find_next_observer_params(node)
                        s_w, z_w = self._get_weight_qparams(conv_mod)
                        x_scale, x_zp = self._find_prev_observer_params(
                            x_node_obs)
                        conv2_out_scale, _ = self.obs_params.get(
                            y_node_obs.name, (1.0, 0))

                        from .utils import approximate_scale_ratio
                        target_scale = s_in * s_w
                        exact_ratio = x_scale / target_scale if target_scale != 0 else 0
                        M, n = approximate_scale_ratio(exact_ratio)

                        custom_name = f"manual_add_relu_{count}"

                        print(f"[{custom_name}] (AddReLU Fusion)")
                        print(
                            f"  Shortcut Input : scale={x_scale:.6f}, zp={x_zp}")
                        print(
                            f"  Conv2 Input    : scale={s_in:.6f}, zp={z_in}")
                        print(f"  Conv2 Weight   : scale={s_w:.6f}, zp={z_w}")
                        print(
                            f"  Conv2 Output   : scale={conv2_out_scale:.6f}")
                        print(
                            f"  Final Output   : scale={s_out:.6f}, zp={z_out}")
                        print(
                            f"  Approximation  : M={M}, n={n}, exact_ratio={exact_ratio:.6f}, approx_ratio={M*(2**n):.6f}, error={exact_ratio - M*(2**n):.6f}")

                        manual_block = ManualIntAddResidual(
                            original_conv2=conv_mod,
                            y_in_scale=s_in, y_in_zp=z_in,
                            w_scale=s_w, w_zp=z_w,
                            conv2_out_scale=conv2_out_scale,
                            out_scale=s_out, out_zp=z_out,
                            x_scale=x_scale, x_zp=x_zp,
                            activation_symmetric=self.activation_symmetric
                        )
                        self.model.add_submodule(custom_name, manual_block)

                        with self.model.graph.inserting_before(node):
                            conv_in = conv_node.args[0]
                            new_node = self.model.graph.call_module(
                                custom_name, args=(x_node_obs, conv_in))

                        node.replace_all_uses_with(new_node)
                        # Do NOT remove from self.modules because the SAME module might be called by another node (reused modules in FX)
                        # self.modules[conv_node.target] = None
                        self.model.graph.erase_node(node)

                        # Cleanup replaced nodes to prevent ShapeProp from executing them
                        if len(add_node.users) == 0:
                            self.model.graph.erase_node(add_node)
                        if y_node_obs is not None and len(y_node_obs.users) == 0:
                            self.model.graph.erase_node(y_node_obs)
                        if len(conv_node.users) == 0:
                            self.model.graph.erase_node(conv_node)

                        count += 1

        print(f"Replaced {count} instances of 'conv2 -> add -> relu'.")

    def _cleanup_graph(self):
        """Remove observers."""
        # Remove Observers (replace with identity/passthrough)
        nodes_to_pass = []
        for node in self.model.graph.nodes:
            if node.op == 'call_module' and node.name in self.obs_params:
                if len(node.args) > 0:
                    node.replace_all_uses_with(node.args[0])
                    nodes_to_pass.append(node)

        for n in nodes_to_pass:
            self.model.graph.erase_node(n)

        self.model.graph.lint()
        self.model.recompile()

    def convert(self):
        """Execute the conversion process."""
        self._collect_calibration_stats()

        self._replace_add_relu()

        # Iterate over a copy of nodes to allow modification
        for node in list(self.model.graph.nodes):
            if node.op == 'call_module':
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


def convert_fx_to_manual(prepared_model, use_lut=False, activation_symmetric=False):
    """
    Legacy wrapper for FxGraphConverter.
    """
    converter = FxGraphConverter(
        prepared_model, use_lut=use_lut, activation_symmetric=activation_symmetric)
    return converter.convert()
