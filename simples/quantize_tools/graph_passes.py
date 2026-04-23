import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple

from .graph_analysis import (
    find_next_observer_params,
    find_prev_observer_params,
    get_weight_qparams,
)
from .ops import ManualIntAddResidual
from .utils import approximate_scale_ratio


class AddReluResidualFusionPass:
    """Fuse conv2 -> add -> relu into a manual integer residual block."""

    def __init__(
        self,
        model: torch.fx.GraphModule,
        modules: Dict[str, nn.Module],
        obs_params: Dict[str, Tuple[float, int]],
        activation_symmetric: bool = False,
    ):
        self.model = model
        self.modules = modules
        self.obs_params = obs_params
        self.activation_symmetric = activation_symmetric

    def _strip_activation_postprocess(self, node: torch.fx.Node) -> torch.fx.Node:
        current = node
        while current.op == "call_module" and "activation_post_process" in current.name:
            if not current.args:
                break
            first_arg = current.args[0]
            if not isinstance(first_arg, torch.fx.Node):
                break
            current = first_arg
        return current

    def run(self) -> int:
        import operator

        count = 0
        nodes = list(self.model.graph.nodes)

        for node in nodes:
            if node.op == "call_function" and node.target in (F.relu, torch.relu):
                if not node.args:
                    continue

                arg_node = node.args[0]
                if not isinstance(arg_node, torch.fx.Node):
                    continue

                # Verify trailing add
                add_node = self._strip_activation_postprocess(arg_node)

                if add_node.op == "call_function" and add_node.target in (operator.add,):

                    conv_node = None
                    x_node_obs = None
                    y_node_obs = None

                    for arg in add_node.args:
                        if not isinstance(arg, torch.fx.Node):
                            continue
                        curr = self._strip_activation_postprocess(arg)

                        if curr.op == "call_module":
                            if not isinstance(curr.target, str):
                                continue
                            mod = self.modules.get(curr.target)
                            # Pytorch prepared_model uses nn.Conv2d natively unquantized
                            if isinstance(mod, nn.Conv2d):
                                conv_node = curr
                                y_node_obs = arg
                                left_arg = add_node.args[0]
                                right_arg = add_node.args[1] if len(add_node.args) > 1 else None
                                x_node_obs = right_arg if left_arg == arg else left_arg
                                break

                    if conv_node and isinstance(x_node_obs, torch.fx.Node) and isinstance(y_node_obs, torch.fx.Node):
                        if not isinstance(conv_node.target, str):
                            continue
                        conv_mod = self.modules.get(conv_node.target)
                        s_in, z_in = find_prev_observer_params(conv_node, self.obs_params)
                        s_out, z_out = find_next_observer_params(node, self.obs_params)
                        s_w, z_w = get_weight_qparams(conv_mod)
                        x_scale, x_zp = find_prev_observer_params(x_node_obs, self.obs_params)
                        conv2_out_scale, _ = self.obs_params.get(y_node_obs.name, (1.0, 0))

                        target_scale = s_in * s_w
                        exact_ratio = x_scale / target_scale if target_scale != 0 else 0
                        M, n = approximate_scale_ratio(exact_ratio)
                        approx_ratio = M * (2 ** n)
                        approx_error = exact_ratio - approx_ratio

                        custom_name = f"mq_addrelu_{count}"

                        print(f"[{custom_name}] (AddReLU Fusion)")
                        print(f"  Shortcut Input : scale={x_scale:.6f}, zp={x_zp}")
                        print(f"  Conv2 Input    : scale={s_in:.6f}, zp={z_in}")
                        print(f"  Conv2 Weight   : scale={s_w:.6f}, zp={z_w}")
                        print(f"  Conv2 Output   : scale={conv2_out_scale:.6f}")
                        print(f"  Final Output   : scale={s_out:.6f}, zp={z_out}")
                        print(
                            f"  Approximation  : M={M}, n={n}, exact_ratio={exact_ratio:.6f}, approx_ratio={approx_ratio:.6f}, error={approx_error:.6f}"
                        )

                        manual_block = ManualIntAddResidual(
                            original_conv2=conv_mod,
                            y_in_scale=s_in, y_in_zp=z_in,
                            w_scale=s_w, w_zp=z_w,
                            conv2_out_scale=conv2_out_scale,
                            out_scale=s_out, out_zp=z_out,
                            x_scale=x_scale, x_zp=x_zp,
                            activation_symmetric=self.activation_symmetric,
                        )
                        manual_block.__dict__["approximation"] = {
                            "M": M,
                            "n": n,
                            "exact_ratio": exact_ratio,
                            "approx_ratio": approx_ratio,
                            "error": approx_error,
                        }
                        self.model.add_submodule(custom_name, manual_block)

                        with self.model.graph.inserting_before(node):
                            conv_in = conv_node.args[0]
                            new_node = self.model.graph.call_module(
                                custom_name, args=(x_node_obs, conv_in)
                            )

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
        return count