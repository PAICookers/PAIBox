import torch
import torch.fx

from typing import Dict, Tuple

from .utils import get_obs_params


def collect_observer_qparams(
    model: torch.fx.GraphModule,
    activation_symmetric: bool = False,
) -> Dict[str, Tuple[float, int]]:
    """Collect scale/zp from all observer nodes in the graph."""
    print("\n=== Calibration Statistics ===")
    obs_params: Dict[str, Tuple[float, int]] = {}

    for node in model.graph.nodes:
        if node.op == "call_module":
            if not isinstance(node.target, str):
                continue
            mod = getattr(model, node.target, None)
            if mod and hasattr(mod, "calculate_qparams"):
                s, zp = get_obs_params(mod)

                if activation_symmetric and zp != 0:
                    # PyTorch QNNPACK requires quint8 (0~255) for activations.
                    # We intercept its observer stats and forcefully remap into symmetric int8 (-128~127).
                    float_max = (255 - zp) * s
                    float_min = (0 - zp) * s
                    abs_max = max(abs(float_max), abs(float_min))
                    s = abs_max / 127.0 if abs_max > 0 else 1.0
                    zp = 0

                obs_params[node.name] = (s, zp)
                print(f"Observer [{node.name}]: scale={s:.6f}, zp={zp}")

    return obs_params


def find_next_observer_params(
    node,
    obs_params: Dict[str, Tuple[float, int]],
) -> Tuple[float, int]:
    """Find the output scale/zp by looking ahead in the graph."""
    # Simple heuristic: The first user that is an Observer determines the output params.

    # We might need to skip some pure-functional non-observer nodes like flatten/max_pool2d
    # To do this robustly, we do a BFS
    queue = list(node.users.keys())
    while queue:
        curr = queue.pop(0)
        if curr.name in obs_params:
            return obs_params[curr.name]
        queue.extend(list(curr.users.keys()))

    return (1.0, 0)


def find_prev_observer_params(
    node,
    obs_params: Dict[str, Tuple[float, int]],
) -> Tuple[float, int]:
    """Find the input scale/zp by backtracking the graph."""
    if hasattr(node, "name") and node.name in obs_params:
        return obs_params[node.name]

    if len(node.args) > 0:
        input_node = node.args[0]
        curr = input_node
        # Trace back through non-observer nodes if necessary
        while isinstance(curr, torch.fx.Node) and curr.op != "placeholder":
            if curr.name in obs_params:
                return obs_params[curr.name]
            if len(curr.args) > 0 and isinstance(curr.args[0], torch.fx.Node):
                curr = curr.args[0]
            else:
                break

    return (1.0, 0)


def get_weight_qparams(mod) -> Tuple[float, int]:
    """Extract weight quantization parameters from qconfig."""
    s_w, z_w = 1.0, 0
    if hasattr(mod, "qconfig") and mod.qconfig is not None:
        weight_obs = mod.qconfig.weight()
        weight_obs(mod.weight)
        s_w, z_w = get_obs_params(weight_obs)
    return s_w, z_w