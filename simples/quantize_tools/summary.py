import io
import json

from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn

from .ops import ManualIntAddResidual


def _to_python_scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _to_float_scalar(value) -> float:
    scalar = _to_python_scalar(value)
    if isinstance(scalar, list):
        raise TypeError("Expected a scalar value")
    return float(cast(Any, scalar))


def _to_int_scalar(value) -> int:
    scalar = _to_python_scalar(value)
    if isinstance(scalar, list):
        raise TypeError("Expected a scalar value")
    return int(cast(Any, scalar))


def _qparam_pair(scale, zero_point):
    return {
        "scale": _to_float_scalar(scale),
        "zp": _to_int_scalar(zero_point),
    }


def _approximation_payload(module: ManualIntAddResidual):
    approximation = cast(Optional[Dict[str, Any]], getattr(module, "approximation", None))
    if approximation is None:
        return None

    return {
        "M": _to_int_scalar(approximation.get("M", 0)),
        "n": _to_int_scalar(approximation.get("n", 0)),
        "exact_ratio": _to_float_scalar(approximation.get("exact_ratio", 0.0)),
        "approx_ratio": _to_float_scalar(approximation.get("approx_ratio", 0.0)),
        "error": _to_float_scalar(approximation.get("error", 0.0)),
    }


def _collect_node_quant_record(name: str, module: nn.Module) -> Optional[Dict[str, Any]]:
    if all(hasattr(module, attr) for attr in ("s_in", "z_in", "s_w", "z_w", "s_out", "z_out")):
        return {
            "name": name,
            "type": type(module).__name__,
            "input": _qparam_pair(module.s_in, module.z_in),
            "weight": _qparam_pair(module.s_w, module.z_w),
            "output": _qparam_pair(module.s_out, module.z_out),
        }

    if isinstance(module, ManualIntAddResidual):
        record = {
            "name": name,
            "type": type(module).__name__,
            "shortcut_input": _qparam_pair(module.x_scale, module.x_zp),
            "conv_input": _qparam_pair(module.conv.s_in, module.conv.z_in),
            "weight": _qparam_pair(module.conv.s_w, module.conv.z_w),
            "conv_output": _qparam_pair(module.conv2_out_scale, 0),
            "output": _qparam_pair(module.out_scale, module.out_zp),
        }

        approximation = _approximation_payload(module)
        if approximation is not None:
            record["approximation"] = approximation

        return record

    return None


def collect_quantized_layer_records(model: torch.fx.GraphModule) -> List[Dict[str, Any]]:
    """Collect quantization parameters for each call_module node in a converted model."""
    records: List[Dict[str, Any]] = []
    modules = dict(model.named_modules())

    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        if not isinstance(node.target, str):
            continue
        module = modules.get(node.target)
        if module is None:
            continue

        record = _collect_node_quant_record(node.target, module)
        if record is not None:
            records.append(record)

    return records


def export_quantized_model_summary(
    model: torch.fx.GraphModule,
    json_path: str,
    txt_path: str,
) -> Dict[str, Any]:
    """Export layer qparams to JSON and the quantized graph to TXT."""
    payload: Dict[str, Any] = {
        "model": type(model).__name__,
        "layer_count": 0,
        "layers": collect_quantized_layer_records(model),
    }
    payload["layer_count"] = len(payload["layers"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    graph_buffer = io.StringIO()
    with redirect_stdout(graph_buffer):
        model.graph.print_tabular()

    graph_text = graph_buffer.getvalue().rstrip()
    model_code = getattr(model, "code", "")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Quantized Graph (Tabular) ===\n")
        f.write(graph_text)
        f.write("\n\n=== Generated Code ===\n")
        f.write(str(model_code).rstrip())
        f.write("\n")

    return payload