"""Materialize broadcastable neuron parameters for backend consumption."""

from dataclasses import replace
from math import prod

import torch
from paicorelib import LeakAddMode, LeakMultiMode
from torch import Tensor

from ..ir.calc_params import NeuronParams
from ..ir.graph import PAIIRGraph
from ..ir.op_node import OfflineCoreOp

__all__ = ["materialize_neuron_params"]


def materialize_neuron_params(graph: PAIIRGraph) -> PAIIRGraph:
    """Store scalar or flat neuron parameters on every offline-core node.

    Materialization is atomic: parameters are committed only after every node
    has been checked successfully.
    """
    staged: list[tuple[OfflineCoreOp, NeuronParams]] = []
    for name, node in graph.nodes.items():
        if not isinstance(node, OfflineCoreOp):
            continue
        params, bias = node.src_params()
        logical_shape = _logical_output_shape(name, node)
        staged.append((node, _materialize_params(name, params, bias, logical_shape)))

    for node, params in staged:
        node.neu_params = params
    return graph


def _logical_output_shape(name: str, node: OfflineCoreOp) -> tuple[int, ...]:
    if len(node.output_layouts) != 1 or not node.output_layouts[0].shape:
        raise ValueError(
            f"OfflineCoreOp '{name}' requires one known output shape before "
            "neuron-parameter materialization"
        )

    output_shape = tuple(node.output_layouts[0].shape)
    if len(output_shape) == 1:
        return output_shape
    if output_shape[0] != 1:
        raise ValueError(
            f"OfflineCoreOp '{name}' only supports batch size 1, got "
            f"output_shape={output_shape}"
        )
    return output_shape[1:]


def _materialize_params(
    node_name: str,
    params: NeuronParams,
    bias: Tensor | None,
    logical_shape: tuple[int, ...],
) -> NeuronParams:
    values = {
        field: _materialize_param(
            getattr(params, field), logical_shape, node_name=node_name, field=field
        )
        for field in params.__vectorized_attrs__
    }
    _validate_leak_multi_mode(node_name, values["leak_multi_mode"])
    _validate_leak_tau(node_name, values["leak_tau"])
    _validate_threshold_bounds(node_name, values["thres_pos"], values["thres_neg"])

    if bias is not None:
        if params.leak_add_mode == LeakAddMode.BACKWARD:
            raise ValueError(
                f"OfflineCoreOp '{node_name}' cannot fuse bias when "
                "leak_add_mode is BACKWARD"
            )
        bias_flat = _materialize_output_channel_bias(bias, logical_shape, node_name)
        values["leak_v"] = values["leak_v"] + bias_flat

    return replace(params, **values)


def _validate_leak_multi_mode(
    node_name: str,
    leak_multi_mode: LeakMultiMode | bool | int | Tensor,
) -> None:
    if torch.is_tensor(leak_multi_mode):
        if leak_multi_mode.dtype not in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                f"OfflineCoreOp '{node_name}' leak_multi_mode must have integer "
                "or bool dtype"
            )
        invalid = torch.any((leak_multi_mode != 0) & (leak_multi_mode != 1))
        if invalid.item():
            raise ValueError(
                f"OfflineCoreOp '{node_name}' leak_multi_mode values must be 0 or 1"
            )
        return

    if isinstance(leak_multi_mode, bool):
        return
    if not isinstance(leak_multi_mode, (LeakMultiMode, int)):
        raise TypeError(
            f"OfflineCoreOp '{node_name}' leak_multi_mode must be a LeakMultiMode, "
            "bool, or integer/bool Tensor"
        )
    if int(leak_multi_mode) not in (0, 1):
        raise ValueError(
            f"OfflineCoreOp '{node_name}' leak_multi_mode values must be 0 or 1"
        )


def _validate_leak_tau(node_name: str, leak_tau: int | float | Tensor) -> None:
    if torch.is_tensor(leak_tau):
        if leak_tau.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                f"OfflineCoreOp '{node_name}' leak_tau must have integer dtype"
            )
        return
    if isinstance(leak_tau, bool) or not isinstance(leak_tau, int):
        raise TypeError(f"OfflineCoreOp '{node_name}' leak_tau must be an integer")


def _validate_threshold_bounds(
    node_name: str,
    thres_pos: int | float | Tensor,
    thres_neg: int | float | Tensor,
) -> None:
    if torch.is_tensor(thres_pos):
        other = (
            thres_neg.to(thres_pos.device) if torch.is_tensor(thres_neg) else thres_neg
        )
        invalid = thres_pos < other
    else:
        invalid = thres_pos < thres_neg

    if torch.is_tensor(invalid):
        invalid = bool(torch.any(invalid).item())
    if invalid:
        raise ValueError(
            f"OfflineCoreOp '{node_name}' requires thres_pos >= thres_neg "
            "elementwise after materialization"
        )


def _materialize_param(
    value: int | float | Tensor,
    logical_shape: tuple[int, ...],
    *,
    node_name: str,
    field: str,
) -> int | float | Tensor:
    if not torch.is_tensor(value):
        return value
    if value.numel() == 0:
        raise ValueError(f"OfflineCoreOp '{node_name}' {field} must not be empty")
    if value.ndim == 0:
        return value.item()

    value = value.detach()
    if value.ndim == len(logical_shape) + 1:
        if value.shape[0] != 1:
            raise ValueError(
                f"OfflineCoreOp '{node_name}' {field} has batch-varying shape "
                f"{tuple(value.shape)}"
            )
        value = value[0]

    logical_numel = prod(logical_shape)
    if value.ndim == 1 and value.numel() == logical_shape[0]:
        channel_shape = (logical_shape[0], *(1 for _ in logical_shape[1:]))
        expanded = value.reshape(channel_shape).expand(logical_shape)
    elif value.ndim == 1 and value.numel() == logical_numel:
        return value.reshape(-1).clone()
    else:
        try:
            expanded = torch.broadcast_to(value, logical_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"OfflineCoreOp '{node_name}' {field} shape {tuple(value.shape)} "
                f"cannot broadcast to logical output shape {logical_shape}"
            ) from exc

    return expanded.reshape(-1).clone()


def _materialize_output_channel_bias(
    bias: Tensor, logical_shape: tuple[int, ...], node_name: str
) -> Tensor:
    bias = bias.detach()
    if bias.ndim != 1 or bias.numel() != logical_shape[0]:
        raise ValueError(
            f"OfflineCoreOp '{node_name}' output-channel bias must have shape "
            f"({logical_shape[0]},), got {tuple(bias.shape)}"
        )
    channel_shape = (logical_shape[0], *(1 for _ in logical_shape[1:]))
    return bias.reshape(channel_shape).expand(logical_shape).reshape(-1).clone()
