"""Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

This module provides a high-level entry point that wraps the multi-step
PAIIR conversion pipeline:

1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
2. :func:`fuse_to_offline_cores` -- fuse atomic nodes into offline-core units
3. :func:`propagate_data_format` -- infer and fill data format parameters
4. :func:`assign_tick_params` -- assign timing parameters

Use :func:`compile_to_paiir` for a one-step compilation, or call the
individual passes directly for fine-grained control.

Example::

    from paibox.paiir import compile_to_paiir

    # Simple one-step compilation
    graph = compile_to_paiir(model, torch.randn(1, 3, 32, 32))

    # With custom timing parameters
    graph = compile_to_paiir(
        model,
        torch.randn(1, 3, 32, 32),
        tick_duration=100,
        auto_reset=True,
    )

    # With per-node timing overrides
    graph = compile_to_paiir(
        model,
        torch.randn(1, 3, 32, 32),
        tick_duration=100,
        tick_overrides={
            "seq_op_0": {"tick_start": 5},
            "seq_op_1": {"tick_duration": 200},
        },
    )
"""

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .converter import torch_to_paiir
from .data_format import DataFormat
from .graph import PAIIRGraph
from .passes import (
    TickOverride,
    assign_tick_params,
    fuse_to_offline_cores,
    propagate_data_format,
    validate_graph,
)

__all__ = ["compile_to_paiir", "CompileConfig"]


@dataclass
class CompileConfig:
    """Configuration for :func:`compile_to_paiir`.

    Args:
        tick_duration: Global default for how many time steps each core works.
            0 = always working (default).
        auto_reset: Global default for automatic neuron state reset.
        input_formats: Per-InputNode data format override for
            :func:`propagate_data_format`.
    """

    tick_duration: int = 0
    auto_reset: bool = True
    input_formats: dict[str, DataFormat] | None = None


def compile_to_paiir(
    model: nn.Module,
    *sample_inputs: Tensor,
    tick_duration: int | None = None,
    auto_reset: bool | None = None,
    tick_overrides: dict[str, TickOverride] | None = None,
    input_formats: dict[str, DataFormat] | None = None,
    compile_config: CompileConfig | None = None,
    concrete_args: dict[str, Any] | None = None,
    strict: bool = True,
) -> PAIIRGraph:
    """Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

    This is a high-level wrapper that runs the full PAIIR compilation pipeline:

    1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
    2. :func:`fuse_to_offline_cores` -- fuse atomic nodes into offline-core units
    3. :func:`propagate_data_format` -- infer and fill data format parameters
    4. :func:`assign_tick_params` -- assign timing parameters

    For fine-grained control, call the individual passes directly instead.

    Precedence: explicit keyword arguments > ``config`` fields > built-in
    defaults.

    Args:
        model: The PyTorch model to compile.
        *sample_inputs: Example input tensor(s) for shape and dims inference.
            Omit entirely to skip shape propagation.
        tick_duration: Global default for how many time steps each core works.
            0 means always working. Default: 0.
        auto_reset: Automatic neuron state reset when tick_duration > 0.
            Default: True.
        tick_overrides: Per-node timing overrides, keyed by node name.
            Each value is a dict with optional keys ``tick_start``,
            ``tick_duration``, and ``auto_reset``.
        input_formats: Per-InputNode data format override, keyed by node name.
            Format: ``(DataSign, DataWidth)`` tuples.
        compile_config: Optional :class:`CompileConfig` providing defaults.
            Explicit keyword arguments always take precedence.
        concrete_args: Concrete arguments forwarded to ``fx.Tracer.trace``.
        strict: Raise on unsupported ops (True) or warn and bypass (False).
            Default: True.

    Returns:
        A compiled :class:`PAIIRGraph` with fused nodes and all parameters filled.

    Raises:
        UnsupportedOpError: If ``strict=True`` and an unsupported operator is encountered.
        GraphValidationError: If the graph fails structural validation.
        ValueError: If timing parameters are invalid.

    Example::

        # Simple compilation
        graph = compile_to_paiir(model, torch.randn(1, 3, 32, 32))

        # Custom timing
        graph = compile_to_paiir(
            model,
            torch.randn(1, 3, 32, 32),
            tick_duration=100,
            auto_reset=True,
        )

        # Using CompileConfig with per-call overrides
        cfg = CompileConfig(tick_duration=100, auto_reset=True)
        graph = compile_to_paiir(model, x, config=cfg, strict=False)
    """
    # Resolve: explicit kwarg > config > built-in default
    cfg = compile_config or CompileConfig()
    _tick_duration = tick_duration if tick_duration is not None else cfg.tick_duration
    _auto_reset = auto_reset if auto_reset is not None else cfg.auto_reset
    _input_formats = input_formats if input_formats is not None else cfg.input_formats

    # Step 1: FX trace and 1:1 node mapping
    graph = torch_to_paiir(
        model, *sample_inputs, concrete_args=concrete_args, strict=strict
    )

    # Step 2: Fuse atomic nodes into offline-core units
    graph = fuse_to_offline_cores(graph)

    # Step 3: Validate graph structure
    validate_graph(graph)

    # Step 4: Infer and fill data format parameters
    propagate_data_format(graph, _input_formats)

    # Step 5: Assign timing parameters
    assign_tick_params(graph, _tick_duration, _auto_reset, tick_overrides)

    return graph
