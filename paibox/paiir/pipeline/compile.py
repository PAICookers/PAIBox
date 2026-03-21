"""Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

This module provides a high-level entry point that wraps the multi-step
PAIIR conversion pipeline:

1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
2. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and apply AvgPool deployment params
3. :func:`validate_graph` -- early structural cleanup / validation on the fused graph
4. :func:`propagate_data_format` -- infer and fill data format parameters
5. :func:`assign_tick_params` -- assign timing parameters
6. :func:`calibrate_avgpool_thresholds` -- optional AvgPool threshold refinement
7. :func:`validate_compiled_graph` -- final post-pass validation before returning the graph

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

from ..ir.graph import PAIIRGraph
from ..lowering.converter import torch_to_paiir
from .data_format import DataFormat
from .passes import (
    TickOverride,
    assign_tick_params,
    calibrate_avgpool_thresholds,
    fuse_to_offline_cores,
    propagate_data_format,
    validate_compiled_graph,
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
        enable_avgpool_calibration: Experimental. If True, refine shared-core
            AvgPool+LIF thresholds via offline integer search after topology
            selection. Off by default.
        enable_split_avgpool_lif: Experimental. If True, allow the compiler to choose
            split-core AvgPool+LIF when Core 1 can transmit the exact sum-domain
            code through the current LUT VALUE path. Off by default.
    """

    tick_duration: int = 0
    auto_reset: bool = True
    input_formats: dict[str, DataFormat] | None = None
    enable_avgpool_calibration: bool = False
    enable_split_avgpool_lif: bool = False


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
    enable_avgpool_calibration: bool | None = None,
    enable_split_avgpool_lif: bool | None = None,
) -> PAIIRGraph:
    """Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

    This is a high-level wrapper that runs the full PAIIR compilation pipeline:

    1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
    2. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and
       apply AvgPool deployment params
    3. :func:`validate_graph` -- structural cleanup / validation on the fused
       graph, before later passes add data-format and timing annotations
    4. :func:`propagate_data_format` -- infer and fill data format parameters
    5. :func:`assign_tick_params` -- assign timing parameters
    6. :func:`calibrate_avgpool_thresholds` -- (experimental) refine shared-core
       AvgPool+LIF thresholds via offline integer search
    7. :func:`validate_compiled_graph` -- final post-pass graph validation
       after connectivity cleanup, data-format propagation, and tick assignment

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
        enable_avgpool_calibration: Experimental. If True, refine shared-core
            AvgPool+LIF thresholds via offline integer search after topology
            selection. Off by default. Explicit kwarg overrides
            ``CompileConfig.enable_avgpool_calibration``.
        enable_split_avgpool_lif: Experimental. If True, allow conditional
            split-core AvgPool+LIF deployment when exact sum-domain LUT coding
            is possible. Explicit kwarg overrides
            ``CompileConfig.enable_split_avgpool_lif``.

    Returns:
        A compiled :class:`PAIIRGraph` with fused nodes and all compile-time
        annotations filled.

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

        # With calibration (experimental)
        graph = compile_to_paiir(
            model,
            torch.randn(1, 3, 32, 32),
            enable_avgpool_calibration=True,
        )

        # Using CompileConfig with per-call overrides
        cfg = CompileConfig(tick_duration=100, auto_reset=True)
        graph = compile_to_paiir(model, x, compile_config=cfg, strict=False)
    """
    # Resolve: explicit kwarg > config > built-in default
    cfg = compile_config or CompileConfig()
    _tick_duration = tick_duration if tick_duration is not None else cfg.tick_duration
    _auto_reset = auto_reset if auto_reset is not None else cfg.auto_reset
    _input_formats = input_formats if input_formats is not None else cfg.input_formats
    _enable_avgpool_calibration = (
        enable_avgpool_calibration
        if enable_avgpool_calibration is not None
        else cfg.enable_avgpool_calibration
    )
    _enable_split_avgpool_lif = (
        enable_split_avgpool_lif
        if enable_split_avgpool_lif is not None
        else cfg.enable_split_avgpool_lif
    )

    # Step 1: FX trace and 1:1 node mapping
    graph = torch_to_paiir(
        model, *sample_inputs, concrete_args=concrete_args, strict=strict
    )

    # Step 2: Choose topology, fuse atomic nodes, and prepare AvgPool deployment
    graph = fuse_to_offline_cores(
        graph, _enable_split_avgpool_lif, _enable_avgpool_calibration
    )

    # Step 3: Early validation on the fused graph. This stage is allowed to
    # clean up disconnected regions and enforces only the invariants needed
    # before later passes run.
    validate_graph(graph)

    # Step 4: Infer and fill data format parameters
    propagate_data_format(graph, _input_formats)

    # Step 5: Assign timing parameters
    assign_tick_params(graph, _tick_duration, _auto_reset, tick_overrides)

    # Step 6: Calibrate AvgPool+LIF thresholds (experimental, off by default)
    if _enable_avgpool_calibration:
        calibrate_avgpool_thresholds(graph)

    # Step 7: Final validation after all compile-time annotations are filled.
    # Unlike validate_graph(), this stage assumes the graph is in its final
    # compiled form and checks shape/dims completeness, propagated data formats,
    # tick parameters, and end-to-end connectivity.
    validate_compiled_graph(graph)

    graph.eval()
    return graph
