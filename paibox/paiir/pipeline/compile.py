"""Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

This module provides a high-level entry point that wraps the current PAIIR
conversion pipeline:

1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
2. pre-fusion fixed-point rewrites -- canonicalize transform chains and
   commute transparent pre-activation transforms
3. :func:`specialize_general_adds` -- narrow expression-layer add nodes into
   deployable add IR where possible
4. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and apply
   AvgPool deployment params
5. post-fusion fixed-point rewrites with analysis refresh
6. :func:`assign_tick_params` -- assign timing parameters
7. :func:`calibrate_avgpool_thresholds` -- optional AvgPool threshold refinement
8. :func:`validate_compiled_graph` -- final post-pass validation before returning
   the graph
9. :func:`validate_deployable_graph` -- reject residual expression-layer IR

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
from .avgpool import rewrite_standalone_avgpools
from .data_format import DataFormat
from .layout_chain_canonicalization import canonicalize_layout_chains
from .layout_cross_node_elision import commute_pre_activation_transforms
from .passes import (
    TickOverride,
    assign_tick_params,
    calibrate_avgpool_thresholds,
    fuse_to_offline_cores,
    propagate_data_format,
    propagate_signal_domain,
    specialize_general_adds,
    validate_compiled_graph,
    validate_deployable_graph,
    validate_graph,
)
from .rewrite_phase import RewritePass, run_fixed_point_rewrite_phase

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
    2. pre-fusion fixed-point rewrite phase -- canonicalize transform chains and
       commute transparent pre-activation transforms
    3. :func:`specialize_general_adds` -- narrow expression-layer add nodes
       into deployable add IR where possible
    4. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and
       apply AvgPool deployment params
    5. analysis phase -- validate graph, infer signal domains, infer data formats
    6. standalone AvgPool auto-rewrite that depends on those analyses
    7. re-run the analysis phase if the rewrite changed the graph
    8. :func:`assign_tick_params` -- assign timing parameters
    9. :func:`calibrate_avgpool_thresholds` -- (experimental) refine shared-core
       AvgPool+LIF thresholds via offline integer search
    10. :func:`validate_compiled_graph` -- final post-pass graph validation
       after connectivity cleanup, signal-domain propagation, data-format
       propagation, and tick assignment
    11. :func:`validate_deployable_graph` -- ensure no frontend-only IR remains

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

    # Step 2: Normalize pre-fusion transform topology to a fixed point.
    graph = _run_pre_fusion_rewrite_phase(graph)

    # Step 3: Narrow expression-layer add nodes into deployable add IR where possible.
    graph = specialize_general_adds(graph)

    # Step 4: Choose topology, fuse atomic nodes, and prepare AvgPool deployment
    graph = fuse_to_offline_cores(
        graph, _enable_split_avgpool_lif, _enable_avgpool_calibration
    )

    # Step 5: Run post-fusion rewrites that depend on graph analyses. This
    # phase owns the "analyze -> rewrite -> re-analyze" control flow so future
    # topology rewrites can be added without growing ad-hoc replay logic in the
    # main compile function.
    graph = _run_post_fusion_rewrite_phase(
        graph, _input_formats, _post_fusion_rewrite_passes()
    )

    # Step 6: Assign timing parameters
    assign_tick_params(graph, _tick_duration, _auto_reset, tick_overrides)

    # Step 7: Calibrate AvgPool+LIF thresholds (experimental, off by default)
    if _enable_avgpool_calibration:
        calibrate_avgpool_thresholds(graph)

    # Step 8: Final validation after all compile-time annotations are filled.
    # Unlike validate_graph(), this stage assumes the graph is in its final
    # compiled form and checks shape/dims completeness, propagated signal
    # domains, propagated data formats, tick parameters, and connectivity.
    validate_compiled_graph(graph)

    # Step 9: The backend only accepts deployable IR nodes. Expression-layer
    # nodes such as GeneralAddOp must have been specialized away by now.
    validate_deployable_graph(graph)

    graph.eval()
    return graph


def _run_mid_compile_analyses(
    graph: PAIIRGraph, input_formats: dict[str, DataFormat] | None
) -> PAIIRGraph:
    """Run the standard analysis phase for a topology-stable graph.

    This helper gives compile-time rewrites one explicit place to "rewind" to
    when they invalidate previously-computed graph annotations. Any pass that
    rewrites topology after fusion but before scheduling can re-enter this
    phase instead of manually remembering which analyses must be replayed.
    """

    validate_graph(graph)
    propagate_signal_domain(graph)
    propagate_data_format(graph, input_formats)
    return graph


def _pre_fusion_rewrite_passes() -> tuple[RewritePass, ...]:
    """Return ordered topology rewrites that may interact before fusion."""
    return (
        RewritePass("canonicalize_transform_chains", canonicalize_layout_chains),
        RewritePass(
            "commute_pre_activation_transforms", commute_pre_activation_transforms
        ),
    )


def _run_pre_fusion_rewrite_phase(graph: PAIIRGraph, max_rounds: int = 4) -> PAIIRGraph:
    """Run analysis-independent pre-fusion rewrites to a fixed point."""
    return run_fixed_point_rewrite_phase(
        graph, rewrite_passes=_pre_fusion_rewrite_passes(), max_rounds=max_rounds
    )


def _post_fusion_rewrite_passes() -> tuple[RewritePass, ...]:
    """Return the ordered post-fusion rewrite passes.

    Keeping pass declaration separate from phase execution lets us add future
    rewrites by appending one new spec instead of editing the replay logic.
    """

    return (RewritePass("rewrite_standalone_avgpools", rewrite_standalone_avgpools),)


def _run_post_fusion_rewrite_phase(
    graph: PAIIRGraph,
    input_formats: dict[str, DataFormat] | None,
    rewrite_passes: tuple[RewritePass, ...],
    max_rounds: int = 4,
) -> PAIIRGraph:
    """Run the post-fusion fixed-point rewrite phase with analysis refresh."""
    return run_fixed_point_rewrite_phase(
        graph,
        rewrite_passes,
        lambda g: _run_mid_compile_analyses(g, input_formats),
        max_rounds,
    )
