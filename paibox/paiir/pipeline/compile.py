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
    analyze_graph,
    assign_tick_params,
    calibrate_avgpool_thresholds,
    fuse_to_offline_cores,
    specialize_general_adds,
    validate_compiled_graph,
    validate_deployable_graph,
)
from .rewrite_phase import RewritePass, run_fixed_point_rewrite_phase

__all__ = ["compile_to_paiir", "CompileConfig"]


@dataclass
class CompileConfig:
    """Configuration for :func:`compile_to_paiir`."""

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
    5. analysis phase -- validate graph, infer signal semantics, infer data formats
    6. standalone AvgPool auto-rewrite that depends on those analyses
    7. re-run the analysis phase if a rewrite changed the graph
    8. :func:`assign_tick_params` -- assign timing parameters
    9. :func:`calibrate_avgpool_thresholds` -- (experimental) refine shared-core
       AvgPool+LIF thresholds via offline integer search
    10. :func:`validate_compiled_graph` -- final post-pass graph validation
        after connectivity cleanup, signal-semantics propagation, data-format
        propagation, and tick assignment
    11. :func:`validate_deployable_graph` -- ensure no frontend-only IR remains
    """
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

    graph = torch_to_paiir(
        model, *sample_inputs, concrete_args=concrete_args, strict=strict
    )

    graph = _run_pre_fusion_rewrite_phase(graph)
    graph = specialize_general_adds(graph)
    graph = fuse_to_offline_cores(
        graph, _enable_split_avgpool_lif, _enable_avgpool_calibration
    )
    graph = _run_post_fusion_rewrite_phase(
        graph, _input_formats, _post_fusion_rewrite_passes()
    )

    assign_tick_params(graph, _tick_duration, _auto_reset, tick_overrides)

    if _enable_avgpool_calibration:
        calibrate_avgpool_thresholds(graph)

    validate_compiled_graph(graph)
    validate_deployable_graph(graph)

    graph.eval()
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
    """Return the ordered post-fusion rewrite passes."""
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
        lambda g: analyze_graph(g, input_formats),
        max_rounds,
    )
