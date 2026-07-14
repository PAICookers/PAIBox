"""Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

This module provides a high-level entry point that wraps the current PAIIR
conversion pipeline:

1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
2. pre-fusion fixed-point rewrites -- fold safe zero pads into following convs,
   canonicalize transform chains, and commute transparent pre-activation transforms
3. :func:`flatten_general_add_chains` -- collapse deployable binary add chains
   into n-ary add nodes
4. :func:`specialize_general_adds` -- narrow expression-layer add nodes into
   deployable add IR where possible
5. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and apply
   AvgPool deployment params
6. post-fusion fixed-point rewrites with analysis refresh
7. :func:`assign_tick_params` -- assign timing parameters
8. :func:`calibrate_avgpool_thresholds` -- optional AvgPool threshold refinement
9. :func:`validate_compiled_graph` -- validate the compiled semantic graph
10. neuron-parameter materialization -- expand tensors and merge compute bias
11. :func:`validate_deployable_graph` -- reject residual expression-layer IR

Use :func:`compile_to_paiir` for a one-step compilation, or call the
individual passes directly for fine-grained control.

Public timing configuration uses ``timesteps`` and ``auto_reset``. The
lowering pipeline maps those values to internal ``tick_duration`` /
``tick_initial`` core fields before backend export.
"""

from dataclasses import dataclass
from typing import Any, Final, TypeVar, cast

from torch import Tensor, nn

from ..ir.graph import PAIIRGraph
from ..lowering.converter import torch_to_paiir
from .avgpool import rewrite_delayed_avgpool_division, rewrite_standalone_avgpools
from .avgpool.standalone_rewrite import OutputApprox
from .data_format import DataFormat
from .layout_chain_canonicalization import canonicalize_layout_chains
from .layout_cross_node_elision import commute_pre_activation_transforms
from .neuron_param_materialization import materialize_neuron_params
from .pad_folding import fold_zero_pad_into_convs
from .passes import (
    analyze_graph,
    assign_tick_params,
    calibrate_avgpool_thresholds,
    flatten_general_add_chains,
    fuse_to_offline_cores,
    specialize_general_adds,
    validate_compiled_graph,
    validate_deployable_graph,
)
from .rewrite_phase import RewritePass, run_fixed_point_rewrite_phase

__all__ = ["compile_to_paiir", "CompileConfig"]


_T = TypeVar("_T")


class _UnsetArg:
    """Sentinel type for keyword arguments whose omitted state matters."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


# Distinguish omitted keywords from explicit invalid/default values so
# CompileConfig defaults and explicit keyword overrides can coexist.
_UNSET_ARG: Final = _UnsetArg()


def _resolve_config_arg(value: _T | _UnsetArg, config_value: _T) -> _T:
    """Use the config value only when the keyword argument was omitted."""
    if value is _UNSET_ARG:
        return config_value
    return cast(_T, value)


@dataclass
class CompileConfig:
    """Global configuration defaults for :func:`compile_to_paiir`.

    Keyword arguments passed directly to :func:`compile_to_paiir` override
    matching values in this object when those keywords are explicitly provided.

    Attributes:
        timesteps: Application-side inference sequence length. It must be a
            positive integer. When automatic reset is enabled, the compiled
            cores stay active and reset every ``timesteps`` active ticks. When
            automatic reset is disabled, the compiled cores work for exactly
            ``timesteps`` ticks and leave reset control to the caller.
        auto_reset: Whether compiled cores automatically reinitialise neuron
            state every ``timesteps`` active ticks. The same timing policy is
            applied to all offline cores; it does not branch by ANN/SNN mode.
        input_formats: Optional data-format overrides keyed by ``InputNode``
            name. Each value is a ``(DataSign, DataWidth)`` pair.
        enable_avgpool_calibration: Enable experimental threshold calibration
            for shared-core AvgPool+LIF deployments.
        enable_split_avgpool_lif: Enable experimental split-core deployment
            for supported AvgPool+LIF chains.
        enable_delayed_avgpool_division: Rewrite eligible AvgPool chains to
            keep integer sums and materialize division at a later exact endpoint.
        output_approx: Output-boundary approximation strategy. ``"default"``
            keeps the standard policy; ``"sum_approx_if_avgpool"`` exports
            eligible output-layer AvgPool nodes as unnormalized
            ``SumPool + identity LUT`` counts and emits ``OutputApproxWarning``.
            The application CPU must divide or accumulate those counts according
            to task semantics.
    """

    timesteps: int = 1
    auto_reset: bool = True
    input_formats: dict[str, DataFormat] | None = None
    enable_avgpool_calibration: bool = False
    enable_split_avgpool_lif: bool = False
    enable_delayed_avgpool_division: bool = True
    output_approx: OutputApprox = "default"


def _compile_paiir_graph(
    graph: PAIIRGraph,
    *,
    timesteps: int,
    auto_reset: bool,
    input_formats: dict[str, DataFormat] | None = None,
    enable_avgpool_calibration: bool = False,
    enable_split_avgpool_lif: bool = False,
    enable_delayed_avgpool_division: bool = True,
    output_approx: OutputApprox = "default",
) -> PAIIRGraph:
    """Run compile-time passes on an already-built PAIIR graph."""
    graph = _run_pre_fusion_rewrite_phase(graph)
    graph = flatten_general_add_chains(graph)
    graph = specialize_general_adds(graph)
    graph = fuse_to_offline_cores(
        graph, enable_split_avgpool_lif, enable_avgpool_calibration
    )
    graph = _run_post_fusion_rewrite_phase(
        graph,
        input_formats,
        _post_fusion_rewrite_passes(enable_delayed_avgpool_division, output_approx),
    )

    assign_tick_params(graph, timesteps, auto_reset)

    if enable_avgpool_calibration:
        calibrate_avgpool_thresholds(graph)

    validate_compiled_graph(graph)
    materialize_neuron_params(graph)
    validate_deployable_graph(graph)

    graph.eval()
    return graph


def compile_to_paiir(
    model: nn.Module,
    *sample_inputs: Tensor,
    timesteps: int | _UnsetArg = _UNSET_ARG,
    auto_reset: bool | _UnsetArg = _UNSET_ARG,
    input_formats: dict[str, DataFormat] | None = None,
    compile_config: CompileConfig | None = None,
    concrete_args: dict[str, Any] | None = None,
    strict: bool = True,
    enable_avgpool_calibration: bool | None = None,
    enable_split_avgpool_lif: bool | None = None,
    enable_delayed_avgpool_division: bool | None = None,
    output_approx: OutputApprox | None = None,
) -> PAIIRGraph:
    """Compile a PyTorch model to a ready-to-deploy :class:`PAIIRGraph`.

    The function traces ``model`` with ``sample_inputs``, lowers the traced
    graph to PAIIR nodes, runs deployability rewrites, fuses deployable
    operations into offline cores, assigns core timing/data-format metadata,
    validates the final graph, switches it to eval mode, and returns it.

    Args:
        model: PyTorch module to compile. It should already represent the
            deployment-time model behavior.
        *sample_inputs: Example tensors used for FX tracing, shape propagation,
            and dimension propagation. The current deployment path expects
            batch size 1.
        timesteps: Application-side inference sequence length. It must be a
            positive integer and defaults to ``1`` through ``CompileConfig``.
            With ``auto_reset=True``, cores stay active
            (``tick_duration=0``) and reinitialise every ``timesteps`` active
            ticks (``tick_initial=timesteps``). With ``auto_reset=False``,
            cores work for ``timesteps`` ticks (``tick_duration=timesteps``)
            with reset control left to the caller (``tick_initial=0``).
        auto_reset: Whether compiled cores automatically reinitialise state
            every ``timesteps`` active ticks. Defaults to ``True`` through
            ``CompileConfig`` and applies uniformly to ANN and SNN mode cores.
        input_formats: Optional data-format overrides keyed by ``InputNode``
            name.
        compile_config: Optional object containing global defaults. Explicit
            keyword arguments on this function take priority over matching
            values in ``compile_config``.
        concrete_args: Optional concrete non-tensor arguments passed to FX
            tracing.
        strict: If true, unsupported operations raise immediately. If false,
            unsupported operations are reported as warnings where the lowering
            path can safely continue.
        enable_avgpool_calibration: Optional override for shared-core
            AvgPool+LIF threshold calibration.
        enable_split_avgpool_lif: Optional override for conditional AvgPool+LIF
            split-core deployment.
        enable_delayed_avgpool_division: Optional override for the delayed
            AvgPool division rewrite.
        output_approx: Optional output-boundary approximation strategy.
            ``"default"`` preserves the standard rewrite policy.
            ``"sum_approx_if_avgpool"`` rewrites eligible output-layer
            ``AvgPool1d`` / ``AvgPool2d`` nodes to unnormalized
            ``SumPool + identity LUT`` counts and emits an
            :class:`~paibox.paiir.exceptions.OutputApproxWarning`. CPU-side
            divide/accumulate postprocessing is currently an application
            responsibility rather than a structured graph node.

    Timing defaults:
        If neither keyword arguments nor ``compile_config`` specify timing,
        ``timesteps`` defaults to ``1`` and ``auto_reset`` defaults to
        ``True``. This exports all offline cores with ``tick_duration=0`` and
        ``tick_initial=1``. The same policy applies to ANN and SNN mode cores.

    Pipeline:

    1. :func:`torch_to_paiir` -- FX trace and 1:1 node mapping
    2. pre-fusion fixed-point rewrite phase -- canonicalize transform chains and
       commute transparent pre-activation transforms
    3. :func:`flatten_general_add_chains` -- collapse deployable binary add
       chains into n-ary add nodes
    4. :func:`specialize_general_adds` -- narrow expression-layer add nodes
       into deployable add IR where possible
    5. :func:`fuse_to_offline_cores` -- choose topology, rewrite nodes, and
       apply AvgPool deployment params
    6. analysis phase -- validate graph, infer signal semantics, infer data formats
    7. delayed AvgPool division rewrite when an exact-sum single-consumer chain
       can materialize the divisor safely at a later endpoint
    8. standalone AvgPool auto-rewrite that depends on those analyses
    9. re-run the analysis phase if a rewrite changed the graph
    10. :func:`assign_tick_params` -- assign timing parameters
    11. :func:`calibrate_avgpool_thresholds` -- (experimental) refine shared-core
       AvgPool+LIF thresholds via offline integer search
    12. :func:`validate_compiled_graph` -- post-pass semantic graph validation
        after connectivity cleanup, signal-semantics propagation, data-format
        propagation, and tick assignment
    13. neuron-parameter materialization -- expand tensors against each final
        output shape and merge compute bias into additive leak
    14. :func:`validate_deployable_graph` -- ensure only backend-ready IR and
        flat neuron parameters remain
    """
    cfg = compile_config or CompileConfig()
    _timesteps = _resolve_config_arg(timesteps, cfg.timesteps)
    _auto_reset = _resolve_config_arg(auto_reset, cfg.auto_reset)
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
    _enable_delayed_avgpool_division = (
        enable_delayed_avgpool_division
        if enable_delayed_avgpool_division is not None
        else cfg.enable_delayed_avgpool_division
    )
    _output_approx = output_approx if output_approx is not None else cfg.output_approx

    graph = torch_to_paiir(
        model, *sample_inputs, concrete_args=concrete_args, strict=strict
    )

    return _compile_paiir_graph(
        graph,
        timesteps=_timesteps,
        auto_reset=_auto_reset,
        input_formats=_input_formats,
        enable_avgpool_calibration=_enable_avgpool_calibration,
        enable_split_avgpool_lif=_enable_split_avgpool_lif,
        enable_delayed_avgpool_division=_enable_delayed_avgpool_division,
        output_approx=_output_approx,
    )


def _pre_fusion_rewrite_passes() -> tuple[RewritePass, ...]:
    """Return ordered topology rewrites that may interact before fusion."""
    return (
        RewritePass("fold_zero_pad_into_convs", fold_zero_pad_into_convs),
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


def _post_fusion_rewrite_passes(
    enable_delayed_avgpool_division: bool, output_approx: OutputApprox
) -> tuple[RewritePass, ...]:
    """Return the ordered post-fusion rewrite passes."""
    passes: list[RewritePass] = []
    if enable_delayed_avgpool_division:
        passes.append(
            RewritePass(
                "rewrite_delayed_avgpool_division", rewrite_delayed_avgpool_division
            )
        )
    passes.append(
        RewritePass(
            "rewrite_standalone_avgpools",
            lambda graph: rewrite_standalone_avgpools(graph, output_approx),
        )
    )
    return tuple(passes)


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
