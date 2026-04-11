"""Helpers for analysis-dependent graph rewrite phases.

This module captures one recurring compile-time pattern:

1. run a bundle of analyses on a topology-stable graph
2. execute rewrite passes that depend on those analyses
3. if any rewrite changes topology, refresh the analyses and try again

The design borrows the most useful parts of ``torch.fx`` pass orchestration:

- pass work is represented explicitly as small pass specs
- rewrites report change by object replacement / identity change
- the phase runs to a fixed point rather than assuming one pass is enough

Unlike ``torch.fx.passes.infra.PassManager``, this module stays intentionally
small and tailored to the current PAIIR need: a handful of graph rewrites that
must re-run prior analyses when they mutate topology.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..ir.graph import PAIIRGraph

__all__ = [
    "AnalysisDependentRewritePass",
    "run_analysis_dependent_rewrite_phase",
]


RewriteFunc = Callable[[PAIIRGraph], PAIIRGraph]
AnalysisRefreshFunc = Callable[[PAIIRGraph], PAIIRGraph]


@dataclass(frozen=True, slots=True)
class AnalysisDependentRewritePass:
    """One topology rewrite that requires fresh analysis results first."""

    name: str
    func: RewriteFunc


def run_analysis_dependent_rewrite_phase(
    graph: PAIIRGraph,
    *,
    refresh_analyses: AnalysisRefreshFunc,
    rewrite_passes: tuple[AnalysisDependentRewritePass, ...],
    max_rounds: int = 4,
) -> PAIIRGraph:
    """Run analysis-dependent rewrite passes to a fixed point.

    Args:
        graph: The graph to rewrite.
        refresh_analyses: Function that recomputes the analyses needed by the
            rewrite passes and returns the refreshed graph.
        rewrite_passes: Ordered rewrite pass specs.
        max_rounds: Safety cap for fixed-point iteration.

    Returns:
        The rewritten graph, with analyses refreshed after the last mutation.

    Raises:
        RuntimeError: If the rewrite phase does not converge within
            ``max_rounds``.
    """

    graph = refresh_analyses(graph)

    for _ in range(max_rounds):
        changed_in_round = False
        for rewrite_pass in rewrite_passes:
            rewritten = rewrite_pass.func(graph)
            if rewritten is graph:
                continue
            graph = refresh_analyses(rewritten)
            changed_in_round = True

        if not changed_in_round:
            return graph

    raise RuntimeError(
        "analysis-dependent rewrite phase did not converge within "
        f"{max_rounds} round(s)"
    )
