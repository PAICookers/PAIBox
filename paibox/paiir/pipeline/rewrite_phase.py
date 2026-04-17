"""Helpers for fixed-point graph rewrite phases.

This module captures one recurring compile-time pattern:

1. start from a topology-stable graph
2. execute an ordered set of rewrite passes
3. if any rewrite changes topology, optionally refresh derived graph state
4. repeat until the phase reaches a fixed point

The design borrows the most useful parts of ``torch.fx`` pass orchestration:

- pass work is represented explicitly as small pass specs
- rewrites report change by object replacement / identity change
- the phase runs to a fixed point rather than assuming one pass is enough

Unlike ``torch.fx.passes.infra.PassManager``, this module stays intentionally
small and tailored to the current PAIIR need: a handful of graph rewrites that
may need to re-run cleanup or analysis steps when they mutate topology.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..ir.graph import PAIIRGraph

__all__ = ["RewritePass", "run_fixed_point_rewrite_phase"]


RewriteFunc = Callable[[PAIIRGraph], PAIIRGraph]
RefreshFunc = Callable[[PAIIRGraph], PAIIRGraph]


@dataclass(frozen=True, slots=True)
class RewritePass:
    """One topology rewrite pass used inside a fixed-point rewrite phase."""

    name: str
    func: RewriteFunc


def _no_refresh(graph: PAIIRGraph) -> PAIIRGraph:
    return graph


def run_fixed_point_rewrite_phase(
    graph: PAIIRGraph,
    rewrite_passes: tuple[RewritePass, ...],
    refresh_graph: RefreshFunc = _no_refresh,
    max_rounds: int = 4,
) -> PAIIRGraph:
    """Run ordered topology rewrites until they reach a fixed point.

    ``refresh_graph`` lets the phase re-run prerequisite analyses or cleanup
    between rewrites. When omitted, the phase operates on plain topology
    rewrites only.
    """
    graph = refresh_graph(graph)

    for _ in range(max_rounds):
        changed_in_round = False
        for rewrite_pass in rewrite_passes:
            rewritten = rewrite_pass.func(graph)
            if rewritten is graph:
                continue
            graph = refresh_graph(rewritten)
            changed_in_round = True

        if not changed_in_round:
            return graph

    raise RuntimeError(
        "fixed-point rewrite phase did not converge within " f"{max_rounds} round(s)"
    )
