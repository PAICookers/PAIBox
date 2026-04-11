from paibox.paiir.ir.graph import PAIIRGraph
from paibox.paiir.pipeline.rewrite_phase import (
    AnalysisDependentRewritePass,
    run_analysis_dependent_rewrite_phase,
)


def test_rewrite_phase_replays_analyses_until_fixed_point() -> None:
    graph = PAIIRGraph("g")
    calls: list[str] = []
    state = {"rewritten": False}

    def refresh(g: PAIIRGraph) -> PAIIRGraph:
        calls.append("refresh")
        return g

    def rewrite_once(g: PAIIRGraph) -> PAIIRGraph:
        calls.append("rewrite")
        if state["rewritten"]:
            return g
        state["rewritten"] = True
        return g.clone_shallow()

    result = run_analysis_dependent_rewrite_phase(
        graph,
        refresh_analyses=refresh,
        rewrite_passes=(
            AnalysisDependentRewritePass("rewrite_once", rewrite_once),
        ),
    )

    assert result is not graph
    assert calls == ["refresh", "rewrite", "refresh", "rewrite"]


def test_rewrite_phase_raises_when_not_converged() -> None:
    graph = PAIIRGraph("g")

    def refresh(g: PAIIRGraph) -> PAIIRGraph:
        return g

    def rewrite_always(g: PAIIRGraph) -> PAIIRGraph:
        return g.clone_shallow()

    try:
        run_analysis_dependent_rewrite_phase(
            graph,
            refresh_analyses=refresh,
            rewrite_passes=(
                AnalysisDependentRewritePass("rewrite_always", rewrite_always),
            ),
            max_rounds=2,
        )
    except RuntimeError as exc:
        assert "did not converge" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError for non-convergent rewrite phase")
