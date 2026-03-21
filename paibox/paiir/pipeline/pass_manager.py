"""Compilation pass orchestration with dependency and invalidation tracking.

This module adds a lightweight manager around the existing pass functions used
by :func:`compile_to_paiir`.  The goal is not to replace the current passes,
but to make their ordering explicit and robust when future compensation passes:

- depend on analysis information produced earlier in the pipeline, and/or
- mutate the graph in ways that invalidate previously-computed information.

The manager tracks three things:

1. Which analysis artifacts are currently available.
2. Which pass provides each artifact.
3. Which downstream artifacts become stale when an upstream artifact is
   invalidated.

This allows a compensation pass to invalidate only the information it really
breaks (for example ``VALIDATED``), and let the manager automatically drop and
re-run dependent analyses (for example data format propagation and tick
assignment) before the next pass that needs them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..ir.graph import PAIIRGraph

__all__ = [
    "CompileInfo",
    "CompileInvalidationScope",
    "CompilePass",
    "CompilePassContext",
    "CompilePassManager",
    "CompilePassPhase",
    "CompilePassResult",
]


class CompileInfo(str, Enum):
    """Named analysis artifacts produced during compilation."""

    VALIDATED = "validated"
    DATA_FORMAT = "data_format"
    TICK_PARAMS = "tick_params"


class CompilePassPhase(str, Enum):
    """High-level stage tags for future pipeline planning.

    These tags are intentionally broad.  They provide stable slots for future
    passes without forcing the current implementation to commit to a full pass
    scheduler yet.
    """

    CANONICALIZE = "canonicalize"
    ANALYZE = "analyze"
    COMPENSATE = "compensate"
    FINALIZE = "finalize"


class CompileInvalidationScope(str, Enum):
    """What kind of compiler state a pass may invalidate.

    ``CompileInfo`` tracks concrete analysis artifacts already available.  This
    enum is a coarser design placeholder for future passes that need to express
    their impact before the exact re-analysis strategy is finalized.
    """

    TOPOLOGY = "topology"
    SHAPE = "shape"
    DATA_FORMAT = "data_format"
    SCHEDULE = "schedule"
    DEPLOY_PARAMS = "deploy_params"


@dataclass(slots=True)
class CompilePassContext:
    """Mutable pipeline state shared across pass executions.

    Attributes:
        available_info: Analysis artifacts that are currently valid.
        analysis_data: Optional per-artifact payloads produced by analysis
            passes.  Existing passes do not populate this yet, but future
            passes can store reusable analysis results here.
        metadata: Free-form pipeline metadata for pass coordination.
        executed_passes: Ordered log of executed pass names.
    """

    available_info: set[CompileInfo] = field(default_factory=set)
    analysis_data: dict[CompileInfo, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    executed_passes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompilePassResult:
    """Optional structured result returned by a compilation pass.

    ``invalidated_scopes`` is design-only for now.  The current manager reacts
    to concrete ``invalidated_info`` items; scope-based invalidation is kept as
    an explicit placeholder for future integration once more analyses exist.
    """

    graph: PAIIRGraph | None = None
    provided_info: set[CompileInfo] = field(default_factory=set)
    invalidated_info: set[CompileInfo] = field(default_factory=set)
    invalidated_scopes: set[CompileInvalidationScope] = field(default_factory=set)
    analysis_updates: dict[CompileInfo, Any] = field(default_factory=dict)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


PassFunc = Callable[
    [PAIIRGraph, CompilePassContext], PAIIRGraph | CompilePassResult | None
]


@dataclass(frozen=True, slots=True)
class CompilePass:
    """Declarative pass specification for compilation orchestration.

    ``phase`` and ``impacts`` are currently descriptive metadata.  They are
    intentionally carried in the design now so future real passes can declare
    their position and blast radius without another API churn.
    """

    name: str
    func: PassFunc
    phase: CompilePassPhase = CompilePassPhase.ANALYZE
    requires: frozenset[CompileInfo] = frozenset()
    provides: frozenset[CompileInfo] = frozenset()
    invalidates: frozenset[CompileInfo] = frozenset()
    impacts: frozenset[CompileInvalidationScope] = frozenset()


class CompilePassManager:
    """Run passes while tracking analysis dependencies and invalidations."""

    def __init__(self, graph: PAIIRGraph) -> None:
        self.graph = graph
        self.context = CompilePassContext()
        self._providers: dict[CompileInfo, CompilePass] = {}

    def register_provider(self, pass_spec: CompilePass) -> None:
        """Register *pass_spec* as the provider for its declared artifacts."""

        for info in pass_spec.provides:
            provider = self._providers.get(info)
            if provider is not None and provider.name != pass_spec.name:
                raise ValueError(
                    f"CompileInfo {info.value!r} already provided by pass "
                    f"{provider.name!r}, cannot also register {pass_spec.name!r}"
                )
            self._providers[info] = pass_spec

    def run_pass(self, pass_spec: CompilePass) -> CompilePassResult:
        """Run *pass_spec*, materialising and refreshing its dependencies."""

        self._ensure_requirements(pass_spec.requires)

        raw_result = pass_spec.func(self.graph, self.context)
        result = self._normalize_result(raw_result)

        if result.graph is not None:
            self.graph = result.graph

        invalidated = set(pass_spec.invalidates)
        invalidated.update(result.invalidated_info)
        if invalidated:
            self._invalidate(invalidated)

        provided = set(pass_spec.provides)
        provided.update(result.provided_info)
        if provided:
            self.context.available_info.update(provided)

        if result.analysis_updates:
            self.context.analysis_data.update(result.analysis_updates)

        if result.metadata_updates:
            self.context.metadata.update(result.metadata_updates)

        self.context.executed_passes.append(pass_spec.name)
        return result

    def _ensure_requirements(self, requirements: frozenset[CompileInfo]) -> None:
        for info in requirements:
            if info in self.context.available_info:
                continue

            provider = self._providers.get(info)
            if provider is None:
                raise ValueError(
                    f"pass requirement {info.value!r} has no registered provider"
                )
            self.run_pass(provider)

    def _invalidate(self, invalidated: set[CompileInfo]) -> None:
        expanded = self._expand_invalidation(invalidated)
        self.context.available_info.difference_update(expanded)
        for info in expanded:
            self.context.analysis_data.pop(info, None)

    def _expand_invalidation(self, invalidated: set[CompileInfo]) -> set[CompileInfo]:
        """Cascade invalidation through provider dependencies.

        If artifact A becomes stale, any already-available artifact whose
        provider depends on A is also stale and must be dropped.
        """

        expanded = set(invalidated)
        changed = True
        while changed:
            changed = False
            for info in tuple(self.context.available_info):
                if info in expanded:
                    continue

                provider = self._providers.get(info)
                if provider is None:
                    continue

                if provider.requires & expanded:
                    expanded.add(info)
                    changed = True

        return expanded

    @staticmethod
    def _normalize_result(
        result: PAIIRGraph | CompilePassResult | None,
    ) -> CompilePassResult:
        if result is None:
            return CompilePassResult()
        if isinstance(result, CompilePassResult):
            return result
        if isinstance(result, PAIIRGraph):
            return CompilePassResult(graph=result)
        raise TypeError(
            "compile pass must return None, PAIIRGraph, or CompilePassResult; "
            f"got {type(result).__name__}"
        )
