"""Static lowering frontend registry."""

from collections.abc import Iterable

from torch import nn

from ...ir.core_neuron import CoreNeuronV25
from ...ir.op_node import OpNode, StandaloneActOp
from ..support import ModuleMapper, SourceResolution, format_constraint_error
from . import snntorch, spikingjelly
from .base import FrontendAdapter

__all__ = [
    "build_frontend_module_map",
    "collect_erase_types",
    "describe_source_error",
    "describe_source_context_error",
    "describe_unsupported_module",
    "lower_source_resolution",
    "owned_by_frontend",
    "prepare_model",
    "resolve_source_op",
    "resolve_frontends",
]

_FRONTENDS: tuple[FrontendAdapter, ...] = (spikingjelly, snntorch)


def resolve_frontends(model: nn.Module) -> tuple[FrontendAdapter, ...]:
    return tuple(frontend for frontend in _FRONTENDS if frontend.detect(model))


def prepare_model(model: nn.Module, frontends: Iterable[FrontendAdapter]) -> nn.Module:
    for frontend in frontends:
        model = frontend.prepare(model)
    return model


def collect_erase_types(
    frontends: Iterable[FrontendAdapter],
) -> tuple[type[nn.Module], ...]:
    return tuple(
        erase_type for frontend in frontends for erase_type in frontend.erase_types
    )


def owned_by_frontend(
    module: nn.Module, frontends: Iterable[FrontendAdapter]
) -> FrontendAdapter | None:
    for frontend in frontends:
        if frontend.owns_module(module):
            return frontend
    return None


def resolve_source_op(
    module: nn.Module, frontends: Iterable[FrontendAdapter]
) -> SourceResolution | None:
    for frontend in frontends:
        for schema in frontend.source_schemas:
            resolution = schema.resolve(module)
            if resolution is not None:
                return resolution
    return None


def lower_source_resolution(resolution: SourceResolution) -> OpNode:
    canonical = resolution.canonicalize()
    if isinstance(canonical, OpNode):
        return canonical
    if isinstance(canonical, CoreNeuronV25):
        return StandaloneActOp(canonical)
    raise TypeError(
        "source schema canonicalize() must return OpNode or CoreNeuronV25, "
        f"got {type(canonical).__name__}"
    )


def describe_source_error(resolution: SourceResolution) -> str:
    if resolution.error is None:
        raise ValueError("source resolution has no error")
    return format_constraint_error(
        stage="source",
        frontend=resolution.schema.frontend,
        op=resolution.schema.op,
        result=resolution.error,
    )


def describe_source_context_error(
    resolution: SourceResolution, output_shape
) -> str | None:
    """Return a frontend error that depends on propagated FX output shape."""
    if resolution.schema.frontend != snntorch.name:
        return None
    result = snntorch.validate_source_context(resolution.source_op, output_shape)
    if result is None:
        return None
    return format_constraint_error(
        stage="source",
        frontend=resolution.schema.frontend,
        op=resolution.schema.op,
        result=result,
    )


def describe_unsupported_module(
    module: nn.Module, frontends: Iterable[FrontendAdapter]
) -> str | None:
    for frontend in frontends:
        if not frontend.owns_module(module):
            continue
        result = frontend.describe_unsupported(module)
        if result is None:
            return None
        return format_constraint_error(
            stage="source",
            frontend=frontend.name,
            op=type(module).__name__,
            result=result,
        )
    return None


def build_frontend_module_map(frontends: Iterable[FrontendAdapter]) -> ModuleMapper:
    module_map: ModuleMapper = {}
    for frontend in frontends:
        for module_type, mapper in frontend.build_module_map().items():
            if module_type in module_map:
                raise ValueError(
                    f"frontend module map conflict for {module_type.__name__}"
                )
            module_map[module_type] = mapper
    return module_map
