"""Shared helpers for framework-specific lowering frontends."""

from typing import Protocol

from torch import nn

from ..support import ConstraintResult, ModuleMapper, SourceOpSchema


class FrontendAdapter(Protocol):
    name: str
    erase_types: tuple[type[nn.Module], ...]
    source_schemas: tuple[SourceOpSchema, ...]

    def detect(self, model: nn.Module) -> bool: ...

    def prepare(self, model: nn.Module) -> nn.Module: ...

    def build_module_map(self) -> ModuleMapper: ...

    def owns_module(self, module: nn.Module) -> bool: ...

    def describe_unsupported(self, module: nn.Module) -> ConstraintResult | None: ...
