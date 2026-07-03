"""Source-op schemas and constraint helpers for lowering frontends."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import torch
from torch import nn

if TYPE_CHECKING:
    from ..ir.op_node import OpNode

ModuleMapper: TypeAlias = dict[type[nn.Module], Callable[[nn.Module], "OpNode"]]


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


MISSING: Final = _Missing()


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Result of one source or target legality predicate."""

    supported: bool
    constraint: str | None = None
    field: str | None = None
    value: object | None = None
    reason: str | None = None
    case: str | None = None

    @classmethod
    def ok(cls) -> "ConstraintResult":
        return cls(True)

    @classmethod
    def fail(
        cls,
        *,
        constraint: str,
        field: str | None = None,
        value: object | None = None,
        reason: str,
        case: str | None = None,
    ) -> "ConstraintResult":
        return cls(
            False,
            constraint=constraint,
            field=field,
            value=value,
            reason=reason,
            case=case,
        )

    def with_case(self, case: str | None) -> "ConstraintResult":
        if self.supported or self.case is not None:
            return self
        return replace(self, case=case)


@dataclass(frozen=True, slots=True)
class AttributeReader:
    """Read one named source attribute from a module or canonical subject."""

    read: Callable[[Any], Any]
    default: object = MISSING

    def __call__(self, subject: Any) -> Any:
        try:
            return self.read(subject)
        except AttributeError:
            if self.default is MISSING:
                return MISSING
            return self.default


@dataclass(frozen=True, slots=True)
class AttributeView:
    """A snapshot of attributes read during one legality check."""

    values: dict[str, Any]

    def raw(self, name: str, default: object = MISSING) -> Any:
        return self.values.get(name, default)

    def display(self, name: str) -> object:
        return format_attribute_value(self.raw(name))


def attr(source_name: str, *, default: object = MISSING) -> AttributeReader:
    def _read(subject: Any) -> Any:
        return getattr(subject, source_name)

    return AttributeReader(_read, default)


def reader(
    read: Callable[[Any], Any],
    default: object = MISSING,
) -> AttributeReader:
    return AttributeReader(read, default)


def format_attribute_value(value: object) -> object:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, nn.Parameter):
        return f"Parameter(shape={tuple(value.shape)})"
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return f"{type(value).__name__}(shape={tuple(value.shape)})"
    return value


@dataclass(frozen=True, slots=True)
class Constraint:
    """Named predicate used in schema and capability checks."""

    name: str
    fields: tuple[str, ...]
    check: Callable[[AttributeView], bool | ConstraintResult]
    reason: str

    def evaluate(self, attrs: AttributeView) -> ConstraintResult:
        checked = self.check(attrs)
        if isinstance(checked, ConstraintResult):
            if checked.supported:
                return checked
            return ConstraintResult.fail(
                constraint=checked.constraint or self.name,
                field=checked.field or _first_or_none(self.fields),
                value=(
                    checked.value
                    if checked.value is not None
                    else _display_first_field(attrs, self.fields)
                ),
                reason=checked.reason or self.reason,
                case=checked.case,
            )
        if checked:
            return ConstraintResult.ok()
        return ConstraintResult.fail(
            constraint=self.name,
            field=_first_or_none(self.fields),
            value=_display_first_field(attrs, self.fields),
            reason=self.reason,
        )


def predicate(
    name: str,
    fields: str | Iterable[str],
    check: Callable[[AttributeView], bool | ConstraintResult],
    reason: str,
) -> Constraint:
    return Constraint(
        name=name,
        fields=_fields_tuple(fields),
        check=check,
        reason=reason,
    )


def eq(field: str, expected: object, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} == {expected!r}",
        field,
        lambda attrs: _semantic_value(attrs.raw(field)) == expected,
        reason or f"expected {field} == {expected!r}",
    )


def one_of(
    field: str, choices: Iterable[object], *, reason: str | None = None
) -> Constraint:
    frozen_choices = tuple(choices)
    return predicate(
        f"{field} in {frozen_choices!r}",
        field,
        lambda attrs: _semantic_value(attrs.raw(field)) in frozen_choices,
        reason or f"expected {field} in {frozen_choices!r}",
    )


def is_false(field: str, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} is False",
        field,
        lambda attrs: attrs.raw(field) is False,
        reason or f"expected {field} to be False",
    )


def scalar(field: str, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} is scalar",
        field,
        lambda attrs: _is_scalar_value(attrs.raw(field)),
        reason or f"{field} must be a non-learnable scalar",
    )


def scalar_or_none(field: str, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} is None or scalar",
        field,
        lambda attrs: attrs.raw(field) is None or _is_scalar_value(attrs.raw(field)),
        reason or f"{field} must be None or a non-learnable scalar",
    )


def scalar_or_1d_tensor(field: str, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} is scalar or 1D tensor",
        field,
        lambda attrs: (
            _is_scalar_value(attrs.raw(field))
            or _is_non_parameter_1d_tensor(attrs.raw(field))
        ),
        reason or f"{field} must be a scalar or 1D Tensor",
    )


def gt(field: str, minimum: float, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} > {minimum!r}",
        field,
        lambda attrs: (
            _is_scalar_value(attrs.raw(field))
            and float(_scalar_value(attrs.raw(field))) > minimum
        ),
        reason or f"expected {field} > {minimum!r}",
    )


def is_bool(field: str, *, reason: str | None = None) -> Constraint:
    return predicate(
        f"{field} is bool",
        field,
        lambda attrs: isinstance(attrs.raw(field), bool),
        reason or f"{field} must be bool",
    )


def scalar_value(value: Any) -> float:
    return _scalar_value(value)


@dataclass(frozen=True, slots=True)
class SourceOp:
    """Recognized frontend module plus the attributes read from it."""

    module: nn.Module
    attributes: AttributeView


@dataclass(frozen=True, slots=True)
class SourceCase:
    """One legal source-op variant and its canonicalization function."""

    name: str
    constraints: tuple[Constraint, ...]
    canonicalize: Callable[[SourceOp], Any]


@dataclass(frozen=True, slots=True)
class SourceResolution:
    """Outcome of resolving one module against a source-op schema."""

    schema: "SourceOpSchema"
    source_op: SourceOp
    case: SourceCase | None = None
    error: ConstraintResult | None = None

    @property
    def supported(self) -> bool:
        return self.error is None and self.case is not None

    def canonicalize(self) -> Any:
        if self.case is None:
            raise RuntimeError("cannot canonicalize an unsupported source op")
        return self.case.canonicalize(self.source_op)


@dataclass(frozen=True, slots=True)
class SourceOpSchema:
    """Whitelist schema for one frontend operator type."""

    frontend: str
    op: str
    recognize: Callable[[nn.Module], bool]
    attributes: tuple[tuple[str, AttributeReader], ...]
    generic_constraints: tuple[Constraint, ...]
    cases: tuple[SourceCase, ...]

    def resolve(self, module: nn.Module) -> SourceResolution | None:
        """Return support status for *module*, or ``None`` if not recognized."""

        if not self.recognize(module):
            return None

        source_op = SourceOp(module, _read_attributes(module, self.attributes))

        for constraint in self.generic_constraints:
            result = constraint.evaluate(source_op.attributes)
            if not result.supported:
                return SourceResolution(self, source_op, error=result)

        matched: list[SourceCase] = []
        first_error: ConstraintResult | None = None
        for source_case in self.cases:
            case_error = _evaluate_case(source_case, source_op.attributes)
            if case_error is None:
                matched.append(source_case)
            elif first_error is None:
                first_error = case_error

        if len(matched) == 1:
            return SourceResolution(self, source_op, matched[0])
        if len(matched) > 1:
            case_names = ", ".join(case.name for case in matched)
            result = ConstraintResult.fail(
                constraint="ambiguous case",
                reason=f"multiple source cases match: {case_names}",
            )
            return SourceResolution(self, source_op, error=result)

        result = first_error or ConstraintResult.fail(
            constraint="no matching case",
            reason="no source case can canonicalize this instance",
        )
        return SourceResolution(self, source_op, error=result)


def source_op_schema(
    frontend: str, op: str, *, recognize: Callable[[nn.Module], bool]
) -> "SourceOpSchemaBuilder":
    """Start a source-op schema builder."""
    return SourceOpSchemaBuilder(frontend, op, recognize)


class SourceOpSchemaBuilder:
    """Mutable builder that produces an immutable ``SourceOpSchema``."""

    def __init__(
        self, frontend: str, op: str, recognize: Callable[[nn.Module], bool]
    ) -> None:
        self._frontend = frontend
        self._op = op
        self._recognize = recognize
        self._attributes: list[tuple[str, AttributeReader]] = []
        self._generic_constraints: list[Constraint] = []
        self._cases: list[SourceCase] = []

    def attribute(
        self, name: str, attribute_reader: AttributeReader | None = None
    ) -> "SourceOpSchemaBuilder":
        if attribute_reader is None:
            attribute_reader = attr(name)
        self._attributes.append((name, attribute_reader))
        return self

    def generic(self, constraint: Constraint) -> "SourceOpSchemaBuilder":
        self._generic_constraints.append(constraint)
        return self

    def case(self, name: str) -> "SourceCaseBuilder":
        return SourceCaseBuilder(self, name)

    def build(self) -> SourceOpSchema:
        return SourceOpSchema(
            frontend=self._frontend,
            op=self._op,
            recognize=self._recognize,
            attributes=tuple(self._attributes),
            generic_constraints=tuple(self._generic_constraints),
            cases=tuple(self._cases),
        )

    def _add_case(self, source_case: SourceCase) -> None:
        self._cases.append(source_case)


class SourceCaseBuilder:
    """Builder for one source-op case."""

    def __init__(self, parent: SourceOpSchemaBuilder, name: str) -> None:
        self._parent = parent
        self._name = name
        self._constraints: list[Constraint] = []

    def when(self, constraint: Constraint) -> "SourceCaseBuilder":
        self._constraints.append(constraint)
        return self

    def canonicalize_to(
        self, canonicalize: Callable[[SourceOp], Any]
    ) -> SourceOpSchemaBuilder:
        self._parent._add_case(
            SourceCase(
                name=self._name,
                constraints=tuple(self._constraints),
                canonicalize=canonicalize,
            )
        )
        return self._parent


def format_constraint_error(
    *, stage: str, frontend: str | None, op: str, result: ConstraintResult
) -> str:
    parts = [f"stage={stage}"]
    if frontend is not None:
        parts.append(f"frontend={frontend}")
    parts.extend(
        [
            f"op={op}",
            f"case={result.case}",
            f"constraint={result.constraint}",
            f"field={result.field}",
            f"value={result.value!r}",
            f"reason={result.reason}",
        ]
    )
    return " ".join(parts)


def _read_attributes(
    subject: Any, attributes: tuple[tuple[str, AttributeReader], ...]
) -> AttributeView:
    return AttributeView(
        {name: attribute_reader(subject) for name, attribute_reader in attributes}
    )


def _evaluate_case(
    source_case: SourceCase, attrs: AttributeView
) -> ConstraintResult | None:
    for constraint in source_case.constraints:
        result = constraint.evaluate(attrs)
        if not result.supported:
            return result.with_case(source_case.name)
    return None


def _fields_tuple(fields: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        return (fields,)
    return tuple(fields)


def _first_or_none(fields: tuple[str, ...]) -> str | None:
    return fields[0] if fields else None


def _display_first_field(
    attrs: AttributeView, fields: tuple[str, ...]
) -> object | None:
    return attrs.display(fields[0]) if fields else None


def _semantic_value(value: Any) -> Any:
    if value is MISSING:
        return value
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    return value


def _is_scalar_value(value: Any) -> bool:
    if isinstance(value, nn.Parameter):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if torch.is_tensor(value):
        return value.numel() == 1
    return False


def _is_non_parameter_1d_tensor(value: Any) -> bool:
    return (
        torch.is_tensor(value)
        and not isinstance(value, nn.Parameter)
        and value.ndim == 1
        and value.numel() > 0
    )


def _scalar_value(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    raise TypeError(f"expected scalar, got {type(value).__name__}")
