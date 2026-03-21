"""PAIIR exception and warning types."""

from ..exceptions import PAIBoxError, PAIBoxWarning

__all__ = [
    "GraphCleanupWarning",
    "GraphValidationError",
    "PAIIRError",
    "PAIIRWarning",
    "UnsupportedFusionError",
    "UnsupportedOpError",
    "UnsupportedOpWarning",
]


class PAIIRError(PAIBoxError):
    """Base exception for all PAIIR errors."""

    pass


class UnsupportedOpError(PAIIRError):
    """Raised when an unsupported operator is encountered in strict mode.

    This exception indicates that the model contains an operator that PAIIR
    does not know how to convert. The operator will not be added to the IR
    graph.

    To resolve this error, either:
    1. Register a converter via :func:`~paibox.paiir.register_neuron`
    2. Set strict=False to bypass unsupported operators (with warnings)
    3. Modify the model to avoid using the unsupported operator

    Attributes:
        node_name: The name of the FX node that caused the error.
        operator_desc: A description of the unsupported operator.
    """

    def __init__(self, node_name: str, operator_desc: str) -> None:
        self.node_name = node_name
        self.operator_desc = operator_desc
        super().__init__(
            f"Unsupported operator '{operator_desc}' at node '{node_name}'. "
            f"Use register_neuron() to add a converter, or set strict=False to bypass."
        )


class UnsupportedFusionError(PAIIRError):
    """Raised when an unsupported operator combination or configuration is encountered.

    This exception indicates that while individual operators are valid, their
    combination or input configuration is not supported by the PAIIR fusion
    pipeline.

    Attributes:
        fusion_desc: A description of the unsupported fusion scenario.
    """

    def __init__(self, fusion_desc: str) -> None:
        self.fusion_desc = fusion_desc
        super().__init__(f"Unsupported fusion configuration: {fusion_desc}")


class PAIIRWarning(PAIBoxWarning):
    """Base warning for all PAIIR warnings."""

    pass


class UnsupportedOpWarning(PAIIRWarning):
    """Warning emitted when an unsupported operator is bypassed in non-strict mode.

    This warning indicates that the model contains an operator that PAIIR
    does not know how to convert. The operator will be bypassed (data flow
    will skip this node), which may affect model behavior.

    Attributes:
        unsupported_ops: A list of (node_name, operator_desc) tuples for
            all unsupported operators that were bypassed.
    """

    def __init__(self, unsupported_ops: list[tuple[str, str]]) -> None:
        self.unsupported_ops = unsupported_ops
        ops_list = "\n".join(f"  - {name}: {desc}" for name, desc in unsupported_ops)
        super().__init__(
            f"Encountered {len(unsupported_ops)} unsupported operator(s), "
            f"which will be bypassed (data flow will skip these nodes):\n{ops_list}"
        )


class GraphValidationError(PAIIRError):
    """Raised when a PAIIR graph has unrecoverable structural problems."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(
            f"Graph validation failed with {len(errors)} error(s):\n{bullet_list}"
        )


class GraphCleanupWarning(PAIIRWarning):
    """Warning emitted when disconnected nodes are removed during validation."""

    def __init__(self, removed: list[str]) -> None:
        self.removed = removed
        names = ", ".join(removed)
        super().__init__(
            f"Removed {len(removed)} disconnected node(s) from the graph: {names}"
        )
