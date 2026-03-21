"""PAIIR base types: node base class and graph boundary nodes."""

from ._namespace import IRNamespace

__all__ = ["PAIIRNode", "InputNode", "OutputNode"]

_ir_namespace = IRNamespace()


class PAIIRNode:
    """Base class for all PAIIR nodes.

    Each node is automatically assigned a unique name for identification
    within the computation graph.
    """

    def __init__(self) -> None:
        self.name: str = _ir_namespace.create_name(self)

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        if hasattr(self, "shape") and self.shape:
            parts.append(f"shape={self.shape}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class InputNode(PAIIRNode):
    """Graph input placeholder carrying shape information."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.shape = shape


class OutputNode(PAIIRNode):
    """Graph output node."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.shape = shape
