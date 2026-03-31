"""PAIIR base types: node base class and graph boundary nodes."""

import torch

from ._namespace import IRNamespace
from .signal_domain import SignalDomain

__all__ = ["PAIIRNode", "InputNode", "OutputNode"]

_ir_namespace = IRNamespace()


class PAIIRNode:
    """Base class for all PAIIR nodes.

    Each node is automatically assigned a unique name for identification
    within the computation graph.
    """

    def __init__(self) -> None:
        self.name: str = _ir_namespace.create_name(self)
        self.output_domain: SignalDomain | None = None

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        if hasattr(self, "shape") and self.shape:
            parts.append(f"shape={self.shape}")
        if self.output_domain is not None:
            parts.append(f"output_domain={self.output_domain.name}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class InputNode(PAIIRNode):
    """Graph input placeholder carrying shape information."""

    def __init__(self, shape: torch.Size = torch.Size()) -> None:
        super().__init__()
        self.shape = shape


class OutputNode(PAIIRNode):
    """Graph output node."""

    def __init__(self, shape: torch.Size = torch.Size()) -> None:
        super().__init__()
        self.shape = shape
