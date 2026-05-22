"""PAIIR base types: node base class, tensor layout, and graph boundaries."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

import torch

from ._namespace import IRNamespace
from .signal_domain import SignalSemantics

__all__ = ["FormatFlow", "TensorLayout", "PAIIRNode", "InputNode", "OutputNode"]

_ir_namespace = IRNamespace()


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """Immutable tensor metadata pairing shape with logical axis order."""

    shape: torch.Size = torch.Size()
    dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.shape and self.dims and len(self.shape) != len(self.dims):
            raise ValueError(
                f"shape rank {len(self.shape)} != dims rank {len(self.dims)}"
            )

    def __bool__(self) -> bool:
        return bool(self.shape)


class FormatFlow(Enum):
    """How a node propagates scalar data format through graph edges."""

    NONE = auto()
    PASS_THROUGH = auto()
    MERGE = auto()


class PAIIRNode:
    """Base class for all PAIIR nodes.

    Each node is automatically assigned a unique name for identification
    within the computation graph.

    ``signal_semantics.output_domain`` is a node-level semantic annotation
    describing the signal domain of the node's output:

    - :class:`~paibox.paiir.ir.signal_domain.SignalDomain.VALUE`
    - :class:`~paibox.paiir.ir.signal_domain.SignalDomain.POTENTIAL`

    The current IR treats this as one value per node, not one value per output
    port. This remains valid for today's multi-output ``SplitOp`` because all
    split branches inherit the same output domain from the split input.

    ``signal_semantics.known_code_range`` stores an optional exact VALUE code
    range for the node's output. ``None`` means the current compile-time
    analyses cannot determine one precisely.
    """

    __format_flow__: ClassVar[FormatFlow] = FormatFlow.NONE
    __tick_depth__: ClassVar[int] = 1

    def __init__(self) -> None:
        self.name: str = _ir_namespace.create_name(self)
        self.signal_semantics = SignalSemantics()

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        if hasattr(self, "shape") and self.shape:
            parts.append(f"shape={self.shape}")
        if self.signal_semantics.output_domain is not None:
            parts.append(f"output_domain={self.signal_semantics.output_domain.name}")
        if self.signal_semantics.known_code_range is not None:
            parts.append(f"known_code_range={self.signal_semantics.known_code_range}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class InputNode(PAIIRNode):
    """Graph input placeholder carrying explicit boundary layout."""

    def __init__(
        self, shape: torch.Size = torch.Size(), dims: tuple[int, ...] | None = None
    ) -> None:
        super().__init__()
        if dims is None:
            dims = tuple(range(len(shape))) if shape else ()
        self.layout = TensorLayout(shape=shape, dims=dims)

    @property
    def shape(self) -> torch.Size:
        return self.layout.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return self.layout.dims


class OutputNode(PAIIRNode):
    """Graph output node carrying explicit boundary layout."""

    __format_flow__: ClassVar[FormatFlow] = FormatFlow.PASS_THROUGH
    __tick_depth__: ClassVar[int] = 0

    def __init__(
        self, shape: torch.Size = torch.Size(), dims: tuple[int, ...] | None = None
    ) -> None:
        super().__init__()
        if dims is None:
            dims = tuple(range(len(shape))) if shape else ()
        self.layout = TensorLayout(shape=shape, dims=dims)

    @property
    def shape(self) -> torch.Size:
        return self.layout.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return self.layout.dims
