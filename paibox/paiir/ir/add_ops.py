"""Add-family IR nodes.

This module contains both:

- expression-layer add nodes that preserve original PyTorch semantics
- deployable add nodes that represent the current chip-supported subset
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

import torch
from paicorelib import AddPotentialMode
from torch import Tensor

from .calc_params import OfflineCoreParams
from .op_node import OfflineCoreOp, OpNode

__all__ = ["AddOperandKind", "AddOperandSpec", "GeneralAddOp", "PotentialAddOp"]


class AddOperandKind(Enum):
    """Operand categories for expression-layer add nodes."""

    TENSOR = auto()
    CONST = auto()


@dataclass(frozen=True)
class AddOperandSpec:
    """One operand of a :class:`GeneralAddOp`.

    Only the non-derivable facts are stored here:

    - coefficient/sign applied to the operand
    - whether the operand comes from a tensor input path or an embedded const
    - the tensor input port, if tensor-backed
    - the constant value, if const-backed

    Shape/dims/broadcast properties are derived from node-level metadata and
    the current output shape rather than duplicated here.
    """

    coeff: int
    kind: AddOperandKind
    tensor_port: int | None = None
    const_value: Tensor | int | float | None = None

    def __post_init__(self) -> None:
        if self.kind is AddOperandKind.TENSOR:
            if self.tensor_port is None:
                raise ValueError(f"{self.__class__.__name__}: tensor_port is required")
            if self.const_value is not None:
                raise ValueError(
                    f"{self.__class__.__name__}: const_value must be omitted for tensor operands"
                )
            return

        if self.kind is AddOperandKind.CONST:
            if self.const_value is None:
                raise ValueError(f"{self.__class__.__name__}: const_value is required")
            if self.tensor_port is not None:
                raise ValueError(
                    f"{self.__class__.__name__}: tensor_port must be omitted for const operands"
                )
            return

        raise ValueError(
            f"{self.__class__.__name__}: unknown operand kind {self.kind!r}"
        )


def _constant_shape(value: Tensor | int | float) -> torch.Size:
    if torch.is_tensor(value):
        return value.shape
    return torch.Size()


def _constant_dims(value: Tensor | int | float) -> tuple[int, ...]:
    if torch.is_tensor(value):
        return tuple(range(value.ndim))
    return ()


class GeneralAddOp(OpNode):
    """Expression-layer element-wise add / subtract.

    This node preserves the original PyTorch add/sub semantics after FX
    lowering, including const operands and broadcasted operands. It is a
    frontend/general IR node rather than a deployable offline-core operator.
    Compile-time specialization passes may later rewrite a subset of
    ``GeneralAddOp`` instances into deployable forms such as
    :class:`PotentialAddOp`.
    """

    deploy: ClassVar[bool] = False

    def __init__(self, operands: Sequence[AddOperandSpec]) -> None:
        super().__init__()
        if len(operands) < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least two operands"
            )
        self.operands = tuple(operands)

    @property
    def coeffs(self) -> tuple[int, ...]:
        return tuple(operand.coeff for operand in self.operands)

    @property
    def tensor_operands(self) -> tuple[AddOperandSpec, ...]:
        return tuple(
            operand
            for operand in self.operands
            if operand.kind is AddOperandKind.TENSOR
        )

    @property
    def tensor_coeffs(self) -> tuple[int, ...]:
        return tuple(operand.coeff for operand in self.tensor_operands)

    @property
    def has_const_operands(self) -> bool:
        return any(operand.kind is AddOperandKind.CONST for operand in self.operands)

    def operand_shape(self, operand: AddOperandSpec) -> torch.Size:
        if operand.kind is AddOperandKind.TENSOR:
            assert operand.tensor_port is not None
            if operand.tensor_port >= len(self.input_shapes):
                return torch.Size()
            return self.input_shapes[operand.tensor_port]

        assert operand.const_value is not None
        return _constant_shape(operand.const_value)

    def operand_dims(self, operand: AddOperandSpec) -> tuple[int, ...]:
        if operand.kind is AddOperandKind.TENSOR:
            assert operand.tensor_port is not None
            if operand.tensor_port >= len(self.input_dims):
                return ()
            return self.input_dims[operand.tensor_port]

        assert operand.const_value is not None
        return _constant_dims(operand.const_value)

    def is_operand_broadcasted(self, operand: AddOperandSpec) -> bool:
        return (
            bool(self.output_shape) and self.operand_shape(operand) != self.output_shape
        )

    @property
    def has_broadcasted_operands(self) -> bool:
        return any(self.is_operand_broadcasted(operand) for operand in self.operands)

    def forward(self, *xs: Tensor) -> Tensor:
        tensor_idx = 0
        acc: Tensor | None = None
        ref_tensor = xs[0] if xs else None

        for operand in self.operands:
            if operand.kind is AddOperandKind.TENSOR:
                if tensor_idx >= len(xs):
                    raise ValueError(
                        f"{self.__class__.__name__} expected {len(self.tensor_operands)} tensor input(s), "
                        f"got {len(xs)}"
                    )
                value: Tensor | int | float = xs[tensor_idx]
                tensor_idx += 1
            else:
                assert operand.const_value is not None
                value = operand.const_value
                if torch.is_tensor(value) and ref_tensor is not None:
                    value = value.to(device=ref_tensor.device)

            term = operand.coeff * value
            if not torch.is_tensor(term):
                if acc is not None:
                    term = torch.as_tensor(term, dtype=acc.dtype, device=acc.device)
                elif ref_tensor is not None:
                    term = torch.as_tensor(
                        term, dtype=ref_tensor.dtype, device=ref_tensor.device
                    )
                else:
                    term = torch.as_tensor(term)

            acc = term if acc is None else acc + term

        if tensor_idx != len(xs):
            raise ValueError(
                f"{self.__class__.__name__} consumed {tensor_idx} tensor input(s), got {len(xs)}"
            )
        if acc is None:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one materialized term"
            )

        return acc

    def extra_repr(self) -> str:
        operand_parts = []
        for operand in self.operands:
            if operand.kind is AddOperandKind.TENSOR:
                operand_parts.append(
                    f"tensor(coeff={operand.coeff}, port={operand.tensor_port})"
                )
            else:
                operand_parts.append(f"const(coeff={operand.coeff})")
        return f"{super().extra_repr()}, operands=[{', '.join(operand_parts)}]"


class PotentialAddOp(OfflineCoreOp):
    """Deployable element-wise potential add / subtract.

    This IR node intentionally models only the narrow add form that the
    current chip/backend path can deploy:

    - each operand must come from a real tensor-path predecessor
    - all operands must be aligned element-wise in the same shape/layout
    - each path uses a fixed sign in ``{-1, +1}``
    - the result remains in the membrane-potential domain

    The public constructor derives semantic core parameters from ``op_signs``.
    Advanced callers that need to preserve prepared compile-time state should
    construct the node normally, then call
    :meth:`OfflineCoreOp.override_compile_state`.
    """

    def __init__(self, op_signs: tuple[int, ...] = (1, 1)) -> None:
        if len(op_signs) < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least two signed input paths"
            )
        if any(sign not in (-1, 1) for sign in op_signs):
            raise ValueError(
                f"{self.__class__.__name__} signs must be +/-1 only, got {op_signs}"
            )

        core_params = OfflineCoreParams()
        core_params.add_potential = AddPotentialMode.DIRECT_ADD

        super().__init__(core_params)
        self.signs = tuple(op_signs)

    def forward(self, *xs: Tensor) -> Tensor:
        if len(xs) != len(self.signs):
            raise ValueError(
                f"{self.__class__.__name__} expected {len(self.signs)} inputs, got {len(xs)}"
            )
        if not xs:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one runtime input"
            )

        acc = torch.zeros_like(xs[0])
        for sign, x in zip(self.signs, xs):
            acc += sign * x

        return acc

    @property
    def weights(self) -> list[Tensor] | None:
        return None

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, signs={self.signs}"
