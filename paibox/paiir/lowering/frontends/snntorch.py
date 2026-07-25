"""snnTorch lowering frontend."""

import math
import warnings

import torch
from paicorelib import RM, LeakMultiMode
from torch import nn

from ....exceptions import AutoOptimizationWarning
from ...ir.core_neuron import CoreNeuronV25, IFNodeV25, LeakyBeta0NodeV25
from ..support import (
    Constraint,
    ConstraintResult,
    ModuleMapper,
    SourceOp,
    attr,
    eq,
    is_false,
    one_of,
    predicate,
    reader,
    scalar,
    source_op_schema,
)

name = "snntorch"
erase_types: tuple[type[nn.Module], ...] = ()


def detect(model: nn.Module) -> bool:
    """Return whether the model contains any snnTorch-owned module."""
    return any(owns_module(module) for module in model.modules())


def prepare(model: nn.Module) -> nn.Module:
    """snnTorch modules are already instantiated, so no import is needed here."""
    return model


def owns_module(module: nn.Module) -> bool:
    """Return framework ownership, not support status."""
    return any(typ.__module__.startswith("snntorch.") for typ in type(module).__mro__)


def build_module_map() -> ModuleMapper:
    return {}


def describe_unsupported(module: nn.Module) -> ConstraintResult | None:
    if owns_module(module) and not _is_exact_leaky(module):
        return ConstraintResult.fail(
            constraint="supported snnTorch source op",
            field="module",
            value=f"{type(module).__module__}.{type(module).__name__}",
            reason=(
                "only exact snntorch.Leaky is supported; subclass or other "
                "snnTorch module semantics are not assumed equivalent"
            ),
        )
    return None


def validate_source_context(
    source_op: SourceOp, output_shape: torch.Size
) -> ConstraintResult | None:
    """Restrict 1D Leaky parameters to unambiguous per-feature outputs."""
    for field in ("beta", "threshold"):
        value = source_op.attributes.raw(field)
        if torch.is_tensor(value) and value.ndim == 1 and len(output_shape) != 2:
            return ConstraintResult.fail(
                constraint=f"{field} 1D layout",
                field=field,
                value=f"Tensor(shape={tuple(value.shape)})",
                reason=(
                    f"1D {field} is supported only for rank-2 Linear/per-feature "
                    "outputs; Conv tensor broadcasting is not supported"
                ),
            )
    return None


def _scalar_or_1d_numeric(field: str) -> Constraint:
    return predicate(
        f"{field} is scalar or 1D numeric tensor",
        field,
        lambda attrs: _is_scalar_or_1d_numeric(attrs.raw(field)),
        f"{field} must be a scalar or non-empty 1D real-valued Tensor",
    )


def _closed_interval(field: str, lower: float, upper: float) -> Constraint:
    return predicate(
        f"{field} is in [{lower}, {upper}]",
        field,
        lambda attrs: _is_in_closed_interval(attrs.raw(field), lower, upper),
        f"{field} values must be in [{lower}, {upper}]",
    )


def _is_scalar_or_1d_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(
        torch.is_tensor(value)
        and value.ndim <= 1
        and value.numel() > 0
        and not value.is_complex()
        and value.dtype != torch.bool
    )


def _is_in_closed_interval(value: object, lower: float, upper: float) -> bool:
    if torch.is_tensor(value):
        return bool(torch.all((value >= lower) & (value <= upper)).item())
    return lower <= float(value) <= upper


def _snapshot_numeric(value: object) -> int | float | torch.Tensor:
    """Read the current numeric value without preserving training metadata."""
    if torch.is_tensor(value):
        snapshot = value.detach().clone()
        return snapshot.item() if snapshot.ndim == 0 else snapshot
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise TypeError(f"expected numeric value, got {type(value).__name__}")


def _deployment_threshold(source_op: SourceOp) -> tuple[int | torch.Tensor, float]:
    """Convert snnTorch's strict firing threshold to PAICORE's inclusive one.

    snnTorch fires when ``V > T``, while PAICORE compares integer potentials
    using ``V >= T_hw``. Therefore ``T_hw = floor(T) + 1`` preserves the firing
    boundary. The second return value is the maximum ``T_hw - T`` adjustment,
    used to report the extra subtraction introduced by subtract reset.
    """
    value = _snapshot_numeric(source_op.attributes.raw("threshold"))
    if torch.is_tensor(value):
        deployed = torch.floor(value).to(torch.int64) + 1
        return deployed, float(torch.max(deployed - value).item())
    deployed = math.floor(value) + 1
    return deployed, float(deployed - value)


def _beta_to_hardware(
    source_op: SourceOp,
) -> tuple[
    LeakMultiMode | torch.Tensor, int | torch.Tensor, int | torch.Tensor, int, float
]:
    """Encode snnTorch beta values as PAICORE leak mode and shift fields.

    ``beta=0`` maps exactly to complete leakage (multi-leak enabled, shift 0),
    and ``beta=1`` maps exactly to IF behavior (multi-leak disabled, shift 0).
    Each ``0 < beta < 1`` is approximated by the nearest hardware value from
    the direct-shift and retention grids, ``2**-k`` and ``1 - 2**-k`` for
    ``k`` in ``[1, 32]``. A tie selects the retention encoding. The return
    value contains mode, signed shift, equivalent tau, the number of
    approximated elements, and maximum absolute beta error.
    """
    raw_value = _snapshot_numeric(source_op.attributes.raw("beta"))
    beta = torch.as_tensor(raw_value, dtype=torch.float64)

    # Endpoint defaults encode beta=1; beta=0 switches to complete leakage.
    modes = torch.full(beta.shape, int(LeakMultiMode.DISABLE), dtype=torch.int64)
    shifts = torch.zeros(beta.shape, dtype=torch.int64)
    interior = (beta > 0) & (beta < 1)
    modes[beta == 0] = int(LeakMultiMode.ENABLE)

    approximated_count = 0
    max_error = 0.0
    if torch.any(interior).item():
        exponents = torch.arange(1, 33, dtype=torch.int64)
        shifts_grid = -exponents
        retention_grid = 1.0 - torch.pow(
            torch.tensor(2, dtype=torch.float64), -exponents.to(torch.float64)
        )
        direct_grid = torch.pow(
            torch.tensor(2, dtype=torch.float64), -exponents.to(torch.float64)
        )
        grid = torch.cat((retention_grid, direct_grid))
        grid_modes = torch.cat(
            (
                torch.full_like(exponents, int(LeakMultiMode.ENABLE)),
                torch.full_like(exponents, int(LeakMultiMode.DISABLE)),
            )
        )
        grid_shifts = torch.cat((shifts_grid, shifts_grid))
        distances = torch.abs(beta[interior, None] - grid[None, :])
        # argmin retains the first match, preferring retention on ties.
        grid_indices = torch.argmin(distances, dim=1)
        modes[interior] = grid_modes[grid_indices]
        shifts[interior] = grid_shifts[grid_indices]
        errors = distances.gather(1, grid_indices[:, None]).squeeze(1)
        approximated_count = int(torch.count_nonzero(errors).item())
        max_error = float(errors.max().item())

    # Preserve equivalent tau metadata alongside the exact hardware shift.
    tau = torch.bitwise_left_shift(torch.ones_like(shifts, dtype=torch.int64), -shifts)
    if beta.ndim == 0:
        return (
            LeakMultiMode(int(modes.item())),
            int(shifts.item()),
            int(tau.item()),
            approximated_count,
            max_error,
        )
    return modes, shifts, tau, approximated_count, max_error


def _all_equal(value: int | torch.Tensor, expected: int) -> bool:
    if torch.is_tensor(value):
        return bool(torch.all(value == expected).item())
    return value == expected


def _warn_adjustments(
    *,
    reset_mechanism: str,
    approximated_count: int,
    max_error: float,
    max_threshold_adjustment: float,
) -> None:
    """Emit one summary warning for beta approximation and threshold changes."""
    details: list[str] = []
    if approximated_count:
        details.append(
            f"approximated {approximated_count} beta value(s) "
            f"(max absolute error {max_error:.6g})"
        )
    if reset_mechanism == "subtract":
        details.append(
            "raised threshold to floor(threshold)+1 to preserve strict spike "
            "comparison; subtract reset uses the adjusted threshold "
            f"(maximum extra reset {max_threshold_adjustment:.6g})"
        )
    if details:
        warnings.warn(
            "snnTorch Leaky deployment: " + "; ".join(details),
            AutoOptimizationWarning,
            stacklevel=3,
        )


def _map_leaky(source_op: SourceOp) -> CoreNeuronV25:
    """Lower a validated snnTorch Leaky op to a deployable PAIIR neuron."""
    reset_mechanism = str(source_op.attributes.raw("reset_mechanism"))
    threshold, max_threshold_adjustment = _deployment_threshold(source_op)
    mode, shift, tau, approximated_count, max_error = _beta_to_hardware(source_op)
    _warn_adjustments(
        reset_mechanism=reset_mechanism,
        approximated_count=approximated_count,
        max_error=max_error,
        max_threshold_adjustment=max_threshold_adjustment,
    )

    all_if = _all_equal(mode, int(LeakMultiMode.DISABLE)) and _all_equal(shift, 0)
    if all_if and reset_mechanism == "subtract":
        return IFNodeV25(threshold, None)
    if all_if and reset_mechanism == "zero":
        return IFNodeV25(threshold)

    all_beta_zero = _all_equal(mode, int(LeakMultiMode.ENABLE)) and _all_equal(shift, 0)
    if all_beta_zero:
        return LeakyBeta0NodeV25(
            threshold, None if reset_mechanism == "subtract" else 0
        )

    reset_mode = {
        "subtract": RM.MODE_LINEAR,
        "zero": RM.MODE_NORMAL,
        "none": RM.MODE_NONRESET,
    }[reset_mechanism]
    return CoreNeuronV25(
        reset_mode=reset_mode,
        thres_pos=threshold,
        tau=tau,
        leak_tau_shift=shift,
        leak_multi_mode=mode,
    )


def _reset_mechanism(module: nn.Module) -> str:
    value = getattr(module, "reset_mechanism", None)
    if isinstance(value, str):
        return value

    raw = getattr(module, "reset_mechanism_val", None)
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().item()
    mapping = {0: "subtract", 1: "zero", 2: "none"}
    return mapping.get(int(raw), str(value)) if raw is not None else str(value)


def _is_exact_leaky(module: nn.Module) -> bool:
    return (
        type(module).__module__ == "snntorch._neurons.leaky"
        and type(module).__name__ == "Leaky"
    )


source_schemas = (
    source_op_schema("snntorch", "Leaky", recognize=_is_exact_leaky)
    .attribute("reset_delay", attr("reset_delay"))
    .attribute("beta", attr("beta"))
    .attribute("threshold", attr("threshold"))
    .attribute("output", attr("output", default=False))
    .attribute("inhibition", attr("inhibition", default=False))
    .attribute("state_quant", attr("state_quant", default=False))
    .attribute("graded_spikes_factor", attr("graded_spikes_factor", default=1.0))
    .attribute("reset_mechanism", reader(_reset_mechanism))
    .generic(_scalar_or_1d_numeric("beta"))
    .generic(_closed_interval("beta", 0, 1))
    .generic(_scalar_or_1d_numeric("threshold"))
    .generic(
        is_false(
            "output", reason="explicit membrane output is not represented in PAIIR"
        )
    )
    .generic(
        is_false("inhibition", reason="lateral inhibition semantics are not equivalent")
    )
    .generic(
        one_of(
            "state_quant",
            (False, None),
            reason="state quantization is not represented in PAIIR",
        )
    )
    .generic(
        scalar(
            "graded_spikes_factor",
            reason="learnable or non-scalar graded spikes are not supported",
        )
    )
    .generic(
        eq(
            "graded_spikes_factor",
            1,
            reason="graded spike scaling is not represented in PAIIR",
        )
    )
    .generic(
        one_of(
            "reset_mechanism",
            ("subtract", "zero", "none"),
            reason="unsupported snnTorch reset mechanism",
        )
    )
    .case("linear_reset")
    .when(eq("reset_mechanism", "subtract"))
    .when(
        is_false(
            "reset_delay",
            reason="reset_delay=True is not equivalent for subtract reset",
        )
    )
    .canonicalize_to(_map_leaky)
    .case("zero_reset")
    .when(eq("reset_mechanism", "zero"))
    .canonicalize_to(_map_leaky)
    .case("non_reset")
    .when(eq("reset_mechanism", "none"))
    .canonicalize_to(_map_leaky)
    .build(),
)
