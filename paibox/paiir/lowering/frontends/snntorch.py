"""snnTorch lowering frontend."""

import torch
from paicorelib import RM
from torch import nn

from ...ir.core_neuron import CoreNeuronV25, IFNodeV25
from ..support import (
    ConstraintResult,
    ModuleMapper,
    SourceOp,
    attr,
    eq,
    is_false,
    one_of,
    reader,
    scalar,
    scalar_value,
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


def _threshold(source_op: SourceOp) -> float:
    return scalar_value(source_op.attributes.raw("threshold"))


def _map_linear_reset(source_op: SourceOp) -> IFNodeV25:
    return IFNodeV25(v_threshold=_threshold(source_op), v_reset=None)


def _map_zero_reset(source_op: SourceOp) -> IFNodeV25:
    return IFNodeV25(v_threshold=_threshold(source_op))


def _map_nonreset(source_op: SourceOp) -> CoreNeuronV25:
    return CoreNeuronV25(reset_mode=RM.MODE_NONRESET, thres_pos=_threshold(source_op))


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
    .generic(scalar("beta", reason="learnable or non-scalar beta is not equivalent"))
    .generic(
        eq(
            "beta",
            1,
            reason=(
                "beta != 1 leaks only old membrane state in snnTorch and needs "
                "IR support"
            ),
        )
    )
    .generic(
        scalar("threshold", reason="learnable or non-scalar threshold is not supported")
    )
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
    .canonicalize_to(_map_linear_reset)
    .case("zero_reset")
    .when(eq("reset_mechanism", "zero"))
    .canonicalize_to(_map_zero_reset)
    .case("non_reset")
    .when(eq("reset_mechanism", "none"))
    .canonicalize_to(_map_nonreset)
    .build(),
)
