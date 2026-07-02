"""SpikingJelly lowering frontend."""

import warnings
from collections.abc import Callable
from functools import cache

from spikingjelly.activation_based import base, layer, neuron
from spikingjelly.activation_based import functional as sj_F
from torch import nn

from paibox.paiir.lowering.support import SourceOpSchema

from ...ir.core_neuron import IFNodeV25, LIFNodeV25
from ...ir.op_node import StandaloneCompOp
from ..support import (
    ConstraintResult,
    ModuleMapper,
    SourceOp,
    attr,
    eq,
    gt,
    is_bool,
    one_of,
    scalar,
    scalar_or_1d_tensor,
    scalar_or_none,
    scalar_value,
    source_op_schema,
)

name = "spikingjelly"
erase_types = (layer.Dropout, layer.Dropout2d)

_SUPPORTED_LAYER_COMP_TYPES = (
    layer.Conv1d,
    layer.Conv2d,
    layer.Linear,
    layer.MaxPool1d,
    layer.MaxPool2d,
    layer.AvgPool1d,
    layer.AvgPool2d,
    layer.AdaptiveAvgPool1d,
    layer.AdaptiveAvgPool2d,
    layer.VotingLayer,
)

_SUPPORTED_LAYER_TYPES = (*_SUPPORTED_LAYER_COMP_TYPES, layer.Flatten)


@cache
def _legacy_neuron_types() -> tuple[type[nn.Module], ...]:
    try:
        from spikingjelly.clock_driven import neuron as legacy_neuron
    except ImportError:  # pragma: no cover - depends on installed SJ version
        return ()

    return (legacy_neuron.IFNode, legacy_neuron.LIFNode)


def detect(model: nn.Module) -> bool:
    """Return whether the model contains any SpikingJelly-owned module."""
    return any(owns_module(module) for module in model.modules())


def prepare(model: nn.Module) -> nn.Module:
    """Normalize SpikingJelly modules to single-step execution for tracing."""
    multi_step_modules = [
        type(m).__name__
        for m in model.modules()
        if isinstance(m, base.StepModule) and m.step_mode != "s"
    ]
    if multi_step_modules:
        warnings.warn(
            f"Model contains SpikingJelly modules with step_mode='m' "
            f"({', '.join(multi_step_modules)}). "
            f"step_mode has been set to 's' on the internal model copy for "
            f"chip deployment. The original model is not modified.",
            UserWarning,
            stacklevel=3,
        )
        sj_F.set_step_mode(model, "s")
    return model


def owns_module(module: nn.Module) -> bool:
    """Return framework ownership, not support status."""
    if type(module).__module__.startswith("paibox."):
        return False
    return any(
        typ.__module__.startswith(
            ("spikingjelly.activation_based.", "spikingjelly.clock_driven.")
        )
        for typ in type(module).__mro__
    )


def describe_unsupported(module: nn.Module) -> ConstraintResult | None:
    if not owns_module(module) or _is_supported_module(module):
        return None
    return ConstraintResult.fail(
        constraint="supported SpikingJelly source op or layer",
        field="module",
        value=describe(module),
        reason="unsupported SpikingJelly module",
    )


def describe(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__name__}"


def _map_comp(m: nn.Module, **kwargs) -> StandaloneCompOp:
    return StandaloneCompOp(m, **kwargs)


def _map_voting_layer(m: nn.Module, **kwargs) -> StandaloneCompOp:
    return _map_comp(nn.AvgPool1d(m.voting_size, m.voting_size), **kwargs)  # type: ignore


def build_module_map() -> ModuleMapper:
    module_map: ModuleMapper = dict.fromkeys(_SUPPORTED_LAYER_COMP_TYPES, _map_comp)
    module_map[layer.VotingLayer] = _map_voting_layer
    return module_map


def _is_supported_module(module: nn.Module) -> bool:
    return (
        type(module) in _SUPPORTED_LAYER_TYPES
        or isinstance(module, erase_types)
        or _is_activation_ifnode(module)
        or _is_activation_lifnode(module)
        or _is_legacy_ifnode(module)
        or _is_legacy_lifnode(module)
    )


def _is_activation_ifnode(module: nn.Module) -> bool:
    return type(module) is neuron.IFNode


def _is_activation_lifnode(module: nn.Module) -> bool:
    return type(module) is neuron.LIFNode


def _is_legacy_ifnode(module: nn.Module) -> bool:
    legacy_types = _legacy_neuron_types()
    return bool(legacy_types) and type(module) is legacy_types[0]


def _is_legacy_lifnode(module: nn.Module) -> bool:
    legacy_types = _legacy_neuron_types()
    return len(legacy_types) == 2 and type(module) is legacy_types[1]


def _warn_legacy_ifnode() -> None:
    warnings.warn(
        (
            "SpikingJelly's legacy `spikingjelly.clock_driven.neuron.IFNode` usage "
            "is deprecated; please migrate to "
            "`spikingjelly.activation_based.neuron.IFNode`."
        ),
        DeprecationWarning,
        stacklevel=6,
    )


def _warn_legacy_lifnode() -> None:
    warnings.warn(
        (
            "SpikingJelly's legacy `spikingjelly.clock_driven.neuron.LIFNode` usage "
            "is deprecated; please migrate to "
            "`spikingjelly.activation_based.neuron.LIFNode`."
        ),
        DeprecationWarning,
        stacklevel=6,
    )


def _map_ifnode(source_op: SourceOp) -> IFNodeV25:
    attrs = source_op.attributes
    if _is_legacy_ifnode(source_op.module):
        _warn_legacy_ifnode()
    return IFNodeV25(
        attrs.raw("v_threshold"),
        attrs.raw("v_reset"),
        attrs.raw("surrogate_function"),
        attrs.raw("detach_reset"),
    )


def _map_lifnode(source_op: SourceOp) -> LIFNodeV25:
    attrs = source_op.attributes
    if _is_legacy_lifnode(source_op.module):
        _warn_legacy_lifnode()
    return LIFNodeV25(
        tau=scalar_value(attrs.raw("tau")),
        decay_input=bool(attrs.raw("decay_input")),
        v_threshold=attrs.raw("v_threshold"),
        v_reset=attrs.raw("v_reset"),
        surrogate_function=attrs.raw("surrogate_function"),
        detach_reset=attrs.raw("detach_reset"),
    )


def _ifnode_schema(recognize_fn: Callable[[nn.Module], bool]) -> SourceOpSchema:
    return (
        source_op_schema("spikingjelly", "IFNode", recognize=recognize_fn)
        .attribute("v_threshold", attr("v_threshold"))
        .attribute("v_reset", attr("v_reset"))
        .attribute("surrogate_function", attr("surrogate_function"))
        .attribute("detach_reset", attr("detach_reset"))
        .attribute("step_mode", attr("step_mode", default="s"))
        .generic(
            scalar_or_1d_tensor(
                "v_threshold", reason="v_threshold must be a scalar or 1D Tensor"
            )
        )
        .generic(
            scalar_or_none(
                "v_reset", reason="v_reset must be None or a non-learnable scalar"
            )
        )
        .generic(
            eq("step_mode", "s", reason="step_mode must be 's' after frontend prepare")
        )
        .generic(is_bool("detach_reset", reason="detach_reset must be bool"))
        .case("ifnode")
        .canonicalize_to(_map_ifnode)
        .build()
    )


def _lifnode_schema(recognize_fn: Callable[[nn.Module], bool]) -> SourceOpSchema:
    return (
        source_op_schema("spikingjelly", "LIFNode", recognize=recognize_fn)
        .attribute("tau", attr("tau"))
        .attribute("decay_input", attr("decay_input"))
        .attribute("v_threshold", attr("v_threshold"))
        .attribute("v_reset", attr("v_reset"))
        .attribute("surrogate_function", attr("surrogate_function"))
        .attribute("detach_reset", attr("detach_reset"))
        .attribute("step_mode", attr("step_mode", default="s"))
        .generic(scalar("tau", reason="tau must be a non-learnable scalar"))
        .generic(gt("tau", 1.0, reason="tau must be > 1 for LIFNodeV25"))
        .generic(
            one_of(
                "decay_input",
                (True, False, 0, 1, 0.0, 1.0),
                reason="decay_input must be bool or legacy numeric 0/1",
            )
        )
        .generic(
            scalar_or_1d_tensor(
                "v_threshold", reason="v_threshold must be a scalar or 1D Tensor"
            )
        )
        .generic(
            scalar_or_none(
                "v_reset", reason="v_reset must be None or a non-learnable scalar"
            )
        )
        .generic(
            eq("step_mode", "s", reason="step_mode must be 's' after frontend prepare")
        )
        .generic(is_bool("detach_reset", reason="detach_reset must be bool"))
        .case("lifnode")
        .canonicalize_to(_map_lifnode)
        .build()
    )


source_schemas = (
    _ifnode_schema(_is_activation_ifnode),
    _lifnode_schema(_is_activation_lifnode),
    _ifnode_schema(_is_legacy_ifnode),
    _lifnode_schema(_is_legacy_lifnode),
)
