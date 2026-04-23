"""Shared FX tracing helpers used by PAIIR tests."""

from typing import Any

from torch import Tensor, fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from paibox.paiir.lowering.converter import (
    TRACE_LEAF_MODULE_TYPES,
    _EraseModuleTransformer,
    _get_full_module_map,
    _PAIIRTracer,
    propagate_dims,
    propagate_shapes,
)
from paibox.paiir.lowering.dims_prop import DimsProp


def trace_with_fx_shape_and_dims(
    model: nn.Module,
    *sample_inputs: Tensor,
    concrete_args: dict[str, Any] | None = None,
) -> fx.GraphModule:
    """Trace with vanilla FX, then run ShapeProp and DimsProp."""
    if not sample_inputs:
        raise ValueError(
            "trace_with_fx_shape_and_dims() requires at least one sample input."
        )

    gm = fx.symbolic_trace(model, concrete_args=concrete_args)
    ShapeProp(gm).propagate(*sample_inputs)
    DimsProp().propagate(gm)
    return gm


def trace_with_paiir_tracer(
    model: nn.Module, concrete_args: dict[str, Any] | None = None
) -> fx.GraphModule:
    """Trace with the project-specific PAIIR tracer configuration."""
    leaf_types = tuple(_get_full_module_map().keys()) + TRACE_LEAF_MODULE_TYPES
    tracer = _PAIIRTracer(custom_leaf_modules=leaf_types)
    traced = tracer.trace(model, concrete_args)
    return fx.GraphModule(tracer.root, traced)


def trace_for_lowering(
    model: nn.Module,
    *sample_inputs: Tensor,
    concrete_args: dict[str, Any] | None = None,
) -> fx.GraphModule:
    """Trace with the PAIIR tracer and run lowering pre-analysis propagation."""
    if not sample_inputs:
        raise ValueError("trace_for_lowering() requires at least one sample input.")

    gm = trace_with_paiir_tracer(model, concrete_args=concrete_args)
    gm = _EraseModuleTransformer(gm).transform()
    propagate_shapes(gm, *sample_inputs)
    propagate_dims(gm)
    return gm
