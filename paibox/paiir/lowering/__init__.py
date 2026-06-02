"""PAIIR front-end lowering from PyTorch/FX graphs."""

from .converter import register_ir_module, register_module, register_neuron, torch_to_paiir
from .dims_prop import DimsProp, DimsType

__all__ = [
    "DimsProp",
    "DimsType",
    "register_ir_module",
    "register_module",
    "register_neuron",
    "torch_to_paiir",
]
