"""PAIIR intermediate representation building blocks.

This package contains the core IR entities shared by lowering, compilation,
and backend deployment:

- base graph nodes and naming helpers
- chip-accurate neuron / LUT modules
- deployable operator node wrappers
- the PAIIR graph container
- parameter dataclasses used by deployable nodes
"""

from .calc_params import LutData, NeuronParams, OfflineCoreParams, OnlineCoreParams
from .core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from .graph import Edge, PAIIRGraph
from .ir_base import InputNode, OutputNode, PAIIRNode
from .lut_activation import (
    LutActivation,
    LutAdaptiveActivation,
    LutCustom,
    LutLinear,
    LutReLU,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
)
from .op_node import (
    AccumulateOp,
    AddOp,
    CPUOp,
    ConcatOp,
    OfflineCoreOp,
    OnlineCoreOp,
    OpNode,
    ReshapeOp,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)

__all__ = [
    "ANNNodeV25",
    "AccumulateOp",
    "AddOp",
    "CPUOp",
    "ConcatOp",
    "CoreNeuronV25",
    "Edge",
    "IFNodeV25",
    "InputNode",
    "LIFNodeV25",
    "LutActivation",
    "LutAdaptiveActivation",
    "LutCustom",
    "LutData",
    "LutLinear",
    "LutReLU",
    "LutSigmoid",
    "LutSoftsign",
    "LutTanh",
    "NeuronParams",
    "OfflineCoreOp",
    "OfflineCoreParams",
    "OnlineCoreOp",
    "OnlineCoreParams",
    "OpNode",
    "OutputNode",
    "PAIIRGraph",
    "PAIIRNode",
    "ReshapeOp",
    "SequentialOp",
    "StandaloneActOp",
    "StandaloneCompOp",
]
