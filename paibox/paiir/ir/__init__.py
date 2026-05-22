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
from .core_neuron import (
    ANNNodeV25,
    CoreNeuronV25,
    IFNodeV25,
    LIFNodeV25,
)
from .graph import Edge, PAIIRGraph
from .ir_base import FormatFlow, InputNode, OutputNode, PAIIRNode, TensorLayout
from .signal_domain import SignalDomain, SignalSemantics
from .add_ops import (
    AddOperandKind,
    AddOperandSpec,
    GeneralAddOp,
    PotentialAddOp,
)
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
    CPUOp,
    ConcatOp,
    LayoutStage,
    OfflineCoreOp,
    OnlineCoreOp,
    OpNode,
    RoutingOp,
    SequentialOp,
    ShapeStage,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TransformOp,
)

__all__ = [
    "ANNNodeV25",
    "AddOperandKind",
    "AddOperandSpec",
    "AccumulateOp",
    "CPUOp",
    "ConcatOp",
    "CoreNeuronV25",
    "Edge",
    "FormatFlow",
    "IFNodeV25",
    "InputNode",
    "LIFNodeV25",
    "LayoutStage",
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
    "GeneralAddOp",
    "OfflineCoreOp",
    "OfflineCoreParams",
    "OnlineCoreOp",
    "OnlineCoreParams",
    "OpNode",
    "OutputNode",
    "PAIIRGraph",
    "PAIIRNode",
    "PotentialAddOp",
    "RoutingOp",
    "SequentialOp",
    "ShapeStage",
    "SignalDomain",
    "SignalSemantics",
    "SplitOp",
    "StandaloneActOp",
    "StandaloneCompOp",
    "TensorLayout",
    "TransformOp",
]
