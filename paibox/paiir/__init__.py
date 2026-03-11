"""PAIIR -- intermediate representation for chip deployment.

Provides chip-accurate operators, IR nodes, and a computation graph for:

1. Building PyTorch models with exact chip behaviour (v2.0 / v2.5).
2. Representing computation graphs deployable to chip cores.
3. Converting PyTorch models into PAIIR graphs.

The IR is version-agnostic: ``OfflineCoreOp`` and ``OnlineCoreOp``
capture the computation pattern common to all chip generations.
The backend handles target-specific lowering and parameter encoding.

Quick start::

    from paibox.paiir import IFNodeV25, LIFNodeV25, LutReLU

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        LIFNodeV25(tau=2.0, v_threshold=1.0),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10),
        IFNodeV25(v_threshold=0.5),
    )
"""

# Chip-accurate operators (public API)
from .core_neuron import (
    ANNNodeV25,
    CoreNeuronV25,
    IFNodeV25,
    LIFNodeV25,
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

# IR base types
from .ir_base import InputNode, OutputNode, PAIIRNode

# Graph
from .graph import Edge, PAIIRGraph

# Core operator IR nodes
from .op_node import (
    AccumulateOp,
    AddOp,
    ConcatOp,
    CPUOp,
    OfflineCoreOp,
    OnlineCoreOp,
    OpNode,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)

# Parameter definitions
from .calc_params import LutData, NeuronParams, OfflineCoreParams, OnlineCoreParams

# Data format inference
from .data_format import (
    DataFormat,
    infer_output_format,
    infer_weight_format,
    merge_data_formats,
)

# Conversion and passes
from .converter import register_neuron, torch_to_paiir
from .compile import compile_to_paiir, CompileConfig

__all__ = [
    # Chip-accurate operators
    "CoreNeuronV25",
    "ANNNodeV25",
    "IFNodeV25",
    "LIFNodeV25",
    "LutActivation",
    "LutAdaptiveActivation",
    "LutCustom",
    "LutReLU",
    "LutLinear",
    "LutSigmoid",
    "LutTanh",
    "LutSoftsign",
    # IR base types
    "PAIIRNode",
    "InputNode",
    "OutputNode",
    # Graph
    "Edge",
    "PAIIRGraph",
    # Core operator IR nodes
    "OpNode",
    "OfflineCoreOp",
    "SequentialOp",
    "AccumulateOp",
    "AddOp",
    "ConcatOp",
    "StandaloneCompOp",
    "StandaloneActOp",
    "OnlineCoreOp",
    "CPUOp",
    # Parameters
    "LutData",
    "OfflineCoreParams",
    "NeuronParams",
    "OnlineCoreParams",
    # Data format inference
    "DataFormat",
    "infer_output_format",
    "infer_weight_format",
    "merge_data_formats",
    # Conversion and passes
    "register_neuron",
    "torch_to_paiir",
    "compile_to_paiir",
    "CompileConfig",
]
