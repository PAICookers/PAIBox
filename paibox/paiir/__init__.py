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

# Chip-accurate operators and IR entities (public API)
from .ir import (
    ANNNodeV25,
    AddOperandKind,
    AddOperandSpec,
    AccumulateOp,
    CPUOp,
    ConcatOp,
    CoreNeuronV25,
    Edge,
    GeneralAddOp,
    IFNodeV25,
    InputNode,
    LIFNodeV25,
    LutActivation,
    LutAdaptiveActivation,
    LutCustom,
    LutData,
    LutLinear,
    LutReLU,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
    NeuronParams,
    OfflineCoreOp,
    OfflineCoreParams,
    OnlineCoreOp,
    OnlineCoreParams,
    OpNode,
    OutputNode,
    PAIIRGraph,
    PAIIRNode,
    PotentialAddOp,
    SequentialOp,
    SignalDomain,
    StandaloneActOp,
    StandaloneCompOp,
)

# Data format inference and compile pipeline
from .pipeline import (
    CompileConfig,
    DataFormat,
    compile_to_paiir,
    infer_output_format,
    infer_weight_format,
    merge_data_formats,
)

# Conversion entrypoints
from .lowering import register_neuron, torch_to_paiir

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
    "AddOperandKind",
    "AddOperandSpec",
    "GeneralAddOp",
    "OpNode",
    "OfflineCoreOp",
    "SequentialOp",
    "AccumulateOp",
    "PotentialAddOp",
    "ConcatOp",
    "StandaloneCompOp",
    "StandaloneActOp",
    "SignalDomain",
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
