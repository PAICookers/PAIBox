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
    CoreNeuronV25,
    IFNodeV25,
    LUT_TABLE_SIZE,
    LayoutStage,
    LIFNodeV25,
    LutActivation,
    LutAdaptiveActivation,
    LutCustom,
    LutLinear,
    LutReLU,
    LutSigmoid,
    LutSoftsign,
    LutTanh,
    PAIIRGraph,
    ShapeStage,
    TransformOp,
)

# Compile entrypoints
from .pipeline import CompileConfig, compile_to_paiir

# Conversion entrypoints
from .lowering import register_module, register_neuron, torch_to_paiir

__all__ = [
    # Chip-accurate operators
    "CoreNeuronV25",
    "ANNNodeV25",
    "IFNodeV25",
    "LIFNodeV25",
    "LUT_TABLE_SIZE",
    "LutActivation",
    "LutAdaptiveActivation",
    "LutCustom",
    "LayoutStage",
    "LutReLU",
    "LutLinear",
    "LutSigmoid",
    "LutTanh",
    "LutSoftsign",
    "ShapeStage",
    "TransformOp",
    # Graph
    "PAIIRGraph",
    # Conversion and passes
    "register_module",
    "register_neuron",
    "torch_to_paiir",
    "compile_to_paiir",
    "CompileConfig",
]
