# isort: skip_file

# Frontend context
from .context import FRONTEND_ENV as FRONTEND_ENV

# Backend context & mapper
from .backend import BACKEND_CONFIG as BACKEND_CONFIG
from .backend import Mapper as Mapper

# Functional modules in ANN mode only
from .components.functional import Linear as Linear
from .components.functional import LinearSemiFolded as LinearSemiFolded
from .components.functional import Conv2dSemiFolded as Conv2dSemiFolded
from .components.functional import MaxPool1d as MaxPool1d
from .components.functional import MaxPool2d as MaxPool2d
from .components.functional import MaxPool2dSemiFolded as MaxPool2dSemiFolded
from .components.functional import AvgPool1d as AvgPool1d
from .components.functional import AvgPool2d as AvgPool2d
from .components.functional import AvgPool2dSemiFolded as AvgPool2dSemiFolded

# Functional modules in SNN mode only
from .components.functional import BitwiseAND as BitwiseAND
from .components.functional import BitwiseNOT as BitwiseNOT
from .components.functional import BitwiseOR as BitwiseOR
from .components.functional import BitwiseXOR as BitwiseXOR
from .components.functional import SpikingAdd as SpikingAdd
from .components.functional import SpikingAvgPool1d as SpikingAvgPool1d
from .components.functional import SpikingAvgPool1dWithV as SpikingAvgPool1dWithV
from .components.functional import SpikingAvgPool2d as SpikingAvgPool2d
from .components.functional import SpikingAvgPool2dWithV as SpikingAvgPool2dWithV
from .components.functional import SpikingMaxPool1d as SpikingMaxPool1d
from .components.functional import SpikingMaxPool2d as SpikingMaxPool2d
from .components.functional import SpikingSub as SpikingSub

# Recued neurons in ANN mode only
from .components.neuron import ANNBypassNeuron as ANNBypassNeuron
from .components.neuron import ANNNeuron as ANNNeuron

# Reduced neurons
from .components.neuron import IF as IF
from .components.neuron import LIF as LIF
from .components.neuron import BypassNeuron as BypassNeuron
from .components.neuron import PhasicSpiking as PhasicSpiking
from .components.neuron import TonicSpiking as TonicSpiking
from .components.neuron import StoreVoltageNeuron as StoreVoltageNeuron

# STDP neurons in SNN mode only
from .components.neuron import STDPLIF as STDPLIF

# Input projection
from .components.projection import InputProj as InputProj

# Connection types of synapses
from .components.synapses import ConnType as SynConnType

# Synapses
from .components.synapses import Conv1d as Conv1d
from .components.synapses import Conv2d as Conv2d
from .components.synapses import ConvTranspose1d as ConvTranspose1d
from .components.synapses import ConvTranspose2d as ConvTranspose2d
from .components.synapses import FullConn as FullConn
from .components.synapses import MatMul2d as MatMul2d

# STDP synapses
from .components.synapses import STDPFullConn as STDPFullConn

# Network
from .network import DynSysGroup as DynSysGroup
from .network import Network as Network  # alias for DynSysGroup

# Auxiliary containers
from .node import NodeDict as NodeDict
from .node import NodeList as NodeList

# Simulation
from .simulator import Probe as Probe
from .simulator import Simulator as Simulator

# Log
from . import _logging

# Version
from importlib.metadata import version
from .tools import PLIB_INSTALL_INTRO, PLIB_UPDATE_INTRO

_logging._init_logs()


try:
    __version__ = version("paibox")
except Exception:
    __version__ = None

# Minimum required version of paicorelib
__plib_minimum_version__ = "1.5.0"

try:
    import paicorelib as plib

    if plib.__version__ < __plib_minimum_version__:
        raise ImportError(
            PLIB_UPDATE_INTRO.format(__plib_minimum_version__, plib.__version__)
        ) from None

    del plib

except ModuleNotFoundError:
    raise ModuleNotFoundError(PLIB_INSTALL_INTRO) from None
