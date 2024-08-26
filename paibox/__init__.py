from importlib.metadata import version

from .backend import BACKEND_CONFIG as BACKEND_CONFIG
from .backend import Mapper as Mapper

# Functional modules
from .components.functional import AvgPool2dSemiMap as AvgPool2dSemiMap
from .components.functional import BitwiseAND as BitwiseAND
from .components.functional import BitwiseNOT as BitwiseNOT
from .components.functional import BitwiseOR as BitwiseOR
from .components.functional import BitwiseXOR as BitwiseXOR
from .components.functional import Conv2dSemiMap as Conv2dSemiMap
from .components.functional import Delay_FullConn as DelayFullConn
from .components.functional import DelayChain as DelayChain
from .components.functional import Filter as Filter
from .components.functional import Linear as Linear
from .components.functional import MaxPool2dSemiMap as MaxPool2dSemiMap
from .components.functional import SpikingAdd as SpikingAdd
from .components.functional import SpikingAvgPool1d as SpikingAvgPool1d
from .components.functional import SpikingAvgPool1dWithV as SpikingAvgPool1dWithV
from .components.functional import SpikingAvgPool2d as SpikingAvgPool2d
from .components.functional import SpikingAvgPool2dWithV as SpikingAvgPool2dWithV
from .components.functional import SpikingMaxPool1d as SpikingMaxPool1d
from .components.functional import SpikingMaxPool2d as SpikingMaxPool2d
from .components.functional import SpikingSub as SpikingSub
from .components.functional import Transpose2d as Transpose2d
from .components.functional import Transpose3d as Transpose3d

# Reduced neurons
from .components.neuron.neurons import IF as IF
from .components.neuron.neurons import LIF as LIF
from .components.neuron.neurons import PhasicSpiking as PhasicSpiking
from .components.neuron.neurons import SpikingRelu as SpikingRelu
from .components.neuron.neurons import TonicSpiking as TonicSpiking

# Input projection
from .components.projection import InputProj as InputProj

# Synapses
from .components.synapses import ConnType as SynConnType
from .components.synapses.synapses import Conv1d as Conv1d
from .components.synapses.synapses import Conv2d as Conv2d
from .components.synapses.synapses import ConvTranspose1d as ConvTranspose1d
from .components.synapses.synapses import ConvTranspose2d as ConvTranspose2d
from .components.synapses.synapses import FullConn as FullConn
from .components.synapses.synapses import MatMul2d as MatMul2d
from .context import FRONTEND_ENV as FRONTEND_ENV
from .network import DynSysGroup as DynSysGroup
from .network import Network as Network
from .node import NodeDict as NodeDict
from .node import NodeList as NodeList
from .simulator import Probe as Probe
from .simulator import Simulator as Simulator

try:
    __version__ = version("paibox")
except Exception:
    __version__ = None

from paibox import tools

# Minimum required version of paicorelib
__plib_minimum_version__ = "1.3.0"

try:
    import paicorelib as plib

    if plib.__version__ < __plib_minimum_version__:  # For paicorelib > 0.0.12
        raise ImportError(
            tools.PLIB_UPDATE_INTRO.format(__plib_minimum_version__, plib.__version__)
        ) from None

    del tools, plib

except ModuleNotFoundError:
    raise ModuleNotFoundError(tools.PLIB_INSTALL_INTRO) from None
