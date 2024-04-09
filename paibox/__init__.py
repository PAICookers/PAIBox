from importlib.metadata import version

from .backend import BACKEND_CONFIG as BACKEND_CONFIG
from .backend import Mapper as Mapper
from .base import *
from .components.functional import (
    BitwiseAND as BitwiseAND,
    BitwiseNOT as BitwiseNOT,
    BitwiseOR as BitwiseOR,
    BitwiseXOR as BitwiseXOR,
    DelayChain as DelayChain,
    SpikingAdd as SpikingAdd,
    SpikingAvgPool2d as SpikingAvgPool2d,
    SpikingMaxPool2d as SpikingMaxPool2d,
    SpikingSub as SpikingSub,
    Transpose2d as Transpose2d,
    Transpose3d as Transpose3d,
)
from .components.neuron.neurons import (
    Always1Neuron as Always1Neuron,
    IF as IF,
    LIF as LIF,
    PhasicSpiking as PhasicSpiking,
    SpikingRelu as SpikingRelu,
    TonicSpiking as TonicSpiking,
)
from .components.projection import InputProj as InputProj
from .components.synapses.synapses import (
    Conv1d as Conv1d,
    Conv2d as Conv2d,
    FullConn as FullConn,
    NoDecay as NoDecay,
    GConnType as SynConnType,
)
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
__plib_minimum_version__ = "1.0.0"

try:
    import paicorelib as plib

    if hasattr(plib, "get_version"):  # For plib <= 0.0.12
        raise ImportError(
            tools.PLIB_UPDATE_INTRO.format(
                __plib_minimum_version__, ".".join(map(str, plib.get_version()))  # type: ignore
            )
        ) from None

    if plib.__version__ < __plib_minimum_version__:  # For plib > 0.0.12
        raise ImportError(
            tools.PLIB_UPDATE_INTRO.format(__plib_minimum_version__, plib.__version__)
        ) from None

    del tools, plib

except ModuleNotFoundError:
    raise ModuleNotFoundError(tools.PLIB_INSTALL_INTRO) from None
