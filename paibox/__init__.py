from importlib.metadata import version

from .backend import BACKEND_CONFIG as BACKEND_CONFIG
from .backend import Mapper as Mapper
from .base import *
from .context import FRONTEND_ENV as FRONTEND_ENV
from .network import DynSysGroup as DynSysGroup
from .network import Network as Network
from .neuron import *
from .node import NodeDict as NodeDict
from .node import NodeList as NodeList
from .projection import InputProj as InputProj
from .simulator import Probe as Probe
from .simulator import Simulator as Simulator
from .synapses import Conv1d as Conv1d
from .synapses import Conv2d as Conv2d
from .synapses import ConvTranspose1d as ConvTranspose1d
from .synapses import ConvTranspose2d as ConvTranspose2d
from .synapses import FullConn as FullConn
from .synapses import GeneralConnType as SynConnType
from .synapses import NoDecay as NoDecay

try:
    __version__ = version("paibox")
except Exception:
    __version__ = None

from paibox import tools

# Minimum required version of paicorelib
__plib_minimum_version__ = "0.0.13"

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
