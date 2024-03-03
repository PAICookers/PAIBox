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
from .synapses import NoDecay as NoDecay

__all__ = [
    "Mapper",
    "DynSysGroup",
    "Network",
    "NodeDict",
    "NodeList",
    "InputProj",
    "Simulator",
    "Probe",
    "BACKEND_CONFIG",
    "FRONTEND_ENV",
]

from importlib.metadata import version

try:
    __version__ = version("paibox")
except Exception:
    __version__ = None
