from .coordinate import Coord as Coord
from .coordinate import ReplicationId as ReplicationId
from .hw_defs import HwConfig as HwConfig
from .hw_types import (
    AxonCoord as AxonCoord,
    AxonSegment as AxonSegment,
    NeuronSegment as NeuronSegment,
)
from .ram_model import NeuronAttrs as NeuronAttrs
from .ram_model import NeuronDestInfo as NeuronDestInfo
from .ram_types import LeakingComparisonMode as LCM
from .ram_types import LeakingDirectionMode as LDM
from .ram_types import LeakingIntegrationMode as LIM
from .ram_types import NegativeThresholdMode as NTM
from .ram_types import ResetMode as RM
from .ram_types import SynapticIntegrationMode as SIM
from .ram_types import ThresholdMode as TM
from .reg_model import ParamsReg as ParamsReg
from .reg_types import InputWidthFormatType as InputWidthFormat
from .reg_types import LCNExtensionType as LCN_EX
from .reg_types import MaxPoolingEnableType as MaxPoolingEnable
from .reg_types import SNNModeEnableType as SNNModeEnable
from .reg_types import SpikeWidthFormatType as SpikeWidthFormat
from .reg_types import WeightPrecisionType as WeightPrecision
from .routing_defs import RoutingNodeCoord as RoutingNodeCoord, get_replication_id

v_major = 2
v_minor = 0
v_revision = 0
version = f"{v_major}.{v_minor}.{v_revision}"
