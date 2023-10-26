from .coordinate import Coord as Coord
from .coordinate import ReplicationId as ReplicationId
from .hw_defs import HwConfig as HwConfig
from .ram_model import NeuronSelfConfig as NeuronSelfConfig
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
from .route import RoutingNodeCoord as RoutingNodeCoord

v_major = 2
v_minor = 0
v_revision = 0
version = f"{v_major}.{v_minor}.{v_revision}"
