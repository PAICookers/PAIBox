from .identifier import Coord as Coord
from .identifier import ReplicationId as ReplicationId
from .ram_model import NeuronSelfConfig as NeuronSelfConfig
from .ram_types import LeakingComparisonMode as LCM
from .ram_types import LeakingDirectionMode as LDM
from .ram_types import LeakingIntegrationMode as LIM
from .ram_types import NegativeThresholdMode as NTM
from .ram_types import ResetMode as RM
from .ram_types import SynapticIntegrationMode as SIM
from .ram_types import ThresholdMode as TM
from .reg_model import ParamsReg as ParamsReg
from .reg_types import InputWidthFormatType as InputWidthFormatType
from .reg_types import LCNExtensionType as LCN_EX
from .reg_types import MaxPoolingEnableType as MaxPoolingEnableType
from .reg_types import SNNModeEnableType as SNNModeEnableType
from .reg_types import SpikeWidthFormatType as SpikeWidthFormatType
from .reg_types import WeightPrecisionType as WeightPrecisionType

v_major = 2
v_minor = 0
v_revision = 0
version = f"{v_major}.{v_minor}.{v_revision}"
